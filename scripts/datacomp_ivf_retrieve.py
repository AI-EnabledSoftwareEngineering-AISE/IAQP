#!/usr/bin/env python3
"""
DataComp FAISS-GPU IVF retrieval script (self-contained, no project imports).

Uses only original CLIP ViT-B/32 (512-d) from datacomp_features.pkl (from
datacomp_reader.py). No PCA. Loads features pkl, L2-normalizes, builds/loads
FAISS-GPU IVF, runs retrieval, returns top-k with image file paths.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import faiss
except Exception:
    faiss = None

# ---------- Local helpers (no imports from projector) ----------
def l2n_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize rows."""
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)).astype(np.float32)


def _sha1(x: bytes) -> str:
    return hashlib.sha1(x).hexdigest()


def bank_fingerprint(X_bank: np.ndarray) -> str:
    """Stable id for bank (shape + content). No PCA."""
    N, D = X_bank.shape
    buf = np.ascontiguousarray(X_bank, dtype=np.float32).tobytes()
    sample = buf[:10**6] + buf[-10**6:] if len(buf) > 2_000_000 else buf
    h = _sha1(sample + str((N, D)).encode())
    return f"N{N}_D{D}_{h[:12]}"


def suggested_nlist(N: int, hint: int) -> int:
    """Auto-scale ivf_nlist based on dataset size."""
    sug = max(16, int(4 * math.sqrt(N)))
    if hint and hint > 0:
        return min(sug, int(hint))
    return sug


def resolve_nlist(N: int, hint: int) -> int:
    return max(16, min(int(4 * math.sqrt(N)), int(hint)))


# Original 512-d features from datacomp_reader.py (no PCA)
DEF_FEATURES_PKL = "/ssd/hamed/ann/datacomp_small/datacomp_features.pkl"
DEF_INDICES_DIR = "/ssd/hamed/ann/datacomp_small/precompute/indexes"
DEF_OUTPUT = "outputs/datacomp_ivf_retrieval.jsonl"
NUM_TEST_QUERIES = 10
ADD_CHUNK_SIZE = 500_000  # vectors per chunk for GPU add (avoid OOM)

# Test captions for --test: encoded with CLIP ViT-B/32 (same as DataComp), then L2-normalized.
TEST_CAPTIONS = [
    "a dog running on the beach at sunset",
    "red sports car on a mountain road",
    "woman reading a book in a coffee shop",
    "plate of pasta with basil and tomatoes",
    "snowy mountain peak under blue sky",
    "child playing with a yellow balloon",
    "modern kitchen with white countertops",
    "cat sleeping on a windowsill",
    "city skyline at night with lights",
    "person riding a bicycle in the park",
]


def get_image_path_from_uid(uid: Any, images_dir: Path) -> Path:
    """Resolve image path from UID (tries .jpg, .jpeg, .png, .webp)."""
    uid_str = str(uid)
    if hasattr(uid, "item"):
        uid_str = str(uid.item())
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        candidate = images_dir / f"{uid_str}{ext}"
        if candidate.exists():
            return candidate
    return images_dir / f"{uid_str}.jpg"


def load_features_pkl(features_path: str) -> Tuple[np.ndarray, List[Any], List[Any], np.ndarray]:
    """
    Load original 512-d CLIP from datacomp_features.pkl (from datacomp_reader.py).
    Returns (X_bank, uids_list, texts, T_all) all L2-normalized. No PCA.
    """
    import pickle

    path = Path(features_path)
    if not path.exists():
        raise FileNotFoundError(f"Features pkl not found: {features_path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    uids = data["uid"]
    texts = data["text"]
    b32_img = np.asarray(data["b32_img"], dtype=np.float32)
    b32_txt = np.asarray(data["b32_txt"], dtype=np.float32)
    if "image_exists" in data:
        mask = np.asarray(data["image_exists"], dtype=bool)
        uids = uids[mask]
        texts = texts[mask]
        b32_img = b32_img[mask]
        b32_txt = b32_txt[mask]
    n = len(uids)
    assert b32_img.shape[0] == n and b32_img.shape[1] == 512, f"b32_img shape {b32_img.shape}"
    assert b32_txt.shape[0] == n and b32_txt.shape[1] == 512, f"b32_txt shape {b32_txt.shape}"
    uids_list = uids.tolist() if hasattr(uids, "tolist") else list(uids)
    texts_list = [str(t) for t in (texts.tolist() if hasattr(texts, "tolist") else list(texts))]
    X_bank = l2n_np(b32_img)
    T_all = l2n_np(b32_txt)
    return X_bank, uids_list, texts_list, T_all


def build_bank_paths(uids: List[Any], images_dir: Path) -> List[str]:
    """Build list of image paths for each bank index."""
    return [
        str(get_image_path_from_uid(uid, images_dir))
        for uid in tqdm(uids, desc="Building image paths", unit="img")
    ]


def _ivf_build_gpu_train_add(
    X_bank: np.ndarray,
    nlist: int,
    add_chunk_size: int = ADD_CHUNK_SIZE,
) -> Any:
    """Build IVF on GPU: train on GPU, add vectors on GPU in chunks. Returns GPU index."""
    N, D = X_bank.shape
    Xf = X_bank.astype(np.float32, order="C")
    quant = faiss.IndexFlatIP(D)
    cpu = faiss.IndexIVFFlat(quant, D, int(nlist), faiss.METRIC_INNER_PRODUCT)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu)
    tqdm.write("    Training IVF index on GPU...")
    gpu_index.train(Xf)
    tqdm.write("    Adding vectors to IVF index on GPU...")
    for start in tqdm(
        range(0, N, add_chunk_size),
        desc="    Add vectors (GPU)",
        unit="chunk",
    ):
        end = min(N, start + add_chunk_size)
        chunk = Xf[start:end]
        gpu_index.add(chunk)
    return gpu_index


def get_or_build_ivf(
    X_bank: np.ndarray,
    indices_dir: str,
    bank_fp: str,
    nlist_hint: int,
    force_cpu: bool = False,
    num_threads: int = 0,
) -> Any:
    """Build or load IVF index. Build path: train on GPU, add on GPU (chunked), save CPU copy to disk, return GPU index."""
    os.makedirs(indices_dir, exist_ok=True)
    N, D = X_bank.shape
    nrm = np.linalg.norm(X_bank, axis=1)
    if not np.allclose(nrm, 1.0, atol=1e-4):
        raise ValueError("X_bank must be L2-normalized for IVF (IP metric).")
    nlist = resolve_nlist(N, nlist_hint)
    tqdm.write(f"Building IVF index with nlist={nlist}")

    meta = {
        "lib": "faiss",
        "type": "IVFFlat",
        "metric": "IP",
        "nlist": int(nlist),
        "dtype": "f32",
        "N": int(N),
        "D": int(D),
    }
    try:
        meta["faiss_ver"] = getattr(faiss, "__version__", "unknown")
    except Exception:
        pass
    fp = _sha1((bank_fp + json.dumps(meta, sort_keys=True)).encode())
    fname = f"ivf_{bank_fp}_{fp[:8]}.faiss"
    meta_name = f"ivf_{bank_fp}_{fp[:8]}.json"
    cpu_path = os.path.join(indices_dir, fname)
    meta_path = os.path.join(indices_dir, meta_name)

    # Try load from disk
    if os.path.exists(cpu_path):
        try:
            cpu_idx = faiss.read_index(cpu_path)
            if cpu_idx.d == D and cpu_idx.ntotal == N:
                if not force_cpu and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
                    res = faiss.StandardGpuResources()
                    gpu_idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
                    tqdm.write("    Loaded index from disk, moved to GPU.")
                    return gpu_idx
                tqdm.write("    Loaded index from disk.")
                return cpu_idx
            cpu_idx = None
            os.remove(cpu_path)
        except Exception:
            try:
                os.remove(cpu_path)
            except Exception:
                pass

    # Build: GPU train + GPU add
    use_gpu = (
        not force_cpu
        and hasattr(faiss, "get_num_gpus")
        and faiss.get_num_gpus() > 0
    )
    if use_gpu:
        try:
            gpu_index = _ivf_build_gpu_train_add(X_bank, nlist)
            cpu_idx = faiss.index_gpu_to_cpu(gpu_index)
            try:
                faiss.write_index(cpu_idx, cpu_path)
                with open(meta_path, "w") as f:
                    json.dump({**meta, "bank_fp": bank_fp, "file": fname}, f)
            except Exception:
                pass
            return gpu_index
        except Exception as e:
            tqdm.write(f"  GPU build failed ({e}), fallback to CPU")
            use_gpu = False

    # CPU build
    if num_threads > 0:
        faiss.omp_set_num_threads(num_threads)
        tqdm.write(f"    Using {num_threads} threads for FAISS")
    else:
        max_threads = os.cpu_count() or 1
        faiss.omp_set_num_threads(max_threads)
        tqdm.write(f"    Using all {max_threads} CPU cores for FAISS")
    quant = faiss.IndexFlatIP(D)
    cpu = faiss.IndexIVFFlat(quant, D, int(nlist), faiss.METRIC_INNER_PRODUCT)
    Xf = X_bank.astype(np.float32, order="C")
    cpu.train(Xf)
    tqdm.write("    Adding vectors to IVF index on CPU...")
    for start in tqdm(range(0, N, ADD_CHUNK_SIZE), desc="    Add vectors (CPU)", unit="chunk"):
        end = min(N, start + ADD_CHUNK_SIZE)
        cpu.add(Xf[start:end])
    try:
        faiss.write_index(cpu, cpu_path)
        with open(meta_path, "w") as f:
            json.dump({**meta, "bank_fp": bank_fp, "file": fname}, f)
    except Exception:
        pass
    if not force_cpu and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, cpu)
    return cpu


def build_ivf_in_memory(
    X_bank: np.ndarray,
    nlist: int,
) -> Any:
    """Build FAISS IVF in memory: GPU train + GPU add (chunked), then return GPU index; else CPU."""
    if faiss is None:
        raise RuntimeError("faiss not installed")
    N, D = X_bank.shape
    nrm = np.linalg.norm(X_bank, axis=1)
    if not np.allclose(nrm, 1.0, atol=1e-4):
        raise ValueError("Bank must be L2-normalized for IVF (IP metric).")
    try:
        if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
            return _ivf_build_gpu_train_add(X_bank, nlist)
    except Exception as e:
        tqdm.write(f"  GPU build failed ({e}), fallback to CPU")
    # CPU fallback: train on CPU, add on CPU
    quant = faiss.IndexFlatIP(D)
    cpu = faiss.IndexIVFFlat(quant, D, int(nlist), faiss.METRIC_INNER_PRODUCT)
    Xf = X_bank.astype(np.float32, order="C")
    cpu.train(Xf)
    tqdm.write("    Adding vectors to IVF index on CPU...")
    cpu.add(Xf)
    return cpu


def load_queries_from_file(path_str: str) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Load query matrix from .npy or .npz; L2-normalize. No texts."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path_str}")
    if path.suffix == ".npy":
        Q = np.load(path).astype(np.float32)
    else:
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            keys = [k for k in loaded.files if not k.startswith("_")]
            Q = loaded["queries"] if "queries" in keys else loaded[keys[0]]
        else:
            Q = loaded
        Q = np.asarray(Q, dtype=np.float32)
    if Q.ndim != 2:
        raise ValueError(f"Query matrix must be 2D [Q, D], got shape {Q.shape}")
    Q = l2n_np(Q)
    return Q, None


def search_ivf_batch(
    index: Any,
    Q: np.ndarray,
    k: int,
    nprobe: int,
    batch_size: int,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run batch IVF search; returns (indices [Q,k], scores [Q,k])."""
    nlist = getattr(index, "nlist", nprobe)
    nprobe = max(1, min(nprobe, nlist))
    try:
        index.nprobe = nprobe
    except Exception:
        pass
    nq = Q.shape[0]
    all_I = []
    all_D = []
    starts = list(range(0, nq, batch_size))
    it = tqdm(starts, desc="Search batches", unit="batch") if show_progress else starts
    for start in it:
        end = min(nq, start + batch_size)
        Q_batch = Q[start:end].astype(np.float32, order="C")
        D_batch, I_batch = index.search(Q_batch, k)
        all_I.append(I_batch)
        all_D.append(D_batch)
    I = np.vstack(all_I)
    D = np.vstack(all_D)
    return I, D


def run_retrieval(
    index: Any,
    Q: np.ndarray,
    k: int,
    nprobe: int,
    batch_size: int,
    bank_paths: List[str],
    uids: List[Any],
    query_texts: Optional[List[str]] = None,
    show_progress: bool = True,
) -> List[dict]:
    """Run retrieval and return list of per-query result dicts with image paths."""
    I, D = search_ivf_batch(index, Q, k, nprobe, batch_size, show_progress=show_progress)
    nq = Q.shape[0]
    out = []
    it = tqdm(range(nq), desc="Building result rows", unit="query") if show_progress else range(nq)
    for i in it:
        row = {
            "query_id": int(i),
            "results": [],
        }
        if query_texts is not None and i < len(query_texts):
            row["query_text"] = str(query_texts[i])
        for r in range(I.shape[1]):
            idx = int(I[i, r])
            score = float(D[i, r])
            path = bank_paths[idx] if idx < len(bank_paths) else ""
            uid = uids[idx] if idx < len(uids) else None
            if hasattr(uid, "item"):
                uid = uid.item()
            row["results"].append({
                "rank": r + 1,
                "score": score,
                "image_path": path,
                "uid": uid,
            })
        out.append(row)
    return out


def write_output(results: List[dict], output_path: str, fmt: str) -> None:
    """Write results to JSON or JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for row in tqdm(results, desc="Writing output", unit="row"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    tqdm.write(f"Wrote {len(results)} queries to {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DataComp FAISS-GPU IVF: index + batch retrieval with image paths"
    )
    p.add_argument(
        "--features_pkl",
        type=str,
        default=DEF_FEATURES_PKL,
        help=f"Path to datacomp_features.pkl (original 512-d from datacomp_reader.py). Default: {DEF_FEATURES_PKL}",
    )
    p.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images (UID-based filenames)",
    )
    p.add_argument(
        "--queries",
        type=str,
        default="",
        help="Path to .npy/.npz query matrix [Q, 512] (required when not using --test)",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of results per query",
    )
    p.add_argument(
        "--nprobe",
        type=int,
        default=50,
        help="IVF nprobe for search",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Query batch size for search",
    )
    p.add_argument(
        "--output",
        type=str,
        default=DEF_OUTPUT,
        help="Output file path (JSON or JSONL)",
    )
    p.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl"],
        default="jsonl",
        help="Output format",
    )
    p.add_argument(
        "--indices_dir",
        type=str,
        default=DEF_INDICES_DIR,
        help=f"Directory to save/load IVF index (default: {DEF_INDICES_DIR}). Set to empty to build in memory only.",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help="Run on 10 fixed caption queries (CLIP ViT-B/32)",
    )
    p.add_argument(
        "--ivf_nlist",
        type=int,
        default=None,
        help="IVF nlist override (default: auto from bank size)",
    )
    return p.parse_args()


def encode_test_captions_with_clip(
    captions: List[str],
    device: str = "cuda",
) -> Tuple[np.ndarray, List[str]]:
    """Encode caption strings with CLIP ViT-B/32 (same as DataComp), L2-normalize, return (Q, captions)."""
    try:
        import torch
        import clip
    except ImportError as e:
        raise ImportError(
            "For --test with CLIP encoding, install: pip install torch clip (or open_clip_torch). "
            f"Missing: {e}"
        ) from e
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    tokenize = clip.tokenize
    texts = captions
    # Encode in one batch if small, else chunk
    batch_size = 64
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            toks = tokenize(batch, truncate=True).to(device)
            feats = model.encode_text(toks)
            feats = feats.float()
            all_feats.append(feats.cpu().numpy())
    Q = np.vstack(all_feats).astype(np.float32)
    Q = l2n_np(Q)
    return Q, texts


def generate_test_queries(dim: int, num: int = NUM_TEST_QUERIES, seed: int = 42) -> Tuple[np.ndarray, List[str]]:
    """Generate num random unit vectors and placeholder labels (fallback if CLIP not used)."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((num, dim)).astype(np.float32)
    Q = l2n_np(Q)
    texts = [f"test query {i}" for i in range(num)]
    return Q, texts


def main() -> None:
    args = parse_args()
    if faiss is None:
        print("ERROR: faiss not installed. Install faiss-gpu or faiss-cpu.")
        sys.exit(1)

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"ERROR: images_dir is not a directory: {args.images_dir}")
        sys.exit(1)

    use_index_dir = args.indices_dir and args.indices_dir.strip()

    with tqdm(total=1, desc="Loading features (original 512-d)", unit="step") as pbar:
        t0 = time.time()
        X_bank, uids_list, _, _ = load_features_pkl(args.features_pkl)
        N, D = X_bank.shape
        pbar.set_postfix_str(f"N={N:,} D={D} ({time.time() - t0:.2f}s)")
        pbar.update(1)

    bank_paths = build_bank_paths(uids_list, images_dir)

    nlist_hint = args.ivf_nlist if args.ivf_nlist is not None else suggested_nlist(N, 0)
    bank_fp = bank_fingerprint(X_bank)

    if use_index_dir:
        os.makedirs(args.indices_dir, exist_ok=True)
        with tqdm(total=1, desc="Building/loading IVF index", unit="step") as pbar:
            t0 = time.time()
            index = get_or_build_ivf(
                X_bank,
                args.indices_dir,
                bank_fp,
                nlist_hint,
                force_cpu=False,
            )
            pbar.set_postfix_str(f"{time.time() - t0:.2f}s")
            pbar.update(1)
    else:
        with tqdm(total=1, desc="Building IVF index (memory)", unit="step") as pbar:
            t0 = time.time()
            nlist = resolve_nlist(N, nlist_hint)
            index = build_ivf_in_memory(X_bank, nlist)
            pbar.set_postfix_str(f"{time.time() - t0:.2f}s nlist={nlist}")
            pbar.update(1)

    if args.test:
        captions = TEST_CAPTIONS[:NUM_TEST_QUERIES]
        with tqdm(total=1, desc="Encoding test captions with CLIP", unit="step") as pbar:
            Q, query_texts = encode_test_captions_with_clip(captions)
            pbar.set_postfix_str(f"{Q.shape[0]} queries, dim={Q.shape[1]}")
            pbar.update(1)
    else:
        if not (args.queries and args.queries.strip()):
            raise ValueError("Without --test you must pass --queries /path/to/queries.npy (or .npz)")
        with tqdm(total=1, desc="Loading queries", unit="step") as pbar:
            Q, query_texts = load_queries_from_file(args.queries.strip())
            pbar.set_postfix_str(f"{Q.shape[0]:,} x {Q.shape[1]}")
            pbar.update(1)
    nq = Q.shape[0]

    t0 = time.time()
    results = run_retrieval(
        index,
        Q,
        k=args.top_k,
        nprobe=args.nprobe,
        batch_size=args.batch_size,
        bank_paths=bank_paths,
        uids=uids_list,
        query_texts=query_texts,
        show_progress=True,
    )
    elapsed = time.time() - t0
    tqdm.write(f"Retrieval: {elapsed:.2f}s ({nq / max(elapsed, 1e-9):,.0f} QPS)")

    write_output(results, args.output, args.format)
    tqdm.write("Done.")


if __name__ == "__main__":
    main()

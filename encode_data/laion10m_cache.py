# scripts/make_laion1m_cache.py
# Usage: python scripts/make_laion1m_cache.py --up_to 1 --max_items 1000000 --k_exact 100
import os, sys, math, pickle, shutil, argparse, subprocess
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PCARSpace

# ---------------------------
# Defaults
# ---------------------------
DEF_ROOT = "/ssd/hamed/ann/laion"
DEF_OUT  = "laion_cache_up_to_{up_to}.pkl"
DEF_KEX  = 100
DEF_UP_TO = 0

# Acceptable part IDs (excluding 8)
ACCEPTABLE_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]

BASE_URL = "https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-embeddings"
QUERY_URL = "https://zenodo.org/records/11090378/files/laion.query.10k.fbin"

# ---------------------------
# Download helpers
# ---------------------------
def have_wget() -> bool:
    return shutil.which("wget") is not None

def wget_download(url: str, dst: str) -> None:
    cmd = ["wget", "-t", "0", "--no-check-certificate", url, "-O", dst]
    print(f"Downloading with wget: {url} -> {dst}")
    subprocess.check_call(cmd)

def py_download(url: str, dst: str, chunk: int = 1 << 20) -> None:
    import urllib.request
    import ssl
    print(f"Downloading (python) {url} -> {dst}")
    # Create SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    with urllib.request.urlopen(url, context=ssl_context) as r, open(dst, "wb") as f:
        total = r.length or 0
        read = 0
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf); read += len(buf)
            if total:
                pct = 100.0 * read / total
                sys.stdout.write(f"\r  {read/1e6:7.2f}MB / {total/1e6:7.2f}MB  ({pct:5.1f}%)")
            else:
                sys.stdout.write(f"\r  {read/1e6:7.2f}MB")
            sys.stdout.flush()
    sys.stdout.write("\n")

def ensure_file(url: str, path: Path) -> None:
    if path.exists() and path.is_file():
        print(f"✓ Found {path.name} at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if have_wget():
            wget_download(url, str(path))
        else:
            py_download(url, str(path))
    except Exception as e:
        if path.exists():
            try: path.unlink()
            except Exception: pass
        raise RuntimeError(f"Failed to download {url} -> {path}: {e}")

def check_part_availability(part_id: int) -> bool:
    """Check if a part is available for download."""
    import urllib.request
    import ssl
    
    # Create SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    img_url = f"{BASE_URL}/images/img_emb_{part_id}.npy"
    try:
        with urllib.request.urlopen(img_url, context=ssl_context) as response:
            return response.status == 200
    except Exception:
        return False

def ensure_data(root: Path, up_to: int) -> None:
    """Download all parts from 0 to up_to (excluding 8) to data subdirectory."""
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of parts to download
    parts_to_download = [pid for pid in ACCEPTABLE_IDS if pid <= up_to]
    
    for part_id in parts_to_download:
        img_name = f"img_emb_{part_id}.npy"
        txt_name = f"text_emb_{part_id}.npy"
        img_url = f"{BASE_URL}/images/{img_name}"
        txt_url = f"{BASE_URL}/texts/{txt_name}"
        
        img_path = data_dir / img_name
        txt_path = data_dir / txt_name
        
        # Check if files already exist locally
        if img_path.exists() and txt_path.exists():
            print(f"✓ Found existing {img_name} at {img_path}")
            print(f"✓ Found existing {txt_name} at {txt_path}")
            print(f"✓ Using existing shard {part_id} from {data_dir}")
            continue
        
        # If files don't exist, check online availability before attempting download
        if not check_part_availability(part_id):
            print(f"Warning: Part {part_id} is not available for download and files don't exist locally. Skipping.")
            continue
        
        # Download missing files
        if not img_path.exists():
            ensure_file(img_url, img_path)
        else:
            print(f"✓ Found existing {img_name} at {img_path}")
            
        if not txt_path.exists():
            ensure_file(txt_url, txt_path)
        else:
            print(f"✓ Found existing {txt_name} at {txt_path}")
        
        print(f"✓ Downloaded shard {part_id} to {data_dir}")

# ---------------------------
# fbin file handling (from t2i_precompute.py)
# ---------------------------
def read_fbin_header(path: Path):
    with open(path, 'rb') as f:
        hdr = np.frombuffer(f.read(8), dtype=np.uint32)
    return int(hdr[0]), int(hdr[1])


def read_fbin_block(path: Path, start: int, count: int, dim: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, dim), dtype=np.float32)
    off = 8 + start * dim * 4
    nbytes = count * dim * 4
    with open(path, 'rb') as f:
        f.seek(off)
        buf = f.read(nbytes)
    total = len(buf)
    if total == 0:
        return np.empty((0, dim), dtype=np.float32)
    full_elems = (total // 4)
    full_rows = full_elems // dim
    use_bytes = full_rows * dim * 4
    arr = np.frombuffer(memoryview(buf)[:use_bytes], dtype=np.float32)
    return arr.reshape((full_rows, dim))


def download_query_file(root: Path) -> Path:
    """Download query.10k.fbin file to data subdirectory."""
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    query_path = data_dir / "query.10k.fbin"
    
    if not query_path.exists():
        print(f"Downloading query file from {QUERY_URL}...")
        ensure_file(QUERY_URL, query_path)
        print(f"✓ Downloaded query file to {query_path}")
    else:
        print(f"✓ Found existing query file at {query_path}")
    
    return query_path


# ---------------------------
# Math / exact@K helpers
# ---------------------------
def l2n(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return (x / n).astype(np.float32)



def topk_exact(Q: np.ndarray, X: np.ndarray, k: int, batch: int = 8192, index_batch: int = 50000, use_fp16: bool = False) -> np.ndarray:
    """
    Exact top-k by cosine; Q and X must be L2-normalized float32.
    Streams over index in chunks and maintains a running top-k per query.
    Uses GPU if available but keeps memory bounded (no full [B,N] matrix).
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Validate k parameter
    k = int(min(k, X.shape[0]))
    if k < 1:
        raise ValueError("k must be >= 1")
    
    # Enable TF32 for better performance on Ampere+ GPUs (keeps FP32 correctness)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # helper to (vals, ids) -> topk merge
    def merge_topk(best_vals, best_ids, new_vals, new_ids, k):
        # best_*: [B,k], new_*: [B,k]
        vals_all = torch.cat([best_vals, new_vals], dim=1)   # [B, 2k]
        ids_all  = torch.cat([best_ids,  new_ids],  dim=1)   # [B, 2k]
        topk_vals, topk_pos = torch.topk(vals_all, k=k, dim=1, largest=True, sorted=True)
        # gather matching ids
        bidx = torch.arange(vals_all.size(0), device=vals_all.device).unsqueeze(1).expand_as(topk_pos)
        topk_ids = ids_all[bidx, topk_pos]
        return topk_vals, topk_ids

    all_out = []
    n_batches = (Q.shape[0] + batch - 1) // batch
    with tqdm(total=Q.shape[0], desc=f"Computing exact@{k}", unit="queries",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for s in range(0, Q.shape[0], batch):
            e = min(Q.shape[0], s + batch)

            Qt = torch.from_numpy(np.asarray(Q[s:e], dtype=np.float32)).to(dev, non_blocking=True).contiguous()
            if use_fp16 and torch.cuda.is_available():
                Qt = Qt.half()

            # initialize running top-k for this query batch
            B = Qt.size(0)
            # Use dtype-aware negative infinity with FP16 guard
            if Qt.dtype == torch.float16:
                neg_inf = torch.tensor(-1e4, device=dev, dtype=Qt.dtype)
            else:
                neg_inf = torch.finfo(Qt.dtype).min
            best_vals = torch.full((B, k), neg_inf, device=dev, dtype=Qt.dtype)
            best_ids  = torch.full((B, k), -1,   device=dev, dtype=torch.long)

            for idx_s in range(0, X.shape[0], index_batch):
                idx_e = min(X.shape[0], idx_s + index_batch)
                Xt_chunk = torch.from_numpy(np.asarray(X[idx_s:idx_e], dtype=np.float32)).to(dev, non_blocking=True).contiguous()
                if use_fp16 and torch.cuda.is_available():
                    Xt_chunk = Xt_chunk.half()

                # [B, chunk] similarities
                S_chunk = Qt @ Xt_chunk.T

                # local top-k in this chunk
                k_local = min(k, S_chunk.size(1))
                local_vals, local_pos = torch.topk(S_chunk, k=k_local, dim=1, largest=True, sorted=True)
                local_ids = (local_pos + idx_s).to(torch.long)

                # pad local top-k to length k if chunk smaller than k
                if k_local < k:
                    pad = k - k_local
                    pad_vals = torch.full((B, pad), neg_inf, device=dev, dtype=S_chunk.dtype)
                    pad_ids  = torch.full((B, pad), -1,   device=dev, dtype=torch.long)
                    local_vals = torch.cat([local_vals, pad_vals], dim=1)
                    local_ids  = torch.cat([local_ids,  pad_ids],  dim=1)

                # merge with running best
                best_vals, best_ids = merge_topk(best_vals, best_ids, local_vals, local_ids, k)

                # cleanup
                del Xt_chunk, S_chunk, local_vals, local_pos, local_ids

            # store batch result (move to CPU, convert to float32 indices)
            all_out.append(best_ids.cpu().numpy().astype(np.int64))

            del Qt, best_vals, best_ids
            pbar.update(e - s)

    return np.vstack(all_out)

# ---------------------------
# Cache builder
# ---------------------------
def build_cache(root: Path, out_path: Path, k_exact: int, up_to: int, 
                query_batch: int = 4096, index_batch: int = 25000, use_fp16: bool = False,
                pca_dim: int = None, pca_device: str = "cuda", pca_center: bool = True) -> None:
    """Build cache from downloaded parts and query file."""
    data_dir = root / "data"
    
    # Load all image and text embeddings from downloaded parts
    print("Loading and concatenating data from all parts...")
    
    parts_to_load = [pid for pid in ACCEPTABLE_IDS if pid <= up_to]
    img_parts = []
    txt_parts = []
    
    for part_id in parts_to_load:
        img_name = f"img_emb_{part_id}.npy"
        txt_name = f"text_emb_{part_id}.npy"
        img_path = data_dir / img_name
        txt_path = data_dir / txt_name
        
        if img_path.exists() and txt_path.exists():
            print(f"  Loading part {part_id}...")
            img_part = np.load(str(img_path), mmap_mode="r")
            txt_part = np.load(str(txt_path), mmap_mode="r")
            assert img_part.shape[0] == txt_part.shape[0], f"mismatched rows in part {part_id}"
            img_parts.append(img_part)
            txt_parts.append(txt_part)
        else:
            print(f"  Warning: Part {part_id} files not found, skipping")
    
    if not img_parts:
        raise RuntimeError("No valid parts found to load")
    
    # Concatenate all parts
    print("Concatenating parts...")
    X_all = np.concatenate([np.array(part, dtype=np.float32) for part in img_parts], axis=0)
    T_all = np.concatenate([np.array(part, dtype=np.float32) for part in txt_parts], axis=0)
    
    # Use all available data
    n_take = X_all.shape[0]
    
    # L2-normalize FIRST (before PCA)
    print("Normalizing data...")
    X_all = l2n(X_all)
    T_all = l2n(T_all)
    
    D = X_all.shape[1]
    print(f"Using {n_take:,} items, dim={D}")

    # Load query file for validation set
    query_path = download_query_file(root)
    print("Loading query file...")
    n_query, d_query = read_fbin_header(query_path)
    assert d_query == D, f"Query dimension {d_query} != image dimension {D}"
    
    # Read query data
    Q_query = read_fbin_block(query_path, 0, n_query, D)
    Q_query = l2n(Q_query.astype(np.float32))
    print(f"Loaded {n_query:,} query vectors")

    # Compute PCA on normalized training images (if pca_dim is specified)
    R = None
    if pca_dim is not None and pca_dim > 0:
        print(f"\n🔧 Computing PCA on normalized training images (target dim: {pca_dim})...")
        print(f"   Device: {pca_device}, Center: {pca_center}")
        
        # Fit PCA on normalized training images
        R = PCARSpace(d_keep=pca_dim, center_for_fit=pca_center, device=pca_device, chunk=262144)
        R.fit(X_all)
        
        print(f"✅ PCA fitted: W.shape={R.W.shape}, mu={'None' if R.mu is None else R.mu.shape}")
        print(f"   Original dim: {D} → Reduced dim: {pca_dim}")
        
        # Apply PCA transform to all data (transform also normalizes)
        print("🔄 Applying PCA transform to all data...")
        X_all = R.transform(X_all)  # [N, pca_dim]
        T_all = R.transform(T_all)  # [N, pca_dim]
        Q_query = R.transform(Q_query)  # [N_query, pca_dim]
        
        print(f"✅ PCA transform applied")
        print(f"   X_all: {X_all.shape}, T_all: {T_all.shape}, Q_query: {Q_query.shape}")
        
        # Update D to reflect reduced dimension
        D = pca_dim
    else:
        print("⚠️  No PCA reduction requested (pca_dim=None). Using original features.")

    print("\nComputing exact@K for train and val sets (on PCA-reduced features if applicable):")
    
    # Compute exact@K for training set (self-similarity)
    print("  • train: queries = train texts, index = train images")
    knn_train = topk_exact(T_all, X_all, k=k_exact, batch=query_batch, index_batch=index_batch, use_fp16=use_fp16)
    
    # Compute exact@K for validation set (query against train images)
    print("  • val: queries = external query set, index = train images")
    knn_val = topk_exact(Q_query, X_all, k=k_exact, batch=query_batch, index_batch=index_batch, use_fp16=use_fp16)

    # assemble cache
    data = {
        "train": {
            "image_features": X_all,           # All downloaded images (indexed)
            "text_features":  T_all,           # All downloaded texts
            "knn_indices":    knn_train,       # exact@K vs TRAIN images
        },
        "val": {
            "text_features":  Q_query,         # External query set
            "knn_indices":    knn_val,         # exact@K vs TRAIN images
            "index_ref":      "train",         # explicit reference
        },
        # Metadata breadcrumb for training code adaptation
        "_meta": {
            "index_bank_split": "train",
            "k_exact": k_exact,
            "dim": int(X_all.shape[1]),  # Final dimension (after PCA if applied)
            "normalized": True,
            "up_to": up_to,
            "parts_used": parts_to_load,
            "n_train_items": int(X_all.shape[0]),
            "n_val_items": int(Q_query.shape[0]),
            "pca_applied": R is not None,
            "pca_dim": pca_dim if R is not None else None,
            "pca_center": pca_center if R is not None else None,
        }
    }
    
    # Add PCA parameters if PCA was applied
    if R is not None:
        data["pca"] = {
            "W": R.W.astype(np.float32),
            "mu": R.mu.astype(np.float32) if R.mu is not None else None,
            "center_for_fit": bool(R.center_for_fit),
    }

    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✓ Wrote {out_path}")
    print("Summary:")
    print("  train:",
          data["train"]["image_features"].shape,
          data["train"]["text_features"].shape,
          data["train"]["knn_indices"].shape)
    print("  val:  texts", data["val"]["text_features"].shape,
          "knn", data["val"]["knn_indices"].shape,
          " (index_ref=train)")
    print("  metadata:", data["_meta"])

# ---------------------------
# CLI
# ---------------------------
def list_available_parts():
    """List all available parts for download."""
    print("Checking available parts...")
    available_parts = []
    
    for i in range(15):  # Check parts 0-14
        if check_part_availability(i):
            available_parts.append(i)
            print(f"✓ Part {i} available")
        else:
            print(f"✗ Part {i} not available")
    
    print(f"\nAvailable parts: {available_parts}")
    return available_parts

def parse_args():
    p = argparse.ArgumentParser(description="Build LAION cache (downloads parts 0 to up_to if missing).")
    p.add_argument("--root", type=str, default=DEF_ROOT, help="Dataset root directory.")
    p.add_argument("--out",  type=str, default=None,  help="Output pickle filename (under precompute subdir). If not provided, uses default naming with up_to.")
    p.add_argument("--up_to", type=int, default=DEF_UP_TO, choices=ACCEPTABLE_IDS, help="Download parts 0 to up_to (excluding 8).")
    p.add_argument("--k_exact",   type=int, default=DEF_KEX, help="Exact@K to store in cache.")
    p.add_argument("--query_batch", type=int, default=4096, help="Batch size for query processing.")
    p.add_argument("--index_batch", type=int, default=25000, help="Batch size for index processing.")
    p.add_argument("--use_fp16", action="store_true", help="Do matmul in FP16 (faster but not perfectly exact). Default: FP32 exact.")
    p.add_argument("--pca_dim", type=int, default=None, help="Target PCA dimension (e.g., 200). If None, no PCA reduction is applied.")
    p.add_argument("--pca_device", type=str, default="cuda", help="Device for PCA computation (cuda or cpu).")
    p.add_argument("--pca_center", action="store_true", default=True, help="Center data before PCA (subtract mean). Default: True.")
    p.add_argument("--no_pca_center", dest="pca_center", action="store_false", help="Disable centering for PCA.")
    p.add_argument("--force", action="store_true", help="Overwrite existing output file if it exists.")
    p.add_argument("--list_parts", action="store_true", help="List available parts and exit.")
    return p.parse_args()

def main():
    args = parse_args()
    
    # Handle list_parts option
    if args.list_parts:
        list_available_parts()
        return 0
    
    root = Path(args.root)
    
    # Create precompute subdirectory for output
    precompute_dir = root / "precompute"
    precompute_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename if not provided
    if args.out is None:
        out_filename = DEF_OUT.format(up_to=args.up_to)
    else:
        out_filename = args.out
    
    out_path = precompute_dir / out_filename

    print(f"Root: {root}")
    print(f"Up to: {args.up_to}")
    print(f"Output: {out_path}")
    print(f"Parts to download: {[pid for pid in ACCEPTABLE_IDS if pid <= args.up_to]}")
    
    # Check if output file exists and handle accordingly
    if out_path.exists() and not args.force:
        print(f"Error: Output file {out_path} already exists.")
        print("Use --force to overwrite or choose a different output name with --out.")
        return 1
    
    print("Checking data files...")
    ensure_data(root, args.up_to)

    print("Building cache...")
    build_cache(root, out_path, k_exact=args.k_exact, 
                up_to=args.up_to, query_batch=args.query_batch, 
                index_batch=args.index_batch, use_fp16=args.use_fp16,
                pca_dim=args.pca_dim, pca_device=args.pca_device, pca_center=args.pca_center)
    
    return 0

if __name__ == "__main__":
    main()

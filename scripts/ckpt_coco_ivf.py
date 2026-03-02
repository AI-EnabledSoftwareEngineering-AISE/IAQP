#!/usr/bin/env python3
# ckpt_coco_ivf.py
#
# COCO-specific IVF checkpoint evaluation
# Tests only ONE checkpoint (coco-80k_ivf_up_to_e3) on IVF backend only
#
# Hard-coded for COCO dataset with 82,783 images
# Evaluates on IVF search only (no CAGRA)
# Budgets 10..100 step 10

import os, sys, time, random, math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# --- repo imports (match your tree) ---
current_dir = Path(__file__).resolve().parents[2]  # .../t2i_code
sys.path.insert(0, str(current_dir))

from projector.dataset_loader import CocoDatasetLoader
from projector.utils import ResidualProjector, PCARSpace, brute_force_topk_streaming

# --- FAISS IVF ---
try:
    import faiss
except Exception:
    faiss = None
    print("[ERROR] FAISS not installed. Install with: conda install -c pytorch faiss-gpu")
    exit(1)

# Hard-coded COCO configuration
DATA_PATH = "/home/hamed/projects/SPIN/adapter/data/coco_cache/coco_ranking.pkl"
HIDDEN_DIM = 512  # COCO uses 512 hidden dim (standard CLIP ViT-B/32)
FEATURE_DIM = 512  # COCO feature dim

# Checkpoint paths (all 3 epochs)
CKPTS = {
    "ep1": "outputs/checkpoints/coco-80k_ivf_up_to_e3/epoch_1.pt",
    "ep2": "outputs/checkpoints/coco-80k_ivf_up_to_e3/epoch_2.pt",
    "ep3": "outputs/checkpoints/coco-80k_ivf_up_to_e3/epoch_3.pt",
}

print(f"🔧 Dataset: COCO")
print(f"📁 Data path: {DATA_PATH}")
print(f"📦 Checkpoints: {list(CKPTS.keys())}")

# Choose which checkpoint's PCA defines the bank/search space
USE_PCA_FROM = "ep2"

# Evaluation sizes (use full dataset)
N_BANK = None  # Will be full dataset
NQ = None      # Will be full val/test
K = 10

# Budgets to evaluate
BUDGETS = list(range(10, 101, 10))

# ----------------- helpers -----------------
def l2n(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def assert_unit(name, A):
    n = np.linalg.norm(A, axis=1)
    print(f"[norm] {name}: mean={n.mean():.6f} std={n.std():.6f} min={n.min():.6f} max={n.max():.6f}")
    if not np.allclose(n, 1.0, atol=1e-4):
        raise AssertionError(f"{name} not unit-normalized")

def recall_at_k(pred, gt, k):
    k = min(k, pred.shape[1], gt.shape[1])
    hits = 0
    for i in range(pred.shape[0]):
        hits += len(set(pred[i,:k]) & set(gt[i,:k]))
    return hits/(pred.shape[0]*k)

def load_ckpt(ckpt_path, device="cuda", dim=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden = ck.get("hidden", dim or 512)
    alpha = ck.get("alpha", 0.25)
    model_dim = dim or 512
    model = ResidualProjector(dim=model_dim, hidden=hidden, alpha=alpha)
    msd = ck["model_state_dict"]
    if any(k.startswith("module.") for k in msd):
        msd = {k.replace("module.",""):v for k,v in msd.items()}
    model.load_state_dict(msd)
    model.to(device).eval()
    R = PCARSpace(d_keep=None, center_for_fit=True, device=device)
    R.W = ck["W"].astype(np.float32)
    R.mu = ck.get("mu", None)
    if R.mu is not None:
        R.mu = R.mu.astype(np.float32)
    return model, R

@torch.no_grad()
def project_np(model, T_np, device="cuda", batch=65536):
    dev = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    out = np.empty_like(T_np, dtype=np.float32)
    for s in range(0, T_np.shape[0], batch):
        e = min(T_np.shape[0], s+batch)
        xb = torch.from_numpy(T_np[s:e]).to(dev).float()
        xb = F.normalize(xb, dim=1)
        out[s:e] = model(xb).cpu().numpy().astype(np.float32)
    return out

# ---------- IVF (FAISS) ----------
def suggested_nlist(N: int) -> int:
    return 512

def build_ivf_index_gpu(X_R_bank: np.ndarray, nlist_hint: int = None):
    assert faiss is not None, "faiss not installed"
    N, D = X_R_bank.shape
    nrm = np.linalg.norm(X_R_bank, axis=1)
    if not np.allclose(nrm, 1.0, atol=1e-4):
        raise ValueError("IVF build expects L2-normalized inputs.")
    nlist = nlist_hint if (nlist_hint and nlist_hint>0) else suggested_nlist(N)
    print(f"Building IVF index (IP) with nlist={nlist}")

    quant = faiss.IndexFlatIP(D)
    cpu = faiss.IndexIVFFlat(quant, D, int(nlist), faiss.METRIC_INNER_PRODUCT)

    Xf = X_R_bank.astype(np.float32)
    # train on GPU if possible
    try:
        if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            gpu = faiss.index_cpu_to_gpu(res, 0, cpu)
            print("  • Training IVF on GPU…")
            gpu.train(Xf)
            cpu = faiss.index_gpu_to_cpu(gpu)
        else:
            print("  • Training IVF on CPU…")
            cpu.train(Xf)
    except Exception as e:
        print(f"  ! GPU train failed ({e}), fallback to CPU train")
        cpu.train(Xf)

    print("  • Adding vectors to IVF (CPU)…")
    cpu.add(Xf)

    # move to GPU for search if possible
    if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        gpu = faiss.index_cpu_to_gpu(res, 0, cpu)
        print("  • IVF moved to GPU for search")
        return gpu, nlist
    else:
        print("  • Using CPU IVF (no GPU found)")
        return cpu, nlist

def ivf_search(index, Q: np.ndarray, k: int, nprobe: int):
    nlist = index.nlist if hasattr(index, "nlist") else nprobe
    nprobe = int(max(1, min(nprobe, nlist)))
    try:
        index.nprobe = nprobe
    except Exception:
        pass
    t0 = time.time()
    _D, I = index.search(Q.astype(np.float32, order="C"), k)
    dt = time.time() - t0
    qps = Q.shape[0] / max(dt, 1e-9)
    return I.astype(np.int64, copy=False), qps, nprobe

# Parse command line arguments
parser = argparse.ArgumentParser(description="COCO IVF Checkpoint Evaluation")
parser.add_argument("--split", type=str, choices=["val", "test", "inbound"], default="val",
                   help="Split to evaluate on: 'val', 'test', or 'inbound'")
parser.add_argument("--full_dataset", action="store_true",
                   help="Use full dataset for evaluation (no subsampling). Default: True (full dataset)")
args = parser.parse_args()

# Default to full dataset if not specified
if not args.full_dataset:
    args.full_dataset = True

# ----------------- main -----------------
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    np.random.seed(123)
    random.seed(123)

    print("=" * 60)
    print("COCO IVF CHECKPOINT EVALUATION")
    print(f"Evaluating on: {args.split} split")
    print("=" * 60)
    
    print(f"Loading COCO dataset…")
    ds = CocoDatasetLoader(DATA_PATH)
    X_train_full, T_train_full, _ = ds.get_train_data()
    X_split_full, T_split_full, _, _ = ds.get_split_data(args.split)
    print(f"Train: X={X_train_full.shape}, T={T_train_full.shape}")
    print(f"{args.split}: X={X_split_full.shape}, T={T_split_full.shape}")

    # Determine sampling strategy
    if args.full_dataset:
        print("🔍 Using FULL dataset for evaluation")
        N_BANK = X_train_full.shape[0]
        NQ = T_split_full.shape[0]
        X_bank = l2n(X_train_full.astype(np.float32))
        T_split = l2n(T_split_full.astype(np.float32))
        print(f"📊 Full evaluation: Bank={N_BANK:,} images, Queries={NQ:,} texts")
    else:
        print("🔍 Using SUBSAMPLED dataset for evaluation")
        N_BANK = min(50000, X_train_full.shape[0])  # Limit to 50k for speed
        NQ = min(5000, T_split_full.shape[0])     # Limit to 5k for speed
        # Sample fixed subsets (deterministic)
        N_tr = X_train_full.shape[0]
        N_split = T_split_full.shape[0]
        bank_ids = np.random.RandomState(123).choice(N_tr, size=min(N_BANK, N_tr), replace=False)
        q_ids = np.random.RandomState(123).choice(N_split, size=min(NQ, N_split), replace=False)
        X_bank = l2n(X_train_full[bank_ids].astype(np.float32))
        T_split = l2n(T_split_full[q_ids].astype(np.float32))
        print(f"📊 Subsampled evaluation: Bank={N_BANK:,} images, Queries={NQ:,} texts")

    # Load all checkpoints
    print("\nLoading checkpoints…")
    models = {}
    rotations = {}
    for tag, path in CKPTS.items():
        print(f"  • {tag}: {path}")
        if not os.path.exists(path):
            print(f"[ERROR] Checkpoint not found: {path}")
            exit(1)
        models[tag], rotations[tag] = load_ckpt(path, device="cuda", dim=FEATURE_DIM)

    # Fix bank space with PCA from USE_PCA_FROM
    print(f"\nUsing PCA from {USE_PCA_FROM} to define the bank/search space")
    R_ref = rotations[USE_PCA_FROM]

    print("Rotating bank…")
    X_R_bank = R_ref.transform(X_bank)
    X_R_bank = l2n(X_R_bank)
    assert_unit("X_R_bank", X_R_bank)

    # Build IVF index
    print("\nBuilding IVF index…")
    ivf_index, nlist = build_ivf_index_gpu(X_R_bank, nlist_hint=512)
    print(f"✓ IVF ready (nlist={nlist})")

    # Prepare 4 query variants (baseline + 3 epochs)
    print("\nPreparing queries (4 variants)…")
    variants = {}
    Q_base = R_ref.transform(T_split)
    Q_base = l2n(Q_base)
    assert_unit("Q_base (baseline)", Q_base)
    variants["baseline"] = Q_base
    
    for tag in ["ep1", "ep2", "ep3"]:
        Qp = project_np(models[tag], T_split, device="cuda", batch=65536)
        Qp_R = R_ref.transform(Qp)
        Qp_R = l2n(Qp_R)
        assert_unit(f"Q_proj_{tag}", Qp_R)
        variants[f"proj_{tag}"] = Qp_R

    # Compute exact GTs per variant
    print("\nComputing exact GTs (projected space) for all variants…")
    GTs = {}
    for name, Q in variants.items():
        print(f"  • GT for {name} …")
        GTs[name] = brute_force_topk_streaming(Q, X_R_bank, k=K, q_batch=min(8192, Q.shape[0]), x_batch=50_000)

    # ----------------- EVAL: IVF (nprobe = budget) -----------------
    print("\n=== IVF: Recall@10 & QPS vs nprobe ===")
    ivf_results = {}
    for name in ["baseline", "proj_ep1", "proj_ep2", "proj_ep3"]:
        print(f"\nVariant: {name}")
        rows = []
        Q_cpu = variants[name]
        for B in BUDGETS:
            I_pred, qps, used_nprobe = ivf_search(ivf_index, Q_cpu, K, nprobe=B)
            r = recall_at_k(I_pred, GTs[name], K)
            rows.append((used_nprobe, r, qps))
            print(f"  nprobe={used_nprobe:4d}  R@{K}={r:.4f}  QPS={qps:,.0f}")
        ivf_results[name] = rows

    # Summary
    print("\n=== SUMMARY (IVF, R@10 vs nprobe) ===")
    for name, rows in ivf_results.items():
        line = f"{name:12s}: " + "  ".join([f"n{nb:3d}:{r:.3f}" for (nb,r,_) in rows])
        print(line)

    print("\nDone.")

if __name__ == "__main__":
    main()

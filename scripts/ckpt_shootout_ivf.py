#!/usr/bin/env python3
# ckpt_shootout_ivf.py
#
# TRAINING: IVF backend (GPU-sharded IVF from FAISS)
# TESTING: IVF + CAGRA (generalization test across backends)
#
# One fixed bank (500k), 5k queries, 4 variants (baseline + 3 ckpts).
# Builds ONE CAGRA index (on GPU) and ONE IVF index (GPU if available).
# Budgets 10..100 step 10. IVF uses nprobe as the "budget".
# Tests generalization: IVF-trained model → IVF/CAGRA evaluation
# Supports both 'laion' and 't2i' datasets
# Minimal + hard-coded as requested.

import os, sys, time, random, math
import argparse
import numpy as np
import cupy as cp
import torch
import torch.nn.functional as F
from pathlib import Path

# --- repo imports (match your tree) ---
current_dir = Path(__file__).resolve().parents[2]  # .../t2i_code
sys.path.insert(0, str(current_dir))

from projector.dataset_loader import LaionDatasetLoader, T2IDatasetLoader
from projector.utils import ResidualProjector, PCARSpace, brute_force_topk_streaming

# --- cuVS CAGRA ---
from cuvs.neighbors import cagra

# --- FAISS IVF ---
try:
    import faiss
except Exception:
    faiss = None

# ----------------- dataset configuration -----------------
def get_dataset_config(dataset_name):
    """Get dataset-specific configuration"""
    if dataset_name.lower() == "laion":
        return {
            "data_path": "/ssd/hamed/ann/laion/precompute/laion_cache_up_to_2.pkl",
            "dataset_loader": LaionDatasetLoader,
            "hidden_dim": 512,  # LAION uses 512 hidden dim
            "feature_dim": 512  # LAION feature dim
        }
    elif dataset_name.lower() == "t2i":
        return {
            "data_path": "/ssd/hamed/ann/t2i-10M/precompute/t2i_cache_up_to_3.pkl",
            "dataset_loader": T2IDatasetLoader,
            "hidden_dim": 200,  # T2I uses 200 hidden dim
            "feature_dim": 200  # T2I feature dim
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Use 'laion' or 't2i'")

# Parse command line arguments
parser = argparse.ArgumentParser(description="IVF Checkpoint Shootout")
parser.add_argument("--dataset", type=str, choices=["laion", "t2i"], default="laion",
                   help="Dataset to use: 'laion' or 't2i'")
parser.add_argument("--exact_k", action="store_true", 
                   help="Load exact_k checkpoints instead of IVF checkpoints")
parser.add_argument("--ann_only", action="store_true", 
                   help="Load ANN-only checkpoints (trained with --disable_exactK_teacher)")
parser.add_argument("--full_dataset", action="store_true", 
                   help="Use full dataset for evaluation (no subsampling)")
args = parser.parse_args()

# Get dataset configuration
config = get_dataset_config(args.dataset)
DATA_PATH = config["data_path"]
DatasetLoader = config["dataset_loader"]
HIDDEN_DIM = config["hidden_dim"]
FEATURE_DIM = config["feature_dim"]

# Generate checkpoint paths based on dataset and flags
if args.exact_k:
    base_path = f"outputs/checkpoints/{args.dataset}-3m_exact_k_up_to_e3"
elif args.ann_only:
    base_path = f"outputs/checkpoints/{args.dataset}-3m_ivf_ann_only_up_to_e3"
else:
    base_path = f"outputs/checkpoints/{args.dataset}-3m_ivf_up_to_e3"

CKPTS = {
    "ep1": f"{base_path}/epoch_1.pt",
    "ep2": f"{base_path}/epoch_2.pt", 
    "ep3": f"{base_path}/epoch_3.pt",
}

print(f"🔧 Dataset: {args.dataset}")
if args.exact_k:
    checkpoint_type = "exact_k"
elif args.ann_only:
    checkpoint_type = "ivf_ann_only"
else:
    checkpoint_type = "ivf"
print(f"🔧 Checkpoint type: {checkpoint_type}")
print(f"📁 Data path: {DATA_PATH}")
print(f"📦 Checkpoints: {list(CKPTS.keys())}")

# Choose which checkpoint's PCA defines the bank/search space (fixed index space)
USE_PCA_FROM = "ep2"

# Subsampling sizes (can be overridden by --full_dataset)
N_BANK = None  # Will be set based on --full_dataset flag
NQ     = None  # Will be set based on --full_dataset flag
K      = 10

# CAGRA build config (as requested)
CAGRA_CFG = dict(
    graph_degree=32,
    intermediate_graph_degree=64,
    nn_descent_niter=30,
    build_algo="nn_descent",
    metric="inner_product",  # rank-equivalent to cosine for L2n inputs
)

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

def cpu2gpu(a: np.ndarray) -> cp.ndarray:
    a = np.asarray(a, dtype=np.float32, order="C")
    return cp.ascontiguousarray(cp.asarray(a, dtype=cp.float32))

def load_ckpt(ckpt_path, device="cuda", dim=None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden = ck.get("hidden", dim or 512); alpha = ck.get("alpha", 0.25)
    model_dim = dim or 512  # Use provided dim or default to 512
    model = ResidualProjector(dim=model_dim, hidden=hidden, alpha=alpha)
    msd = ck["model_state_dict"]
    if any(k.startswith("module.") for k in msd): msd = {k.replace("module.",""):v for k,v in msd.items()}
    model.load_state_dict(msd); model.to(device).eval()
    R = PCARSpace(d_keep=None, center_for_fit=True, device=device)
    R.W = ck["W"].astype(np.float32)
    R.mu = ck.get("mu", None)
    if R.mu is not None: R.mu = R.mu.astype(np.float32)
    return model, R

@torch.no_grad()
def project_np(model, T_np, device="cuda", batch=65536):
    dev = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    out = np.empty_like(T_np, dtype=np.float32)
    for s in range(0, T_np.shape[0], batch):
        e=min(T_np.shape[0], s+batch)
        xb=torch.from_numpy(T_np[s:e]).to(dev).float()
        xb=F.normalize(xb, dim=1)
        out[s:e]=model(xb).cpu().numpy().astype(np.float32)
    return out

# ---------- CAGRA ----------
def build_cagra_index(X_gpu):
    p = cagra.IndexParams(
        metric=CAGRA_CFG["metric"],
        intermediate_graph_degree=CAGRA_CFG["intermediate_graph_degree"],
        graph_degree=CAGRA_CFG["graph_degree"],
        build_algo=CAGRA_CFG["build_algo"],
        nn_descent_niter=CAGRA_CFG["nn_descent_niter"],
    )
    t0=time.time()
    idx = cagra.build(p, X_gpu)
    print(f"✓ CAGRA build done in {time.time()-t0:.2f}s  (gdeg={p.graph_degree}/{p.intermediate_graph_degree}, algo={p.build_algo})")
    return idx

def cagra_search(idx, Q_gpu, k, iters, itopk=256, width=1):
    itopk = max(itopk, k)
    sp = cagra.SearchParams(itopk_size=itopk, search_width=max(1,width), algo="auto", max_iterations=max(1,int(iters)))
    t0=time.time(); D,I = cagra.search(sp, idx, Q_gpu, k); dt=time.time()-t0
    qps = Q_gpu.shape[0] / max(dt, 1e-9)
    return cp.asnumpy(I), qps

# ---------- IVF (FAISS) ----------
def suggested_nlist(N: int) -> int:
    # simple heuristic: 4 * sqrt(N), round to nearest power-of-two-ish bucket
    raw = int(4 * math.sqrt(max(1, N)))
    # snap to {1024, 2048, 4096, 8192}
    if raw <= 1024: return 1024
    if raw <= 2048: return 2048
    if raw <= 4096: return 4096
    return 8192

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
            # move trained back to CPU for add (safer memory-wise), then to GPU for search
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
    # clamp nprobe to valid range
    nlist = index.nlist if hasattr(index, "nlist") else nprobe
    nprobe = int(max(1, min(nprobe, nlist)))
    # set nprobe on both CPU and GPU index types
    try:
        index.nprobe = nprobe
    except Exception:
        # some GPU wrappers use ParameterSpace, but most expose nprobe directly
        pass
    t0 = time.time()
    _D, I = index.search(Q.astype(np.float32, order="C"), k)
    dt = time.time() - t0
    qps = Q.shape[0] / max(dt, 1e-9)
    return I.astype(np.int64, copy=False), qps, nprobe

# ----------------- main -----------------
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    np.random.seed(123); random.seed(123)

    print("=" * 60)
    print("IVF CHECKPOINT SHOOTOUT")
    print("TRAINING: IVF backend (GPU-sharded IVF from FAISS)")
    print("TESTING: IVF + CAGRA (generalization test across backends)")
    print("=" * 60)
    
    print(f"Loading {args.dataset.upper()} dataset…")
    ds = DatasetLoader(DATA_PATH)
    X_train_full, T_train_full, _ = ds.get_train_data()
    X_val_full,   T_val_full,   *_ = ds.get_split_data("val")
    print(f"Train: X={X_train_full.shape}, T={T_train_full.shape} | Val: T={T_val_full.shape}")

    # Determine sampling strategy based on --full_dataset flag
    if args.full_dataset:
        print("🔍 Using FULL dataset for evaluation (no subsampling)")
        N_BANK = X_train_full.shape[0]  # Use all training data as bank
        NQ = T_val_full.shape[0]        # Use all validation data as queries
        X_bank = l2n(X_train_full.astype(np.float32))
        T_val = l2n(T_val_full.astype(np.float32))
        print(f"📊 Full evaluation: Bank={N_BANK:,} images, Queries={NQ:,} texts")
    else:
        print("🔍 Using SUBSAMPLED dataset for evaluation")
        N_BANK = 500_000
        NQ = 5_000
        # Sample fixed subsets (deterministic)
        N_tr = X_train_full.shape[0]
        N_val = T_val_full.shape[0]
        bank_ids = np.random.RandomState(123).choice(N_tr, size=min(N_BANK, N_tr), replace=False)
        q_ids    = np.random.RandomState(123).choice(N_val, size=min(NQ, N_val), replace=False)
        X_bank = l2n(X_train_full[bank_ids].astype(np.float32))
        T_val  = l2n(T_val_full[q_ids].astype(np.float32))
        print(f"📊 Subsampled evaluation: Bank={N_BANK:,} images, Queries={NQ:,} texts")

    # Load checkpoints
    print("\nLoading checkpoints…")
    models = {}
    rotations = {}
    for tag, path in CKPTS.items():
        print(f"  • {tag}: {path}")
        models[tag], rotations[tag] = load_ckpt(path, device="cuda", dim=FEATURE_DIM)

    # Fix bank space with PCA from USE_PCA_FROM
    print(f"\nUsing PCA from {USE_PCA_FROM} to define the bank/search space")
    R_ref = rotations[USE_PCA_FROM]

    print("Rotating bank…")
    X_R_bank = R_ref.transform(X_bank)
    X_R_bank = l2n(X_R_bank)
    assert_unit("X_R_bank", X_R_bank)

    # ---- Build CAGRA once
    print("\nMoving bank to GPU & building CAGRA index…")
    X_gpu = cpu2gpu(X_R_bank)
    print(f"GPU pool used={cp.get_default_memory_pool().used_bytes()/1e9:.3f} GB  ptr={hex(X_gpu.data.ptr)}")
    cagra_index = build_cagra_index(X_gpu)
    _KEEPALIVE = [X_gpu]  # keep bank alive for CAGRA index lifetime

    # ---- Build IVF once (GPU if possible)
    print("\nBuilding IVF index…")
    ivf_index, nlist = build_ivf_index_gpu(X_R_bank)
    print(f"✓ IVF ready (nlist={nlist})")

    # Prepare 4 query variants (same bank space)
    print("\nPreparing queries (4 variants)…")
    variants = {}
    Q_base = R_ref.transform(T_val); Q_base = l2n(Q_base); assert_unit("Q_base (baseline)", Q_base)
    variants["baseline"] = Q_base
    for tag in ["ep1","ep2","ep3"]:
        Qp = project_np(models[tag], T_val, device="cuda", batch=65536)
        Qp_R = R_ref.transform(Qp); Qp_R = l2n(Qp_R); assert_unit(f"Q_proj_{tag}", Qp_R)
        variants[f"proj_{tag}"] = Qp_R

    # Exact GTs per variant
    print("\nComputing exact GTs (projected space) for all variants…")
    GTs = {}
    for name, Q in variants.items():
        print(f"  • GT for {name} …")
        GTs[name] = brute_force_topk_streaming(Q, X_R_bank, k=K, q_batch=min(8192, Q.shape[0]), x_batch=50_000)

    # Move queries to GPU for CAGRA
    Q_gpu = {name: cpu2gpu(Q) for name, Q in variants.items()}

    # ----------------- EVAL: CAGRA -----------------
    print("\n=== CAGRA: Recall@10 & QPS vs Budget ===")
    cagra_results = {}
    for name in ["baseline","proj_ep1","proj_ep2","proj_ep3"]:
        print(f"\nVariant: {name}")
        rows = []
        for B in BUDGETS:
            I_pred, qps = cagra_search(cagra_index, Q_gpu[name], K, iters=B, itopk=256, width=1)
            r = recall_at_k(I_pred, GTs[name], K)
            rows.append((B, r, qps))
            print(f"  B={B:3d}  R@{K}={r:.4f}  QPS={qps:,.0f}")
        cagra_results[name] = rows

    # ----------------- EVAL: IVF (nprobe = budget) -----------------
    print("\n=== IVF (GPU): Recall@10 & QPS vs nprobe ===")
    ivf_results = {}
    for name in ["baseline","proj_ep1","proj_ep2","proj_ep3"]:
        print(f"\nVariant: {name}")
        rows = []
        Q_cpu = variants[name]  # FAISS takes numpy (CPU)
        for B in BUDGETS:
            I_pred, qps, used_nprobe = ivf_search(ivf_index, Q_cpu, K, nprobe=B)
            r = recall_at_k(I_pred, GTs[name], K)
            rows.append((used_nprobe, r, qps))
            print(f"  nprobe={used_nprobe:4d}  R@{K}={r:.4f}  QPS={qps:,.0f}")
        ivf_results[name] = rows

    # Compact summaries
    print("\n=== SUMMARY (CAGRA, R@10) ===")
    for name, rows in cagra_results.items():
        line = f"{name:10s}: " + "  ".join([f"B{B:3d}:{r:.3f}" for (B,r,_) in rows])
        print(line)

    print("\n=== SUMMARY (IVF, R@10 vs nprobe) ===")
    for name, rows in ivf_results.items():
        line = f"{name:10s}: " + "  ".join([f"n{nb:3d}:{r:.3f}" for (nb,r,_) in rows])
        print(line)

    print("\nDone.")

if __name__ == "__main__":
    main()

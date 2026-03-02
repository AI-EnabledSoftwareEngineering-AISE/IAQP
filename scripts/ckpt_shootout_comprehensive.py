#!/usr/bin/env python3
# ckpt_shootout_cagara.py
# 
# TRAINING: CAGRA backend (GPU-accelerated CAGRA from cuVS)
# TESTING: CAGRA + IVF (generalization test across backends)
#
# One fixed bank (500k), 5k queries, 4 variants (baseline + 3 ckpts).
# Builds ONE CAGRA index (on GPU) and ONE IVF index (GPU if available).
# Budgets 10..100 step 10. IVF uses nprobe as the "budget".
# Tests generalization: CAGRA-trained model → CAGRA/IVF evaluation
# Supports both 'laion' and 't2i' datasets
# Minimal + hard-coded as requested.

import os, sys, time, random, math
import argparse
import json
import numpy as np
import cupy as cp
import torch
import torch.nn.functional as F
from pathlib import Path

# --- repo imports (match current tree) ---
current_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(current_dir))
ARTIFACT_ROOT = current_dir / "notebooks" / "outputs"

from dataset_loader import LaionDatasetLoader, T2IDatasetLoader
from core.utils import ResidualProjector, PCARSpace, brute_force_topk_streaming

# --- cuVS CAGRA ---
from cuvs.neighbors import cagra

# --- FAISS IVF ---
try:
    import faiss
except Exception:
    faiss = None

# ----------------- dataset configuration -----------------
def get_dataset_config(dataset_name, size="3m"):
    """Get dataset-specific configuration based on dataset and size.
    
    Args:
        dataset_name: Dataset name (laion, datacomp, t2i)
        size: Model size ('3m', '8.2m' for datacomp, or '10m')
    """
    if dataset_name.lower() == "laion":
        if size == "10m":
            data_path = "/ssd/hamed/ann/laion/precompute/laion_cache_up_to_10.pkl"
        else:  # 3m
            data_path = "/ssd/hamed/ann/laion/precompute/laion_cache_up_to_3.pkl"
        return {
            "data_path": data_path,
            "dataset_loader": LaionDatasetLoader,
            "hidden_dim": 512,  # LAION uses 512 hidden dim (original)
            "feature_dim": None  # Will be detected from cache (may be PCA-reduced)
        }
    elif dataset_name.lower() == "datacomp":
        # DataComp cache is LAION-compatible; reuse LaionDatasetLoader
        if size == "8.2m":
            data_path = "/ssd/hamed/ann/datacomp_small/precompute/datacomp_cache_up_to_8200000.pkl"
        else:  # 3m
            data_path = "/ssd/hamed/ann/datacomp_small/precompute/datacomp_cache_up_to_3000000.pkl"
        return {
            "data_path": data_path,
            "dataset_loader": LaionDatasetLoader,
            "hidden_dim": 512,  # Original hidden dim
            "feature_dim": None  # Will be detected from cache (may be PCA-reduced)
        }
    elif dataset_name.lower() == "t2i":
        if size == "10m":
            data_path = "/ssd/hamed/ann/t2i-10M/precompute/t2i_cache_up_to_0.pkl"
        else:  # 3m
            data_path = "/ssd/hamed/ann/t2i-10M/precompute/t2i_cache_up_to_3.pkl"
        return {
            "data_path": data_path,
            "dataset_loader": T2IDatasetLoader,
            "hidden_dim": 200,  # T2I uses 200 hidden dim (original)
            "feature_dim": None  # Will be detected from cache (may be PCA-reduced)
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Use 'laion', 'datacomp' or 't2i'")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Comprehensive Checkpoint Shootout")
parser.add_argument("--dataset", type=str, choices=["laion", "datacomp", "t2i"], default="laion",
                   help="Dataset to use: 'laion', 'datacomp' or 't2i'")
parser.add_argument("--train_backend", type=str, 
                   choices=["ivf", "cagra", "exact_k", "ivf_only", "cagra_only"], 
                   default="cagra",
                   help="Training backend: ivf, cagra, exact_k, ivf_only, cagra_only")
parser.add_argument("--size", type=str, choices=["3m", "8.2m", "10m"], default=None,
                   help="Model size: '3m', '8.2m' (for datacomp), or '10m'. Default: dataset and backend dependent")
parser.add_argument("--full_dataset", action="store_true", 
                   help="Use full dataset for evaluation (no subsampling)")
parser.add_argument("--data_path", type=str, default=None,
                   help="Override data path (default: use dataset-specific default)")
parser.add_argument("--at_k", type=str, default="10",
                   help="Comma-separated list of k values for Recall@k (e.g., 10,20,30). Default: 10.")
args = parser.parse_args()

# Set default size based on dataset and backend if not specified
if args.size is None:
    if args.dataset == "datacomp":
        # For datacomp, use 8.2m for CAGRA (larger), 3m for others
        if args.train_backend in ["cagra", "cagra_only"]:
            args.size = "8.2m"
        else:
            args.size = "3m"
    else:
        # For laion and t2i, use 10m for CAGRA, 3m for others
        if args.train_backend in ["cagra", "cagra_only"]:
            args.size = "10m"
        else:
            args.size = "3m"

# Get dataset configuration (size-aware)
config = get_dataset_config(args.dataset, size=args.size)
DATA_PATH = args.data_path if args.data_path is not None else config["data_path"]
DatasetLoader = config["dataset_loader"]
HIDDEN_DIM = config["hidden_dim"]
# FEATURE_DIM will be detected from actual data (may be PCA-reduced)
FEATURE_DIM = config.get("feature_dim")  # May be None, will be detected

# Parse Recall@k values (comma-separated)
def parse_at_k(at_k_str: str):
    parts = at_k_str.replace(",", " ").split()
    vals = []
    for p in parts:
        try:
            v = int(p)
            if v > 0:
                vals.append(v)
        except:
            pass
    if not vals:
        vals = [10]
    return sorted(set(vals))

AT_K = parse_at_k(args.at_k)
# Choose smallest as primary for selection; largest for search budget
PRIMARY_K = AT_K[0]
MAX_K = max(AT_K)

# Generate checkpoint paths based on dataset and flags
def detect_backend_from_args(args):
    """Detect backend from command line arguments."""
    return args.train_backend

def get_checkpoint_paths(dataset, backend, size="3m"):
    """Get checkpoint paths for a specific dataset/backend combination.
    
    Args:
        dataset: Dataset name (laion, datacomp, t2i)
        backend: Backend type (exact_k, cagra, cagra_only, ivf, ivf_only)
        size: Model size ('3m', '8.2m' for datacomp, or '10m')
    """
    checkpoint_root = ARTIFACT_ROOT / "checkpoints"
    if backend == "exact_k":
        base_path = checkpoint_root / f"{dataset}-{size}_exact_k_up_to_e3"
    elif backend == "cagra":
        base_path = checkpoint_root / f"{dataset}-{size}_cuvs_cagra_up_to_e3"
    elif backend == "cagra_only":
        # Special case: t2i uses "cagra_only" without "cuvs_" prefix
        if dataset == "t2i":
            base_path = checkpoint_root / f"{dataset}-{size}_cagra_only_up_to_e3"
        else:
            base_path = checkpoint_root / f"{dataset}-{size}_cuvs_cagra_only_up_to_e3"
    elif backend == "ivf":
        base_path = checkpoint_root / f"{dataset}-{size}_ivf_up_to_e3"
    elif backend == "ivf_only":
        base_path = checkpoint_root / f"{dataset}-{size}_ivf_only_up_to_e3"
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    return {
        "ep1": str(base_path / "epoch_1.pt"),
        "ep2": str(base_path / "epoch_2.pt"),
        "ep3": str(base_path / "epoch_3.pt"),
    }

# Detect backend and get checkpoint paths
backend = detect_backend_from_args(args)
CKPTS = get_checkpoint_paths(args.dataset, backend, size=args.size)

print(f"🔧 Dataset: {args.dataset}")
print(f"🔧 Backend: {backend}")
print(f"🔧 Model size: {args.size}")
print(f"📁 Data path: {DATA_PATH}")
print(f"📦 Checkpoints: {list(CKPTS.keys())}")

# Choose which checkpoint's PCA defines the bank/search space (fixed index space)
USE_PCA_FROM = "ep2"

# Subsampling sizes (can be overridden by --full_dataset)
N_BANK = None  # Will be set based on --full_dataset flag
NQ     = None  # Will be set based on --full_dataset flag

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

def cagra_search(idx, Q_gpu, k, iters, itopk=256, width=1, warmup=True):
    itopk = max(itopk, k)
    sp = cagra.SearchParams(itopk_size=itopk, search_width=max(1,width), algo="auto", max_iterations=max(1,int(iters)))
    
    # Warmup: run a few searches to initialize GPU state
    if warmup:
        for _ in range(3):
            _ = cagra.search(sp, idx, Q_gpu[:min(100, Q_gpu.shape[0])], k)
        cp.cuda.Stream.null.synchronize()  # Synchronize GPU before timing
    
    # Actual measurement with GPU synchronization
    cp.cuda.Stream.null.synchronize()  # Ensure previous operations complete
    t0 = time.perf_counter()
    D, I = cagra.search(sp, idx, Q_gpu, k)
    cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
    dt = time.perf_counter() - t0
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

def ivf_search(index, Q: np.ndarray, k: int, nprobe: int, warmup=True):
    # clamp nprobe to valid range
    nlist = index.nlist if hasattr(index, "nlist") else nprobe
    nprobe = int(max(1, min(nprobe, nlist)))
    # set nprobe on both CPU and GPU index types
    try:
        index.nprobe = nprobe
    except Exception:
        # some GPU wrappers use ParameterSpace, but most expose nprobe directly
        pass
    
    # Warmup: run a few searches to initialize GPU/CPU state
    if warmup:
        Q_warmup = Q[:min(100, Q.shape[0])].astype(np.float32, order="C")
        for _ in range(3):
            _ = index.search(Q_warmup, k)
        # Synchronize if GPU index
        is_gpu_index = hasattr(index, "getDevice")
        if not is_gpu_index and faiss is not None:
            try:
                is_gpu_index = hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0
            except:
                pass
        if is_gpu_index:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except:
                pass
    
    # Actual measurement
    t0 = time.perf_counter()
    _D, I = index.search(Q.astype(np.float32, order="C"), k)
    # Synchronize if GPU index
    is_gpu_index = hasattr(index, "getDevice")
    if not is_gpu_index and faiss is not None:
        try:
            is_gpu_index = hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0
        except:
            pass
    if is_gpu_index:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except:
            pass
    dt = time.perf_counter() - t0
    qps = Q.shape[0] / max(dt, 1e-9)
    return I.astype(np.int64, copy=False), qps, nprobe

# ----------------- results saving -----------------
def save_results_to_json(
    cagra_results,
    ivf_results,
    backend,
    dataset,
    selector,
    size="3m",
    output_dir=None,
    at_k=None,
):
    """
    Save evaluation results to JSON file with checkpoint selection results.
    
    Args:
        cagra_results: CAGRA evaluation results
        ivf_results: IVF evaluation results  
        backend: Training backend
        dataset: Dataset name
        selector: CheckpointSelector instance
        size: Model size ('3m', '8.2m' for datacomp, or '10m')
        output_dir: Output directory for results
    """
    at_k = at_k or [10]
    # Create output directory
    output_dir = Path(output_dir) if output_dir else ARTIFACT_ROOT / "eval_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename based on checkpoint root
    # Map backend to checkpoint naming convention
    backend_map = {
        "exact_k": "exact_k",
        "cagra": "cuvs_cagra",
        "cagra_only": "cuvs_cagra_only",
        "ivf": "ivf",
        "ivf_only": "ivf_only"
    }
    backend_name = backend_map.get(backend, backend)
    # Special case: t2i cagra_only uses "cagra_only" without "cuvs_" prefix in checkpoint dir
    # but we still use "cuvs_cagra_only" in the results filename for consistency
    checkpoint_root = f"{dataset}-{size}_{backend_name}_up_to_e3"
    filename = f"{checkpoint_root}_comprehensive_results.json"
    filepath = output_dir / filename
    
    # Get checkpoint selection results
    selection_summary = selector.get_selection_summary(
        cagra_results, ivf_results, backend, at_k=AT_K
    )
    
    # Prepare results structure
    results = {
        "metadata": {
            "dataset": dataset,
            "backend": backend,
            "checkpoint_root": checkpoint_root,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_type": "comprehensive_checkpoint_shootout",
            "at_k": at_k,
        },
        "checkpoint_selection": {
            "selected_checkpoints": selection_summary,
            "selection_logic": {
                "best_mode": "Selects checkpoint with best performance on its own backend (weighted by budget)",
                "generalization_mode": "Selects checkpoint with best cross-backend performance (weighted by budget) with self-backend constraint"
            }
        },
        "all_results": {
            "cagra_results": {},
            "ivf_results": {}
        },
        "selected_checkpoints": {
            "best_performance": {},
            "best_generalization": {}
        },
        "summaries": {
            "cagra_summary": {},
            "ivf_summary": {}
        }
    }
    
    # Convert results to JSON-serializable format
    for variant, data in cagra_results.items():
        rows = []
        for row in data:
            rec = row["recall"]
            rows.append({
                "budget": int(row["budget"]),
                "recall": {str(k): float(rec[k]) for k in rec},
                "qps": float(row["qps"]),
            })
        results["all_results"]["cagra_results"][variant] = rows
    
    for variant, data in ivf_results.items():
        rows = []
        for row in data:
            rec = row["recall"]
            rows.append({
                "nprobe": int(row["nprobe"]),
                "recall": {str(k): float(rec[k]) for k in rec},
                "qps": float(row["qps"]),
            })
        results["all_results"]["ivf_results"][variant] = rows
    
    # Extract best checkpoint results per k
    if "best" in selection_summary:
        results["selected_checkpoints"]["best_performance"] = {}
        for k, best_epoch in selection_summary["best"].items():
            best_variant = f"proj_{best_epoch}"
            if backend in ["cagra", "cagra_only"]:
                results["selected_checkpoints"]["best_performance"][str(k)] = {
                    "checkpoint": best_epoch,
                    "selection_criteria": "best performance on training backend",
                    "training_backend": "cagra",
                    "results": {
                        "cagra": results["all_results"]["cagra_results"][best_variant]
                    }
                }
            elif backend in ["ivf", "ivf_only"]:
                results["selected_checkpoints"]["best_performance"][str(k)] = {
                    "checkpoint": best_epoch,
                    "selection_criteria": "best performance on training backend",
                    "training_backend": "ivf",
                    "results": {
                        "ivf": results["all_results"]["ivf_results"][best_variant]
                    }
                }
    
    if "generalization" in selection_summary:
        results["selected_checkpoints"]["best_generalization"] = {}
        for k, gen_epoch in selection_summary["generalization"].items():
            gen_variant = f"proj_{gen_epoch}"
            results["selected_checkpoints"]["best_generalization"][str(k)] = {
                "checkpoint": gen_epoch,
                "selection_criteria": "best cross-backend generalization",
                "results": {
                    "cagra": results["all_results"]["cagra_results"][gen_variant],
                    "ivf": results["all_results"]["ivf_results"][gen_variant]
                }
            }
    
    # Add summary statistics
    for variant, data in cagra_results.items():
        k_stats = {}
        for k in at_k:
            recalls = [row["recall"][k] for row in data]
            k_stats[str(k)] = {
            "avg_recall": float(np.mean(recalls)),
            "max_recall": float(np.max(recalls)),
            "min_recall": float(np.min(recalls))
        }
        results["summaries"]["cagra_summary"][variant] = k_stats
    
    for variant, data in ivf_results.items():
        k_stats = {}
        for k in at_k:
            recalls = [row["recall"][k] for row in data]
            k_stats[str(k)] = {
            "avg_recall": float(np.mean(recalls)),
            "max_recall": float(np.max(recalls)),
            "min_recall": float(np.min(recalls))
        }
        results["summaries"]["ivf_summary"][variant] = k_stats
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Results saved to: {filepath}")
    return str(filepath)

# ----------------- main -----------------
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    np.random.seed(123); random.seed(123)

    print("=" * 60)
    print("CAGRA CHECKPOINT SHOOTOUT")
    print("TRAINING: CAGRA backend (GPU-accelerated CAGRA from cuVS)")
    print("TESTING: CAGRA + IVF (generalization test across backends)")
    print("=" * 60)
    
    print(f"Loading {args.dataset.upper()} dataset…")
    ds = DatasetLoader(DATA_PATH)
    X_train_full, T_train_full, _ = ds.get_train_data()
    X_val_full,   T_val_full,   *_ = ds.get_split_data("val")
    print(f"Train: X={X_train_full.shape}, T={T_train_full.shape} | Val: T={T_val_full.shape}")

    # Ensure dataset data is loaded for PCA check
    if ds.data is None:
        ds.data = ds.load_data()

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

    # Check if cache has PCA (features already PCA-reduced)
    cache_has_pca = False
    if hasattr(ds, 'data') and ds.data is not None:
        if "pca" in ds.data and ds.data["pca"] is not None:
            cache_has_pca = True
            print("✅ Cache has PCA: features are already PCA-reduced")
            print(f"   Feature dim: {X_train_full.shape[1]} (PCA-reduced)")
        else:
            print("⚠️  Cache does not have PCA: features are in original space")
            print(f"   Feature dim: {X_train_full.shape[1]} (original)")

    # Detect actual feature dimension from loaded data (may be PCA-reduced)
    actual_feature_dim = X_train_full.shape[1]
    # Use local variable for feature_dim (may override global)
    feature_dim = FEATURE_DIM if FEATURE_DIM is not None else actual_feature_dim
    if feature_dim != actual_feature_dim:
        print(f"⚠️  Warning: Config feature_dim={feature_dim} != actual={actual_feature_dim}, using actual")
        feature_dim = actual_feature_dim
    print(f"✅ Using feature dimension: {feature_dim}")

    # Load checkpoints
    print("\nLoading checkpoints…")
    models = {}
    rotations = {}
    for tag, path in CKPTS.items():
        print(f"  • {tag}: {path}")
        # Use detected feature dim (may be PCA-reduced)
        models[tag], rotations[tag] = load_ckpt(path, device="cuda", dim=feature_dim)

    # Fix bank space with PCA from USE_PCA_FROM
    print(f"\nUsing PCA from {USE_PCA_FROM} to define the bank/search space")
    R_ref = rotations[USE_PCA_FROM]

    print("Rotating bank…")
    if cache_has_pca:
        # Features are already PCA-reduced, use as-is
        X_R_bank = X_bank.astype(np.float32, copy=False)
        print("  ✅ Using pre-computed PCA-reduced features (no transform needed)")
    else:
        # Apply PCA transform
        X_R_bank = R_ref.transform(X_bank)
        print("  ✅ Applied PCA transform to features")
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
    if cache_has_pca:
        # Features are already PCA-reduced, use as-is
        Q_base = T_val.astype(np.float32, copy=False)
        print("  ✅ Baseline queries: using pre-computed PCA-reduced features")
    else:
        # Apply PCA transform
        Q_base = R_ref.transform(T_val)
        print("  ✅ Baseline queries: applied PCA transform")
    Q_base = l2n(Q_base)
    assert_unit("Q_base (baseline)", Q_base)
    variants["baseline"] = Q_base
    
    for tag in ["ep1","ep2","ep3"]:
        Qp = project_np(models[tag], T_val, device="cuda", batch=65536)
        if cache_has_pca:
            # Model output is already in PCA space (same as input), use as-is
            Qp_R = Qp.astype(np.float32, copy=False)
            print(f"  ✅ Projected queries {tag}: using model output (already in PCA space)")
        else:
            # Apply PCA transform to model output
            Qp_R = R_ref.transform(Qp)
            print(f"  ✅ Projected queries {tag}: applied PCA transform")
        Qp_R = l2n(Qp_R)
        assert_unit(f"Q_proj_{tag}", Qp_R)
        variants[f"proj_{tag}"] = Qp_R

    # Exact GTs per variant
    print("\nComputing exact GTs (projected space) for all variants…")
    GTs = {k: {} for k in AT_K}
    for k in AT_K:
        for name, Q in variants.items():
            print(f"  • GT for {name} @k={k} …")
            GTs[k][name] = brute_force_topk_streaming(Q, X_R_bank, k=k, q_batch=min(8192, Q.shape[0]), x_batch=50_000)

    # Move queries to GPU for CAGRA
    Q_gpu = {name: cpu2gpu(Q) for name, Q in variants.items()}

    # ----------------- EVAL: CAGRA -----------------
    print(f"\n=== CAGRA: Recall@{AT_K} & QPS vs Budget ===")
    cagra_results = {}
    for name in ["baseline","proj_ep1","proj_ep2","proj_ep3"]:
        print(f"\nVariant: {name}")
        # Warmup once per variant (use first budget for warmup)
        print(f"  Warming up GPU...")
        first_budget = BUDGETS[0]
        sp_warmup = cagra.SearchParams(itopk_size=256, search_width=1, algo="auto", max_iterations=max(1, int(first_budget)))
        for _ in range(5):
            _ = cagra.search(sp_warmup, cagra_index, Q_gpu[name][:min(100, Q_gpu[name].shape[0])], MAX_K)
        cp.cuda.Stream.null.synchronize()
        
        rows = []
        for B in BUDGETS:
            I_pred, qps = cagra_search(cagra_index, Q_gpu[name], MAX_K, iters=B, itopk=256, width=1, warmup=False)
            recalls = {k: recall_at_k(I_pred, GTs[k][name], k) for k in AT_K}
            rows.append({"budget": B, "recall": recalls, "qps": qps})
            recall_str = "  ".join([f"R@{k}={recalls[k]:.4f}" for k in AT_K])
            print(f"  B={B:3d}  {recall_str}  QPS={qps:,.0f}")
        cagra_results[name] = rows

    # ----------------- EVAL: IVF (nprobe = budget) -----------------
    print(f"\n=== IVF (GPU): Recall@{AT_K} & QPS vs nprobe ===")
    ivf_results = {}
    for name in ["baseline","proj_ep1","proj_ep2","proj_ep3"]:
        print(f"\nVariant: {name}")
        # Warmup once per variant (use first budget for warmup)
        print(f"  Warming up...")
        Q_cpu = variants[name]  # FAISS takes numpy (CPU)
        first_budget = BUDGETS[0]
        nlist = ivf_index.nlist if hasattr(ivf_index, "nlist") else first_budget
        nprobe_warmup = int(max(1, min(first_budget, nlist)))
        try:
            ivf_index.nprobe = nprobe_warmup
        except Exception:
            pass
        Q_warmup = Q_cpu[:min(100, Q_cpu.shape[0])].astype(np.float32, order="C")
        for _ in range(5):
            _ = ivf_index.search(Q_warmup, MAX_K)
        # Synchronize if GPU index
        is_gpu_index = hasattr(ivf_index, "getDevice")
        if not is_gpu_index and faiss is not None:
            try:
                is_gpu_index = hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0
            except:
                pass
        if is_gpu_index:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except:
                pass
        
        rows = []
        for B in BUDGETS:
            I_pred, qps, used_nprobe = ivf_search(ivf_index, Q_cpu, MAX_K, nprobe=B, warmup=False)
            recalls = {k: recall_at_k(I_pred, GTs[k][name], k) for k in AT_K}
            rows.append({"nprobe": used_nprobe, "recall": recalls, "qps": qps})
            recall_str = "  ".join([f"R@{k}={recalls[k]:.4f}" for k in AT_K])
            print(f"  nprobe={used_nprobe:4d}  {recall_str}  QPS={qps:,.0f}")
        ivf_results[name] = rows

    # Compact summaries
    print(f"\n=== SUMMARY (CAGRA, R@{AT_K}) ===")
    for name, rows in cagra_results.items():
        line_parts = []
        for row in rows:
            recall_part = ",".join([f"k{k}={row['recall'][k]:.3f}" for k in AT_K])
            line_parts.append(f"B{row['budget']:3d}:{recall_part}")
        line = f"{name:10s}: " + "  ".join(line_parts)
        print(line)

    print(f"\n=== SUMMARY (IVF, R@{AT_K} vs nprobe) ===")
    for name, rows in ivf_results.items():
        line_parts = []
        for row in rows:
            recall_part = ",".join([f"k{k}={row['recall'][k]:.3f}" for k in AT_K])
            line_parts.append(f"n{row['nprobe']:3d}:{recall_part}")
        line = f"{name:10s}: " + "  ".join(line_parts)
        print(line)

    print("\nDone.")
    
    # Checkpoint selection
    print("\n=== CHECKPOINT SELECTION ===")
    from .checkpoint_selector import CheckpointSelector
    
    selector = CheckpointSelector()
    
    # Select best checkpoint based on backend (per k)
    selection_summary = {}
    if backend == "exact_k":
        selection_summary["generalization"] = selector.select_best_checkpoint(
            cagra_results, ivf_results, backend="exact_k", mode="generalization", at_k=AT_K
        )
        print(f"🎯 Best checkpoints (generalization) per k: {selection_summary['generalization']}")
    elif backend in ["cagra", "cagra_only"]:
        selection_summary["best"] = selector.select_best_checkpoint(
            cagra_results, ivf_results, backend="cuvs_cagra", mode="best", at_k=AT_K
        )
        print(f"🎯 Best checkpoints (CAGRA self-backend) per k: {selection_summary['best']}")
        
        selection_summary["generalization"] = selector.select_best_checkpoint(
            cagra_results, ivf_results, backend="cuvs_cagra", mode="generalization", at_k=AT_K
        )
        print(f"🎯 Best checkpoints (generalization) per k: {selection_summary['generalization']}")
    elif backend in ["ivf", "ivf_only"]:
        selection_summary["best"] = selector.select_best_checkpoint(
            cagra_results, ivf_results, backend="ivf", mode="best", at_k=AT_K
        )
        print(f"🎯 Best checkpoints (IVF self-backend) per k: {selection_summary['best']}")
        
        selection_summary["generalization"] = selector.select_best_checkpoint(
            cagra_results, ivf_results, backend="ivf", mode="generalization", at_k=AT_K
        )
        print(f"🎯 Best checkpoints (generalization) per k: {selection_summary['generalization']}")
    
    # Save results to JSON
    print("\n=== SAVING RESULTS ===")
    results_file = save_results_to_json(
        cagra_results, ivf_results, backend, args.dataset, selector, size=args.size, at_k=AT_K
    )
    
    print(f"\n✅ Comprehensive evaluation complete!")
    print(f"📊 Results saved to: {results_file}")

if __name__ == "__main__":
    main()

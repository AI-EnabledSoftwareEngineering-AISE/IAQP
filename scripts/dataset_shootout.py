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
from core.utils import ResidualProjector, PCARSpace, brute_force_topk_streaming, suggested_nlist

# --- cuVS CAGRA ---
from cuvs.neighbors import cagra

# --- FAISS IVF ---
try:
    import faiss
except Exception:
    faiss = None

# ----------------- dataset configuration -----------------
def get_dataset_config(dataset_name, size="3m", shared_pca: bool = False):
    """Get dataset-specific configuration based on dataset and size.
    
    Args:
        dataset_name: Dataset name (laion, datacomp, t2i)
        size: Model size ('3m', '4m', '8.2m' for datacomp, or '10m')
    """
    if dataset_name.lower() == "laion":
        if shared_pca and size == "10m":
            # Shared-PCA LAION-10M cache (dim=256)
            data_path = "/ssd/hamed/ann/laion/precompute/laion_cache_shared_pca.pkl"
            feature_dim = 256
            hidden_dim = 256
        elif size == "10m":
            data_path = "/ssd/hamed/ann/laion/precompute/laion_cache_up_to_10.pkl"
            feature_dim = 512
            hidden_dim = 512
        elif size == "4m":
            data_path = "/ssd/hamed/ann/laion/precompute/laion_cache_up_to_3.pkl"
            feature_dim = 512
            hidden_dim = 512
        else:  # default 3m
            data_path = "/ssd/hamed/ann/laion/precompute/laion_cache_up_to_2.pkl"
            feature_dim = 512
            hidden_dim = 512
        return {
            "data_path": data_path,
            "dataset_loader": LaionDatasetLoader,
            "hidden_dim": hidden_dim,
            "feature_dim": feature_dim
        }
    elif dataset_name.lower() == "datacomp":
        # DataComp cache is LAION-compatible; reuse LaionDatasetLoader
        if shared_pca and size == "8.2m":
            # Shared-PCA DataComp-8.2M cache (dim=256)
            data_path = "/ssd/hamed/ann/datacomp_small/precompute/datacomp_cache_shared_pca.pkl"
            feature_dim = 256
            hidden_dim = 256
        elif size == "8.2m":
            data_path = "/ssd/hamed/ann/datacomp_small/precompute/datacomp_cache_up_to_8200000.pkl"
            feature_dim = 512
            hidden_dim = 512
        elif size == "4m":
            data_path = "/ssd/hamed/ann/datacomp_small/precompute/datacomp_cache_up_to_4000000.pkl"
            feature_dim = 512
            hidden_dim = 512
        else:  # default 3m
            data_path = "/ssd/hamed/ann/datacomp_small/precompute/datacomp_cache_up_to_3000000.pkl"
            feature_dim = 512
            hidden_dim = 512
        return {
            "data_path": data_path,
            "dataset_loader": LaionDatasetLoader,
            "hidden_dim": hidden_dim,
            "feature_dim": feature_dim
        }
    elif dataset_name.lower() == "t2i":
        if size == "10m":
            data_path = "/ssd/hamed/ann/t2i-10M/precompute/t2i_cache_up_to_0.pkl"
        else:  # 3m
            data_path = "/ssd/hamed/ann/t2i-10M/precompute/t2i_cache_up_to_3.pkl"
        return {
            "data_path": data_path,
            "dataset_loader": T2IDatasetLoader,
            "hidden_dim": 200,  # T2I uses 200 hidden dim
            "feature_dim": 200  # T2I feature dim
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Use 'laion', 'datacomp' or 't2i'")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Dataset Shootout (Cross-dataset generalizability)")
parser.add_argument("--dataset_trained", type=str, choices=["laion", "datacomp"], default="laion",
                   help="Dataset the checkpoints were trained on (laion, datacomp)")
parser.add_argument("--dataset_test", type=str, choices=["laion", "datacomp"], default="laion",
                   help="Dataset to evaluate on (laion, datacomp)")
parser.add_argument("--train_backend", type=str, 
                   choices=["ivf", "cagra", "exact_k", "ivf_only", "cagra_only"], 
                   default="cagra",
                   help="Training backend of the checkpoints: ivf, cagra, exact_k, ivf_only, cagra_only")
parser.add_argument("--size_trained", type=str, choices=["3m", "4m", "8.2m", "10m"], default=None,
                   help="Model size for trained dataset: '3m', '4m', '8.2m' (datacomp), or '10m'. Default: dataset dependent")
parser.add_argument("--size_test", type=str, choices=["3m", "4m", "8.2m", "10m"], default=None,
                   help="Model size for test dataset: '3m', '4m', '8.2m' (datacomp), or '10m'. Default: dataset dependent")
parser.add_argument("--full_dataset", action="store_true", 
                   help="Use full dataset for evaluation (no subsampling)")
parser.add_argument("--pca_regime", type=str, choices=["ckpt", "online"], default="ckpt",
                   help="ckpt: use PCA from training checkpoint; online: fit PCA on Dataset B (train)")
parser.add_argument("--ivf_nlist_hint", type=int, default=None,
                   help="Optional ivf_nlist hint for IVF builds (e.g., 8000 for 4M/512-dim runs)")
parser.add_argument("--shared_pca", action="store_true",
                   help="Use shared-PCA caches & checkpoints (dim=256, *_sharedpca_e3 roots)")
args = parser.parse_args()

# Set default size for trained dataset based on dataset if not specified
if args.size_trained is None:
    if args.dataset_trained == "datacomp":
        # For datacomp, default to 8.2m (largest available)
        args.size_trained = "8.2m"
    else:
        # For laion, default to 10m (largest available)
        args.size_trained = "10m"

# Set default size for test dataset based on dataset (use largest available size)
if args.size_test is None:
    if args.dataset_test == "datacomp":
        # For datacomp, default to 8.2m (largest available)
        args.size_test = "8.2m"
    else:
        # For laion, default to 10m (largest available)
        args.size_test = "10m"

# Get dataset configurations (trained-on for ckpts, test-for data) - size-aware
cfg_tr = get_dataset_config(args.dataset_trained, size=args.size_trained, shared_pca=args.shared_pca)
cfg_te = get_dataset_config(args.dataset_test, size=args.size_test, shared_pca=args.shared_pca)

DATA_PATH = cfg_te["data_path"]
DatasetLoader = cfg_te["dataset_loader"]
HIDDEN_DIM = cfg_tr["hidden_dim"]  # projector hidden matches training
FEATURE_DIM = cfg_tr["feature_dim"]

# Generate checkpoint paths based on dataset and flags
def detect_backend_from_args(args):
    """Detect backend from command line arguments."""
    return args.train_backend

def get_checkpoint_paths(dataset, backend, size="3m", shared_pca: bool = False):
    """Get checkpoint paths for a specific dataset/backend combination.
    
    Args:
        dataset: Dataset name (laion, datacomp, t2i)
        backend: Backend type (exact_k, cagra, cagra_only, ivf, ivf_only)
        size: Model size ('3m', '4m', '8.2m' for datacomp, or '10m')
    """
    checkpoint_root = ARTIFACT_ROOT / "checkpoints"
    # Suffix depends on whether training used shared-PCA caches
    suffix = "sharedpca_e3" if shared_pca else "up_to_e3"

    if backend == "exact_k":
        base_path = checkpoint_root / f"{dataset}-{size}_exact_k_{suffix}"
    elif backend == "cagra":
        base_path = checkpoint_root / f"{dataset}-{size}_cuvs_cagra_{suffix}"
    elif backend == "cagra_only":
        # t2i uses "cagra_only" without "cuvs_" prefix, others use "cuvs_cagra_only"
        if dataset == "t2i":
            base_path = checkpoint_root / f"{dataset}-{size}_cagra_only_{suffix}"
        else:
            base_path = checkpoint_root / f"{dataset}-{size}_cuvs_cagra_only_{suffix}"
    elif backend == "ivf":
        base_path = checkpoint_root / f"{dataset}-{size}_ivf_{suffix}"
    elif backend == "ivf_only":
        base_path = checkpoint_root / f"{dataset}-{size}_ivf_only_{suffix}"
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    return {
        "ep1": str(base_path / "epoch_1.pt"),
        "ep2": str(base_path / "epoch_2.pt"),
        "ep3": str(base_path / "epoch_3.pt"),
    }

"""Use the TRAINED dataset name here for checkpoint roots."""
# Detect backend and get checkpoint paths
backend = detect_backend_from_args(args)
CKPTS = get_checkpoint_paths(args.dataset_trained, backend, size=args.size_trained, shared_pca=args.shared_pca)

# Convert relative paths to absolute paths and validate checkpoints exist
CKPTS_ABS = {}
missing_checkpoints = []

print(f"🔧 Trained on: {args.dataset_trained} (size: {args.size_trained})")
print(f"🔧 Testing on: {args.dataset_test} (size: {args.size_test})")
print(f"🔧 Backend: {backend}")
print(f"📁 Data path: {DATA_PATH}")
print(f"\n📦 Validating checkpoints...")

for tag, rel_path in CKPTS.items():
    abs_path = (current_dir / rel_path).resolve()
    CKPTS_ABS[tag] = str(abs_path)
    
    if not abs_path.exists():
        missing_checkpoints.append(str(abs_path))
        print(f"  ❌ {tag}: {abs_path} - NOT FOUND")
    else:
        print(f"  ✅ {tag}: {abs_path}")

if missing_checkpoints:
    print(f"\n❌ ERROR: {len(missing_checkpoints)} checkpoint(s) not found!")
    print("Please check the checkpoint paths and ensure they exist.")
    print("\nMissing checkpoints:")
    for path in missing_checkpoints:
        print(f"  - {path}")
    sys.exit(1)

# Use absolute paths
CKPTS = CKPTS_ABS
print(f"\n✅ All checkpoints validated successfully!")

# Choose which checkpoint's PCA defines the bank/search space (fixed index space)
USE_PCA_FROM = "ep2"

# Subsampling sizes (can be overridden by --full_dataset)
N_BANK = None  # Will be set based on --full_dataset flag
NQ     = None  # Will be set based on --full_dataset flag
K      = 10

# CAGRA build config - will be loaded from config file based on dataset and size
CAGRA_CFG = None  # Will be set after loading data

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
    
    # Auto-detect dimension from checkpoint if not provided
    if dim is None:
        # Try to get from checkpoint metadata first
        if "dim" in ck:
            model_dim = ck["dim"]
        else:
            # Infer from model state dict (first layer input dimension)
            msd = ck["model_state_dict"]
            if any(k.startswith("module.") for k in msd):
                msd_clean = {k.replace("module.",""):v for k,v in msd.items()}
            else:
                msd_clean = msd
            # First linear layer: g.0.weight has shape [hidden, input_dim]
            if "g.0.weight" in msd_clean:
                model_dim = msd_clean["g.0.weight"].shape[1]
            else:
                # Fallback to default
                model_dim = 512
    else:
        model_dim = dim
    
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
def load_cagra_config(dataset: str, size: str, feature_dim: int, bank_size: int):
    """Load CAGRA config from cagara_cfg directory based on dataset, size, and feature dimension.
    
    Args:
        dataset: Dataset name (laion, datacomp, t2i)
        size: Model size string (3m, 4m, 8.2m, 10m)
        feature_dim: Feature dimension (e.g., 200, 512)
        bank_size: Actual bank size (N)
    
    Returns:
        dict: CAGRA configuration
    """
    cfg_dir = current_dir / "outputs" / "cagara_cfg" / dataset
    if not cfg_dir.exists():
        print(f"⚠️  CAGRA config directory not found: {cfg_dir}")
        print("   Using default CAGRA config")
        return dict(
            graph_degree=32,
            intermediate_graph_degree=64,
            nn_descent_niter=30,
            build_algo="nn_descent",
            metric="inner_product",
        )
    
    # Find matching config file: {dataset}_N{number}_D{dim}_{hash}.json
    # Match by dataset, dimension, and closest N value
    pattern = f"{dataset}_N*_D{feature_dim}_*.json"
    matches = list(cfg_dir.glob(pattern))
    
    if not matches:
        print(f"⚠️  No CAGRA config found for {dataset}, D={feature_dim}")
        print(f"   Looking in: {cfg_dir}")
        print(f"   Pattern: {pattern}")
        print("   Using default CAGRA config")
        return dict(
            graph_degree=32,
            intermediate_graph_degree=64,
            nn_descent_niter=30,
            build_algo="nn_descent",
            metric="inner_product",
        )
    
    # Find closest match by N value
    best_match = None
    best_diff = float('inf')
    
    for match in matches:
        # Extract N value from filename: {dataset}_N{number}_D{dim}_{hash}.json
        parts = match.stem.split('_')
        for part in parts:
            if part.startswith('N'):
                try:
                    n_value = int(part[1:])
                    diff = abs(n_value - bank_size)
                    if diff < best_diff:
                        best_diff = diff
                        best_match = match
                    break
                except ValueError:
                    continue
    
    if best_match is None:
        # Fallback to first match
        best_match = matches[0]
    
    print(f"📦 Loading CAGRA config from: {best_match.name}")
    with open(best_match, 'r') as f:
        config_data = json.load(f)
    
    # Extract cfg from the JSON structure
    if "cfg" in config_data:
        cfg = config_data["cfg"].copy()
        cfg["metric"] = "inner_product"  # Always use inner_product
        return cfg
    else:
        print(f"⚠️  Config file missing 'cfg' field, using default")
        return dict(
            graph_degree=32,
            intermediate_graph_degree=64,
            nn_descent_niter=30,
            build_algo="nn_descent",
            metric="inner_product",
        )

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
def build_ivf_index_gpu(X_R_bank: np.ndarray, nlist_hint: int = None):
    assert faiss is not None, "faiss not installed"
    N, D = X_R_bank.shape
    nrm = np.linalg.norm(X_R_bank, axis=1)
    if not np.allclose(nrm, 1.0, atol=1e-4):
        raise ValueError("IVF build expects L2-normalized inputs.")
    nlist = suggested_nlist(N, nlist_hint or 0)
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
def save_results_to_json(cagra_results, ivf_results, backend, dataset, selector, size_trained="3m", size_test="3m", output_dir=None):
    """
    Save evaluation results to JSON file with checkpoint selection results.
    
    Args:
        cagra_results: CAGRA evaluation results
        ivf_results: IVF evaluation results  
        backend: Training backend
        dataset: Dataset name (test dataset)
        selector: CheckpointSelector instance
        size_trained: Model size for trained dataset ('3m', '4m', '8.2m', or '10m')
        size_test: Model size for test dataset ('3m', '4m', '8.2m', or '10m')
        output_dir: Output directory for results
    """
    # Create output directory
    output_dir = Path(output_dir) if output_dir else ARTIFACT_ROOT / "eval_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename reflecting TEST domain, TRAIN domain, backend, and PCA regime
    # Map backend to checkpoint naming convention (dataset-aware for cagra_only)
    if backend == "cagra_only":
        # t2i uses "cagra_only" without "cuvs_" prefix, others use "cuvs_cagra_only"
        if args.dataset_trained == "t2i":
            backend_name = "cagra_only"
        else:
            backend_name = "cuvs_cagra_only"
    else:
        backend_map = {
            "exact_k": "exact_k",
            "cagra": "cuvs_cagra",
            "ivf": "ivf",
            "ivf_only": "ivf_only"
        }
        backend_name = backend_map.get(backend, backend)
    
    checkpoint_root = (
        f"cross-dataset_{dataset}-{size_test}_{backend_name}_from-{args.dataset_trained}-{size_trained}_{args.pca_regime}_up_to_e3"
    )
    filename = f"{checkpoint_root}_comprehensive_results.json"
    filepath = output_dir / filename
    
    # Get checkpoint selection results
    selection_summary = selector.get_selection_summary(cagra_results, ivf_results, backend)
    
    # Prepare results structure
    results = {
        "metadata": {
            "dataset_test": dataset,                 # target dataset used to build bank+queries
            "dataset_trained": args.dataset_trained, # source dataset of checkpoints
            "train_backend": backend,                # backend family of checkpoints
            "checkpoint_root": checkpoint_root,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_type": "comprehensive_dataset_shootout",
            "pca_regime": args.pca_regime,
            "pca_source": ("checkpoint:" + USE_PCA_FROM) if args.pca_regime == "ckpt" else "online_fit_on_B_train",
            "indices_built": ["cagra", "ivf"],
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
        results["all_results"]["cagra_results"][variant] = [
            {"budget": budget, "recall": float(recall), "qps": float(qps)}
            for budget, recall, qps in data
        ]
    
    for variant, data in ivf_results.items():
        results["all_results"]["ivf_results"][variant] = [
            {"nprobe": nprobe, "recall": float(recall), "qps": float(qps)}
            for nprobe, recall, qps in data
        ]
    
    # Extract best checkpoint results
    if "best" in selection_summary:
        best_val = selection_summary["best"]
        # handle per-k dict or single value
        if isinstance(best_val, dict):
            results["selected_checkpoints"]["best_performance"] = {}
            for k, best_epoch in best_val.items():
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
        else:
            best_epoch = best_val
            best_variant = f"proj_{best_epoch}"
            if backend in ["cagra", "cagra_only"]:
                results["selected_checkpoints"]["best_performance"] = {
                    "checkpoint": best_epoch,
                    "selection_criteria": "best performance on training backend",
                    "training_backend": "cagra",
                    "results": {
                        "cagra": results["all_results"]["cagra_results"][best_variant]
                    }
                }
            elif backend in ["ivf", "ivf_only"]:
                results["selected_checkpoints"]["best_performance"] = {
                    "checkpoint": best_epoch,
                    "selection_criteria": "best performance on training backend",
                    "training_backend": "ivf",
                    "results": {
                        "ivf": results["all_results"]["ivf_results"][best_variant]
                    }
                }
    
    if "generalization" in selection_summary:
        gen_val = selection_summary["generalization"]
        if isinstance(gen_val, dict):
            results["selected_checkpoints"]["best_generalization"] = {}
            for k, gen_epoch in gen_val.items():
                gen_variant = f"proj_{gen_epoch}"
                results["selected_checkpoints"]["best_generalization"][str(k)] = {
                    "checkpoint": gen_epoch,
                    "selection_criteria": "best cross-backend generalization",
                    "results": {
                        "cagra": results["all_results"]["cagra_results"][gen_variant],
                        "ivf": results["all_results"]["ivf_results"][gen_variant]
                    }
                }
        else:
            gen_epoch = gen_val
            gen_variant = f"proj_{gen_epoch}"
            results["selected_checkpoints"]["best_generalization"] = {
                "checkpoint": gen_epoch,
                "selection_criteria": "best cross-backend generalization",
                "results": {
                    "cagra": results["all_results"]["cagra_results"][gen_variant],
                    "ivf": results["all_results"]["ivf_results"][gen_variant]
                }
            }
    
    # Add summary statistics
    for variant, data in cagra_results.items():
        recalls = [recall for _, recall, _ in data]
        results["summaries"]["cagra_summary"][variant] = {
            "avg_recall": float(np.mean(recalls)),
            "max_recall": float(np.max(recalls)),
            "min_recall": float(np.min(recalls))
        }
    
    for variant, data in ivf_results.items():
        recalls = [recall for _, recall, _ in data]
        results["summaries"]["ivf_summary"][variant] = {
            "avg_recall": float(np.mean(recalls)),
            "max_recall": float(np.max(recalls)),
            "min_recall": float(np.min(recalls))
        }
    
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
    
    print(f"Loading {args.dataset_test.upper()} dataset…")
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

    # Load checkpoints (auto-detect dimension from first checkpoint)
    print("\nLoading checkpoints…")
    models = {}
    rotations = {}
    detected_feature_dim = None
    for tag, path in CKPTS.items():
        print(f"  • {tag}: {path}")
        # Load first checkpoint to detect dimension, then use that for all
        if detected_feature_dim is None:
            # Load checkpoint to detect dimension
            ck = torch.load(path, map_location="cpu", weights_only=False)
            if "dim" in ck:
                detected_feature_dim = ck["dim"]
            else:
                msd = ck["model_state_dict"]
                if any(k.startswith("module.") for k in msd):
                    msd_clean = {k.replace("module.",""):v for k,v in msd.items()}
                else:
                    msd_clean = msd
                if "g.0.weight" in msd_clean:
                    detected_feature_dim = msd_clean["g.0.weight"].shape[1]
                else:
                    detected_feature_dim = FEATURE_DIM  # Fallback to config
            print(f"  ✓ Detected feature dimension from checkpoint: {detected_feature_dim}")
        
        models[tag], rotations[tag] = load_ckpt(path, device="cuda", dim=detected_feature_dim)
    
    # Check dimension compatibility between checkpoint and test data
    test_data_dim = X_train_full.shape[1]
    if detected_feature_dim != test_data_dim:
        print(f"\n⚠️  WARNING: Dimension mismatch detected!")
        print(f"   Checkpoint expects: {detected_feature_dim} dim")
        print(f"   Test data has: {test_data_dim} dim")
        print(f"   This may cause issues if PCA regimes don't match.")
        print(f"   Make sure both datasets use the same PCA reduction (or both use original features).")

    # Fix bank space: explicit PCA regime
    if args.shared_pca:
        print("\nShared-PCA regime detected: using cache feature space directly (no extra PCA).")
        R_ref = None
    else:
        if args.pca_regime == "online":
            print("\nFitting PCA on TEST dataset train split to define bank/search space…")
            R_ref = PCARSpace(d_keep=None, center_for_fit=True, device="cuda")
            # Fit on full train or bank subset depending on --full_dataset
            X_for_fit = X_train_full if args.full_dataset else X_bank
            R_ref.fit(X_for_fit.astype(np.float32))
        else:
            print(f"\nUsing PCA from {USE_PCA_FROM} (loaded from checkpoint) to define the bank/search space")
            R_ref = rotations[USE_PCA_FROM]

    print("Rotating bank…")
    if args.shared_pca:
        # Shared-PCA caches are already in the final 256-D space
        print(f"  ✅ Shared-PCA cache: using bank as-is (dim={X_bank.shape[1]})")
        X_R_bank = l2n(X_bank.astype(np.float32))
    else:
        # Check if test data is already PCA-reduced (dimension matches checkpoint input dim)
        ckpt_pca_input_dim = rotations[USE_PCA_FROM].W.shape[1]  # Input dimension for PCA
        test_data_dim = X_bank.shape[1]
        
        if test_data_dim == detected_feature_dim and detected_feature_dim < ckpt_pca_input_dim:
            # Test data is already PCA-reduced to match checkpoint dimension
            print(f"  ✅ Test data is already PCA-reduced ({test_data_dim} dim), using as-is")
            X_R_bank = X_bank.astype(np.float32, copy=False)
        else:
            # Apply PCA transform from checkpoint
            if test_data_dim != ckpt_pca_input_dim:
                raise ValueError(
                    f"Dimension mismatch: Checkpoint PCA expects input D={ckpt_pca_input_dim}, "
                    f"but test features have D={test_data_dim}. "
                    f"Make sure test data matches the training data dimension."
                )
            print(f"  ✅ Applying PCA transform from checkpoint (input: {ckpt_pca_input_dim} → output: {detected_feature_dim})")
            X_R_bank = R_ref.transform(X_bank)
        
        X_R_bank = l2n(X_R_bank)
    assert_unit("X_R_bank", X_R_bank)

    # Load CAGRA config based on dataset, size, and feature dimension
    global CAGRA_CFG
    if CAGRA_CFG is None:
        CAGRA_CFG = load_cagra_config(
            dataset=args.dataset_test,
            size=args.size_test,
            feature_dim=detected_feature_dim,
            bank_size=X_R_bank.shape[0]
        )
        print(f"✅ CAGRA config loaded: {CAGRA_CFG}")

    # ---- Build CAGRA once
    print("\nMoving bank to GPU & building CAGRA index…")
    X_gpu = cpu2gpu(X_R_bank)
    print(f"GPU pool used={cp.get_default_memory_pool().used_bytes()/1e9:.3f} GB  ptr={hex(X_gpu.data.ptr)}")
    cagra_index = build_cagra_index(X_gpu)
    _KEEPALIVE = [X_gpu]  # keep bank alive for CAGRA index lifetime

    # ---- Build IVF once (GPU if possible)
    print("\nBuilding IVF index…")
    ivf_index, nlist = build_ivf_index_gpu(X_R_bank, nlist_hint=args.ivf_nlist_hint)
    print(f"✓ IVF ready (nlist={nlist})")

    # Prepare 4 query variants (same bank space)
    print("\nPreparing queries (4 variants)…")
    variants = {}
    
    # Check if queries are already PCA-reduced
    query_dim = T_val.shape[1]
    if args.shared_pca:
        # Shared-PCA: queries are already in the same 256-D space as the bank
        print(f"  ✅ Shared-PCA: queries are already in final space ({query_dim} dim), using as-is")
        Q_base = l2n(T_val.astype(np.float32, copy=False))
    else:
        if query_dim == detected_feature_dim and detected_feature_dim < ckpt_pca_input_dim:
            # Queries are already PCA-reduced
            print(f"  ✅ Queries are already PCA-reduced ({query_dim} dim), using as-is")
            Q_base = T_val.astype(np.float32, copy=False)
        else:
            # Apply PCA transform
            Q_base = R_ref.transform(T_val)
        Q_base = l2n(Q_base)
    assert_unit("Q_base (baseline)", Q_base)
    variants["baseline"] = Q_base
    
    for tag in ["ep1","ep2","ep3"]:
        Qp = project_np(models[tag], T_val, device="cuda", batch=65536)
        if args.shared_pca:
            # Shared-PCA: projector outputs already live in the same 256-D space
            Qp_R = l2n(Qp.astype(np.float32, copy=False))
        else:
            if query_dim == detected_feature_dim and detected_feature_dim < ckpt_pca_input_dim:
                # Model output is already in PCA space, use as-is
                Qp_R = Qp.astype(np.float32, copy=False)
            else:
                # Apply PCA transform to model output
                Qp_R = R_ref.transform(Qp)
            Qp_R = l2n(Qp_R)
        assert_unit(f"Q_proj_{tag}", Qp_R)
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
        # Warmup once per variant (use first budget for warmup)
        print(f"  Warming up GPU...")
        first_budget = BUDGETS[0]
        sp_warmup = cagra.SearchParams(itopk_size=256, search_width=1, algo="auto", max_iterations=max(1, int(first_budget)))
        for _ in range(5):
            _ = cagra.search(sp_warmup, cagra_index, Q_gpu[name][:min(100, Q_gpu[name].shape[0])], K)
        cp.cuda.Stream.null.synchronize()
        
        rows = []
        for B in BUDGETS:
            I_pred, qps = cagra_search(cagra_index, Q_gpu[name], K, iters=B, itopk=256, width=1, warmup=False)
            r = recall_at_k(I_pred, GTs[name], K)
            rows.append((B, r, qps))
            print(f"  B={B:3d}  R@{K}={r:.4f}  QPS={qps:,.0f}")
        cagra_results[name] = rows

    # ----------------- EVAL: IVF (nprobe = budget) -----------------
    print("\n=== IVF (GPU): Recall@10 & QPS vs nprobe ===")
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
            _ = ivf_index.search(Q_warmup, K)
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
            I_pred, qps, used_nprobe = ivf_search(ivf_index, Q_cpu, K, nprobe=B, warmup=False)
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
    
    # Checkpoint selection
    print("\n=== CHECKPOINT SELECTION ===")
    from .checkpoint_selector import CheckpointSelector
    
    selector = CheckpointSelector()
    
    # Select best checkpoint based on backend
    if backend == "exact_k":
        # For exact_k, only generalization mode is valid
        best_epoch = selector.select_best_checkpoint(
            cagra_results, ivf_results, backend="exact_k", 
            mode="generalization"
        )
        print(f"🎯 Best checkpoint (generalization): {best_epoch}")
    elif backend in ["cagra", "cagra_only"]:
        # For CAGRA backends, try both modes
        best_epoch = selector.select_best_checkpoint(
            cagra_results, ivf_results, backend="cuvs_cagra", 
            mode="best"
        )
        print(f"🎯 Best checkpoint (CAGRA self-backend): {best_epoch}")
        
        gen_epoch = selector.select_best_checkpoint(
            cagra_results, ivf_results, backend="cuvs_cagra", 
            mode="generalization"
        )
        print(f"🎯 Best checkpoint (generalization): {gen_epoch}")
    elif backend in ["ivf", "ivf_only"]:
        # For IVF backends, try both modes
        best_epoch = selector.select_best_checkpoint(
            cagra_results, ivf_results, backend="ivf", 
            mode="best"
        )
        print(f"🎯 Best checkpoint (IVF self-backend): {best_epoch}")
        
        gen_epoch = selector.select_best_checkpoint(
            cagra_results, ivf_results, backend="ivf", 
            mode="generalization"
        )
        print(f"🎯 Best checkpoint (generalization): {gen_epoch}")
    
    # Save results to JSON
    print("\n=== SAVING RESULTS ===")
    results_file = save_results_to_json(cagra_results, ivf_results, backend, args.dataset_test, selector, size_trained=args.size_trained, size_test=args.size_test)
    
    print(f"\n✅ Comprehensive evaluation complete!")
    print(f"📊 Results saved to: {results_file}")

if __name__ == "__main__":
    main()

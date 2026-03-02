#!/usr/bin/env python3
"""
QPS-Focused Evaluation Script

Simplified version focused on accurate QPS measurements:
- Runs each budget 10 times (default, configurable)
- Warmup per budget (not just once per variant)
- Uses full dataset by default
- Reports QPS statistics (mean, median, std, min, max)

Supports 6 experiments:
- IVF: laion-10m_ivf, t2i-10m_ivf, datacomp-8.2m_ivf
- CAGRA: laion-10m_cuvs_cagra, t2i-10m_cuvs_cagra, datacomp-8.2m_cuvs_cagra
"""

import os, sys, time, random, math
import argparse
import json
import numpy as np
import cupy as cp
import torch
import torch.nn.functional as F
from pathlib import Path
from statistics import median

# --- repo imports ---
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
    """Get dataset-specific configuration."""
    if dataset_name.lower() == "laion":
        if size == "10m":
            data_path = "/ssd/hamed/ann/laion/precompute/laion_cache_up_to_10.pkl"
        else:
            data_path = "/ssd/hamed/ann/laion/precompute/laion_cache_up_to_3.pkl"
        return {
            "data_path": data_path,
            "dataset_loader": LaionDatasetLoader,
            "hidden_dim": 512,
            "feature_dim": None
        }
    elif dataset_name.lower() == "datacomp":
        if size == "8.2m":
            data_path = "/ssd/hamed/ann/datacomp_small/precompute/datacomp_cache_up_to_8200000.pkl"
        else:
            data_path = "/ssd/hamed/ann/datacomp_small/precompute/datacomp_cache_up_to_3000000.pkl"
        return {
            "data_path": data_path,
            "dataset_loader": LaionDatasetLoader,
            "hidden_dim": 512,
            "feature_dim": None
        }
    elif dataset_name.lower() == "t2i":
        if size == "10m":
            data_path = "/ssd/hamed/ann/t2i-10M/precompute/t2i_cache_up_to_0.pkl"
        else:
            data_path = "/ssd/hamed/ann/t2i-10M/precompute/t2i_cache_up_to_3.pkl"
        return {
            "data_path": data_path,
            "dataset_loader": T2IDatasetLoader,
            "hidden_dim": 200,
            "feature_dim": None
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

# Parse command line arguments
parser = argparse.ArgumentParser(description="QPS-Focused Evaluation")
parser.add_argument("--dataset", type=str, choices=["laion", "datacomp", "t2i"], required=True,
                   help="Dataset: 'laion', 'datacomp' or 't2i'")
parser.add_argument("--train_backend", type=str, choices=["ivf", "cagra"], required=True,
                   help="Training backend: 'ivf' or 'cagra'")
parser.add_argument("--size", type=str, choices=["8.2m", "10m"], default=None,
                   help="Model size: '8.2m' (datacomp) or '10m' (laion/t2i)")
parser.add_argument("--n_runs", type=int, default=10,
                   help="Number of runs per budget for QPS measurement (default: 10)")
parser.add_argument("--data_path", type=str, default=None,
                   help="Override data path")
args = parser.parse_args()

# Set default size
if args.size is None:
    if args.dataset == "datacomp":
        args.size = "8.2m"
    else:
        args.size = "10m"

# Get dataset configuration
config = get_dataset_config(args.dataset, size=args.size)
DATA_PATH = args.data_path if args.data_path is not None else config["data_path"]
DatasetLoader = config["dataset_loader"]
HIDDEN_DIM = config["hidden_dim"]
FEATURE_DIM = config.get("feature_dim")

# Get checkpoint paths
def get_checkpoint_paths(dataset, backend, size):
    checkpoint_root = ARTIFACT_ROOT / "checkpoints"
    if backend == "cagra":
        base_path = checkpoint_root / f"{dataset}-{size}_cuvs_cagra_up_to_e3"
    elif backend == "ivf":
        base_path = checkpoint_root / f"{dataset}-{size}_ivf_up_to_e3"
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    
    return {
        "ep1": str(base_path / "epoch_1.pt"),
        "ep2": str(base_path / "epoch_2.pt"),
        "ep3": str(base_path / "epoch_3.pt"),
    }

CKPTS = get_checkpoint_paths(args.dataset, args.train_backend, args.size)

print(f"🔧 Dataset: {args.dataset}")
print(f"🔧 Backend: {args.train_backend}")
print(f"🔧 Model size: {args.size}")
print(f"📁 Data path: {DATA_PATH}")
print(f"📦 Checkpoints: {list(CKPTS.keys())}")
print(f"🔄 QPS runs per budget: {args.n_runs}")

USE_PCA_FROM = "ep2"
K = 10
BUDGETS = list(range(10, 101, 10))

# CAGRA config
CAGRA_CFG = dict(
    graph_degree=32,
    intermediate_graph_degree=64,
    nn_descent_niter=30,
    build_algo="nn_descent",
    metric="inner_product",
)

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

# ---------- CAGRA ----------
def build_cagra_index(X_gpu):
    p = cagra.IndexParams(
        metric=CAGRA_CFG["metric"],
        intermediate_graph_degree=CAGRA_CFG["intermediate_graph_degree"],
        graph_degree=CAGRA_CFG["graph_degree"],
        build_algo=CAGRA_CFG["build_algo"],
        nn_descent_niter=CAGRA_CFG["nn_descent_niter"],
    )
    t0 = time.time()
    idx = cagra.build(p, X_gpu)
    print(f"✓ CAGRA build done in {time.time()-t0:.2f}s")
    return idx

def cagra_search_multiple_runs(idx, Q_gpu, k, iters, itopk=256, width=1, n_runs=10):
    """Run CAGRA search multiple times and return QPS statistics."""
    itopk = max(itopk, k)
    sp = cagra.SearchParams(itopk_size=itopk, search_width=max(1,width), algo="auto", max_iterations=max(1,int(iters)))
    
    # Warmup per budget
    for _ in range(5):
        _ = cagra.search(sp, idx, Q_gpu[:min(100, Q_gpu.shape[0])], k)
    cp.cuda.Stream.null.synchronize()
    
    # Run multiple times
    qps_values = []
    for run_idx in range(n_runs):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        D, I = cagra.search(sp, idx, Q_gpu, k)
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        qps = Q_gpu.shape[0] / max(dt, 1e-9)
        qps_values.append(qps)
        if run_idx == 0:
            results_I = cp.asnumpy(I)  # Save first run's results for recall
    
    # Calculate statistics
    qps_mean = np.mean(qps_values)
    qps_median = median(qps_values)
    qps_std = np.std(qps_values)
    qps_min = np.min(qps_values)
    qps_max = np.max(qps_values)
    
    return results_I, {
        "mean": float(qps_mean),
        "median": float(qps_median),
        "std": float(qps_std),
        "min": float(qps_min),
        "max": float(qps_max),
        "values": [float(v) for v in qps_values]
    }

# ---------- IVF (FAISS) ----------
def suggested_nlist(N: int) -> int:
    raw = int(4 * math.sqrt(max(1, N)))
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

    if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        gpu = faiss.index_cpu_to_gpu(res, 0, cpu)
        print("  • IVF moved to GPU for search")
        return gpu, nlist
    else:
        print("  • Using CPU IVF (no GPU found)")
        return cpu, nlist

def ivf_search_multiple_runs(index, Q: np.ndarray, k: int, nprobe: int, n_runs=10):
    """Run IVF search multiple times and return QPS statistics."""
    nlist = index.nlist if hasattr(index, "nlist") else nprobe
    nprobe = int(max(1, min(nprobe, nlist)))
    try:
        index.nprobe = nprobe
    except Exception:
        pass
    
    # Warmup per budget
    Q_warmup = Q[:min(100, Q.shape[0])].astype(np.float32, order="C")
    for _ in range(5):
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
    
    # Run multiple times
    qps_values = []
    for run_idx in range(n_runs):
        if is_gpu_index:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        t0 = time.perf_counter()
        _D, I = index.search(Q.astype(np.float32, order="C"), k)
        if is_gpu_index:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except:
                pass
        dt = time.perf_counter() - t0
        qps = Q.shape[0] / max(dt, 1e-9)
        qps_values.append(qps)
        if run_idx == 0:
            results_I = I.astype(np.int64, copy=False)  # Save first run's results for recall
    
    # Calculate statistics
    qps_mean = np.mean(qps_values)
    qps_median = median(qps_values)
    qps_std = np.std(qps_values)
    qps_min = np.min(qps_values)
    qps_max = np.max(qps_values)
    
    return results_I, {
        "mean": float(qps_mean),
        "median": float(qps_median),
        "std": float(qps_std),
        "min": float(qps_min),
        "max": float(qps_max),
        "values": [float(v) for v in qps_values]
    }, nprobe

# ----------------- results saving -----------------
def save_results_to_json(cagra_results, ivf_results, backend, dataset, size, n_runs, output_dir=None):
    """Save QPS evaluation results to JSON file."""
    output_dir = Path(output_dir) if output_dir else ARTIFACT_ROOT / "qps_results"
    os.makedirs(output_dir, exist_ok=True)
    
    backend_map = {
        "cagra": "cuvs_cagra",
        "ivf": "ivf"
    }
    backend_name = backend_map.get(backend, backend)
    filename = f"{dataset}-{size}_{backend_name}_qps_results.json"
    filepath = output_dir / filename
    
    results = {
        "metadata": {
            "dataset": dataset,
            "backend": backend,
            "size": size,
            "n_runs_per_budget": n_runs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_type": "qps_focused"
        },
        "all_results": {
            "cagra_results": {},
            "ivf_results": {}
        }
    }
    
    # Convert results to JSON-serializable format
    for variant, data in cagra_results.items():
        results["all_results"]["cagra_results"][variant] = [
            {
                "budget": budget,
                "recall": float(recall),
                "qps_mean": float(qps_stats["mean"]),
                "qps_median": float(qps_stats["median"]),
                "qps_std": float(qps_stats["std"]),
                "qps_min": float(qps_stats["min"]),
                "qps_max": float(qps_stats["max"])
            }
            for budget, recall, qps_stats in data
        ]
    
    for variant, data in ivf_results.items():
        results["all_results"]["ivf_results"][variant] = [
            {
                "nprobe": nprobe,
                "recall": float(recall),
                "qps_mean": float(qps_stats["mean"]),
                "qps_median": float(qps_stats["median"]),
                "qps_std": float(qps_stats["std"]),
                "qps_min": float(qps_stats["min"]),
                "qps_max": float(qps_stats["max"])
            }
            for nprobe, recall, qps_stats in data
        ]
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 QPS results saved to: {filepath}")
    return str(filepath)

# ----------------- main -----------------
def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    np.random.seed(123)
    random.seed(123)

    print("=" * 60)
    print("QPS-FOCUSED EVALUATION")
    print(f"Dataset: {args.dataset.upper()}, Backend: {args.train_backend.upper()}")
    print(f"Runs per budget: {args.n_runs}")
    print("=" * 60)
    
    print(f"Loading {args.dataset.upper()} dataset…")
    ds = DatasetLoader(DATA_PATH)
    X_train_full, T_train_full, _ = ds.get_train_data()
    X_val_full, T_val_full, *_ = ds.get_split_data("val")
    print(f"Train: X={X_train_full.shape}, T={T_train_full.shape} | Val: T={T_val_full.shape}")

    if ds.data is None:
        ds.data = ds.load_data()

    # Use FULL dataset (no subsampling)
    print("🔍 Using FULL dataset for evaluation")
    N_BANK = X_train_full.shape[0]
    NQ = T_val_full.shape[0]
    X_bank = l2n(X_train_full.astype(np.float32))
    T_val = l2n(T_val_full.astype(np.float32))
    print(f"📊 Full evaluation: Bank={N_BANK:,} images, Queries={NQ:,} texts")

    # Check if cache has PCA
    cache_has_pca = False
    if hasattr(ds, 'data') and ds.data is not None:
        if "pca" in ds.data and ds.data["pca"] is not None:
            cache_has_pca = True
            print("✅ Cache has PCA: features are already PCA-reduced")
        else:
            print("⚠️  Cache does not have PCA: features are in original space")

    actual_feature_dim = X_train_full.shape[1]
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
        models[tag], rotations[tag] = load_ckpt(path, device="cuda", dim=feature_dim)

    # Fix bank space with PCA from USE_PCA_FROM
    print(f"\nUsing PCA from {USE_PCA_FROM} to define the bank/search space")
    R_ref = rotations[USE_PCA_FROM]

    print("Rotating bank…")
    if cache_has_pca:
        X_R_bank = X_bank.astype(np.float32, copy=False)
        print("  ✅ Using pre-computed PCA-reduced features")
    else:
        X_R_bank = R_ref.transform(X_bank)
        print("  ✅ Applied PCA transform to features")
    X_R_bank = l2n(X_R_bank)
    assert_unit("X_R_bank", X_R_bank)

    # Build index only for the training backend (no cross-backend evaluation)
    if args.train_backend == "cagra":
        print("\nBuilding CAGRA index…")
        X_gpu = cpu2gpu(X_R_bank)
        print(f"GPU pool used={cp.get_default_memory_pool().used_bytes()/1e9:.3f} GB")
        cagra_index = build_cagra_index(X_gpu)
        _KEEPALIVE = [X_gpu]
        ivf_index = None
        nlist = None
    elif args.train_backend == "ivf":
        print("\nBuilding IVF index…")
        ivf_index, nlist = build_ivf_index_gpu(X_R_bank)
        print(f"✓ IVF ready (nlist={nlist})")
        cagra_index = None
        X_gpu = None
        _KEEPALIVE = []
    else:
        raise ValueError(f"Unsupported backend: {args.train_backend}")

    # Prepare query variants
    print("\nPreparing queries (4 variants)…")
    variants = {}
    if cache_has_pca:
        Q_base = T_val.astype(np.float32, copy=False)
        print("  ✅ Baseline queries: using pre-computed PCA-reduced features")
    else:
        Q_base = R_ref.transform(T_val)
        print("  ✅ Baseline queries: applied PCA transform")
    Q_base = l2n(Q_base)
    assert_unit("Q_base (baseline)", Q_base)
    variants["baseline"] = Q_base
    
    for tag in ["ep1", "ep2", "ep3"]:
        Qp = project_np(models[tag], T_val, device="cuda", batch=65536)
        if cache_has_pca:
            Qp_R = Qp.astype(np.float32, copy=False)
            print(f"  ✅ Projected queries {tag}: using model output")
        else:
            Qp_R = R_ref.transform(Qp)
            print(f"  ✅ Projected queries {tag}: applied PCA transform")
        Qp_R = l2n(Qp_R)
        assert_unit(f"Q_proj_{tag}", Qp_R)
        variants[f"proj_{tag}"] = Qp_R

    # Compute exact GTs
    print("\nComputing exact GTs (projected space) for all variants…")
    GTs = {}
    for name, Q in variants.items():
        print(f"  • GT for {name} …")
        GTs[name] = brute_force_topk_streaming(Q, X_R_bank, k=K, q_batch=min(8192, Q.shape[0]), x_batch=50_000)

    # Evaluate only on the training backend (no cross-backend)
    cagra_results = {}
    ivf_results = {}
    
    if args.train_backend == "cagra":
        # Only evaluate on CAGRA
        Q_gpu = {name: cpu2gpu(Q) for name, Q in variants.items()}
        
        print("\n=== CAGRA: Recall@10 & QPS vs Budget (multiple runs) ===")
        for name in ["baseline", "proj_ep1", "proj_ep2", "proj_ep3"]:
            print(f"\nVariant: {name}")
            rows = []
            for B in BUDGETS:
                print(f"  B={B:3d} (running {args.n_runs} times)...", end=" ", flush=True)
                I_pred, qps_stats = cagra_search_multiple_runs(cagra_index, Q_gpu[name], K, iters=B, itopk=256, width=1, n_runs=args.n_runs)
                r = recall_at_k(I_pred, GTs[name], K)
                rows.append((B, r, qps_stats))
                print(f"R@{K}={r:.4f}  QPS_mean={qps_stats['mean']:,.0f}  QPS_std={qps_stats['std']:,.0f}")
            cagra_results[name] = rows
    
    elif args.train_backend == "ivf":
        # Only evaluate on IVF
        print("\n=== IVF: Recall@10 & QPS vs nprobe (multiple runs) ===")
        for name in ["baseline", "proj_ep1", "proj_ep2", "proj_ep3"]:
            print(f"\nVariant: {name}")
            rows = []
            Q_cpu = variants[name]
            for B in BUDGETS:
                print(f"  nprobe={B:3d} (running {args.n_runs} times)...", end=" ", flush=True)
                I_pred, qps_stats, used_nprobe = ivf_search_multiple_runs(ivf_index, Q_cpu, K, nprobe=B, n_runs=args.n_runs)
                r = recall_at_k(I_pred, GTs[name], K)
                rows.append((used_nprobe, r, qps_stats))
                print(f"R@{K}={r:.4f}  QPS_mean={qps_stats['mean']:,.0f}  QPS_std={qps_stats['std']:,.0f}")
            ivf_results[name] = rows

    # Save results
    print("\n=== SAVING RESULTS ===")
    results_file = save_results_to_json(cagra_results, ivf_results, args.train_backend, args.dataset, args.size, args.n_runs)
    
    print(f"\n✅ QPS evaluation complete!")
    print(f"📊 Results saved to: {results_file}")

if __name__ == "__main__":
    main()

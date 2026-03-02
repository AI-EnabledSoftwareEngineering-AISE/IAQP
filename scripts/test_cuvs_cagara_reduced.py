#!/usr/bin/env python3
"""
cuvs_cagra_debug.py
"""

from calendar import c
import os, sys, time, math, argparse, itertools
import numpy as np
import cupy as cp
import torch
from typing import Dict, Any, List, Tuple
from pathlib import Path
from tqdm.auto import tqdm
import json
import pickle
from datetime import datetime

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from cuvs.neighbors import cagra
from adapter.t2i_code.projector.dataset_loader import LaionDatasetLoader, T2IDatasetLoader
from adapter.t2i_code.projector.utils import PCARSpace, brute_force_topk_streaming, project_np, ResidualProjector, CoarseCellHead, QuantityHead, bank_fingerprint, _sha1

# ---------- helpers ----------

def unit_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32, copy=False)

def cpu_to_gpu(a: np.ndarray) -> cp.ndarray:
    a = np.asarray(a, dtype=np.float32, order="C")
    return cp.asarray(a, dtype=cp.float32, order="C")

def recall_at_k(pred_idx: np.ndarray, gt_idx: np.ndarray, k: int) -> float:
    k = min(k, pred_idx.shape[1], gt_idx.shape[1])
    hits = 0
    for i in range(pred_idx.shape[0]):
        hits += len(set(pred_idx[i, :k]) & set(gt_idx[i, :k]))
    return hits / (pred_idx.shape[0] * k)

def builder_param_grid(builder: str) -> List[Dict[str, Any]]:
    shared = {"graph_degree": [32, 64, 96],
              "intermediate_graph_degree": [64, 128, 192]}
    if builder == "nn_descent":
        base = dict(shared, **{"nn_descent_niter": [10, 20, 30]})
    else:
        base = dict(shared)
    grid = []
    keys, vals = zip(*base.items())
    for combo in itertools.product(*vals):
        row = dict(zip(keys, combo))
        row["build_algo"] = builder
        grid.append(row)
    return grid

def build_cagra_index(X_gpu: cp.ndarray, build_cfg: Dict[str, Any]):
    params = cagra.IndexParams(
        metric="inner_product",
        intermediate_graph_degree=int(build_cfg["intermediate_graph_degree"]),
        graph_degree=int(build_cfg["graph_degree"]),
        build_algo=build_cfg.get("build_algo", "ivf_pq"),
        nn_descent_niter=int(build_cfg.get("nn_descent_niter", 20)),
    )
    t0 = time.time()
    index = cagra.build(params, X_gpu)
    t = time.time() - t0
    return index, t, params

def cagra_search(index, Q_gpu: cp.ndarray, k: int, search_cfg: Dict[str, Any]):
    # Safety: k <= itopk_size
    itopk = max(int(search_cfg["itopk_size"]), int(k))
    sp = cagra.SearchParams(
        itopk_size=itopk,
        search_width=int(search_cfg["search_width"]),
        algo=search_cfg.get("algo", "auto"),
        max_iterations=int(search_cfg.get("max_iterations", 0)),
    )
    t0 = time.time()
    D, I = cagra.search(sp, index, Q_gpu, k)
    t = time.time() - t0
    return D, I, t, sp

# ---------- budget mapping strategies ----------

def tune_only(X_gpu: cp.ndarray, Q_gpu: cp.ndarray, GT: np.ndarray, k: int, 
              builder: str, max_build_trials: int = 3, skip_configs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Tune CAGRA parameters using original data only (no building).
    This is the separated tuning function that only does parameter optimization.
    
    Args:
        skip_configs: List of configs to skip (e.g., ones that failed with OOM)
    """
    from tqdm.auto import tqdm
    import time
    
    skip_configs = skip_configs or []
    
    # Get all possible configs
    all_builds = builder_param_grid(builder)
    
    # Filter out skipped configs
    filtered_builds = []
    for cfg in all_builds:
        skip = False
        for skip_cfg in skip_configs:
            if (cfg.get("graph_degree") == skip_cfg.get("graph_degree") and
                cfg.get("intermediate_graph_degree") == skip_cfg.get("intermediate_graph_degree") and
                cfg.get("nn_descent_niter") == skip_cfg.get("nn_descent_niter")):
                skip = True
                break
        if not skip:
            filtered_builds.append(cfg)
    
    # Take first max_build_trials from filtered list
    builds = filtered_builds[:max_build_trials]
    
    if len(builds) < max_build_trials:
        print(f"⚠️  Warning: Only {len(builds)} configs available after filtering (requested {max_build_trials})")
    
    fixed_search = {"itopk_size": 64, "search_width": 1, "algo": "auto", "max_iterations": 0}
    
    best = {"recall": -1.0, "qps": 0.0, "cfg": None, "build_time": None}
    
    print(f"🔧 Tuning CAGRA parameters (original data only)...")
    print(f"   Data: X={X_gpu.shape}, Q={Q_gpu.shape}, GT={GT.shape}")
    if skip_configs:
        print(f"   Skipping {len(skip_configs)} config(s) that previously failed")
    
    tested = 0
    for i, cfg in enumerate(builds, 1):
        print(f"\n[BUILD {i:02d}/{len(builds)}] Testing config: {cfg}")
        
        try:
            # Build index
            build_start = time.time()
            index, bt, _ = build_cagra_index(X_gpu, cfg)
            build_time = time.time() - build_start
            
            # Test search performance
            _, I, t, _ = cagra_search(index, Q_gpu, k, fixed_search)
            I_host = cp.asnumpy(I)
            rec = recall_at_k(I_host, GT, k)
            qps = Q_gpu.shape[0] / max(t, 1e-9)
            
            print(f"  ✓ build={build_time:.1f}s  R@{k}={rec:.4f}  QPS={qps:,.0f}")
            
            if rec > best["recall"] or (math.isclose(rec, best["recall"]) and qps > best["qps"]):
                best.update({"recall": rec, "qps": qps, "cfg": cfg, "build_time": build_time})
            
            tested += 1
            
            # CRITICAL: Free previous index before next iteration to prevent OOM
            del index
            cp.cuda.runtime.deviceSynchronize()
            cp.get_default_memory_pool().free_all_blocks()
            
        except Exception as e:
            print(f"  ✗ FAILED: {str(e)}")
            print(f"  Skipping this config and continuing...")
            # Free memory and continue
            cp.cuda.runtime.deviceSynchronize()
            cp.get_default_memory_pool().free_all_blocks()
            continue
    
    if best["cfg"] is None:
        raise RuntimeError("All configs failed! Try smaller parameters or reduce dataset size.")
    
    print(f"\n→ Best tuning cfg: {best['cfg']}  R@{k}={best['recall']:.4f}  build={best['build_time']:.1f}s (tested {tested}/{len(builds)} configs)")
    return best

# ---------- CLI ----------
def load_model_for_dataset(dataset: str, checkpoint_path: str = None, epochs: int = None):
    """
    Load the appropriate model based on dataset type and checkpoint path.
    
    Args:
        dataset: Either 't2i' or 'laion'
        checkpoint_path: Optional path to specific checkpoint file
        epochs: Number of epochs for the checkpoint (for display purposes)
        
    Returns:
        Tuple of (model, R, coarse_head, qty_head, dim)
    """
    # Use provided checkpoint path or fall back to hard-coded paths
    if checkpoint_path:
        model_path = checkpoint_path
        print(f"📦 Loading model from provided checkpoint: {model_path}")
    else:
        # Hard-coded model paths based on dataset (fallback)
        if dataset.lower() == 't2i':
            model_path = "outputs/projectors/cuvs_cagra_nn_descent_2_epoch_3m_t2i.pt"
            dim = 200
        elif dataset.lower() == 'laion':
            model_path = "outputs/projectors/cuvs_cagra_nn_descent_2_epochs_3m_laion.pt"
            dim = 512
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Must be 't2i' or 'laion'")
        print(f"📦 Loading model for {dataset.upper()} from {model_path}...")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    print("✅ Model loaded successfully")
    
    # Get saved parameters from checkpoint
    saved_M_cells = ckpt.get("M_cells", 2048)
    saved_hidden = ckpt.get("hidden", 200 if dataset.lower() == 't2i' else 512)
    saved_alpha = ckpt.get("alpha", 0.1)
    dim = 200 if dataset.lower() == 't2i' else 512
    
    # Create models with saved parameters
    R = PCARSpace(d_keep=None, center_for_fit=True, device="cpu")
    model = ResidualProjector(dim=dim, hidden=saved_hidden, alpha=saved_alpha)
    coarse_head = CoarseCellHead(dim=dim, M=saved_M_cells)
    qty_head = QuantityHead(dim=dim)
    
    # Load saved parameters
    if "W" in ckpt:
        R.W = ckpt["W"].astype(np.float32)
        # PCA/rotation sanity checks
        assert R.W.shape[1] == dim, f"PCA W dim mismatch: {R.W.shape[1]} != {dim}"
        # rows should be orthonormal
        ortho = R.W @ R.W.T
        assert np.allclose(ortho, np.eye(ortho.shape[0], dtype=np.float32), atol=1e-3), "PCA W not orthonormal enough"
    if "center_for_fit" in ckpt:
        R.center_for_fit = ckpt["center_for_fit"]
    if "mu" in ckpt:
        R.mu = ckpt["mu"]
    
    # Load model state dicts
    if "model_state_dict" in ckpt:
        # Handle DDP-wrapped models by removing 'module.' prefix
        model_state = ckpt["model_state_dict"]
        if any(key.startswith('module.') for key in model_state.keys()):
            print("Detected DDP-wrapped model, removing 'module.' prefix...")
            model_state = {key.replace('module.', ''): value for key, value in model_state.items()}
        model.load_state_dict(model_state)
        
        coarse_state = ckpt["coarse_head_state_dict"]
        if any(key.startswith('module.') for key in coarse_state.keys()):
            coarse_state = {key.replace('module.', ''): value for key, value in coarse_state.items()}
        coarse_head.load_state_dict(coarse_state)
        
        if "qty_head_state_dict" in ckpt and ckpt["qty_head_state_dict"] is not None:
            qty_state = ckpt["qty_head_state_dict"]
            if any(key.startswith('module.') for key in qty_state.keys()):
                qty_state = {key.replace('module.', ''): value for key, value in qty_state.items()}
            qty_head.load_state_dict(qty_state)
    elif "state_dict" in ckpt:
        model_state = ckpt["state_dict"]
        if any(key.startswith('module.') for key in model_state.keys()):
            model_state = {key.replace('module.', ''): value for key, value in model_state.items()}
        model.load_state_dict(model_state) 
    else:
        if any(key.startswith('module.') for key in ckpt.keys()):
            ckpt = {key.replace('module.', ''): value for key, value in ckpt.items()}
        model.load_state_dict(ckpt)
    
    epoch_info = f" ({epochs} epochs)" if epochs is not None else ""
    print(f"✅ Model components loaded{epoch_info}:")
    print(f"   - ResidualProjector: dim={dim}, hidden={saved_hidden}, alpha={saved_alpha}")
    print(f"   - CoarseCellHead: dim={dim}, M={saved_M_cells}")
    print(f"   - QuantityHead: dim={dim}")
    print(f"   - PCA rotation: W.shape={R.W.shape if hasattr(R, 'W') and R.W is not None else 'None'}")
    
    return model, R, coarse_head, qty_head, dim

def load_cached_config(cfg_path: Path):
    """Load cached config if exists, return None otherwise."""
    if not cfg_path.exists():
        return None
    try:
        with open(cfg_path, "r") as f:
            data = json.load(f)
        if "cfg" in data and isinstance(data["cfg"], dict):
            return data
    except Exception:
        pass
    return None

def save_cached_config(cfg_path: Path, dataset: str, k: int, builder: str, best: dict):
    """Save config to cache."""
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset, "k": int(k), "builder": builder,
        "cfg": best.get("cfg"), "recall": float(best.get("recall", -1.0)),
        "qps": float(best.get("qps", 0.0)), "build_time": float(best.get("build_time", 0.0)),
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(cfg_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"💾 Saved config to {cfg_path}")

def load_gt_data(gt_path: Path):
    """Load GT data from pickle file if exists."""
    if not gt_path.exists():
        return None
    try:
        with open(gt_path, "rb") as f:
            gt_data = pickle.load(f)
        print(f"🟢 Found cached GT data: {gt_path}")
        return gt_data
    except Exception as e:
        print(f"⚠️ Failed to load GT data from {gt_path}: {e}")
        return None

def save_gt_data(gt_path: Path, gt_data: np.ndarray):
    """Save GT data to pickle file."""
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "wb") as f:
        pickle.dump(gt_data, f)
    print(f"💾 Saved GT data to {gt_path}")



def sweep_budgets(index,
                  Q_gpu_all: cp.ndarray,
                  GT_all: np.ndarray,
                  k: int,
                  budgets: List[int],
                  mode: str,
                  itopk: int = 256,
                  width: int = 1) -> List[Tuple[int, Dict[str, Any], float, float]]:
    """
    For each budget B, run a search using the selected mapping mode.
    Returns list of (B, cfg, recall, qps).
    """
    results = []
    nq = Q_gpu_all.shape[0]
    for B in budgets:
        cfg = {"itopk_size": max(itopk, k), "search_width": max(1, width), "algo": "auto", "max_iterations": max(1, B)}
        _, I, t, _ = cagra_search(index, Q_gpu_all, k, cfg)
        I_host = cp.asnumpy(I)
        rec = recall_at_k(I_host, GT_all, k)
        qps = nq / max(t, 1e-9)
        print(f"[SEARCH/{mode:10s}] B={B:<3}  itopk={cfg['itopk_size']:<3}  width={cfg['search_width']:<3}  "
              f"iters={cfg['max_iterations']:<3}  R@{k}={rec:.4f}  QPS={qps:,.0f}")
        results.append((B, cfg, rec, qps))
    return results

def is_normalized(A: np.ndarray, tol: float = 1e-3) -> bool:
    if A.size == 0:
        return True
    n = np.linalg.norm(A, axis=1)
    if not np.all(np.isfinite(n)):
        return False
    return np.all(np.abs(n - 1.0) <= tol)

CFG_DIR = Path("/home/hamed/projects/SPIN/adapter/t2i_code/outputs/cagara_cfg")
CFG_DIR.mkdir(parents=True, exist_ok=True)

# --- keep your existing imports and helpers (unit_normalize, is_normalized, etc.) ---

CFG_DIR = Path("/home/hamed/projects/SPIN/adapter/t2i_code/outputs/cagara_cfg")
CFG_DIR.mkdir(parents=True, exist_ok=True)


def check_for_config(cfg_path: Path) -> Dict[str, Any]:
    """Check for cached config or tune new one if not found."""
    cached = load_cached_config(cfg_path)
    
    if cached is not None:
        print(f"🟢 Found cached CAGRA cfg: {cfg_path}")
        return {
            "cfg": cached["cfg"],
            "recall": cached.get("recall", -1.0),
            "qps": cached.get("qps", 0.0),
            "build_time": cached.get("build_time", 0.0),
        }
    
    # No cached config - compute GT and tune
    return None


def apply_pca_transform(data: np.ndarray, R: PCARSpace, use_gpu: bool = False,
                        chunk: int = 262144, device: str = "cuda") -> np.ndarray:
    """
    Apply PCA transform (and mean-centering) optionally on GPU in chunks.
    """
    if data.size == 0:
        return data.astype(np.float32, copy=False)

    if use_gpu and torch.cuda.is_available():
        if device.startswith("cuda"):
            dev = torch.device(device)
        else:
            dev = torch.device("cuda")
        W = torch.from_numpy(R.W.astype(np.float32)).to(dev)
        mu = None
        if getattr(R, "mu", None) is not None:
            mu = torch.from_numpy(R.mu.astype(np.float32)).to(dev)
            if mu.ndim == 2 and mu.shape[0] == 1:
                mu = mu.squeeze(0)
        out = np.empty((data.shape[0], W.shape[0]), dtype=np.float32)
        for s in range(0, data.shape[0], chunk):
            e = min(data.shape[0], s + chunk)
            batch = torch.from_numpy(data[s:e]).to(dev, non_blocking=True).float()
            if getattr(R, "center_for_fit", False) and mu is not None:
                batch = batch - mu
            batch = torch.matmul(batch, W.T)
            out[s:e] = batch.cpu().numpy().astype(np.float32, copy=False)
        return out

    # CPU fallback
    return R.transform(data)


def build_cagra_index_wrapper(X_gpu: cp.ndarray, T_train_index_tune: np.ndarray, 
                             dataset: str, k: int, builder: str, 
                             max_build_trials: int, bank_fp_train: str, 
                             force_tune: bool = False) -> cagra.Index:
    """
    Build CAGRA index with tuning. Returns only the index.
    
    CRITICAL MEMORY LIFETIME ISSUE:
    CAGRA index holds pointers to GPU memory (X_gpu) rather than copying the data.
    If X_gpu goes out of scope and gets garbage collected, the index becomes invalid
    with dangling pointers, causing:
    - Dramatically lower recall (e.g., 0.15 instead of 0.89)
    - Recall that decreases with more iterations (more garbage traversals)
    - Same QPS but completely wrong results
    
    SOLUTION: The caller MUST keep X_gpu alive for the entire lifetime of the index.
    This wrapper takes X_gpu as input (not recreating it) to ensure the GPU memory
    stays valid as long as the returned index is used.
    
    Args:
        X_gpu: Training data matrix on GPU (already normalized if needed)
               MUST stay alive in caller scope for index lifetime
        T_train_index_tune: Training text data matrix (already normalized if needed)
        dataset: Dataset name for config caching
        k: Number of neighbors
        builder: Builder algorithm
        max_build_trials: Max trials for tuning
        bank_fp_train: Bank fingerprint for config path
        
    Returns:
        CAGRA index (valid only as long as X_gpu stays alive in caller)
    """
    # Load-or-tune best config
    cfg_path = CFG_DIR / dataset / f"{dataset}_{bank_fp_train}.json"
    print(f"Config path: {cfg_path}")
    
    _tuning = None
    if not force_tune:
        _tuning = check_for_config(cfg_path)
    
    if _tuning is not None:
        print(f"🟢 Using cached config from {cfg_path}")
        best_tuning = _tuning
    else:
        if force_tune:
            print(f"🔧 Force tuning: ignoring any existing configs")
        else:
            print(f"🟡 No cached config found, tuning new config...")
        # Try to load cached GT data first
        pkl_path = CFG_DIR / dataset / f"{dataset}_{bank_fp_train}_gt.pkl"
        GT_original_index_tune = load_gt_data(pkl_path)
        
        if GT_original_index_tune is None:
            print(f"🟡 No cached GT data found at {pkl_path}")
            print("   Computing GT for tuning (this may take a while for large datasets)...")
            # Convert X_gpu back to CPU for ground truth computation
            X_cpu = cp.asnumpy(X_gpu)
            GT_original_index_tune = brute_force_topk_streaming(T_train_index_tune, X_cpu, 
                k=k, q_batch=16384, x_batch=50000, show_progress=True
            )
            print(f"✅ GT computed: shape={GT_original_index_tune.shape}")
            
            # Save GT data for future use
            save_gt_data(pkl_path, GT_original_index_tune)
            print(f"💾 GT saved to {pkl_path} for future runs")
        else:
            print(f"✅ Loaded cached GT data from {pkl_path}")
            print(f"   GT shape: {GT_original_index_tune.shape}")

        T_train_index_tune_gpu = cpu_to_gpu(T_train_index_tune)
        print("\n🔧 Tuning CAGRA parameters using original data...")
        best_tuning = tune_only(
            X_gpu, T_train_index_tune_gpu, GT_original_index_tune, k,
            builder=builder, max_build_trials=max_build_trials
        )

        save_cached_config(cfg_path, dataset, k, builder, best_tuning)
    
    # Build final index with tuned parameters
    print(f"\n🔄 Building final CAGRA index with tuned parameters...")
    print(f"   Best config: {best_tuning['cfg']}")
    index, build_time, params = build_cagra_index(X_gpu, best_tuning['cfg'])
    print(f"✅ Final index built in {build_time:.2f}s")
    
    return index

def get_or_build_cuvs_cagra(X_gpu: cp.ndarray, dataset: str, bank_fp_train: str) -> cagra.Index:
    """
    Simple wrapper to read cached config and build CAGRA index.
    
    CRITICAL MEMORY LIFETIME ISSUE:
    CAGRA index holds pointers to GPU memory (X_gpu) rather than copying the data.
    If X_gpu goes out of scope and gets garbage collected, the index becomes invalid
    with dangling pointers, causing dramatically lower recall and wrong results.
    
    SOLUTION: The caller MUST keep X_gpu alive for the entire lifetime of the index.
    This wrapper takes X_gpu as input to ensure the GPU memory stays valid.
    
    Args:
        X_gpu: Training data matrix on GPU (MUST stay alive in caller scope for index lifetime)
        dataset: Dataset name for config caching
        bank_fp_train: Bank fingerprint for config path
        
    Returns:
        CAGRA index (valid only as long as X_gpu stays alive in caller)
    """
    # Load cached config
    cfg_path = CFG_DIR / dataset / f"{dataset}_{bank_fp_train}.json"
    print(f"Cagara Config path: {cfg_path}")
    
    cached = load_cached_config(cfg_path)
    if cached is None:
        raise FileNotFoundError(f"No cached config found at {cfg_path}. Run tuning first.")
    
    best_tuning = {
        "cfg": cached["cfg"],
        "recall": cached.get("recall", -1.0),
        "qps": cached.get("qps", 0.0),
        "build_time": cached.get("build_time", 0.0),
    }
    
    # Build index with cached parameters
    print(f"\n🔄 Building CAGRA index with cached parameters...")
    print(f"   Config: {best_tuning['cfg']}")
    index, build_time, params = build_cagra_index(X_gpu, best_tuning['cfg'])
    print(f"✅ Index built in {build_time:.2f}s")
    
    return index


def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--use_cosine", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dataset", type=str, choices=["t2i", "laion", "datacomp"], default="t2i", 
                   help="Dataset type to determine which model to load")
    ap.add_argument("--device", type=str, default="cuda", help="Device to use for model inference")
    
    # Baseline mode (no checkpoint, just raw features)
    ap.add_argument("--baseline", action="store_true",
                   help="Use baseline encoding (no model, just raw features). Requires --pca_path if you want PCA transform.")
    ap.add_argument("--pca_path", type=str, default=None,
                   help="Path to PCA npz file (for baseline mode with PCA transform)")
    ap.add_argument("--pca_dim", type=int, default=None,
                   help="Target PCA dimension (reduce to this many components). If None, uses full PCA dimension.")
    ap.add_argument("--use_pca_gpu", action="store_true",
                   help="Apply PCA transform on GPU (faster for large datasets).")
    ap.add_argument("--pca_chunk", type=int, default=262144,
                   help="Rows per chunk when applying PCA on GPU.")
    
    # Checkpoint
    ap.add_argument("--checkpoint_path", type=str, default=None, 
                   help="Path to specific checkpoint file (overrides default model selection)")
    ap.add_argument("--epochs", type=int, default=None, 
                   help="Number of epochs for the checkpoint (for display purposes)")

    # Algo
    ap.add_argument("--builder", choices=["ivf_pq", "nn_descent"], default="nn_descent")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max_build_trials", type=int, default=9)

    # Budgets
    ap.add_argument("--budgets", type=str, default="10,20,30,40,50,60,70,80,90,100")



    args = ap.parse_args()

    print("🚀 cuVS CAGRA — Budget Mapping Comparison")
    print("="*72)
    print(f"builder={args.builder} | k={args.k} | GPU={args.gpu}")
    print(f"data_path={args.data_path}")
    print(f"use_cosine={args.use_cosine} | max_build_trials={args.max_build_trials}")
    if args.baseline:
        print("🔧 BASELINE MODE: Using raw features (no model projection)")
        if args.pca_path:
            print(f"   PCA path: {args.pca_path}")
    elif args.checkpoint_path:
        epoch_info = f" ({args.epochs} epochs)" if args.epochs is not None else ""
        print(f"checkpoint={args.checkpoint_path}{epoch_info}")
    
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]

    # ---- Load dataset
    if args.dataset == "t2i":
        ds = T2IDatasetLoader(args.data_path)
    else:
        ds = LaionDatasetLoader(args.data_path)
    X_train, T_train, _ = ds.get_train_data()
    X_val, T_val, *_ = ds.get_split_data("val")
    print(f"Train: X={X_train.shape}  T={T_train.shape} | Val: X={X_val.shape}  T={T_val.shape}")
    
    # ---- Setup PCA and model (or baseline mode)
    if args.baseline:
        # Baseline mode: no model, optionally use PCA from npz
        model = None
        R = None
        dim = X_train.shape[1]
        
        if args.pca_path:
            print(f"\n🔄 Loading PCA from {args.pca_path}...")
            pca_data = np.load(args.pca_path)
            R = PCARSpace(d_keep=None, center_for_fit=bool(pca_data.get("center_for_fit", [False])[0]), device="cpu")
            R.W = pca_data["W"].astype(np.float32)
            if "mu" in pca_data and pca_data["mu"] is not None:
                R.mu = pca_data["mu"].astype(np.float32)
            
            # Apply dimension reduction if requested
            if args.pca_dim is not None and args.pca_dim < R.W.shape[0]:
                print(f"🔧 Reducing PCA dimension: {R.W.shape[0]} → {args.pca_dim}")
                R.W = R.W[:args.pca_dim, :]  # Keep only first pca_dim components
                print(f"✅ PCA reduced: W.shape={R.W.shape}, center_for_fit={R.center_for_fit}")
            else:
            print(f"✅ PCA loaded: W.shape={R.W.shape}, center_for_fit={R.center_for_fit}")
            
            # Update dim to match reduced PCA dimension
            dim = R.W.shape[0]
            
            # Apply PCA transform (with GPU acceleration for large datasets)
            print(f"🔄 Applying PCA transform (GPU={args.use_pca_gpu}, chunk={args.pca_chunk})...")
            t0 = time.time()
            device_str = f"cuda:{args.gpu}" if args.use_pca_gpu else args.device
            X_train = apply_pca_transform(X_train, R, use_gpu=args.use_pca_gpu,
                                          chunk=args.pca_chunk, device=device_str)
            T_val = apply_pca_transform(T_val, R, use_gpu=args.use_pca_gpu,
                                        chunk=args.pca_chunk, device=device_str)
            T_train = apply_pca_transform(T_train, R, use_gpu=args.use_pca_gpu,
                                          chunk=args.pca_chunk, device=device_str)
            print(f"✅ PCA transform done in {time.time()-t0:.2f}s")
            print(f"   Output shapes: X_train={X_train.shape}, T_train={T_train.shape}, T_val={T_val.shape}")
        else:
            print("\n🔧 Using raw features (no PCA transform)")
        
        # Generate bank fingerprint
        if R is not None:
            bank_fp_train = bank_fingerprint(X_train, R)
        else:
            # Use raw data fingerprint if no PCA (simple hash of shape and sample)
            N, D = X_train.shape
            buf = np.ascontiguousarray(X_train, dtype=np.float32).tobytes()
            sample = buf[:10**6] + buf[-10**6:] if len(buf) > 2_000_000 else buf
            bank_fp_train = _sha1(sample)[:12]  # Use first 12 chars of hash
        print(f"bank_fp_train: {bank_fp_train}")
        
        # In baseline mode, we use raw text features (no projection)
        T_train_index_tune = T_train.astype(np.float32, copy=False)
        T_val_baseline = T_val.astype(np.float32, copy=False)
        
    else:
        # Normal mode: load model and checkpoint
        print(f"\n🔧 Loading model for {args.dataset.upper()} dataset...")
        model, R, coarse_head, qty_head, dim = load_model_for_dataset(
            args.dataset, 
            checkpoint_path=args.checkpoint_path, 
            epochs=args.epochs
        )
        print(f"✅ Model loaded with feature dimension: {dim}")
        
        # Project texts then rotate for ANN search
        print(f"\n🔄 Projecting texts using loaded model...")
        print(f"   Device: {args.device}")
        print(f"   Text shape: {T_val.shape}")
        
        # Move model to device
        model = model.to(args.device)
        R.device = args.device
        
        # Project texts
        T_val_proj = project_np(model, T_val, device=args.device, batch=65536)
        print(f"   Projected text shape: {T_val_proj.shape}")
        
        # Apply PCA rotation
        T_val_proj_R = apply_pca_transform(
            T_val_proj, R, use_gpu=args.use_pca_gpu,
            chunk=args.pca_chunk,
            device=f"cuda:{args.gpu}" if args.use_pca_gpu else args.device
        )
        print(f"   Rotated text shape: {T_val_proj_R.shape}")
        print(f"✅ Text projection and rotation completed")

        # Generate bank fingerprint BEFORE PCA transform (same as trainer)
        bank_fp_train = bank_fingerprint(X_train, R)
        print(f"bank_fp_train: {bank_fp_train}")
        
        # Apply PCA transform using parameters from the loaded checkpoint
        print("🔄 Applying PCA transform using checkpoint parameters...")
        t0 = time.time()
        X_train = apply_pca_transform(X_train, R, use_gpu=args.use_pca_gpu,
                                      chunk=args.pca_chunk,
                                      device=f"cuda:{args.gpu}" if args.use_pca_gpu else args.device)
        T_val   = apply_pca_transform(T_val, R, use_gpu=args.use_pca_gpu,
                                      chunk=args.pca_chunk,
                                      device=f"cuda:{args.gpu}" if args.use_pca_gpu else args.device)
        T_train = apply_pca_transform(T_train, R, use_gpu=args.use_pca_gpu,
                                      chunk=args.pca_chunk,
                                      device=f"cuda:{args.gpu}" if args.use_pca_gpu else args.device)
        print(f"✅ PCA transform done in {time.time()-t0:.2f}s")
        
        T_train_index_tune = T_train.astype(np.float32, copy=False)



    # Prepare data for index building
    X = X_train.astype(np.float32, copy=False)
    
    if args.use_cosine:
        print("🔄 Normalizing (cosine via L2)…")
        if not is_normalized(X):
            X = unit_normalize(X)
        if not is_normalized(T_train_index_tune):
            T_train_index_tune = unit_normalize(T_train_index_tune)
    
    # Convert to GPU and keep alive for the lifetime of the index
    X_gpu = cpu_to_gpu(X)
    
    # Build CAGRA index using wrapper function to get the best config
    # ALWAYS tune (force_tune=True) since the whole point of this script is to find best config
    # Use the actual dataset name from args (datacomp should have its own directory)
    index = build_cagra_index_wrapper(
        X_gpu, T_train_index_tune, args.dataset, args.k, args.builder, 
        args.max_build_trials, bank_fp_train, force_tune=True
    )
    
    # CRITICAL: Keep GPU data alive to prevent index from holding dangling pointers
    # Store reference in a variable that stays alive for the entire function
    _keepalive_X_gpu = X_gpu

    if args.baseline:
        # Baseline mode: use raw text features as queries
        Q_baseline = T_val_baseline.astype(np.float32, copy=False)
        if args.use_cosine:
            print("🔄 Normalizing queries (cosine via L2)…")
            if not is_normalized(Q_baseline):
                Q_baseline = unit_normalize(Q_baseline)
        
        # Compute GT for baseline queries
        GT_baseline = brute_force_topk_streaming(Q_baseline, X, k=args.k)
        Q_baseline_gpu = cpu_to_gpu(Q_baseline)
        
        print(f"\n--- Mode: iterations (Baseline Queries) ---")
        _ = sweep_budgets(index, Q_baseline_gpu, GT_baseline, args.k, budgets, "iterations",
                            itopk=256,
                            width=1)
    else:
        # Normal mode: keep both original and projected queries
        Q_original = T_val.astype(np.float32, copy=False)
        Q_projected = T_val_proj_R.astype(np.float32, copy=False)
        if args.use_cosine:
            print("🔄 Normalizing queries (cosine via L2)…")
            if not is_normalized(Q_original):
                Q_original = unit_normalize(Q_original)
            if not is_normalized(Q_projected):
                Q_projected = unit_normalize(Q_projected)

        # Test with original queries
        GT_original = brute_force_topk_streaming(Q_original, X, k=args.k)
        GT_projected = brute_force_topk_streaming(Q_projected, X, k=args.k)
        Q_original_gpu = cpu_to_gpu(Q_original)
        Q_projected_gpu = cpu_to_gpu(Q_projected)

        print(f"\n--- Mode: iterations (Original Queries) ---")
        _ = sweep_budgets(index, Q_original_gpu, GT_original, args.k, budgets, "iterations",
                            itopk=256,
                            width=1)
        
        # Test with projected queries
        print(f"\n--- Mode: iterations (Projected Queries) ---")
        _ = sweep_budgets(index, Q_projected_gpu, GT_projected, args.k, budgets, "iterations",
                            itopk=256,
                            width=1)

    print("\nDone.")

if __name__ == "__main__":
    main()

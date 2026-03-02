"""
trainer.py
"""


import os
import math
import pickle
import time
import json
from pathlib import Path
from typing import Tuple, Any, NamedTuple
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
# Import from our modules
from .utils import (
    l2n_np,
    build_pack_topC_streaming, AnnHeadCache,
    lambda_ann_for_budget, pack_size_for, _fill_scores_from_head,
    exact_teacher_in_pack, exact_teacher_in_pack_ranked,
    brute_force_topk_streaming,
    precompute_regions_and_residuals,
    get_or_build_ivf, bank_fingerprint,
    PCARSpace, ResidualProjector, CoarseCellHead, QuantityHead,
    setup_ivf_gpu_sharding
)
from .scripts.test_cuvs_cagara import get_or_build_cuvs_cagra
from .scripts.em_update import em_exact_topk_streaming
from .losses import (
    listwise_kld, frontier_gap_loss, cell_ce_loss,
    identity_loss_cone, identity_loss_barycentric, identity_loss_legacy
)
from .dataset_loader import LaionDatasetLoader, T2IDatasetLoader

# CAGRA memory lifetime helper
class CagraHandle(NamedTuple):
    index: Any          # cagra.Index
    bank_gpu: Any       # CuPy array kept alive

# EM refresh parameter helpers for near-brute-force recall
def _cand_target(N: int, K: int) -> int:
    """
    Target #candidates to explore per query for near-BF EM refresh.
    Heuristic: grow with K and (weakly) with N, with sane caps.
    """
    # Base on K
    by_k = max(2 * K, K + 64, 256)          # robust when K is small (≥256)
    # Weakly scale with N but cap to stay practical (chunked refresh anyway)
    by_n = int(max(512, min(2048, 0.0005 * N)))   # ~0.05% of N, clamped [512, 2048]
    return min(4096, max(by_k, by_n))       # hard cap 4096 for stability


def em_params_cagra(N: int, K: int):
    """
    Return CAGRA params for EM refresh that are close to brute-force but fast.
    """
    cand = _cand_target(N, K)
    itopk_size = cand                       # how many candidate nodes to track
    # Wider walk helps when K is small or graph is harder
    search_width = 3 if K <= 10 else 2
    # iterations ~ log-scale in N; 60@1M, 80@10M, 100@100M; clamp 60..140
    import math
    miter = int(round(60 + 20 * max(0.0, math.log10(max(1.0, N)) - 6.0)))
    miter = max(60, min(140, miter))
    return dict(itopk_size=itopk_size, search_width=search_width, max_iterations=miter)




def _resolve_data_path(cfg) -> str:
    """
    Support either a single path or a dict keyed by dataset name.

    Useful when pairing caches produced by encode_data/shared_pca_cache.py
    where both LAION & DataComp caches share the same PCA.
    """
    data_path = getattr(cfg, "data_path", None)
    if isinstance(data_path, dict):
        dataset_key = getattr(cfg, "dataset", "laion")
        resolved = data_path.get(dataset_key) or data_path.get("default")
        if resolved is None:
            raise KeyError(f"No data_path entry for dataset '{dataset_key}'.")
        print(f"[data] Resolved dataset '{dataset_key}' → {resolved}")
        cfg.data_path_resolved = resolved
        return resolved
    cfg.data_path_resolved = data_path
    return data_path


def _timing_log_path(cfg, dataset_name: str, dataset_size: int):
    """
    Resolve timing log destination.
    If cfg.timing_log_file is unset, create an auto-named path using backend, dataset, size, date.
    """
    explicit = getattr(cfg, "timing_log_file", None)
    base_dir = getattr(cfg, "timing_log_dir", None)
    if explicit:
        path = Path(explicit)
    else:
        default_dir = Path(base_dir) if base_dir else Path("adapter/t2i_code/projector/outputs/timing")
        date_tag = time.strftime("%Y%m%d")
        safe_dataset = str(dataset_name).replace("/", "_").replace(" ", "_")
        backend_name = getattr(cfg, "backend", "unknown")
        filename = f"{backend_name}_{safe_dataset}_N{dataset_size}_{date_tag}.jsonl"
        path = default_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_timing(log_path: Path, event: str, duration_sec: float, extra: dict | None = None):
    """
    Append a JSONL timing entry to log_path.
    """
    if log_path is None:
        return
    entry = {
        "event": event,
        "duration_sec": float(duration_sec),
        "ts": time.time(),
    }
    if extra:
        entry["extra"] = extra
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def train_budget_aware(cfg, backend_config, dataset_loader=None):
    overall_t0 = time.time()
    # ---- Single GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set device in config
    cfg.device = str(device)
    
    print(f"Starting single GPU training on {device}")
    
    # --- EM/teacher refresh knobs (defaults) ---
    cfg.em_refresh_every = getattr(cfg, "em_refresh_every", 1)          # refresh each epoch ≥2
    cfg.em_refresh_method = getattr(cfg, "em_refresh_method", "ann")    # use ANN-based refresh
    cfg.em_chunk = getattr(cfg, "em_chunk", 200_000)                    # chunk size for EM refresh
    
    # ---- Load data using dataset loader
    if dataset_loader is None:
        from .dataset_loader import LaionDatasetLoader, T2IDatasetLoader
        resolved_data_path = _resolve_data_path(cfg)
        dataset_loader = LaionDatasetLoader(resolved_data_path)
    
    # Get training data
    data_t0 = time.time()
    X_train, T_train, exact_idx_train = dataset_loader.get_train_data()
    data_load_duration = time.time() - data_t0
    print("X dtype/contig:", X_train.dtype, X_train.flags['C_CONTIGUOUS'])
    print("T dtype/contig:", T_train.dtype, T_train.flags['C_CONTIGUOUS'])
    print("||X|| head:", np.linalg.norm(X_train[:5], axis=1))
    print("||T|| head:", np.linalg.norm(T_train[:5], axis=1))
    dim = dataset_loader.get_feature_dim()
    
    if cfg.hidden is None:
        cfg.hidden = dim
    
    # Print dataset info
    dataset_info = dataset_loader.get_dataset_info()
    dataset_size = dataset_info['train']['num_images']
    timing_log_path = _timing_log_path(cfg, getattr(cfg, "dataset", "dataset"), dataset_size)
    _append_timing(
        timing_log_path,
        "meta",
        0.0,
        extra={
            "backend": getattr(cfg, "backend", "unknown"),
            "dataset": getattr(cfg, "dataset", "unknown"),
            "num_images": dataset_info["train"]["num_images"],
            "num_texts": dataset_info["train"]["num_texts"],
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
        },
    )
    _append_timing(
        timing_log_path,
        "data_load",
        data_load_duration,
        extra={"num_images": dataset_info["train"]["num_images"], "num_texts": dataset_info["train"]["num_texts"]},
    )
    print(f"Train data: {dataset_info['train']['num_images']} images, {dataset_info['train']['num_texts']} texts")
    print(f"Feature dim: {dim}, Hidden dim: {cfg.hidden}")
    print(f"Exact indices shape: {exact_idx_train.shape}")
    print(f"KNN K: {dataset_info['train']['knn_k']}")

    # ---- Check if cache has PCA (features already PCA-reduced) ----
    cache_has_pca = False
    R = None
    pca_source = "unset"
    pca_start = time.time()
    
    # Try to get PCA from cache first
    if hasattr(dataset_loader, 'data') and dataset_loader.data is None:
        dataset_loader.data = dataset_loader.load_data()
    if hasattr(dataset_loader, 'data') and dataset_loader.data is not None:
        if "pca" in dataset_loader.data and dataset_loader.data["pca"] is not None:
            pca_source = "cache"
            pca_data = dataset_loader.data["pca"]
            cache_has_pca = True
            print("✅ Found PCA in cache pkl file (features are already PCA-reduced)")
            
            # Load PCA from cache
            R = PCARSpace(d_keep=pca_data["W"].shape[0], 
                         center_for_fit=bool(pca_data.get("center_for_fit", True)),
                         device=cfg.device)
            R.W = pca_data["W"].astype(np.float32)
            R.mu = (pca_data["mu"].astype(np.float32) if pca_data.get("mu") is not None else None)
            
            print(f"Loaded PCA from cache: W={R.W.shape}, mu={'None' if R.mu is None else R.mu.shape}")
            print(f"  Features are already PCA-reduced, no transform needed")
    
    # If cache doesn't have PCA, try pca_path or fit new PCA
    if not cache_has_pca:
        print("⚠️  No PCA found in cache, checking pca_path or fitting new PCA...")
        R = PCARSpace(d_keep=None, center_for_fit=True, device=cfg.device)
        
        if getattr(cfg, "pca_path", None):
            pca_source = "pca_path"
            print(f"Loading PCA from {cfg.pca_path}")
            z = np.load(cfg.pca_path, allow_pickle=True)
            R.W  = z["W"].astype(np.float32)
            R.mu = (None if ("mu" not in z or z["mu"] is None)
                    else z["mu"].astype(np.float32))
            if "center_for_fit" in z:
                R.center_for_fit = bool(z["center_for_fit"][0])
            print(f"Loaded PCA: W={R.W.shape}, mu={'None' if R.mu is None else R.mu.shape}")
            
            # Check if PCA is identity (no dimension reduction): output_dim == input_dim == feature_dim
            # This handles cases like T2I where features are already at target dimension
            if R.W.shape[0] == R.W.shape[1] == X_train.shape[1]:
                print(f"✅ PCA is identity (no reduction): output_dim={R.W.shape[0]} == input_dim={R.W.shape[1]} == feature_dim={X_train.shape[1]}")
                print("   Treating as no-op: features will be used as-is (similar to cache_has_pca)")
                cache_has_pca = True  # Treat as no-op, skip transform
            else:
                # PCA/rotation sanity checks (only if features are NOT already PCA-reduced)
                assert R.W.shape[1] == X_train.shape[1], f"PCA W dim mismatch: W.shape[1]={R.W.shape[1]} != X_train.shape[1]={X_train.shape[1]}"
                # rows should be orthonormal
                ortho = R.W @ R.W.T
                assert np.allclose(ortho, np.eye(ortho.shape[0], dtype=np.float32), atol=1e-3), "PCA W not orthonormal enough"
        else:
            # Fallback (single-process fit)
            pca_source = "pca_fit"
            print("Fitting new PCA on training data...")
            R = PCARSpace(d_keep=None, center_for_fit=True, device=cfg.device).fit(X_train)
    _append_timing(
        timing_log_path,
        f"pca_{pca_source}",
        time.time() - pca_start,
        extra={
            "d_in": int(X_train.shape[1]),
            "d_keep": (int(R.W.shape[0]) if getattr(R, "W", None) is not None else None),
            "cache_hit": cache_has_pca,
        },
    )
    
    # Use features as-is if already PCA-reduced, otherwise apply PCA transform
    if cache_has_pca:
        # Features are already PCA-reduced and normalized, use directly
        X_train_R = X_train.astype(np.float32, copy=False)
        print("✅ Using pre-computed PCA-reduced features from cache")
    else:
        # Apply PCA transform to original features
        bank_fp_train = bank_fingerprint(X_train, R)
        X_train_R = R.transform(X_train)
        print("✅ Applied PCA transform to features")
    
    # Create normalized R-space bank for consistent cosine metric during EM refresh
    # (features should already be normalized, but ensure it)
    X_train_R_norm = l2n_np(X_train_R.astype(np.float32, copy=False))
    
    # Compute bank fingerprint (needed for cache keys)
    if not cache_has_pca:
        bank_fp_train = bank_fingerprint(X_train, R)
    else:
        # For PCA-reduced features, use the reduced features directly for fingerprint
        bank_fp_train = bank_fingerprint(X_train_R, R)
    
    
    # Auto-scale ivf_nlist if N≠3M (keeps you future-proof)
    from .utils import suggested_nlist
    nlist_eff = suggested_nlist(X_train_R.shape[0], backend_config.ivf_nlist)
    if nlist_eff != backend_config.ivf_nlist:
        print(f"[ivf] overriding ivf_nlist {backend_config.ivf_nlist} → {nlist_eff} for N={X_train_R.shape[0]:,}")
        backend_config.ivf_nlist = nlist_eff
    
    print("✅ Rotated images cached")
    
    # ---- Build ANN indices on rotated images
    ivf_index = None
    cuvs_cagra_handle = None
    build_start_time = time.time()
    
    # Set TF32 for performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set FAISS/HNSW threads (so one node doesn't idle)
    if backend_config.num_threads and backend_config.num_threads > 0:
        try:
            import faiss
            faiss.omp_set_num_threads(backend_config.num_threads)
        except Exception:
            pass
    
    if getattr(cfg, "disable_ann_teacher", False):
        print("ANN teacher disabled: skipping ANN index build and ANN searches during training")
    elif getattr(cfg, "disable_exactK_teacher", False):
        print("Exact teacher disabled: using ANN-only training (IVF or cuvs_cagra backend required)")
    
    # Build ANN indices (skip only if disable_ann_teacher=True)
    # For IVF with large datasets (>9M), force CPU index to save GPU memory
    N_train = X_train_R.shape[0]
    ivf_force_cpu_for_large = (cfg.backend == "ivf" and N_train > 9_000_000)
    
    if not getattr(cfg, "disable_ann_teacher", False):
        if cfg.backend == "ivf":
            print("Building IVF index...")
            ivf_cpu_index = get_or_build_ivf(X_train_R, cfg.indices_dir, bank_fp_train, nlist_hint=backend_config.ivf_nlist, force_cpu=True, num_threads=backend_config.num_threads)
            # For large datasets, keep IVF on CPU to save GPU memory
            if ivf_force_cpu_for_large:
                print(f"[IVF] Large dataset (N={N_train:,}) detected - keeping index on CPU for memory efficiency")
                ivf_index = ivf_cpu_index
            else:
                # Setup GPU sharding for IVF
                ivf_index = setup_ivf_gpu_sharding(ivf_cpu_index, 1, 0, force_cpu=backend_config.force_cpu_eval)
            
            # Print IVF device sanity check
            try:
                import faiss
                is_gpu = hasattr(faiss, "GpuIndex") and isinstance(ivf_index, faiss.GpuIndex)
                print(f"[IVF] index device = {'GPU' if is_gpu else 'CPU'}")
            except Exception:
                pass
            
        
        elif cfg.backend == "cuvs_cagra":
            print("\n🔧 cuVS CAGRA Backend Configuration:")
            print(f"  Build algorithm: {backend_config.cuvs_cagra_build_algo}, metric: {backend_config.cuvs_cagra_metric}")
            print("  Parameters will be automatically tuned during build")
            print("  Using cuVS CAGRA with cuPy for memory lifetime management")
            print("Building cuVS CAGRA index...")
            
            # CRITICAL: Prepare data for CAGRA with proper memory lifetime handling
            import cupy as cp
            
            # Finalize X_train_R on CPU first (all transforms/normalization done)
            X_for_cagra = X_train_R.astype(np.float32, copy=False)
            # ensure queries and bank use same normalization policy
            X_for_cagra = l2n_np(X_for_cagra)
            
            # Move once to GPU and keep it alive for the entire training
            X_bank_gpu = cp.asarray(X_for_cagra, dtype=cp.float32, order="C")
            
            # Build via cached config (or tune offline once)
            cagra_index = get_or_build_cuvs_cagra(X_bank_gpu, cfg.dataset, bank_fp_train)
            
            # Stash in a handle that lives for the entire training
            cuvs_cagra_handle = CagraHandle(index=cagra_index, bank_gpu=X_bank_gpu)
            
            # Warm up once after build: run a tiny dummy query so kernels & caches are primed
            print("Warming up CAGRA index...")
            from cuvs.neighbors import cagra
            # light, budget-band warmups: primes kernels & caches at shallow & mid budgets
            for b in (10, 40, 80):
                w = 2 if b < 50 else 1
                sp = cagra.SearchParams(itopk_size=256, search_width=w, algo="auto", max_iterations=b)
                _ = cagra.search(sp, cagra_index, X_bank_gpu[:1], 10)
            print("✅ CAGRA index warmed up")
            
            # Sanity check: verify CAGRA vs brute force overlap
            print("Running CAGRA sanity check...")
            q_np = l2n_np(X_train_R[:1024].astype(np.float32))
            import cupy as cp
            q_gpu = cp.asarray(q_np)
            from cuvs.neighbors import cagra
            sp = cagra.SearchParams(itopk_size=256, search_width=2, algo="auto", max_iterations=60)
            Dg, Ig = cagra.search(sp, cuvs_cagra_handle.index, q_gpu[:64], 10)
            Ibf = brute_force_topk_streaming(q_np[:64], X_train_R, k=10)
            # quick recall@10
            overlap = sum(len(set(cp.asnumpy(Ig)[i]).intersection(set(Ibf[i]))) for i in range(64)) / (64*10)
            print(f"[Sanity] CAGRA vs BF R@10 ~ {overlap:.3f}")
            if overlap < 0.85:
                print("⚠️  Warning: Low CAGRA-BF overlap. Check metric/normalization settings.")
        else:
            if not getattr(cfg, "disable_ann_teacher", False) or cfg.backend != "exact_k":
                raise ValueError("cfg.backend must be 'ivf', 'cuvs_cagra', or 'exact_k' for training (GPU-supported backends only)")
    
    build_end_time = time.time()
    print(f"Built ANN indices for training in {build_end_time - build_start_time:.2f} seconds")
    _append_timing(
        timing_log_path,
        "ann_index_build",
        build_end_time - build_start_time,
        extra={
            "backend": cfg.backend,
            "N_train": N_train,
            "ivf_nlist": getattr(backend_config, "ivf_nlist", None),
            "force_cpu_for_large": bool(ivf_force_cpu_for_large),
        },
    )
    # sys.exit()
    # ---- Host vs device placement (fast path if VRAM allows)
    bytes_X = X_train.nbytes
    bytes_T = T_train.nbytes
    # heuristics: allow up to ~75% of free VRAM for bank+texts
    # For IVF with large datasets (>9M), force CPU-pinned memory to save GPU VRAM
    bank_on_gpu_default = False
    if torch.cuda.is_available() and not ivf_force_cpu_for_large:
        free, total = torch.cuda.mem_get_info()
        bank_on_gpu_default = (bytes_X + bytes_T) < int(0.75 * free)
    
    # Force CPU-pinned for IVF with large datasets
    if ivf_force_cpu_for_large:
        bank_on_gpu_default = False
        print(f"[Memory] Forcing CPU-pinned memory for bank (IVF + large dataset N={N_train:,})")

    cfg.bank_on_gpu = getattr(cfg, "bank_on_gpu", bank_on_gpu_default)

    if cfg.bank_on_gpu:
        X_base = torch.from_numpy(X_train).to(device, non_blocking=True)
        T_base_all = torch.from_numpy(T_train).to(device, non_blocking=True)
        X_base_cos = torch.from_numpy(l2n_np(X_train.astype(np.float32))).half().to(device, non_blocking=True)
        print(f"  ✅ Bank & texts on GPU: X={tuple(X_base.shape)}, T={tuple(T_base_all.shape)}")
    else:
        X_base = torch.from_numpy(X_train).pin_memory()
        T_base_all = torch.from_numpy(T_train).pin_memory()
        X_base_cos = torch.from_numpy(l2n_np(X_train.astype(np.float32))).half().pin_memory()
        print(f"  🧲 Bank & texts on CPU-pinned (streaming): X={tuple(X_base.shape)}, T={tuple(T_base_all.shape)}")

    # For pack building, use the normalized cosine version
    X_bank_for_packs = X_base_cos
    
    
    # Single GPU setup - use full bank
    print(f"Using full bank {X_base.shape}")
    
    # Precompute W^T for query rotation on-the-fly
    # When features are already PCA-reduced, W_T maps from original to PCA space
    # When features are NOT PCA-reduced, W_T maps from original to PCA space (same)
    # But if features are already PCA-reduced, queries are also in PCA space, so we need identity
    if cache_has_pca:
        # Features are already in PCA space, so queries will be too
        # W_T should be identity (or we skip the transform)
        # Actually, model outputs in same space as input, so if input is PCA space, output is PCA space
        # But we still need W_T for the case where we want to project from original to PCA
        # The issue is: t_b is already in PCA space, so we don't need W_T
        # Let's make W_T identity when cache_has_pca
        pca_dim = X_train_R.shape[1]
        W_T = torch.eye(pca_dim, dtype=torch.float32, device=device)  # Identity: [pca_dim, pca_dim]
        print(f"✅ Using identity W_T (features already PCA-reduced, dim={pca_dim})")
    else:
        W_T = torch.from_numpy(R.W.T.copy()).to(device)  # [original_dim, pca_dim]
        print(f"✅ Using PCA projection W_T: {W_T.shape}")

    exact_idx_train_t = torch.from_numpy(exact_idx_train.astype(np.uint32))  # [N,K_exact] on CPU with compact dtype
    exact_idx_proj_train_t = None  # defensive initialization
    
    # Known sizes (avoid shape broadcasts later)
    N, K_exact = exact_idx_train_t.shape
    
    # For 10M+ scale, use more conservative defaults
    if N > 5_000_000:
        cfg.em_refresh_every = getattr(cfg, "em_refresh_every", 2)      # every other epoch for 10M+
        cfg.em_chunk = getattr(cfg, "em_chunk", 100_000)                # smaller chunks for 10M+
        print(f"  Large dataset detected (N={N:,}), using conservative EM refresh settings")
    
    print(f"EM refresh: every {cfg.em_refresh_every} epochs, method={cfg.em_refresh_method}, chunk={cfg.em_chunk}")

    # ---- Helper functions
    def unwrap(m): return m.module if hasattr(m, "module") else m
    
    def compute_head_k(mask_b: torch.Tensor, cfg) -> int:
        # Compute effective pack size based on actual valid candidates
        eff_C = int(mask_b.sum(dim=1).max().item()) if mask_b.numel() > 0 else 0
        
        # Base: at least the effective pack size so heads can cover the pack,
        # clamp to [128, 512] for stability, and ensure >= k_eval_inside_pack.
        hk = max(eff_C, 128)
        hk = min(hk, 512)
        hk = max(hk, cfg.k_eval_inside_pack)
        return int(hk)
    
    # ---- Model, opt, EMA
    model = ResidualProjector(dim=dim, hidden=cfg.hidden, alpha=cfg.alpha).to(device)
    
    # Add new heads
    coarse_head = CoarseCellHead(dim=dim, M=cfg.M_cells).to(device)
    qty_head = QuantityHead(dim=dim).to(device) if cfg.enable_quantity_head else None
    
    # Single GPU setup - no DDP wrapping needed
    model_module = model
    coarse_head_module = coarse_head
    qty_head_module = qty_head
    
    # Compute regions (centroids) and image->cell assignment; build y per-batch on-the-fly
    print("Computing regions and cell targets on-the-fly...")
    Cents_np, labels_img_np = precompute_regions_and_residuals(
        X_train_R, exact_idx_train, M=cfg.M_cells, save_dir=None, device=str(device)
    )
    # labels_img_np is already [N] int32 array (image->cell assignments)
    Cents = Cents_np
    labels_img_d = torch.from_numpy(labels_img_np).to(device)
    
    with torch.no_grad():
        unwrap(coarse_head).C.copy_(torch.from_numpy(Cents).to(device))

    # Prepare helper to build y_cells per batch on device
    def build_y_cells_batch(exact_idx_batch: torch.Tensor, B: int) -> torch.Tensor:
        # exact_idx_batch: [B,K] on device; labels_img_d: [N_images]
        lab = labels_img_d[exact_idx_batch]  # [B,K]
        y = y_buf[:B]
        y.zero_()
        ones = torch.ones_like(lab, dtype=y.dtype)
        y.scatter_add_(1, lab.long(), ones)
        y /= (y.sum(dim=1, keepdim=True) + 1e-12)
        return y
    
    # Combine all parameters
    params = list(model.parameters()) + list(coarse_head.parameters())
    if qty_head is not None:
        params += list(qty_head.parameters())
    
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    
    # AMP scaler for mixed precision training (only on CUDA)
    use_cuda = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    # EMA state for all models
    
    ema_state = {**model.state_dict(),
                 **coarse_head.state_dict(),
                 **({} if qty_head is None else qty_head.state_dict())}
    ema_state = {k: v.detach().clone() for k, v in ema_state.items()}
    
    def ema_update():
        with torch.no_grad():
            for name, param in model.state_dict().items():
                ema_state[name].mul_(cfg.ema_decay).add_(param, alpha=1-cfg.ema_decay)
            for name, param in coarse_head.state_dict().items():
                ema_state[name].mul_(cfg.ema_decay).add_(param, alpha=1-cfg.ema_decay)
            if qty_head is not None:
                for name, param in qty_head.state_dict().items():
                    ema_state[name].mul_(cfg.ema_decay).add_(param, alpha=1-cfg.ema_decay)

    # ---- Epoch loop (Query Parallelism with DDP)
    N = T_base_all.size(0)
    order_all = torch.arange(N, device=device)
    
    budgets = list(cfg.budgets)
    # Skewed budget sampler (replace round-robin)
    p_budgets = np.asarray([1.0 / b for b in budgets], dtype=np.float64)
    p_budgets /= p_budgets.sum()
    
    # Single GPU batch processing
    M_cells = int(cfg.M_cells)
    max_B = int(cfg.batch_size)
    y_buf = torch.empty(max_B, M_cells, device=device, dtype=torch.float32)
    
    # Simple batch processing for single GPU
    batches_per_gpu = (N + cfg.batch_size - 1) // cfg.batch_size
    start_batch = 0
    end_batch = batches_per_gpu
    actual_batches_per_gpu = batches_per_gpu
    
    print(f"Training with backend={cfg.backend}, budgets={budgets}")
    print(f"Total samples: {N}, Batch size: {cfg.batch_size}")
    print(f"Total batches: {batches_per_gpu}")
    
    # Print trust-region activation strategy
    if cfg.backend == "cuvs_cagra":
        print(f"🔒 Trust-region: Enabled from epoch 1 (CAGRA backend - sensitive to embedding changes)")
    elif cfg.backend == "ivf":
        print(f"🔒 Trust-region: Enabled from epoch 1 (IVF backend - consistent protection)")
    else:
        print(f"🔒 Trust-region: Disabled (exact_k backend - no ANN teacher)")

    # ---- ANN head cache for 10M+ scale
    # Ensure cache can hold biggest head_k (up to 512)
    cfg.ann_khead = max(getattr(cfg, "ann_khead", 0) or 0, 512)   # guard
    ann_cache = AnnHeadCache(enable=cfg.ann_cache_enable, khead=cfg.ann_khead, device=device)
    
    # Staging buffers for unique-then-gather (reuse to avoid allocator churn)
    staging_Xuniq = None
    
    # ---- Loss analysis tracking
    loss_history = {
        "iterations": [],
        "epochs": [],
        "losses": {
            "total": [],
            "kld": [],
            "id": [],
            "gap": [],
            "cell": []
        },
        "epoch_losses": {
            "total": [],
            "kld": [],
            "id": [],
            "gap": [],
            "cell": []
        },
        "metadata": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "total_samples": N,
            "backend": cfg.backend,
            "budgets": budgets,
            "loss_weights": {
                "w_kld": cfg.w_kld,
                "w_id": cfg.w_id,
                "w_gap": cfg.w_gap,
                "w_cell": cfg.w_cell
            }
        }
    } if cfg.loss_analysis else None
    
    # Initialize iteration counter
    iteration_counter = 0
    
    # Checkpoint management
    checkpoint_dir = None
    if cfg.save_path:
        checkpoint_dir = Path(cfg.save_path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Checkpoint directory: {checkpoint_dir}")
    
    for ep in range(1, cfg.epochs + 1):
        epoch_t0 = time.time()
        em_refreshed = False
        # Clear cache each epoch to avoid reusing stale ANN neighborhoods
        if ann_cache is not None:
            ann_cache.clear()
        
        # For large datasets, also clear cache every few batches to manage memory
        clear_every = None
        if N > 5_000_000 and ep > 1:
            clear_every = max(100, N // (cfg.batch_size * 10))  # clear every ~10% of epoch
            print(f"  Large dataset: will clear ANN cache every {clear_every} batches")
        
        # === One-time teacher init + optional EM refresh ===
        if ep == 1:
            exact_idx_proj_train_t = exact_idx_train_t.clone()
            print("✓ Init projected-teacher from base exact@K")
        else:
            do_refresh = ((ep - 1) % cfg.em_refresh_every == 0)
            if do_refresh:
                em_refreshed = True
                torch.cuda.synchronize()
                t0 = time.time()
                
                # Set eval mode for consistent BatchNorm/Dropout behavior
                was_training = model.training
                model.eval()
                
                # Use em_exact_topk_streaming for ALL backends (memory-efficient, avoids OOM)
                print(f"  EM refresh: Using em_exact_topk_streaming for all backends (memory-efficient)")
                
                new_rows = []
                # Add progress bar for EM refresh
                num_chunks = (N + cfg.em_chunk - 1) // cfg.em_chunk
                pbar_refresh = tqdm(range(0, N, cfg.em_chunk), 
                                  desc=f"EM refresh (epoch {ep})", 
                                  unit="chunk", 
                                  total=num_chunks)
                
                for s in pbar_refresh:
                    e = min(N, s + cfg.em_chunk)
                    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_cuda, dtype=torch.float16):
                        q = model(T_base_all[s:e].to(device, non_blocking=True)).float()
                    qR = F.normalize(q @ W_T, dim=1).detach()
                    
                    # EM refresh using em_exact_topk_streaming for ALL backends (avoids OOM)
                    # This is more memory-efficient than using CAGRA/IVF native search for EM refresh
                    qR_np = qR.detach().cpu().numpy().astype(np.float32)
                    chunk_results = em_exact_topk_streaming(
                        Q=qR_np,
                        X=X_train_R_norm,
                        k=K_exact,
                        q_batch=min(cfg.em_chunk, 65536),
                        x_batch=None,  # Auto-tune from free VRAM
                        allow_tf32=True,  # Faster matmul on Ampere+
                        almost_exact_bf16=False,  # Keep exact results
                        overlap_io=True,  # Double-buffered H2D copies
                        show_progress=False
                    )
                    new_rows.append(chunk_results.astype(np.uint32))
                    
                    del q, qR
                
                pbar_refresh.close()
                
                new_exact_np = np.vstack(new_rows)

                # Use compact dtype for 10M+ scale (uint32 fits up to 4B ids)
                exact_idx_proj_train_t = torch.from_numpy(new_exact_np.astype(np.uint32))  # stay on CPU to save VRAM
                
                # Restore training mode
                if was_training:
                    model.train()
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                print(f"✓ Refreshed projected exact@{K_exact} in {time.time()-t0:.1f}s (epoch {ep})")
                _append_timing(
                    timing_log_path,
                    f"em_refresh_epoch_{ep}",
                    time.time() - t0,
                    extra={"chunk": cfg.em_chunk, "N": N, "K_exact": K_exact},
                )

        # Single GPU permutation
        perm = order_all[torch.randperm(N, device=device)]

        model.train()
        total_loss = 0.0
        delta_sum = 0.0
        n_seen = 0
        
        # Initialize epoch-level loss tracking
        epoch_losses = {
            "total": [],
            "kld": [],
            "id": [],
            "gap": [],
            "cell": []
        } if cfg.loss_analysis else None

        # Progress bar for batches in this epoch
        pbar = tqdm(range(actual_batches_per_gpu), 
                   desc=f"Epoch {ep}/{cfg.epochs}", 
                   unit="batch", 
                   total=actual_batches_per_gpu)
        
        # Process batches
        for batch_idx in pbar:
            s = batch_idx * cfg.batch_size
            e = min(N, s + cfg.batch_size)
            idx = perm[s:e]
            # If T_base_all is already on GPU, index it directly; otherwise do the old CPU→GPU hop
            if T_base_all.device.type == "cuda":
                t_b = T_base_all.index_select(0, idx)                         # stays on device
                idx_cpu = idx.detach().cpu()  # still need for CPU tensor indexing
            else:
                idx_cpu = idx.detach().cpu()
                t_b = T_base_all[idx_cpu].to(device, non_blocking=True)      # [B,D], base space (L2n)
            
            # Choose budget with skewed sampling
            B = int(np.random.choice(budgets, p=p_budgets))
            lam_ann = lambda_ann_for_budget(B, epoch=ep)
            
            # AMP autocast for mixed precision training
            with torch.amp.autocast('cuda', enabled=use_cuda, dtype=torch.float16):
                q_b = model(t_b)           # [B,D], student projection (L2n)
                
                # ----- Teachers -----

            # --- Unified pack construction (no hard switch) ---
            C_cur = max(pack_size_for(B), 192)

            # Smoothly increase student share: ep1→0.0, ep2→0.5, ep3+→1.0
            p_student = min(1.0, max(0.0, (ep - 1) / 2.0))
            C_s = int(round(p_student * C_cur))
            C_b = C_cur - C_s

            # Build packs in fp32 for stable similarities
            t32 = t_b.float()
            q32 = q_b.detach().float()
            
            # Only build packs that are needed (avoid double work at schedule ends)
            # Use GPU mirror for pack building (device-aware)
            parts = []
            if C_b > 0:
                ids_base = build_pack_topC_streaming(t32, X_bank_for_packs, min(cfg.C_pack, C_b), x_batch=100_000, use_fp16=True)
                parts.append(ids_base)
            if C_s > 0:
                ids_stud = build_pack_topC_streaming(q32, X_bank_for_packs, C_s, x_batch=100_000, use_fp16=True)
                parts.append(ids_stud)

            if len(parts) == 2:
                ids_mix = torch.cat([parts[0], parts[1]], dim=1)  # base first, then student
            else:
                ids_mix = parts[0]  # only one side needed

            # ids_mix: [B, Ctot] (base ids first, then student ids), may contain -1
            Bsz, Ctot = ids_mix.shape

            valid = ids_mix.ge(0)

            # Row-wise unique with order preserved:
            # (1) Make row-unique keys
            offset = torch.arange(Bsz, device=ids_mix.device).unsqueeze(1) * (X_base.size(0) + 1)
            keys   = torch.where(valid, ids_mix + offset, torch.full_like(ids_mix, -1))

            # (2) Group duplicates by sorting by key, break ties by original position
            pos    = torch.arange(Ctot, device=ids_mix.device).unsqueeze(0).expand_as(ids_mix)
            comb   = (keys.to(torch.int64) * (Ctot + 1)) + pos.to(torch.int64)
            order1 = comb.argsort(dim=1, stable=True)

            keys1  = keys.gather(1, order1)
            pos1   = pos.gather(1, order1)
            ids1   = ids_mix.gather(1, order1)

            # (3) Keep first occurrence in each key group
            is_first = torch.ones_like(keys1, dtype=torch.bool)
            is_first[:, 1:] = keys1[:, 1:] != keys1[:, :-1]

            # (4) Throw away invalid key == -1 entirely
            is_first = is_first & keys1.ne(-1)

            # (5) Stable compaction: push drops to the right, keep earlier positions' order
            big_pos  = Ctot + Ctot
            rank     = torch.where(is_first, pos1, torch.full_like(pos1, big_pos))
            order2   = rank.argsort(dim=1, stable=True)
            ids_comp = ids1.gather(1, order2)  # compacted left, drops on the right

            # (6) Final pack and mask
            if ids_comp.size(1) < C_cur:
                pad = torch.full((Bsz, C_cur - ids_comp.size(1)), -1, device=ids_comp.device, dtype=ids_comp.dtype)
                ids_b = torch.cat([ids_comp, pad], dim=1)
            else:
                ids_b = ids_comp[:, :C_cur]

            mask_b = ids_b.ge(0)

            # Fast path: if bank is on GPU, gather directly; otherwise use unique→gather streaming
            if X_bank_for_packs.device.type == "cuda":
                Xc_b = X_bank_for_packs.index_select(0, ids_b.clamp_min(0).view(-1))\
                                         .view(ids_b.size(0), ids_b.size(1), -1)
            else:
                # Unique-then-gather to minimize PCIe traffic (with staging buffers)
                ids_flat = ids_b[mask_b]                     # 1D kept ids
                if ids_flat.numel() > 0:
                    uniq_ids, inv = torch.unique(ids_flat, return_inverse=True)
                    # 1) Pull unique rows from CPU -> GPU once (reuse staging buffer)
                    # Use consistent bank source for cosine similarity
                    bank_src = X_base_cos if (X_base_cos is not None) else X_base
                    Xuniq_cpu = bank_src.index_select(0, uniq_ids.cpu())
                    if staging_Xuniq is None or staging_Xuniq.size() != Xuniq_cpu.size():
                        staging_Xuniq = torch.empty_like(Xuniq_cpu, device=device)
                    staging_Xuniq.copy_(Xuniq_cpu, non_blocking=True)
                    # 2) Reconstruct [B, C, D] by indexing the compact buffer
                    Xc_b = torch.zeros((ids_b.size(0), ids_b.size(1), staging_Xuniq.size(1)), 
                                       device=device, dtype=staging_Xuniq.dtype)
                    Xc_b[mask_b] = staging_Xuniq[inv]
                else:
                    # Empty pack case
                    Xc_b = torch.zeros((ids_b.size(0), ids_b.size(1), X_base.size(1)), 
                                       device=device, dtype=X_base.dtype)

            # Annealed query mixing instead of hard switch
            p_q = min(1.0, max(0.0, (ep - 1) / 2.0))   # 0.0 → 1.0 by ep3
            # Query mix in fp32 before W_T for dtype safety
            q_mix = ((1.0 - p_q) * t_b + p_q * q_b).float()
            # Note: mu is intentionally not subtracted for queries to keep training consistent with ANN policy
            t_b_R = F.normalize(q_mix @ W_T, dim=1)

            # Always do ANN unless disabled; cache-then-run ANN heads
            row_ids = idx_cpu
            if getattr(cfg, "disable_ann_teacher", False):
                cached_ids, cached_sims, miss = None, None, None
            else:
                cached_ids, cached_sims, miss = ann_cache.get(B, row_ids)
            ids_head_full = []
            sims_head_full = []
            
            # Define head_k consistently per batch (identical to single GPU)
            head_k = compute_head_k(mask_b, cfg)
            
            # Quick guardrails - debug print for head_k and pack construction (gated for 10M scale)
            if batch_idx == 0 and ep % 5 == 1:
                print(f"Epoch {ep}: C_cur={C_cur} C_b={C_b} C_s={C_s} p_student={p_student:.2f} head_k={head_k}")
            
            if (not getattr(cfg, "disable_ann_teacher", False)) and (miss is not None and miss.any()):
                t_miss = t_b_R[miss]
                if cfg.backend == "cuvs_cagra":
                    # cuVS CAGRA native search with budget-aware itopk_size
                    import cupy as cp
                    from cuvs.neighbors import cagra
                    
                    # t_miss: torch [b,D] on device; convert with zero-copy DLPack (faster for big batches)
                    from torch.utils.dlpack import to_dlpack
                    t_miss_gpu = cp.fromDlpack(to_dlpack(t_miss.contiguous()))  # float32 already; ensure contiguous

                    # itopk must comfortably cover head_k; widen early walks for small budgets
                    # NOTE: head_k is already computed above; C_pack is your candidate pack size
                    itopk = min(max(2 * head_k, int(cfg.C_pack)), 1024)  # gentle cap to prevent explosion
                    w = 2 if B < 50 else 1                             # wider graph walk for shallow budgets

                    search_params = cagra.SearchParams(
                        itopk_size=itopk,
                        search_width=w,
                        algo="auto",
                        max_iterations=max(1, int(B)),                 # budget-aware mapping remains
                    )
                    
                    D_gpu, I_gpu = cagra.search(search_params, cuvs_cagra_handle.index, t_miss_gpu, head_k)
                    D_cpu = cp.asnumpy(D_gpu)
                    I_cpu = cp.asnumpy(I_gpu)
                    
                    ids_head_m = torch.from_numpy(I_cpu.astype(np.int64)).to(device)
                    # cuVS CAGRA returns (negative) inner-product distances for IP → sims = -D
                    sims_head_m = torch.from_numpy((-D_cpu).astype(np.float32)).to(device)
                else:
                    faiss_index = ivf_index
                    faiss_index.nprobe = int(B)
                    sims, lab = faiss_index.search(t_miss.detach().cpu().numpy().astype(np.float32), head_k)
                    ids_head_m  = torch.from_numpy(lab.astype(np.int64)).to(device)
                    sims_head_m = torch.from_numpy(sims.astype(np.float32)).to(device)
                # row_ids is on CPU; ensure boolean mask is also on CPU for indexing
                miss_cpu = miss.detach().cpu() if miss.is_cuda else miss
                ann_cache.put_batch(B, row_ids[miss_cpu], ids_head_m, sims_head_m)
            # ---- assemble full heads (robust to None/shape) ----
            def _fit_head_shape(id_row, sim_row, head_k):
                # id_row: LongTensor [head_k] or [1, head_k] or None
                # sim_row: FloatTensor [head_k] or [1, head_k] or None
                if id_row is None or sim_row is None:
                    ids_out = torch.full((head_k,), -1, device=device, dtype=torch.long)
                    sims_out = torch.full((head_k,), -1e9, device=device, dtype=torch.float32)
                    return ids_out, sims_out

                if id_row.dim() == 2:
                    id_row = id_row[0]
                if sim_row.dim() == 2:
                    sim_row = sim_row[0]

                # truncate or pad to head_k
                if id_row.numel() > head_k:
                    id_row = id_row[:head_k]
                    sim_row = sim_row[:head_k]
                elif id_row.numel() < head_k:
                    pad = head_k - id_row.numel()
                    id_pad = torch.full((pad,), -1, device=device, dtype=id_row.dtype)
                    sim_pad = torch.full((pad,), -1e9, device=device, dtype=sim_row.dtype)
                    id_row = torch.cat([id_row, id_pad], dim=0)
                    sim_row = torch.cat([sim_row, sim_pad], dim=0)
                return id_row, sim_row

            ptr_m = 0
            ids_head_full = []
            sims_head_full = []

            # Note: 'miss' is a 1-D bool tensor per-row; cached_* are Python lists
            miss_list = (miss.tolist() if miss is not None else [False] * len(row_ids))
            cached_ids_list = (cached_ids or [None] * len(row_ids))
            cached_sims_list = (cached_sims or [None] * len(row_ids))

            for is_m, cid, csim in zip(miss_list, cached_ids_list, cached_sims_list):
                if is_m:
                    # take the freshly computed row from *_m
                    id_row = ids_head_m[ptr_m]
                    sim_row = sims_head_m[ptr_m]
                    ptr_m += 1
                else:
                    id_row = cid
                    sim_row = csim
                id_row, sim_row = _fit_head_shape(id_row, sim_row, head_k)
                ids_head_full.append(id_row)
                sims_head_full.append(sim_row)

            if getattr(cfg, "disable_ann_teacher", False):
                P_ann = None
            else:
                ids_head_full  = torch.stack(ids_head_full,  dim=0)  # [B, head_k]
                sims_head_full = torch.stack(sims_head_full, dim=0)  # [B, head_k]
                if batch_idx == 0:
                    print(f"  Head shapes: ids={ids_head_full.shape}, sims={sims_head_full.shape}, head_k={head_k}")
                scores_ann = _fill_scores_from_head(ids_head_full, sims_head_full, ids_b)
                P_ann = torch.softmax(scores_ann.float() / cfg.ann_tau, dim=1)

            # Exact teacher: different logic for epoch 1 vs epochs ≥2
            if getattr(cfg, "disable_exactK_teacher", False):
                # ANN-only training: skip exact teacher computation
                P_exact = None
            elif ep == 1:
                # Epoch 1: uniform exact teacher from base space
                # Move only current batch to GPU to save VRAM
                exact_ids_b = exact_idx_train_t.index_select(0, idx_cpu).to(device, non_blocking=True).long()  # [B,K_exact]
                P_exact = exact_teacher_in_pack(ids_b, exact_ids_b)
            else:
                # Epochs ≥2: rank-weighted exact teacher from projected space
                # Move only current batch to GPU to save VRAM
                exact_ids_b = exact_idx_proj_train_t.index_select(0, idx_cpu).to(device, non_blocking=True).long()  # [B,K_exact] from EM step
                P_exact = exact_teacher_in_pack_ranked(ids_b, exact_ids_b, gamma=0.15)

            # Mixed teacher (or exact-only if ANN disabled, or ANN-only if exact disabled)
            if getattr(cfg, "disable_ann_teacher", False):
                P_teach = P_exact
            elif getattr(cfg, "disable_exactK_teacher", False):
                P_teach = P_ann
            else:
                P_teach = lam_ann * P_ann + (1.0 - lam_ann) * P_exact
            P_teach = P_teach / (P_teach.sum(dim=1, keepdim=True) + 1e-12)

            # ----- Losses (4 core losses only) -----
            # Force all computations to fp32 for stable ranking
            q32, Xc32 = q_b.float(), Xc_b.float()
            scores = torch.einsum('bd,bcd->bc', q32, Xc32)     # [B,C]
            
            # 1. KL Divergence Loss (KLD) - Main ranking loss
            loss_kld = listwise_kld(q32, Xc32, mask_b, P_teach, tau=cfg.kl_tau)
            
            # 2. Smart Identity Loss (ID) - Semantic preservation with helpful moves
            if cfg.use_smart_identity:
                if cfg.identity_mode == "cone":
                    # Cone-based identity loss (adaptive threshold)
                    loss_id_core = identity_loss_cone(q_b, t_b, budget=B, epoch=ep)
                elif cfg.identity_mode == "barycentric":
                    # Barycentric semantic anchor
                    pos_ids_b = exact_idx_train_t.index_select(0, idx_cpu)[:, :8].to(device, non_blocking=True)  # [B, 8] top positives
                    loss_id_core = identity_loss_barycentric(q_b, t_b, pos_ids_b, X_base, beta=0.15)
                else:
                    # Fallback to legacy
                    loss_id_core = identity_loss_legacy(q_b, t_b)
            else:
                # Legacy L2 identity loss
                loss_id_core = identity_loss_legacy(q_b, t_b)
            
            # Trust-region loss: Backend-specific activation for optimal performance
            loss_tr = torch.zeros((), device=q_b.device)
            
            # Trust-region activation strategy:
            # - CAGRA: Enable from epoch 1 (sensitive to embedding changes, needs early protection)
            # - IVF: Enable from epoch 1 (consistent protection for all ANN backends)
            # - Exact: Never enable (no ANN teacher to protect)
            exact_only = getattr(cfg, "disable_ann_teacher", False) and not getattr(cfg, "disable_exactK_teacher", False)
            tr_enable = getattr(cfg, "tr_enable", True)  # optional cfg knob
            
            # Determine when to activate trust-region based on backend
            if cfg.backend == "cuvs_cagra":
                tr_epoch_threshold = 1  # Enable from epoch 1 for CAGRA
            elif cfg.backend == "ivf":
                tr_epoch_threshold = 1  # Enable from epoch 1 for IVF (consistent with CAGRA)
            else:
                tr_epoch_threshold = 999  # Never enable for exact_k
            
            if tr_enable and (not exact_only) and ep >= tr_epoch_threshold:
                # Scale margin by feature dimension to be dataset-agnostic
                base_margin = getattr(cfg, "tr_margin", 0.20)
                tr_margin = base_margin * math.sqrt(dim / 512.0)  # scale by sqrt(dim/512)
                move = (q_b - t_b).pow(2).sum(dim=1).sqrt()  # ||Δ||
                loss_tr = F.relu(move - tr_margin).mean()
                
                # Debug: print trust-region stats for first batch of activation epoch
                if batch_idx == 0 and ep == tr_epoch_threshold:
                    print(f"  [TR Debug] Backend={cfg.backend}, ep={ep}, margin={tr_margin:.4f}, mean_move={move.mean():.4f}, loss_tr={loss_tr:.6f}")
            
            # combine ID pieces (weights applied in the final sum)
            loss_id = loss_id_core
            
            # 3. Gap Loss (Frontier Gap) - Ranking stability
            m = 0.05 if B <= 30 else 0.03
            loss_gap = frontier_gap_loss(scores, mask_b, k=cfg.k_eval_inside_pack, m=m, extra=32)
            
            # 4. Cell Loss (Coarse Cell Routing) - ANN routing efficiency
            # Move only current batch to GPU to save VRAM
            exact_ids_for_cells = exact_idx_train_t.index_select(0, idx_cpu).to(device, non_blocking=True).long()  # [B,K] on device
            y_cells_batch = build_y_cells_batch(exact_ids_for_cells, B=idx.size(0))  # [B, M]
            loss_cell_raw = cell_ce_loss(q32, y_cells_batch, coarse_head, tau=cfg.kl_tau)
            
            # Trust-region weight: Consistent ramping for all ANN backends
            tr_w = getattr(cfg, "tr_weight", 0.5) * cfg.w_id
            
            if cfg.backend == "cuvs_cagra":
                # CAGRA: Gradual ramp from epoch 1 to 3 (early protection, gentle activation)
                tr_w *= min(1.0, max(0.0, (ep - 1) / 2.0))  # ramp from ep1 to ep3
            elif cfg.backend == "ivf":
                # IVF: Same gradual ramp from epoch 1 to 3 (consistent with CAGRA)
                tr_w *= min(1.0, max(0.0, (ep - 1) / 2.0))  # ramp from ep1 to ep3
            else:
                # Exact: No trust-region
                tr_w = 0.0
            
            # Cell loss weight ramps 1.0→0.6 by ep3
            w_cell_eff = cfg.w_cell * (1.0 - 0.4 * min(1.0, ep / 3.0))

            # Total loss with 4 core losses + trust-region
            loss = (cfg.w_kld * loss_kld +
                    cfg.w_id * loss_id +
                    cfg.w_gap * loss_gap +
                    w_cell_eff * loss_cell_raw +
                    tr_w * loss_tr)

            # Delta computation outside autocast
            with torch.no_grad():
                delta_sum += (q_b - t_b).norm(dim=1).sum().item()
                n_seen += q_b.size(0)

            opt.zero_grad()
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            ema_update()

            total_loss += float(loss.item()) * idx.size(0)
            
            # Track losses for analysis (4 core losses only)
            if cfg.loss_analysis and loss_history is not None:
                iteration_counter += 1
                loss_history["iterations"].append(iteration_counter)
                loss_history["losses"]["total"].append(float(loss.item()))
                loss_history["losses"]["kld"].append(float(loss_kld.item()))
                loss_history["losses"]["id"].append(float(loss_id.item()))
                loss_history["losses"]["gap"].append(float(loss_gap.item()))
                loss_history["losses"]["cell"].append(float(loss_cell_raw.item()))
                
                # Also track for epoch-level analysis
                if epoch_losses is not None:
                    epoch_losses["total"].append(float(loss.item()))
                    epoch_losses["kld"].append(float(loss_kld.item()))
                    epoch_losses["id"].append(float(loss_id.item()))
                    epoch_losses["gap"].append(float(loss_gap.item()))
                    epoch_losses["cell"].append(float(loss_cell_raw.item()))
            
            # Periodic cache clearing for large datasets
            if clear_every is not None and batch_idx > 0 and batch_idx % clear_every == 0:
                if ann_cache is not None:
                    ann_cache.clear()
                    print(f"  Cleared ANN cache at batch {batch_idx}")
            
            # Update progress bar with current metrics
            current_loss = total_loss / max(1, s + cfg.batch_size)
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'budget': B,
                'λ_ann': f'{lam_ann:.2f}'
            })

        pbar.close()
        sched.step()
        mean_delta = delta_sum / max(1, n_seen)
        avg_loss = total_loss / N
        if (ep % cfg.print_every) == 0:
            print(f"[ep {ep}] loss={avg_loss:.5f}  mean||Δ||={mean_delta:.4f}  (backend={cfg.backend})")
        
        _append_timing(
            timing_log_path,
            f"epoch_{ep}",
            time.time() - epoch_t0,
            extra={
                "batches": actual_batches_per_gpu,
                "em_refreshed": em_refreshed,
                "backend": cfg.backend,
            },
        )
        
        # Save intermediate checkpoint
        if checkpoint_dir is not None:
            checkpoint_path = checkpoint_dir / f"epoch_{ep}.pt"
            to_save = {
                "model_state_dict": model.state_dict(),
                "coarse_head_state_dict": coarse_head.state_dict(),
                "qty_head_state_dict": (qty_head.state_dict() if qty_head is not None else None),
                "W": R.W, "mu": R.mu, "center_for_fit": R.center_for_fit,
                "dim": dim, "hidden": cfg.hidden, "alpha": cfg.alpha, "M_cells": cfg.M_cells,
                "epoch": ep,
                "avg_loss": avg_loss,
                "mean_delta": mean_delta,
                "backend": cfg.backend,
                "dataset": cfg.dataset,
            }
            torch.save(to_save, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Aggregate epoch-level losses (4 core losses only)
        if cfg.loss_analysis and loss_history is not None and epoch_losses is not None:
            loss_history["epochs"].append(ep)
            for loss_key in ["total", "kld", "id", "gap", "cell"]:
                if loss_key in epoch_losses and len(epoch_losses[loss_key]) > 0:
                    # Calculate mean loss for this epoch
                    epoch_mean = sum(epoch_losses[loss_key]) / len(epoch_losses[loss_key])
                    loss_history["epoch_losses"][loss_key].append(epoch_mean)
                else:
                    loss_history["epoch_losses"][loss_key].append(0.0)
            
            # Print epoch-level loss summary
            print(f"📊 Epoch {ep} Loss Summary:")
            for loss_key in ["total", "kld", "id", "gap", "cell"]:
                if loss_key in epoch_losses and len(epoch_losses[loss_key]) > 0:
                    epoch_mean = sum(epoch_losses[loss_key]) / len(epoch_losses[loss_key])
                    print(f"  • {loss_key}: {epoch_mean:.6f}")


    # load EMA for final
    model.load_state_dict({k: v for k, v in ema_state.items() if k in model.state_dict()})
    coarse_head.load_state_dict({k: v for k, v in ema_state.items() if k in coarse_head.state_dict()})
    if qty_head is not None:
        qty_head.load_state_dict({k: v for k, v in ema_state.items() if k in qty_head.state_dict()})
    
    # Save final checkpoint as "best.pt" (can be overridden by evaluation)
    if checkpoint_dir is not None:
        final_checkpoint_path = checkpoint_dir / "best.pt"
        to_save = {
            "model_state_dict": model.state_dict(),
            "coarse_head_state_dict": coarse_head.state_dict(),
            "qty_head_state_dict": (qty_head.state_dict() if qty_head is not None else None),
            "W": R.W, "mu": R.mu, "center_for_fit": R.center_for_fit,
            "dim": dim, "hidden": cfg.hidden, "alpha": cfg.alpha, "M_cells": cfg.M_cells,
            "epoch": cfg.epochs,
            "backend": cfg.backend,
            "dataset": cfg.dataset,
            "final": True,
        }
        torch.save(to_save, final_checkpoint_path)
        print(f"✓ Saved final checkpoint: {final_checkpoint_path}")
        
        # Also save to original path for backward compatibility (if it's a file path)
        if cfg.save_path and not Path(cfg.save_path).is_dir():
            torch.save(to_save, cfg.save_path)
            print(f"✓ Saved projector and heads to {cfg.save_path}")
        
        # Print checkpoint summary
        print(f"\n📊 Checkpoint Summary:")
        print(f"  Directory: {checkpoint_dir}")
        print(f"  Intermediate: epoch_1.pt, epoch_2.pt, ..., epoch_{cfg.epochs}.pt")
        print(f"  Final: best.pt")
        print(f"  Use evaluation script to select best checkpoint")
    
    # Save loss analysis data
    if cfg.loss_analysis and loss_history is not None:
        import json
        Path(os.path.dirname(cfg.loss_analysis_file)).mkdir(parents=True, exist_ok=True)
        with open(cfg.loss_analysis_file, 'w') as f:
            json.dump(loss_history, f, indent=2)
        print(f"✓ Saved loss analysis data to {cfg.loss_analysis_file}")
        print(f"  - Total iterations: {len(loss_history['iterations'])}")
        print(f"  - Total epochs: {len(loss_history['epochs'])}")
        print(f"  - Loss components tracked: {list(loss_history['losses'].keys())}")
        print(f"  - Epoch-level losses tracked: {list(loss_history['epoch_losses'].keys())}")

    _append_timing(
        timing_log_path,
        "train_budget_aware_total",
        time.time() - overall_t0,
        extra={"epochs": cfg.epochs, "backend": getattr(cfg, "backend", "unknown")},
    )
    return model_module, coarse_head, qty_head, R, (ivf_index, cuvs_cagra_handle), (X_train, T_train)


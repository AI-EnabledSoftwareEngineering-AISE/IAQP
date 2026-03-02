#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
--------------
Budget-aware projector training for text→image retrieval with 4 core losses and smart identity loss.

- Loads pre-encoded CLIP features and exact brute-force t2i@K from:
    data/coco_cache/coco_ranking.pkl

- Query-side residual MLP projector with 4 core losses:
  1. KL Divergence Loss (KLD): Main ranking loss - teaches model to rank like teacher
  2. Smart Identity Loss (ID): Semantic preservation with helpful moves allowed:
     - Cone mode: One-sided cosine barrier with adaptive threshold (default)
     - Subspace mode: Only penalizes orthogonal drift from positive manifold
     - Barycentric mode: Moves identity target toward relevant images
     - Legacy mode: Traditional L2 distance (for comparison)
  3. Gap Loss (Frontier Gap): Ranking stability - clear margin between top-K and rest
  4. Cell Loss (Coarse Cell Routing): ANN routing efficiency - predicts relevant regions

- Smart Identity Loss Features:
  - Budget-aware: More permissive for small budgets (need more steering)
  - Epoch decay: Starts permissive, gradually tightens constraints
  - Allows helpful moves while preventing semantic drift

- Rotation-only PCARSpace (same W applied to images and queries).
- Candidate pack = top-C by base cosine per batch (computed on-the-fly).
- Mixed teacher inside pack:
    P_teach(B) = lambda_ann(B) * P_ANN@B + (1 - lambda_ann(B)) * P_exact
  where P_ANN@B is derived from HNSW(ef=B) or IVF(nprobe=B), and P_exact
  comes from exact top-K intersected with the pack.

- Evaluates recall@K across B ∈ {10,20,...,100} for both HNSW and IVF.

Dependencies: torch, numpy, faiss (FAISS), hnswlib, tqdm
"""

import os
import math
import argparse
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import time
import json
import math
import hashlib
try:
    from faiss.contrib.torch_utils import using_stream as faiss_using_stream
except Exception:
    faiss_using_stream = None

# Optional ANN libs (install as needed)
try:
    import faiss  # pip install faiss-gpu or faiss-cpu (depending on environment)
except Exception as e:
    faiss = None
try:
    import hnswlib  # pip install hnswlib
except Exception as e:
    hnswlib = None

# Import from our modules
from .trainer import train_budget_aware
from .evaluator import evaluate
from .utils import (
    set_seed, l2n_np, bank_fingerprint,
    get_or_build_hnsw, get_or_build_ivf, PCARSpace, ResidualProjector,
    CoarseCellHead, QuantityHead
)
from .dataset_loader import LaionDatasetLoader, CocoDatasetLoader, Flickr30KDatasetLoader, T2IDatasetLoader, BaseDatasetLoader
from .config.factory import create_config, apply_dataset_recommendations


# -------------------- Dataset Loader Factory --------------------
def create_dataset_loader(dataset_type: str, data_path: str, **kwargs) -> BaseDatasetLoader:
    """
    Factory function to create dataset loaders based on dataset type.
    
    Args:
        dataset_type: Type of dataset ('laion', 'coco', etc.)
        data_path: Path to the dataset cache file
        **kwargs: Additional dataset-specific parameters
        
    Returns:
        Dataset loader instance
        
    Raises:
        ValueError: If dataset_type is not supported
    """
    if dataset_type.lower() == "laion":
        print("Loading Laion dataset...")
        return LaionDatasetLoader(data_path, **kwargs)
    elif dataset_type.lower() == "datacomp":
        print("Loading DataComp dataset...")
        # DataComp uses the same cache format as LAION, so reuse the LAION loader
        return LaionDatasetLoader(data_path, **kwargs)
    elif dataset_type.lower() == "coco":
        return CocoDatasetLoader(data_path, **kwargs)
    elif dataset_type.lower() == "flickr30k":
        return Flickr30KDatasetLoader(data_path, **kwargs)
    elif dataset_type.lower() == "t2i":
        print("Loading T2I dataset...")
        return T2IDatasetLoader(data_path, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                        f"Supported types: 'laion', 'datacomp', 'coco', 'flickr30k', 't2i'")





# -------------------- Shared Backend Config --------------------
@dataclass
class BackendConfig:
    """Shared configuration for ANN backends (IVF and CAGRA) used in both training and evaluation."""
    # IVF parameters
    ivf_nlist: int = 2048
    
    # CAGRA parameters  
    cuvs_cagra_build_algo: str = "nn_descent"
    cuvs_cagra_metric: str = "inner_product"
    
    # Common parameters
    num_threads: int = 0  # 0 = use all available cores
    force_cpu_eval: bool = False

# -------------------- Evaluation Config --------------------
@dataclass
class EvalConfig:
    """Evaluation-specific configuration with sensible defaults."""
    # Evaluation backends
    eval_backend: str = "both"  # "both", "ivf", "cuvs_cagra", "hnsw", "nsg", "imi", "diskann", "cuvs_hnsw"
    
    # Evaluation metrics
    eval_split: str = "val"
    eval_topk: int = 10
    eval_ph: bool = False
    eval_r5: bool = False
    
    # Evaluation-only backend parameters (with sensible defaults)
    # HNSW for evaluation
    hnsw_M: int = 32
    hnsw_efC: int = 200
    
    # NSG for evaluation  
    nsg_R: int = 100
    nsg_L: int = 400
    
    # DiskANN for evaluation
    diskann_graph_degree: int = 32
    diskann_build_complexity: int = 80
    diskann_c_multiplier: int = 2
    diskann_max_budget: int = 110
    
    # cuVS HNSW for evaluation
    cuvs_hnsw_M: int = 100
    cuvs_hnsw_efC: int = 400
    cuvs_hnsw_mult_ef: int = 1
    cuvs_hnsw_metric: str = "cosine"
    cuvs_hnsw_hierarchy: str = "none"

# -------------------- Training Config --------------------
@dataclass
class TrainConfig:
    # data
    dataset: str = "laion"
    data_path: str = "data/coco_cache/coco_ranking.pkl"
    indices_dir: str = "indices/"
    
    # model
    alpha: float = 0.25
    hidden: Optional[int] = None  # default = dim

    # training
    device: str = "cuda"
    epochs: int = 8
    batch_size: int = 8192
    lr: float = 1e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    print_every: int = 1

    # candidate pack
    C_pack: int = 256
    K_exact_in_pickle: int = 100

    # teachers
    kl_tau: float = 0.07
    ann_tau: float = 0.07
    disable_ann_teacher: bool = False
    disable_exactK_teacher: bool = False

    # losses (4 core losses only)
    w_kld: float = 1.0
    w_id: float = 1e-3
    w_gap: float = 0.05
    w_cell: float = 0.3
    k_eval_inside_pack: int = 10
    
    # Trust-region parameters (for improved generalization after epoch 2)
    tr_margin: float = 0.20
    tr_weight: float = 0.5
    tr_enable: bool = True
    
    # EM refresh parameters (for teacher-student alignment)
    em_refresh_every: int = 1
    em_refresh_method: str = "ann"
    em_chunk: int = 200_000
    
    # Smart identity loss
    use_smart_identity: bool = True
    identity_mode: str = "cone"
    subspace_optimization: str = "optimized"
    
    # heads
    M_cells: int = 4096
    enable_quantity_head: bool = False

    # budgets
    budgets: Tuple[int, ...] = tuple(range(10, 101, 10))

    # training backend (GPU-supported only)
    backend: str = "cuvs_cagra"  # 'ivf', 'cuvs_cagra', or 'exact_k'

    # misc
    seed: int = 42
    save_path: Optional[str] = "outputs/checkpoints/training"

    # ANN cache
    ann_cache_enable: bool = True
    ann_khead: int = 256
    ann_cache_max_rows: int = 200_000

    # test-only evaluation
    test_only: bool = False
    model_path: Optional[str] = None
    
    # Memory management
    pack_chunk_size: int = 8192
    pack_x_batch_size: int = 25000
    
    # Loss analysis
    loss_analysis: bool = False
    loss_analysis_file: str = "loss_analysis.json"
    
    # Evaluation control
    skip_eval: bool = False
    
    # PCA precomputation
    pca_path: Optional[str] = None

    # Timing/logging
    timing_log_file: Optional[str] = None
    timing_log_dir: Optional[str] = None








# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, choices=["laion", "datacomp", "coco", "flickr30k", "t2i"], default="laion",
                   help="Dataset type to use (laion, datacomp, coco, flickr30k, t2i)")
    p.add_argument("--data_path", type=str, default="/ssd/hamed/ann/laion/precompute/laion_cache_up_to_0.pkl")
    p.add_argument("--root_dir", type=str, default="/ssd/hamed/ann/laion/")
    p.add_argument("--backend", type=str, choices=["ivf", "cuvs_cagra", "exact_k"], default="cuvs_cagra", 
                   help="Training backend: 'ivf' (GPU-sharded FAISS), 'cuvs_cagra' (GPU-accelerated CAGRA), or 'exact_k' (exact search only)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--print_every", type=int, default=1, help="Print training progress every N epochs")
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--C_pack", type=int, default=256)
    p.add_argument("--eval_split", type=str, choices=["val", "test", "inbound"], default="val")
    p.add_argument("--eval_topk", type=int, default=10)
    p.add_argument("--eval_ph", action="store_true", help="Compute PH@10 metric")
    p.add_argument("--eval_r5", action="store_true", help="Compute R@5 metric")
    p.add_argument("--save_path", type=str, default="outputs/checkpoints/training",
                   help="Checkpoint directory path (will save intermediate checkpoints as epoch_N.pt and final best.pt)")
    p.add_argument("--test_only", action="store_true", help="Load projector and eval on split without training")
    p.add_argument("--model_path", type=str, default=None, help="Path to saved projector .pt for test_only")
    p.add_argument("--seed", type=int, default=42)
    # Backend parameters (shared between training and evaluation)
    p.add_argument("--ivf_nlist", type=int, default=2048, help="IVF number of clusters (auto-scaled for dataset size)")
    p.add_argument("--cuvs_cagra_build_algo", type=str, default="nn_descent", choices=["ivf_pq", "nn_descent"], help="cuVS CAGRA build algorithm")
    p.add_argument("--cuvs_cagra_metric", type=str, default="inner_product", choices=["sqeuclidean", "inner_product"], help="cuVS CAGRA distance metric")
    p.add_argument("--num_threads", type=int, default=0, help="Number of CPU threads (0 = use all available cores)")
    p.add_argument("--force_cpu_eval", action="store_true", help="Force CPU evaluation to avoid GPU memory issues")
    
    # Training control
    p.add_argument("--disable_ann_teacher", action="store_true", help="Disable ANN teacher (use exact-only teacher; skips ANN index build and searches)")
    p.add_argument("--disable_exactK_teacher", action="store_true", help="Disable exact teacher (use ANN-only teacher; requires IVF or cuvs_cagra backend)")
    
    # Loss weights (4 core losses only)
    p.add_argument("--w_kld", type=float, default=1.0, help="Weight for KL divergence loss")
    p.add_argument("--w_id", type=float, default=1e-3, help="Weight for identity loss")
    p.add_argument("--w_gap", type=float, default=0.05, help="Weight for frontier gap loss")
    p.add_argument("--w_cell", type=float, default=0.3, help="Weight for coarse-cell CE loss")
    
    # Trust-region parameters (for improved generalization after epoch 2)
    p.add_argument("--tr_margin", type=float, default=0.20, help="Trust region margin (L2 distance) - free move radius before constraint kicks in")
    p.add_argument("--tr_weight", type=float, default=0.5, help="Trust region weight relative to identity loss - how strong the constraint is")
    p.add_argument("--tr_enable", dest="tr_enable", action="store_true", help="Enable trust-region loss after epoch 2 (disabled for exact-K-only training)")
    p.add_argument("--no_tr_enable", dest="tr_enable", action="store_false", help="Disable trust-region loss")
    p.set_defaults(tr_enable=True)
    
    # EM refresh parameters (for teacher-student alignment)
    p.add_argument("--em_refresh_every", type=int, default=1, help="Refresh exact teacher every N epochs (1=every epoch ≥2)")
    p.add_argument("--em_refresh_method", type=str, default="ann", choices=["ann"], help="Method for refreshing exact teacher: ann (uses existing backend)")
    p.add_argument("--em_chunk", type=int, default=200_000, help="Chunk size for EM refresh processing (memory management)")
    
    # Smart identity loss
    p.add_argument("--use_smart_identity", dest="use_smart_identity", action="store_true", 
                   help="Use smart cone-based identity loss instead of L2")
    p.add_argument("--no_smart_identity", dest="use_smart_identity", action="store_false", 
                   help="Use legacy L2 identity loss")
    p.set_defaults(use_smart_identity=True)
    p.add_argument("--identity_mode", type=str, default="cone", 
                   choices=["cone", "legacy", "subspace", "barycentric"],
                   help="Identity loss mode: cone (smart), legacy (L2), subspace, barycentric")
    p.add_argument("--subspace_optimization", type=str, default="optimized",
                   choices=["original", "optimized", "ultra_fast"],
                   help="Subspace optimization level: original (slow), optimized (cached), ultra_fast (maximum speed)")
    p.add_argument("--M_cells", type=int, default=4096, help="Number of coarse cells/regions")
    p.add_argument("--enable_quantity_head", action="store_true", help="Enable quantity head for budget prediction")
    
    
    # Memory management
    p.add_argument("--pack_chunk_size", type=int, default=8192, help="Chunk size for pack precomputation")
    p.add_argument("--pack_x_batch_size", type=int, default=25000, help="X batch size for pack precomputation")
    
    # Loss analysis
    p.add_argument("--loss_analysis", action="store_true", help="Enable loss tracking for analysis and visualization")
    p.add_argument("--loss_analysis_file", type=str, default="loss_analysis.json", help="Output file for loss tracking data")
    
    # Evaluation control
    p.add_argument("--skip_eval", action="store_true", help="Skip evaluation after training (save time)")
    
    # PCA precomputation
    p.add_argument("--pca_path", type=str, default=None,
                   help="Path to precomputed PCA npz (W, mu). If set, skip fitting/broadcasting.")
    
    # Timing logs
    p.add_argument("--timing_log_file", type=str, default=None, help="Explicit path to JSONL timing log")
    p.add_argument("--timing_log_dir", type=str, default=None, help="Directory for auto-named timing log")
    
    args = p.parse_args()

    # Sync backend and disable_ann_teacher flag
    if args.backend == "exact_k":
        args.disable_ann_teacher = True
    elif args.disable_ann_teacher and args.backend != "exact_k":
        print("[info] --disable_ann_teacher set; forcing backend=exact_k to skip ANN entirely")
        args.backend = "exact_k"
    
    # Validate disable_exactK_teacher (ANN-only training) - IVF or cuvs_cagra
    if args.disable_exactK_teacher:
        if args.backend not in ["ivf", "cuvs_cagra"]:
            print(f"[ERROR] --disable_exactK_teacher requires IVF or cuvs_cagra backend, but got {args.backend}")
            print("ANN-only training is supported with IVF backend (best performance) or cuvs_cagra backend")
            exit(1)
        if args.disable_ann_teacher:
            print("[ERROR] Cannot disable both ANN and exact teachers simultaneously")
            exit(1)
    
    # Validate training backend (GPU-supported backends only)
    if args.backend not in ["ivf", "cuvs_cagra", "exact_k"]:
        print(f"[ERROR] Training backend '{args.backend}' not supported. Use 'ivf', 'cuvs_cagra', or 'exact_k'")
        print("Training is restricted to GPU-supported backends for optimal performance")
        exit(1)

    # eval flags
    eval_ph = args.eval_ph
    indices_dir = os.path.join(args.root_dir, "indices/")
    os.makedirs(indices_dir, exist_ok=True)
    
    # Create shared backend config
    backend_config = BackendConfig(
        ivf_nlist=args.ivf_nlist,
        cuvs_cagra_build_algo=args.cuvs_cagra_build_algo,
        cuvs_cagra_metric=args.cuvs_cagra_metric,
        num_threads=args.num_threads,
        force_cpu_eval=args.force_cpu_eval
    )
    
    # Create evaluation config with defaults
    eval_config = EvalConfig()
    
    # Create training config
    cfg = TrainConfig(
        dataset=args.dataset,
        data_path=args.data_path,
        backend=args.backend,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        hidden=args.hidden,
        C_pack=args.C_pack,
        save_path=args.save_path,
        seed=args.seed,
        disable_ann_teacher=args.disable_ann_teacher,
        disable_exactK_teacher=args.disable_exactK_teacher,
        test_only=args.test_only,
        model_path=args.model_path,
        print_every=args.print_every,
        w_kld=args.w_kld,
        w_id=args.w_id,
        w_gap=args.w_gap,
        w_cell=args.w_cell,
        tr_margin=args.tr_margin,
        tr_weight=args.tr_weight,
        tr_enable=args.tr_enable,
        em_refresh_every=args.em_refresh_every,
        em_refresh_method=args.em_refresh_method,
        em_chunk=args.em_chunk,
        use_smart_identity=args.use_smart_identity,
        identity_mode=args.identity_mode,
        subspace_optimization=args.subspace_optimization,
        M_cells=args.M_cells,
        enable_quantity_head=args.enable_quantity_head,
        pack_chunk_size=args.pack_chunk_size,
        pack_x_batch_size=args.pack_x_batch_size,
        indices_dir=indices_dir,
        loss_analysis=args.loss_analysis,
        loss_analysis_file=args.loss_analysis_file,
        skip_eval=args.skip_eval,
        pca_path=args.pca_path,
        timing_log_file=args.timing_log_file,
        timing_log_dir=args.timing_log_dir,
    )
    return cfg, backend_config, eval_config


def main():
    cfg, backend_config, eval_config = parse_args()
    set_seed(cfg.seed)
    
    # Show GPU backend explanation
    if cfg.backend == "cuvs_cagra":
        print("\n🔧 cuVS CAGRA Backend Selected:")
        print(f"  Build algorithm: {backend_config.cuvs_cagra_build_algo}, metric: {backend_config.cuvs_cagra_metric}")
        print(f"  GPU-accelerated CAGRA with cuPy for memory lifetime management")
        print(f"  Budget-aware search: itopk_size and max_iterations scale with budget B")
        print(f"  Example: B=10 -> itopk=256, B=50 -> itopk=512, B=100 -> itopk=1024")
        print("")
    elif cfg.backend == "ivf":
        print("\n🔧 IVF Backend Selected:")
        print(f"  GPU-sharded IVF with nlist={backend_config.ivf_nlist} (auto-scaled for dataset size)")
        print(f"  Budget-aware search: nprobe parameter scales with budget B")
        print(f"  Example: B=10 -> nprobe=10, B=50 -> nprobe=50, B=100 -> nprobe=100")
        print("")
    
     # Show configuration for COCO dataset
    if cfg.dataset == "coco":
         dataset_loader = create_dataset_loader(cfg.dataset, cfg.data_path)
         dataset_info = dataset_loader.get_dataset_info()
         dataset_size = dataset_info.get('train_size', len(dataset_loader.get_train_data()[0]))
         
         print(f"🔧 COCO Dataset Configuration:")
         print(f"  Dataset size: {dataset_size:,} images")
         print(f"  Your parameters:")
         print(f"    batch_size: {cfg.batch_size}")
         print(f"    ivf_nlist: {backend_config.ivf_nlist}")
         print(f"    M_cells: {cfg.M_cells}")
         print(f"    backend: {cfg.backend}")
         print(f"    C_pack: {cfg.C_pack}")
         print(f"    epochs: {cfg.epochs}")
         print(f"    identity_mode: {cfg.identity_mode}")
    
    if cfg.test_only:
        # Load cached data and model; evaluate only
        print("🔍 Test-only mode: Loading dataset...")
        dataset_loader = create_dataset_loader(cfg.dataset, cfg.data_path)
        X_train, _, _ = dataset_loader.get_train_data()
        dim = dataset_loader.get_feature_dim()
        print(f"✅ Dataset loaded, feature dim: {dim}")
        
        if cfg.model_path and Path(cfg.model_path).exists():
            print(f"📦 Loading model from {cfg.model_path}...")
            ckpt = torch.load(cfg.model_path, map_location="cpu", weights_only=False)
            print("✅ Model loaded successfully")
            
            # Get saved parameters from checkpoint
            saved_M_cells = ckpt.get("M_cells", cfg.M_cells)
            saved_hidden = ckpt.get("hidden", cfg.hidden or dim)
            saved_alpha = ckpt.get("alpha", cfg.alpha)
            
            # Create models with saved parameters
            R = PCARSpace(d_keep=None, center_for_fit=True, device=cfg.device)
            model = ResidualProjector(dim=dim, hidden=saved_hidden, alpha=saved_alpha)
            coarse_head = CoarseCellHead(dim=dim, M=saved_M_cells)
            qty_head = QuantityHead(dim=dim) if cfg.enable_quantity_head else None
            
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
                
                if qty_head is not None and "qty_head_state_dict" in ckpt:
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
        else:
            # No model path provided - create default models
            R = PCARSpace(d_keep=None, center_for_fit=True, device=cfg.device)
            model = ResidualProjector(dim=dim, hidden=(cfg.hidden or dim), alpha=cfg.alpha)
            coarse_head = CoarseCellHead(dim=dim, M=cfg.M_cells)
            qty_head = QuantityHead(dim=dim) if cfg.enable_quantity_head else None
            # Fit R if not loaded
            R.fit(X_train)
        
        evaluate(cfg, backend_config, eval_config, model, R, (None, None), (X_train, None), dataset_loader, model_path=cfg.model_path)
    else:
        # Create dataset loader once and pass to both training and evaluation
        dataset_loader = create_dataset_loader(cfg.dataset, cfg.data_path)
        model, coarse_head, qty_head, R, indices, train_data = train_budget_aware(cfg, backend_config, dataset_loader)
        if not cfg.skip_eval:
            evaluate(cfg, backend_config, eval_config, model, R, indices, train_data, dataset_loader, model_path=cfg.model_path)


if __name__ == "__main__":
    main()


# Example launch commands for distributed training:
#
# Single GPU:
# python -m adapter.t2i_code.projector.main --dataset laion --data_path /ssd/hamed/ann/laion/precompute/laion_cache_up_to_0.pkl
#
# Multi-GPU (3 GPUs):
# torchrun --nproc_per_node=3 --master_port=29500 -m adapter.t2i_code.projector.main \
#     --dataset laion \
#     --data_path /ssd/hamed/ann/laion/precompute/laion_cache_up_to_0.pkl \
#     --backend hnsw \
#     --epochs 10 \
#     --batch_size 4096
#
# Multi-GPU with IVF:
# torchrun --nproc_per_node=3 --master_port=29500 -m adapter.t2i_code.projector.main \
#     --dataset laion \
#     --data_path /ssd/hamed/ann/laion/precompute/laion_cache_up_to_0.pkl \
#     --backend ivf \
#     --epochs 10 \
#     --batch_size 4096

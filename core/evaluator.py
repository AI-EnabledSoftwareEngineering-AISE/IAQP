import os
import pickle
import time
from typing import Tuple, Any

import numpy as np
import torch
import torch.nn as nn

# Import from our modules
from .utils import (
    bank_fingerprint,
    get_or_build_hnsw, get_or_build_ivf,
    project_np, recall_at_k, pair_hit_at_k, brute_force_topk_streaming,
    run_evaluation_section, build_ann_indices
)
from .dataset_loader import LaionDatasetLoader, T2IDatasetLoader
from .scripts.ivf_helper import ivf_search
from .scripts.hnsw_helper import hnsw_search
from .scripts.imi_helper import imi_search, get_or_build_imi_ivf_ip
from .scripts.nsg_helper import nsg_search, get_or_build_nsg_ip
from .scripts.test_cuvs_hnsw import cuvs_hnsw_search
# from .scripts.test_cuvs_cagara import cuvs_cagra_search  # Disabled for now
from .scripts.save_eval_results import save_evaluation_results

# Optional DiskANN imports
try:
    from .scripts.diskann_helper import diskann_search, get_or_build_diskann
except ImportError:
    diskann_search = None
    get_or_build_diskann = None

# Optional ANN libs (install as needed)
try:
    import faiss
except Exception as e:
    faiss = None
try:
    import hnswlib
except Exception as e:
    hnswlib = None
try:
    import diskannpy as dap
except Exception as e:
    dap = None

# --- drop-in evaluate -----------------------------------------------

def evaluate(cfg,
             backend_config,
             eval_config,
             model: nn.Module,
             R,
             indices: Tuple[Any, Any],
             data_train_tuple: Tuple[np.ndarray, np.ndarray],
             dataset_loader=None,
             model_path: str = None,
             save_results: bool = True,
             output_dir: str = "/home/hamed/projects/SPIN/adapter/t2i_code/outputs/eval_results/",
             custom_filename: str = None):

    # Load data using dataset loader
    if dataset_loader is None:
        # Use the same factory pattern as main.py
        from .main import create_dataset_loader
        dataset_loader = create_dataset_loader(cfg.dataset, cfg.data_path)
    
    split = eval_config.eval_split if eval_config.eval_split in ("val", "test", "inbound") else "val"

    # Always index on the TRAIN image bank
    X_bank, T_split, exact_ref, gt_img_idx = dataset_loader.get_split_data(split)
    bank_fp_eval = bank_fingerprint(X_bank, R)
    X_R_bank = R.transform(X_bank)
    
    # Ground-truth alignment (when loading split)
    N = X_R_bank.shape[0]
    assert exact_ref.shape[1] >= eval_config.eval_topk, f"exact_ref has {exact_ref.shape[1]} columns but need {eval_config.eval_topk}"
    assert np.max(exact_ref) < N, f"exact_ref contains ids ≥ bank size (max={np.max(exact_ref)}, bank_size={N})"
    
    # Auto-scale ivf_nlist if N≠3M (keeps you future-proof)
    from .utils import suggested_nlist
    nlist_eff = suggested_nlist(X_R_bank.shape[0], backend_config.ivf_nlist)
    if nlist_eff != backend_config.ivf_nlist:
        print(f"[eval ivf] overriding ivf_nlist {backend_config.ivf_nlist} → {nlist_eff} for N={X_R_bank.shape[0]:,}")
        backend_config.ivf_nlist = nlist_eff

    # Build ANN indices using utility function
    indices = build_ann_indices(backend_config, eval_config, X_R_bank, bank_fp_eval)

    # Project texts then rotate for ANN search
    T_proj   = project_np(model, T_split, device=cfg.device, batch=65536)
    T_proj_R = R.transform(T_proj)
    
    # Also evaluate baseline (without projection)
    T_baseline_R = R.transform(T_split)  # Rotate original text features
    
    # Unit-norm checks (once per eval/train)
    from .utils import _check_unit
    _check_unit("X_R_bank", X_R_bank)
    _check_unit("T_baseline_R", T_baseline_R)
    _check_unit("T_proj_R", T_proj_R)
    
    # Recalculate exact ground truth for projected features (like in train_coco_002.py)
    print("Computing exact ground truth for projected features...")
    exact_proj = brute_force_topk_streaming(T_proj_R, X_R_bank, k=eval_config.eval_topk)
    
    # Validate exact_proj ground truth
    assert exact_proj.shape[1] >= eval_config.eval_topk, f"exact_proj has {exact_proj.shape[1]} columns but need {eval_config.eval_topk}"
    assert np.max(exact_proj) < N, f"exact_proj contains ids ≥ bank size (max={np.max(exact_proj)}, bank_size={N})"
    assert exact_proj.shape[0] == T_proj_R.shape[0], f"exact_proj queries mismatch: {exact_proj.shape[0]} != {T_proj_R.shape[0]}"
    
    print("✓ Computed projected-space exact ground truth")

    # Timing-safe search + QPS
    import time
    def time_search_qps(index, backend, Q, K, budget, repeats=3):
        # warmup
        qw = Q[:min(2, len(Q))].copy()
        if backend == "hnsw":
            for _ in range(2):
                hnsw_search(index, qw, K, int(budget))
        elif backend == "ivf":
            for _ in range(2):
                ivf_search(index, qw, K, int(budget))
        elif backend == "imi":
            for _ in range(2):
                imi_search(index, qw, K, int(budget))
        elif backend == "diskann":
            for _ in range(2):
                diskann_search(index, qw, X_R_bank, K, int(budget), cfg.diskann_c_multiplier, cfg.num_threads)
        elif backend == "nsg":
            for _ in range(2):
                nsg_search(index, qw, K, int(budget))
        elif backend == "cuvs_hnsw":
            for _ in range(2):
                cuvs_hnsw_search(index, qw, K, int(budget))
        elif backend == "cuvs_cagra":
            # Disabled for now - cuvs_cagra_search not available
            raise NotImplementedError("cuVS CAGRA search disabled in evaluator for now")

        times = []
        for _ in range(repeats):
            if backend == "hnsw":
                _d, ids, dt = hnsw_search(index, Q, K, int(budget))
            elif backend == "ivf":
                _sims, ids, dt = ivf_search(index, Q, K, int(budget))
            elif backend == "imi":
                _sims, ids, dt = imi_search(index, Q, K, int(budget))
            elif backend == "diskann":
                _sims, ids, dt = diskann_search(index, Q, X_R_bank, K, int(budget), cfg.diskann_c_multiplier, cfg.num_threads)
            elif backend == "nsg":
                _sims, ids, dt = nsg_search(index, Q, K, int(budget))
            elif backend == "cuvs_hnsw":
                _d, ids, dt = cuvs_hnsw_search(index, Q, K, int(budget))
            elif backend == "cuvs_cagra":
                # Disabled for now - cuvs_cagra_search not available
                raise NotImplementedError("cuVS CAGRA search disabled in evaluator for now")
            else:
                raise ValueError(f"Unknown backend: {backend}")
            times.append(dt)
        m = sorted(times)[len(times)//2]
        qps = len(Q) / max(m, 1e-9)
        return ids, qps

    def eval_table(index, backend, Q_R, exact_idx_ref, budgets, topk, gt_img_idx_local,
                   eval_ph=True, eval_r5=False):
        tbl = {}
        for B in budgets:
            need_k = max(topk, 5 if eval_r5 else topk)
            pred_ids, qps = time_search_qps(index, backend, Q_R, need_k, B, repeats=3)

            exact_topk = exact_idx_ref[:, :topk]
            r = recall_at_k(pred_ids, exact_topk, topk)
            ph = pair_hit_at_k(pred_ids, gt_img_idx_local, topk) if eval_ph else 0.0
            r5 = recall_at_k(pred_ids[:, :5], exact_topk[:, :5], 5) if eval_r5 else 0.0
            tbl[B] = (r, ph, qps, r5)
        return tbl

    header_bits = [f"Recall@{eval_config.eval_topk}", "QPS"]
    if eval_config.eval_ph: header_bits.insert(1, f"PH@{eval_config.eval_topk}")
    if eval_config.eval_r5: header_bits.append("R@5")
    print(f"\n=== {' & '.join(header_bits)} vs Budget (split={split}) ===")

    results = {}

    # Run evaluations using utility function
    backend_configs = [
        ("IVF", indices.get('ivf')),
        ("HNSW", indices.get('hnsw')),
        ("IMI", indices.get('imi')),
        ("DiskANN", indices.get('diskann')),
        ("NSG", indices.get('nsg')),
        ("cuVS HNSW", indices.get('cuvs_hnsw')),
        ("cuVS CAGRA", indices.get('cuvs_cagra'))
    ]
    
    for backend_name, index in backend_configs:
        if index is not None:
            # Run baseline evaluation
            run_evaluation_section(index, backend_name, T_baseline_R, T_proj_R, 
                                 exact_ref, exact_proj, eval_config, gt_img_idx, 
                                 eval_table, results, is_baseline=True)
            
            # Run projected evaluation
            run_evaluation_section(index, backend_name, T_baseline_R, T_proj_R, 
                                 exact_ref, exact_proj, eval_config, gt_img_idx, 
                                 eval_table, results, is_baseline=False)

    

    # Save results if requested
    if save_results:
        try:
            saved_path = save_evaluation_results(
                results=results,
                cfg=cfg,
                model_path=model_path or "unknown_model.pt",
                dataset_loader=dataset_loader,
                output_dir=output_dir,
                custom_filename=custom_filename
            )
            print(f"📊 Results saved to: {saved_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save results: {e}")
    
    return results



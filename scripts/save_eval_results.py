# Save evaluation results to JSON file with comprehensive metadata
# Handles automatic path generation and result organization

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

def save_evaluation_results(
    results: Dict[str, Dict[int, float]],
    cfg,
    model_path: str,
    dataset_loader=None,
    output_dir: str = "/home/hamed/projects/SPIN/adapter/t2i_code/outputs/eval_results/",
    custom_filename: Optional[str] = None
) -> str:
    """
    Save evaluation results to JSON file with comprehensive metadata.
    
    Args:
        results: Dictionary of results from evaluator (e.g., {"IVF": {10: 0.5, 20: 0.6}})
        cfg: Configuration object with evaluation settings
        model_path: Path to the model being evaluated
        dataset_loader: Dataset loader instance (optional)
        output_dir: Base directory for saving results
        custom_filename: Custom filename (optional, auto-generated if None)
    
    Returns:
        str: Path to the saved JSON file
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if custom_filename is None:
        # Extract model name from path
        model_name = os.path.basename(model_path).replace('.pt', '')
        
        # Get backend and eval_backend information
        backend = getattr(cfg, 'backend', 'unknown')
        eval_backend = getattr(cfg, 'eval_backend', 'unknown')
        eval_split = getattr(cfg, 'eval_split', 'unknown')
        
        # Create descriptive filename
        custom_filename = f"{model_name}_{backend}_{eval_backend}_{eval_split}_eval_results.json"
    
    # Ensure .json extension
    if not custom_filename.endswith('.json'):
        custom_filename += '.json'
    
    output_path = os.path.join(output_dir, custom_filename)
    
    # Gather dataset information
    dataset_info = {}
    if dataset_loader is not None:
        try:
            # Get dataset statistics
            train_data = dataset_loader.get_split_data("train")
            val_data = dataset_loader.get_split_data("val") if hasattr(dataset_loader, 'get_split_data') else None
            
            dataset_info = {
                "train": {
                    "num_texts": train_data[1].shape[0] if len(train_data) > 1 else 0,
                    "feature_dim": train_data[1].shape[1] if len(train_data) > 1 and train_data[1].shape[0] > 0 else 0,
                    "num_images": train_data[0].shape[0] if len(train_data) > 0 else 0,
                    "knn_k": train_data[2].shape[1] if len(train_data) > 2 else 0
                },
                "val": {
                    "num_texts": val_data[1].shape[0] if val_data and len(val_data) > 1 else 0,
                    "feature_dim": val_data[1].shape[1] if val_data and len(val_data) > 1 and val_data[1].shape[0] > 0 else 0
                } if val_data else {},
                "dataset_type": getattr(cfg, 'dataset', 'unknown').upper(),
                "feature_type": "CLIP"  # Assuming CLIP features
            }
        except Exception as e:
            print(f"Warning: Could not gather dataset info: {e}")
            dataset_info = {"error": str(e)}
    
    # Create comprehensive results structure
    eval_results = {
        "metadata": {
            "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": model_path,
            "dataset": getattr(cfg, 'dataset', 'unknown'),
            "data_path": getattr(cfg, 'data_path', 'unknown'),
            "eval_split": getattr(cfg, 'eval_split', 'val'),
            "eval_topk": getattr(cfg, 'eval_topk', 10),
            "eval_ph": getattr(cfg, 'eval_ph', False),
            "eval_r5": getattr(cfg, 'eval_r5', False),
            "eval_backend": getattr(cfg, 'eval_backend', 'both'),
            "device": getattr(cfg, 'device', 'cuda'),
            "force_cpu_eval": getattr(cfg, 'force_cpu_eval', False),
            "num_threads": getattr(cfg, 'num_threads', 1),
            "training_backend": getattr(cfg, 'backend', 'unknown')
        },
        "dataset_info": dataset_info,
        "training_config": {
            "dataset": getattr(cfg, 'dataset', 'unknown'),
            "data_path": getattr(cfg, 'data_path', 'unknown'),
            "backend": getattr(cfg, 'backend', 'unknown'),
            "epochs": getattr(cfg, 'epochs', 0),
            "batch_size": getattr(cfg, 'batch_size', 0),
            "lr": getattr(cfg, 'lr', 0.0),
            "alpha": getattr(cfg, 'alpha', 0.0),
            "hidden": getattr(cfg, 'hidden', 0),
            "C_pack": getattr(cfg, 'C_pack', 'unknown'),
            "eval_split": getattr(cfg, 'eval_split', 'val'),
            "eval_topk": getattr(cfg, 'eval_topk', 10),
            "eval_ph": getattr(cfg, 'eval_ph', False),
            "eval_r5": getattr(cfg, 'eval_r5', False),
            "eval_backend": getattr(cfg, 'eval_backend', 'both'),
            "budgets": getattr(cfg, 'budgets', []),
            "hnsw_M": getattr(cfg, 'hnsw_M', 0),
            "hnsw_efC": getattr(cfg, 'hnsw_efC', 0),
            "ivf_nlist": getattr(cfg, 'ivf_nlist', 0),
            "nsg_R": getattr(cfg, 'nsg_R', 0),
            "nsg_L": getattr(cfg, 'nsg_L', 0),
            "diskann_graph_degree": getattr(cfg, 'diskann_graph_degree', 0),
            "diskann_build_complexity": getattr(cfg, 'diskann_build_complexity', 0),
            "diskann_c_multiplier": getattr(cfg, 'diskann_c_multiplier', 0),
            "diskann_max_budget": getattr(cfg, 'diskann_max_budget', 0),
            "disable_ann_teacher": getattr(cfg, 'disable_ann_teacher', False),
            "disable_exactK_teacher": getattr(cfg, 'disable_exactK_teacher', False),
            "w_kld": getattr(cfg, 'w_kld', 0.0),
            "w_id": getattr(cfg, 'w_id', 0.0),
            "w_gap": getattr(cfg, 'w_gap', 0.0),
            "w_cell": getattr(cfg, 'w_cell', 0.0),
            "use_smart_identity": getattr(cfg, 'use_smart_identity', False),
            "identity_mode": getattr(cfg, 'identity_mode', 'unknown'),
            "subspace_optimization": getattr(cfg, 'subspace_optimization', 'unknown'),
            "M_cells": getattr(cfg, 'M_cells', 0),
            "enable_quantity_head": getattr(cfg, 'enable_quantity_head', False),
            "em_recompute_every": getattr(cfg, 'em_recompute_every', 0),
            "pack_chunk_size": getattr(cfg, 'pack_chunk_size', 0),
            "pack_x_batch_size": getattr(cfg, 'pack_x_batch_size', 0),
            "force_cpu_eval": getattr(cfg, 'force_cpu_eval', False),
            "num_threads": getattr(cfg, 'num_threads', 1),
            "seed": getattr(cfg, 'seed', 42),
            "pca_path": getattr(cfg, 'pca_path', 'unknown'),
            "dataset_info": getattr(cfg, 'dataset_info', {}),
            "feature_dim": getattr(cfg, 'feature_dim', 0)
        },
        "evaluation_config": {
            "hnsw_M": getattr(cfg, 'hnsw_M', 0),
            "hnsw_efC": getattr(cfg, 'hnsw_efC', 0),
            "ivf_nlist": getattr(cfg, 'ivf_nlist', 0),
            "nsg_R": getattr(cfg, 'nsg_R', 0),
            "nsg_L": getattr(cfg, 'nsg_L', 0),
            "diskann_graph_degree": getattr(cfg, 'diskann_graph_degree', 0),
            "diskann_build_complexity": getattr(cfg, 'diskann_build_complexity', 0),
            "diskann_c_multiplier": getattr(cfg, 'diskann_c_multiplier', 0),
            "diskann_max_budget": getattr(cfg, 'diskann_max_budget', 0),
            "budgets": getattr(cfg, 'budgets', [])
        },
        "results": results,
        "summary": {
            "total_backends": len(results),
            "backends_evaluated": list(results.keys()),
            "budgets_tested": getattr(cfg, 'budgets', []),
            "training_backend": getattr(cfg, 'backend', 'unknown')
        }
    }
    
    # Save to JSON file
    try:
        with open(output_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"✅ Evaluation results saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Error saving evaluation results: {e}")
        raise

# -------------------------
# MAIN TEST (runnable)
# -------------------------
if __name__ == "__main__":
    # Test the save functionality
    class MockConfig:
        def __init__(self):
            self.dataset = "laion"
            self.data_path = "/test/path"
            self.backend = "hnsw"
            self.epochs = 5
            self.batch_size = 2048
            self.lr = 0.001
            self.alpha = 0.25
            self.hidden = 512
            self.eval_split = "val"
            self.eval_topk = 10
            self.eval_ph = False
            self.eval_r5 = False
            self.eval_backend = "both"
            self.device = "cuda"
            self.force_cpu_eval = False
            self.num_threads = 48
            self.budgets = [10, 20, 30, 40, 50]
            self.hnsw_M = 100
            self.hnsw_efC = 400
            self.ivf_nlist = 2048
    
    # Mock results
    mock_results = {
        "IVF_baseline": {10: 0.5, 20: 0.6, 30: 0.7},
        "IVF": {10: 0.6, 20: 0.7, 30: 0.8},
        "HNSW_baseline": {10: 0.55, 20: 0.65, 30: 0.75},
        "HNSW": {10: 0.65, 20: 0.75, 30: 0.85}
    }
    
    cfg = MockConfig()
    model_path = "/test/model.pt"
    
    # Test save
    output_path = save_evaluation_results(
        results=mock_results,
        cfg=cfg,
        model_path=model_path,
        output_dir="./test_outputs/"
    )
    print(f"Test results saved to: {output_path}")
    print(f"Filename format: {os.path.basename(output_path)}")

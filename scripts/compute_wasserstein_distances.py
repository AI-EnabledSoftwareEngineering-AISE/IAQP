#!/usr/bin/env python3
"""
Compute Gaussian 2-Wasserstein (Fréchet / FID-style) distances to quantify OOD gap
and projector effectiveness.

For each dataset, checks both IVF and CAGRA backends to find best generalization epoch,
then computes distances between:
- B1, B2: Two disjoint image bank subsets (in-distribution baseline)
- B1, Q_text: Rotated text queries (OOD gap)
- B1, Q_proj: Projected queries (reduced gap)
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add repo root to path
current_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(current_dir))
ARTIFACT_ROOT = current_dir / "notebooks" / "outputs"

from dataset_loader import LaionDatasetLoader, T2IDatasetLoader
from core.utils import ResidualProjector, PCARSpace, l2n_np
import torch
import torch.nn.functional as F

# Try to import scipy for Gaussian 2-Wasserstein distance
try:
    from scipy import linalg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")

# Dataset configuration
DATASET_CONFIG = {
    't2i': {
        'loader': T2IDatasetLoader,
        'size': '10m',
        'data_paths': {
            'ivf': '/ssd/hamed/ann/t2i-10M/precompute/t2i_cache_up_to_0.pkl',
            'cagra': '/ssd/hamed/ann/t2i-10M/precompute/t2i_cache_up_to_0.pkl',
        },
        'checkpoint_patterns': {
            'ivf': str(ARTIFACT_ROOT / 'checkpoints' / 't2i-10m_ivf_up_to_e3'),
            'cagra': str(ARTIFACT_ROOT / 'checkpoints' / 't2i-10m_cuvs_cagra_up_to_e3'),
        }
    },
    'laion': {
        'loader': LaionDatasetLoader,
        'size': '10m',
        'data_paths': {
            'ivf': '/ssd/hamed/ann/laion/precompute/laion_cache_up_to_10.pkl',
            'cagra': '/ssd/hamed/ann/laion/precompute/laion_cache_up_to_10.pkl',
        },
        'checkpoint_patterns': {
            'ivf': str(ARTIFACT_ROOT / 'checkpoints' / 'laion-10m_ivf_up_to_e3'),
            'cagra': str(ARTIFACT_ROOT / 'checkpoints' / 'laion-10m_cuvs_cagra_up_to_e3'),
        }
    },
    'datacomp': {
        'loader': LaionDatasetLoader,
        'size': '8.2m',
        'data_paths': {
            'ivf': '/ssd/hamed/ann/datacomp_small/precompute/datacomp_cache_up_to_8200000.pkl',
            'cagra': '/ssd/hamed/ann/datacomp_small/precompute/datacomp_cache_up_to_8200000.pkl',
        },
        'checkpoint_patterns': {
            'ivf': str(ARTIFACT_ROOT / 'checkpoints' / 'datacomp-8.2m_ivf_up_to_e3'),
            'cagra': str(ARTIFACT_ROOT / 'checkpoints' / 'datacomp-8.2m_cuvs_cagra_up_to_e3'),
        }
    }
}

RESULTS_DIR = ARTIFACT_ROOT / "eval_results"
SAMPLE_SIZE = 100000  # Sample size for B1, B2, Q_text


def load_best_epoch_from_results(dataset: str, backend: str, results_dir: Path) -> Optional[int]:
    """Load best generalization epoch from evaluation results JSON.
    
    Args:
        dataset: Dataset name ('t2i', 'laion', 'datacomp')
        backend: Backend name ('ivf', 'cagra')
        results_dir: Directory containing results JSON files
        
    Returns:
        Epoch number (1, 2, or 3) or None if not found
    """
    # Map backend names to file patterns
    backend_map = {
        'ivf': 'ivf',
        'cagra': 'cuvs_cagra',
    }
    
    # Handle t2i special case (uses 'cagra' not 'cuvs_cagra')
    if dataset == 't2i' and backend == 'cagra':
        backend_name = 'cuvs_cagra'
    else:
        backend_name = backend_map.get(backend, backend)
    
    # Construct filename
    size = DATASET_CONFIG[dataset]['size']
    filename = f"{dataset}-{size}_{backend_name}_up_to_e3_comprehensive_results.json"
    filepath = results_dir / filename
    
    if not filepath.exists():
        print(f"Warning: Results file not found: {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        gen_entry = data.get('checkpoint_selection', {}).get('selected_checkpoints', {}).get('generalization')
        if gen_entry:
            # Handle per-k dict or single string
            if isinstance(gen_entry, dict):
                first_k = sorted(gen_entry.keys())[0]
                gen_epoch = gen_entry[first_k]
            else:
                gen_epoch = gen_entry
            if isinstance(gen_epoch, str):
                epoch_num = int(gen_epoch.replace('ep', ''))
                return epoch_num
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return None


def load_ckpt(ckpt_path: str, device: str = "cuda", dim: Optional[int] = None):
    """Load checkpoint with model and PCA rotation.
    
    Args:
        ckpt_path: Path to checkpoint file
        device: Device to load model on
        dim: Feature dimension (auto-detected if None)
        
    Returns:
        (model, R) tuple where R is PCARSpace
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden = ck.get("hidden", dim or 512)
    alpha = ck.get("alpha", 0.25)
    
    # Auto-detect dimension
    if dim is None:
        if "dim" in ck:
            model_dim = ck["dim"]
        else:
            msd = ck["model_state_dict"]
            if any(k.startswith("module.") for k in msd):
                msd_clean = {k.replace("module.", ""): v for k, v in msd.items()}
            else:
                msd_clean = msd
            if "g.0.weight" in msd_clean:
                model_dim = msd_clean["g.0.weight"].shape[1]
            else:
                model_dim = 512
    else:
        model_dim = dim
    
    model = ResidualProjector(dim=model_dim, hidden=hidden, alpha=alpha)
    msd = ck["model_state_dict"]
    if any(k.startswith("module.") for k in msd):
        msd = {k.replace("module.", ""): v for k, v in msd.items()}
    model.load_state_dict(msd)
    model.to(device).eval()
    
    R = PCARSpace(d_keep=None, center_for_fit=True, device=device)
    R.W = ck["W"].astype(np.float32)
    R.mu = ck.get("mu", None)
    if R.mu is not None:
        R.mu = R.mu.astype(np.float32)
    
    return model, R


@torch.no_grad()
def project_np(model, T_np: np.ndarray, device: str = "cuda", batch: int = 65536) -> np.ndarray:
    """Project text embeddings through model.
    
    Args:
        model: Trained ResidualProjector
        T_np: Text embeddings array [N, D]
        device: Device to run model on
        batch: Batch size for processing
        
    Returns:
        Projected embeddings [N, D]
    """
    dev = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    out = np.empty_like(T_np, dtype=np.float32)
    
    for s in range(0, T_np.shape[0], batch):
        e = min(T_np.shape[0], s + batch)
        xb = torch.from_numpy(T_np[s:e]).to(dev).float()
        xb = F.normalize(xb, dim=1)
        out[s:e] = model(xb).cpu().numpy().astype(np.float32)
    
    return out


def gaussian_w2_distance(X1: np.ndarray, X2: np.ndarray, eps: float = 1e-6) -> float:
    """
    Gaussian 2-Wasserstein distance (Fréchet / FID-style) between Gaussians fit
    to samples X1 and X2. Returns W2 (not squared).
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for Wasserstein distance computation")

    X1 = np.asarray(X1, dtype=np.float64)
    X2 = np.asarray(X2, dtype=np.float64)

    mu1 = X1.mean(axis=0)
    mu2 = X2.mean(axis=0)

    C1 = np.cov(X1, rowvar=False)
    C2 = np.cov(X2, rowvar=False)

    D = C1.shape[0]
    C1 = C1 + eps * np.eye(D)
    C2 = C2 + eps * np.eye(D)

    sqrtC1 = linalg.sqrtm(C1)
    M = sqrtC1 @ C2 @ sqrtC1
    sqrtM = linalg.sqrtm(M)

    if np.iscomplexobj(sqrtM):
        sqrtM = sqrtM.real

    w2_sq = np.sum((mu1 - mu2) ** 2) + np.trace(C1 + C2 - 2.0 * sqrtM)
    w2_sq = max(float(w2_sq), 0.0)
    if not np.isfinite(w2_sq):
        return float("nan")
    return float(np.sqrt(w2_sq))


def compute_wasserstein_distances_for_dataset(
    dataset: str,
    backend: str,
    epoch: int,
    sample_size: int = SAMPLE_SIZE,
    device: str = "cuda"
) -> Dict[str, float]:
    """Compute Wasserstein distances for a dataset.
    
    Args:
        dataset: Dataset name ('t2i', 'laion', 'datacomp')
        backend: Backend name ('ivf', 'cagra')
        epoch: Epoch number (1, 2, or 3)
        sample_size: Number of samples for B1, B2, Q_text
        device: Device for computation
        
    Returns:
        Dictionary with distances and ratios
    """
    config = DATASET_CONFIG[dataset]
    DatasetLoader = config['loader']
    data_path = config['data_paths'][backend]
    checkpoint_pattern = config['checkpoint_patterns'][backend]
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset.upper()}, Backend: {backend.upper()}, Epoch: {epoch}")
    print(f"{'='*60}")
    
    # Load dataset
    print(f"Loading dataset from {data_path}...")
    ds = DatasetLoader(data_path)
    X_train, T_train, _ = ds.get_train_data()
    split_data = ds.get_split_data("val")
    if len(split_data) == 4:
        X_val, T_val, _, _ = split_data
    else:
        X_val, T_val, _ = split_data[:3]
    
    print(f"Train: X={X_train.shape}, T={T_train.shape}")
    print(f"Val: T={T_val.shape}")
    
    # Load checkpoint
    ckpt_base = Path(current_dir) / checkpoint_pattern
    ckpt_path = ckpt_base / f"epoch_{epoch}.pt"
    
    if not ckpt_path.exists():
        # List available files for debugging
        if ckpt_base.exists():
            available = list(ckpt_base.glob("*.pt"))
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                f"Available files: {[f.name for f in available]}"
            )
        else:
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_base}")
    
    print(f"Loading checkpoint from {ckpt_path}...")
    model, R = load_ckpt(str(ckpt_path), device=device, dim=None)
    # Get model dimension from first layer weight shape (ResidualProjector doesn't store dim as attribute)
    model_dim = model.g[0].weight.shape[1] if hasattr(model, 'g') and len(model.g) > 0 else R.W.shape[0]
    print(f"Model dim: {model_dim}, PCA W shape: {R.W.shape}")
    
    # Check if data is already in PCA space
    # If PCA W is (output_dim, input_dim) and data dim matches output_dim, data is already transformed
    pca_input_dim = R.W.shape[1]  # Original dimension before PCA
    pca_output_dim = R.W.shape[0]  # Dimension after PCA
    data_dim = X_train.shape[1]
    
    print(f"Data dim: {data_dim}, PCA input dim: {pca_input_dim}, PCA output dim: {pca_output_dim}")
    
    # Step 1: Apply PCA rotation FIRST to all raw data (if not already applied)
    if data_dim == pca_output_dim:
        # Data is already in PCA space (cache was pre-computed with PCA)
        print("\nData is already in PCA space, skipping PCA transform...")
        X_train_R = X_train.astype(np.float32)
        T_train_R = T_train.astype(np.float32)
        T_val_R = T_val.astype(np.float32)
    elif data_dim == pca_input_dim:
        # Data is in original space, need to apply PCA
        print("\nApplying PCA rotation to all data...")
        X_train_R = R.transform(X_train.astype(np.float32))
        T_train_R = R.transform(T_train.astype(np.float32))
        T_val_R = R.transform(T_val.astype(np.float32))
    else:
        raise ValueError(
            f"Dimension mismatch: data has {data_dim}D, but PCA expects "
            f"{pca_input_dim}D input or {pca_output_dim}D output. "
            f"Check if data is already PCA-transformed or needs transformation."
        )
    
    # L2-normalize after rotation
    X_train_R = l2n_np(X_train_R)
    T_train_R = l2n_np(T_train_R)
    T_val_R = l2n_np(T_val_R)
    
    print(f"After PCA: X_train_R={X_train_R.shape}, T_val_R={T_val_R.shape}")
    
    # Step 2: Sample B1, B2 from rotated image bank
    print(f"\nSampling {sample_size} images for B1 and B2...")
    N_train = X_train_R.shape[0]
    if N_train < 2 * sample_size:
        sample_size = N_train // 2
        print(f"Reducing sample size to {sample_size} (insufficient data)")
    
    rng = np.random.RandomState(42)
    all_indices = rng.permutation(N_train)
    B1_indices = all_indices[:sample_size]
    B2_indices = all_indices[sample_size:2*sample_size]
    
    B1 = X_train_R[B1_indices]
    B2 = X_train_R[B2_indices]
    
    # Step 3: Sample Q_text from rotated text queries
    print(f"Sampling {sample_size} text queries...")
    N_val = T_val_R.shape[0]
    q_size = min(sample_size, N_val)
    q_indices = rng.choice(N_val, size=q_size, replace=False)
    Q_text = T_val_R[q_indices]
    
    # Step 4: Project Q_text through model (input/output both in rotated space)
    print("Projecting text queries through model...")
    Q_proj = project_np(model, Q_text, device=device, batch=65536)
    Q_proj = l2n_np(Q_proj)  # Normalize projected queries
    
    # Step 5: Compute Gaussian 2-Wasserstein (Fréchet) distances in rotated space
    print("\nComputing Gaussian 2-Wasserstein (Fréchet / FID-style) distances...")
    gw2_b1_b2 = gaussian_w2_distance(B1, B2)
    print(f"  GW2(B1, B2) = {gw2_b1_b2:.6f}")
    
    gw2_b1_qtext = gaussian_w2_distance(B1, Q_text)
    print(f"  GW2(B1, Q_text) = {gw2_b1_qtext:.6f}")
    
    gw2_b1_qproj = gaussian_w2_distance(B1, Q_proj)
    print(f"  GW2(B1, Q_proj) = {gw2_b1_qproj:.6f}")
    
    # Compute ratios
    ratio_text = gw2_b1_qtext / gw2_b1_b2 if gw2_b1_b2 > 0 else float('inf')
    ratio_proj = gw2_b1_qproj / gw2_b1_b2 if gw2_b1_b2 > 0 else float('inf')
    
    print(f"\nRatios:")
    print(f"  GW2(B1, Q_text) / GW2(B1, B2) = {ratio_text:.3f}")
    print(f"  GW2(B1, Q_proj) / GW2(B1, B2) = {ratio_proj:.3f}")
    
    return {
        'gw2_b1_b2': gw2_b1_b2,
        'gw2_b1_qtext': gw2_b1_qtext,
        'gw2_b1_qproj': gw2_b1_qproj,
        'ratio_text': ratio_text,
        'ratio_proj': ratio_proj,
        'sample_size': sample_size,
        'epoch': epoch,
        'backend': backend
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Gaussian 2-Wasserstein (Fréchet / FID-style) distances for OOD analysis")
    parser.add_argument("--datasets", type=str, nargs="+", default=["t2i", "laion", "datacomp"],
                       help="Datasets to process")
    parser.add_argument("--backends", type=str, nargs="+", default=["ivf", "cagra"],
                       help="Backends to check for best epoch")
    parser.add_argument("--sample_size", type=int, default=SAMPLE_SIZE,
                       help="Sample size for B1, B2, Q_text")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for computation")
    parser.add_argument("--output", type=str, default=str(ARTIFACT_ROOT / "wasserstein" / "wasserstein_distances.json"),
                       help="Output JSON file path")
    parser.add_argument("--epoch-override", type=str, nargs="+", default=[],
                       help="Override epoch for specific dataset:backend combinations. "
                            "Format: dataset:backend:epoch (e.g., 'datacomp:ivf:2' or 'datacomp:ivf:2 datacomp:ivf:3')")
    args = parser.parse_args()
    
    # Parse epoch overrides
    epoch_overrides = {}
    for override in args.epoch_override:
        try:
            parts = override.split(':')
            if len(parts) != 3:
                raise ValueError(f"Invalid format: {override}. Expected 'dataset:backend:epoch'")
            dataset, backend, epoch = parts
            epoch = int(epoch)
            key = (dataset.lower(), backend.lower())
            if key not in epoch_overrides:
                epoch_overrides[key] = []
            epoch_overrides[key].append(epoch)
        except ValueError as e:
            print(f"Warning: Invalid epoch override '{override}': {e}. Skipping.")
            continue
    
    if not SCIPY_AVAILABLE:
        print("ERROR: scipy is required. Install with: pip install scipy")
        sys.exit(1)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    all_results = {}
    
    for dataset in args.datasets:
        print(f"\n{'#'*60}")
        print(f"Processing dataset: {dataset.upper()}")
        print(f"{'#'*60}")
        
        # Compute distances for EACH backend separately
        dataset_results = {}
        for backend in args.backends:
            # Check for epoch override
            override_key = (dataset.lower(), backend.lower())
            if override_key in epoch_overrides:
                # Use override epochs (can be multiple)
                epochs_to_test = epoch_overrides[override_key]
                print(f"\n  Using override epochs for {backend.upper()}: {epochs_to_test}")
            else:
                # Default: test all three epochs
                epochs_to_test = [1, 2, 3]
            
            # Compute for each epoch
            for epoch in epochs_to_test:
                epoch_label = f"{backend}_{epoch}" if len(epochs_to_test) > 1 else backend
                print(f"\n  Computing distances for {backend.upper()} (epoch {epoch})...")
                
                try:
                    results = compute_wasserstein_distances_for_dataset(
                        dataset=dataset,
                        backend=backend,
                        epoch=epoch,
                        sample_size=args.sample_size,
                        device=args.device
                    )
                    dataset_results[epoch_label] = results
                except Exception as e:
                    print(f"\n{'='*60}")
                    print(f"ERROR processing {dataset} with {backend.upper()} epoch {epoch}: {e}")
                    print(f"{'='*60}")
                    import traceback
                    traceback.print_exc()
                    print(f"\nStopping execution due to error.")
                    sys.exit(1)
        
        if dataset_results:
            all_results[dataset] = dataset_results
    
    # Save results (support absolute or repo-relative paths)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(current_dir) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary:")
    for dataset, dataset_res in all_results.items():
        print(f"\n{dataset.upper()}:")
        for backend_key, res in dataset_res.items():
            # Parse backend key (might be "backend" or "backend_epoch")
            if '_' in backend_key:
                backend_name, epoch_num = backend_key.rsplit('_', 1)
                print(f"  [{backend_name.upper()}] Epoch {epoch_num}:")
            else:
                print(f"  [{backend_key.upper()}] Epoch {res['epoch']}:")
            print(f"    GW2(B1, B2) = {res['gw2_b1_b2']:.6f}")
            print(f"    GW2(B1, Q_text) = {res['gw2_b1_qtext']:.6f} (ratio: {res['ratio_text']:.3f})")
            print(f"    GW2(B1, Q_proj) = {res['gw2_b1_qproj']:.6f} (ratio: {res['ratio_proj']:.3f})")
            improvement = ((res['gw2_b1_qtext'] - res['gw2_b1_qproj']) / res['gw2_b1_qtext']) * 100
            print(f"    Improvement: {improvement:.2f}% reduction")


if __name__ == "__main__":
    main()

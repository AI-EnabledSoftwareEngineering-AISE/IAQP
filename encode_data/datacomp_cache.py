# scripts/make_datacomp_cache.py
# Usage: python scripts/make_datacomp_cache.py --up_to 3000000 --k_exact 100
import os, sys, math, pickle, argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import brute_force_topk_streaming, PCARSpace

# ---------------------------
# Defaults
# ---------------------------
DEF_ROOT = "/ssd/hamed/ann/datacomp_small"
DEF_FEATURES_PKL = "datacomp_features.pkl"  # Input pickle file
DEF_OUT  = "datacomp_cache_up_to_{up_to}.pkl"
DEF_KEX  = 100
DEF_UP_TO = 3000000  # 3M default
DEF_N_QUERIES = 10000  # 10k queries

# ---------------------------
# Helper functions
# ---------------------------
def l2n(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2 normalize along axis=1"""
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return (x / n).astype(np.float32)

# ---------------------------
def load_datacomp_embeddings(root: Path, features_pkl: Path, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Load (and optionally normalize) DataComp CLIP embeddings from a pickle.

    Returns:
        Tuple (X_all, T_all, uids, texts, filtered_by_image_exists)
    """
    if not features_pkl.exists():
        raise RuntimeError(f"Features pickle file not found: {features_pkl}")
    
    print(f"Loading features from: {features_pkl}")
    with open(features_pkl, 'rb') as f:
        data = pickle.load(f)
    
    uids = data['uid']
    texts = data['text']
    b32_img = data['b32_img'].astype(np.float32)  # [N, 512]
    b32_txt = data['b32_txt'].astype(np.float32)  # [N, 512]
    
    # Check if image_exists field exists and filter to only samples with existing images
    filtered_by_image_exists = False
    if 'image_exists' in data:
        image_exists = data['image_exists']
        n_exist = np.sum(image_exists)
        n_total = len(image_exists)
        print(f"Found image_exists field: {n_exist:,}/{n_total:,} images exist ({n_exist/n_total*100:.2f}%)")
        
        # Filter to only samples where images exist
        valid_mask = image_exists.astype(bool)  # Ensure boolean mask
        print(f"Filtering to samples with existing images...")
        print(f"  Before filtering: {len(uids):,} samples")
        
        # Apply filter to all arrays
        uids = uids[valid_mask]
        texts = texts[valid_mask]
        b32_img = b32_img[valid_mask]
        b32_txt = b32_txt[valid_mask]
        
        filtered_by_image_exists = True
        n_after = len(uids)
        print(f"  After filtering: {n_after:,} samples")
        
        # Safety check: ensure we have samples after filtering
        if n_after == 0:
            raise RuntimeError("No samples with existing images found! Cannot build cache.")
    else:
        print("⚠ Warning: image_exists field not found. Using all samples.")
        print("  Run datacomp_reader.py with --add_image_check to filter by image existence.")
        print("  This may include samples without corresponding image files.")
    
    # Verify alignment
    n_samples = len(uids)
    assert len(texts) == n_samples, "uid-text length mismatch"
    assert b32_img.shape[0] == n_samples, "uid-b32_img length mismatch"
    assert b32_txt.shape[0] == n_samples, "uid-b32_txt length mismatch"
    assert b32_img.shape[1] == 512, f"Expected 512 dims for B/32, got {b32_img.shape[1]}"
    assert b32_txt.shape[1] == 512, f"Expected 512 dims for B/32, got {b32_txt.shape[1]}"
    
    X_all = b32_img
    T_all = b32_txt
    
    if normalize:
        X_all = l2n(X_all)
        T_all = l2n(T_all)

    return X_all, T_all, uids, texts, filtered_by_image_exists


# ---------------------------
# Cache builder
# ---------------------------
def build_cache(root: Path, features_pkl: Path, out_path: Path, k_exact: int, up_to: int, n_queries: int,
                query_batch: int = 4096, index_batch: int = 25000, use_fp16: bool = False,
                pca_dim: int = None, pca_device: str = "cuda", pca_center: bool = True,
                shared_pca: Optional[PCARSpace] = None) -> None:
    """Build cache from DataComp features pickle file."""
    
    X_all, T_all, uids, texts, filtered_by_image_exists = load_datacomp_embeddings(root, features_pkl, normalize=True)
    print(f"Total available: {X_all.shape[0]:,} samples, dim={X_all.shape[1]}")
    
    # Check if we have enough data
    if X_all.shape[0] < up_to + n_queries:
        print(f"Warning: Only {X_all.shape[0]:,} samples available, but requested {up_to:,} training + {n_queries:,} queries")
        print(f"Using all available data...")
        up_to = max(0, X_all.shape[0] - n_queries)
    
    # Randomly shuffle and split
    print(f"Randomly selecting {up_to:,} training samples and {n_queries:,} query samples...")
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    indices = rng.permutation(X_all.shape[0])
    
    # Training indices: first up_to samples
    train_indices = indices[:up_to]
    # Query indices: next n_queries samples  
    query_indices = indices[up_to:up_to + n_queries]
    
    print(f"Selected {len(train_indices):,} training samples")
    print(f"Selected {len(query_indices):,} query samples")
    
    # Extract training data
    X_train = X_all[train_indices]
    T_train = T_all[train_indices]
    uids_train = uids[train_indices]
    texts_train = texts[train_indices]
    
    # Extract query data (text only)
    Q_query = T_all[query_indices]
    uids_query = uids[query_indices]
    texts_query = texts[query_indices]
    
    D = X_train.shape[1]
    print(f"Training set: {X_train.shape[0]:,} items, dim={D}")
    print(f"Query set: {Q_query.shape[0]:,} items, dim={D}")
    
    # L2-normalize FIRST (before PCA)
    # Data already normalized in loader; ensure queries normalized
    # (X_train/T_train/Q_query are subsets of normalized arrays)
    
    R = None
    if shared_pca is not None:
        R = shared_pca
        print(f"\n✅ Using shared PCA: W={R.W.shape}, center={R.center_for_fit}")
    elif pca_dim is not None and pca_dim > 0:
        print(f"\n🔧 Computing PCA on normalized training images (target dim: {pca_dim})...")
        print(f"   Device: {pca_device}, Center: {pca_center}")
        
        R = PCARSpace(d_keep=pca_dim, center_for_fit=pca_center, device=pca_device, chunk=262144)
        R.fit(X_train)
        
        print(f"✅ PCA fitted: W.shape={R.W.shape}, mu={'None' if R.mu is None else R.mu.shape}")
        print(f"   Original dim: {D} → Reduced dim: {pca_dim}")
    else:
        print("⚠️  No PCA reduction requested (pca_dim=None). Using original features.")
        
    if R is not None:
        print("🔄 Applying PCA transform to all data...")
        X_train = R.transform(X_train)  # [N_train, pca_dim]
        T_train = R.transform(T_train)  # [N_train, pca_dim]
        Q_query = R.transform(Q_query)  # [N_query, pca_dim]
        
        print(f"✅ PCA transform applied")
        print(f"   X_train: {X_train.shape}, T_train: {T_train.shape}, Q_query: {Q_query.shape}")
        
        D = X_train.shape[1]
    
    print("\nComputing exact@K for train and val sets (on PCA-reduced features if applicable):")
    
    # Compute exact@K for training set (self-similarity)
    print("  • train: queries = train texts, index = train images")
    knn_train = brute_force_topk_streaming(T_train, X_train, k=k_exact, q_batch=query_batch, x_batch=index_batch, use_fp16=use_fp16, show_progress=True)
    
    # Compute exact@K for validation set (query against train images)
    print("  • val: queries = sampled query set, index = train images")
    knn_val = brute_force_topk_streaming(Q_query, X_train, k=k_exact, q_batch=query_batch, x_batch=index_batch, use_fp16=use_fp16, show_progress=True)

    # assemble cache (same structure as LAION, with additional uid/text fields)
    cache_data = {
        "train": {
            "image_features": X_train,           # Training images (indexed)
            "text_features":  T_train,           # Training texts
            "knn_indices":    knn_train,         # exact@K vs TRAIN images
            # Additional fields (backward compatible - LaionDatasetLoader ignores these)
            "uid": uids_train,                   # UIDs for training samples
            "text": texts_train,                 # Text captions for training samples
        },
        "val": {
            "text_features":  Q_query,           # Query set
            "knn_indices":    knn_val,           # exact@K vs TRAIN images
            "index_ref":      "train",           # explicit reference
            # Additional fields (backward compatible)
            "uid": uids_query,                   # UIDs for query samples
            "text": texts_query,                 # Text captions for query samples
        },
        # Metadata breadcrumb for training code adaptation
        "_meta": {
            "index_bank_split": "train",
            "k_exact": k_exact,
            "dim": int(X_train.shape[1]),  # Final dimension (after PCA if applied)
            "normalized": True,
            "up_to": up_to,
            "n_train_items": int(X_train.shape[0]),
            "n_val_items": int(Q_query.shape[0]),
            "clip_model": "B/32",
            "dataset": "datacomp_small",
            "filtered_by_image_exists": filtered_by_image_exists,
            "pca_applied": R is not None,
            "pca_dim": pca_dim if R is not None else None,
            "pca_center": pca_center if R is not None else None,
        },
    }
    
    # Add PCA parameters if PCA was applied
    if R is not None:
        cache_data["pca"] = {
            "W": R.W.astype(np.float32),
            "mu": R.mu.astype(np.float32) if R.mu is not None else None,
            "center_for_fit": bool(R.center_for_fit),
    }

    with open(out_path, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✓ Wrote {out_path}")
    print("Summary:")
    print("  train:",
          cache_data["train"]["image_features"].shape,
          cache_data["train"]["text_features"].shape,
          cache_data["train"]["knn_indices"].shape)
    print("  val:  texts", cache_data["val"]["text_features"].shape,
          "knn", cache_data["val"]["knn_indices"].shape,
          " (index_ref=train)")
    print("  Additional fields: uid, text (backward compatible)")
    print("  metadata:", cache_data["_meta"])

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build DataComp cache from features pickle file.")
    p.add_argument("--root", type=str, default=DEF_ROOT, help="Dataset root directory.")
    p.add_argument("--features_pkl", type=str, default=None, help="Input features pickle file (default: {root}/datacomp_features.pkl).")
    p.add_argument("--out",  type=str, default=None,  help="Output pickle filename (under precompute subdir). If not provided, uses default naming with up_to.")
    p.add_argument("--up_to", type=int, default=DEF_UP_TO, help="Upper bound of training data (e.g., 3000000 for 3M).")
    p.add_argument("--n_queries", type=int, default=DEF_N_QUERIES, help="Number of query samples for validation set.")
    p.add_argument("--k_exact",   type=int, default=DEF_KEX, help="Exact@K to store in cache.")
    p.add_argument("--query_batch", type=int, default=16384, help="Batch size for query processing.")
    p.add_argument("--index_batch", type=int, default=50000, help="Batch size for index processing.")
    p.add_argument("--use_fp16", action="store_true", help="Do matmul in FP16 (faster but not perfectly exact). Default: FP32 exact.")
    p.add_argument("--pca_dim", type=int, default=None, help="Target PCA dimension (e.g., 200). If None, no PCA reduction is applied.")
    p.add_argument("--pca_device", type=str, default="cuda", help="Device for PCA computation (cuda or cpu).")
    p.add_argument("--pca_center", action="store_true", default=True, help="Center data before PCA (subtract mean). Default: True.")
    p.add_argument("--no_pca_center", dest="pca_center", action="store_false", help="Disable centering for PCA.")
    p.add_argument("--force", action="store_true", help="Overwrite existing output file if it exists.")
    return p.parse_args()

def main():
    args = parse_args()
    
    root = Path(args.root)
    
    # Determine input features pickle path
    if args.features_pkl is None:
        features_pkl = root / DEF_FEATURES_PKL
    else:
        features_pkl = Path(args.features_pkl)
        if not features_pkl.is_absolute():
            features_pkl = root / args.features_pkl
    
    # Create precompute subdirectory for output
    precompute_dir = root / "precompute"
    precompute_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename if not provided
    if args.out is None:
        out_filename = DEF_OUT.format(up_to=args.up_to)
    else:
        out_filename = args.out
    
    out_path = precompute_dir / out_filename

    print(f"Root: {root}")
    print(f"Input features: {features_pkl}")
    print(f"Up to: {args.up_to:,} training samples")
    print(f"Queries: {args.n_queries:,}")
    print(f"Output: {out_path}")
    
    # Check if output file exists and handle accordingly
    if out_path.exists() and not args.force:
        print(f"Error: Output file {out_path} already exists.")
        print("Use --force to overwrite or choose a different output name with --out.")
        return 1
    
    print("Building cache...")
    build_cache(root, features_pkl, out_path, k_exact=args.k_exact, 
                up_to=args.up_to, n_queries=args.n_queries,
                query_batch=args.query_batch, 
                index_batch=args.index_batch, use_fp16=args.use_fp16,
                pca_dim=args.pca_dim, pca_device=args.pca_device, pca_center=args.pca_center)
    
    return 0

if __name__ == "__main__":
    main()


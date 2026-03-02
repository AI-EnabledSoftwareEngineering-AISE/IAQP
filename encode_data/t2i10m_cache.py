#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
t2i10m_cache.py
----------------
Build cache from T2I-10M dataset for ANN evaluation.

Downloads the T2I-10M dataset if missing and builds a cache with exact@K
neighbors for both training and validation sets. Supports partial dataset
loading via --up_to parameter (e.g., up_to=1 means 1M out of 10M).

Usage:
    python t2i10m_cache.py --up_to 1 --k_exact 100
    python t2i10m_cache.py --up_to 0 --k_exact 100 --use_fp16
"""

import os
import sys
import pickle
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
from tqdm import tqdm

# Add path for importing from laion10m_cache.py
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# ---------------------------
# Defaults
# ---------------------------
DEF_ROOT = "/ssd/hamed/ann/t2i-10M"
DEF_OUT = "t2i_cache_up_to_{up_to}.pkl"
DEF_KEX = 100
DEF_UP_TO = 0

# T2I-10M dataset configuration
T2I_CONFIG = {
    "name": "t2i-10M",
    "base_url": "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/",
    "gt_url": "https://zenodo.org/records/11090378/files/t2i.gt.10k.ibin",
    "files": {
        "base.10M.fbin": {
            "url": "base.10M.fbin",
            "size": 200 * 4 * 10000000 + 8 - 1,  # 200 dim * 4 bytes * 10M points + header
            "description": "Base vectors (10M points, 200 dim)"
        },
        "query.train.10M.fbin": {
            "url": "query.learn.50M.fbin", 
            "size": 200 * 4 * 10000000 + 8 - 1,  # First 10M from 50M file
            "description": "Training queries (10M points, 200 dim)"
        },
        "query.10k.fbin": {
            "url": "query.public.100K.fbin",
            "size": 200 * 4 * 10000 + 8 - 1,  # First 10k from 100k file
            "description": "Test queries (10k points, 200 dim)"
        },
        "gt.10k.ibin": {
            "url": "t2i.gt.10k.ibin",
            "size": None,  # Variable size
            "description": "Ground truth (10k queries, 100 neighbors)"
        }
    }
}

# ---------------------------
# Download helpers (from download_t2i_10m.py)
# ---------------------------
def download_file(filepath: Path, url: str, expected_size: Optional[int] = None, max_bytes: Optional[int] = None) -> bool:
    """Download a single file with progress bar."""
    if filepath.exists():
        print(f"✓ {filepath.name} already exists")
        return True
        
    print(f"⬇️ Downloading {filepath.name}...")
    
    try:
        # Use range request if we need to limit the download size
        headers = {}
        if max_bytes:
            headers['Range'] = f'bytes=0-{max_bytes-1}'
            print(f"   Using range request: bytes=0-{max_bytes-1}")
        
        import requests
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()
        
        # Get the actual content length (might be different with range requests)
        content_length = response.headers.get('content-length')
        if content_length:
            total_size = int(content_length)
        else:
            total_size = max_bytes or 0
        
        if expected_size and total_size != expected_size:
            print(f"⚠️ Warning: Expected size {expected_size}, got {total_size}")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✅ Saved {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {filepath.name}: {e}")
        if filepath.exists():
            filepath.unlink()  # Remove partial file
        return False

def check_dataset_files(root: Path) -> bool:
    """Check if T2I-10M dataset files exist and are valid."""
    print("🔍 Checking T2I-10M dataset...")
    print(f"Directory: {root}")
    print()
    
    all_present = True
    
    for filename, config in T2I_CONFIG["files"].items():
        filepath = root / filename
        
        if not filepath.exists():
            print(f"❌ Missing: {filename}")
            all_present = False
            continue
        
        # Check file size
        actual_size = filepath.stat().st_size
        expected_size = config["size"]
        
        if expected_size and abs(actual_size - expected_size) > 1024:  # Allow 1KB tolerance
            print(f"⚠️ Size mismatch: {filename} (expected ~{expected_size}, got {actual_size})")
        else:
            print(f"✅ {filename} ({actual_size:,} bytes)")
    
    if all_present:
        print("\n🎉 Dataset is complete and ready to use!")
    else:
        print("\n❌ Dataset is incomplete. Will download missing files.")
    
    return all_present

def ensure_t2i_data(root: Path) -> None:
    """Download T2I-10M dataset files if missing."""
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if check_dataset_files(data_dir):
        return  # All files exist and are valid
    
    print("Downloading missing T2I-10M files...")
    success = True
    
    for filename, config in T2I_CONFIG["files"].items():
        filepath = data_dir / filename
        
        if filepath.exists():
            continue  # Skip existing files
            
        if filename == "gt.10k.ibin":
            # Ground truth from different source
            url = T2I_CONFIG["gt_url"]
            expected_size = config["size"]
            max_bytes = None
        else:
            # Base files from Yandex storage
            url = T2I_CONFIG["base_url"] + config["url"]
            expected_size = config["size"]
            max_bytes = expected_size  # Limit download to expected size
        
        print(f"📁 {config['description']}")
        if not download_file(filepath, url, expected_size, max_bytes):
            success = False
        print()
    
    if not success:
        raise RuntimeError("Failed to download some T2I-10M files")

# ---------------------------
# FBIN file handling (from assert_t2i_data.py)
# ---------------------------
def read_fbin_header(path: Path) -> Tuple[int, int]:
    """Read [n_points, dim] from FBIN header without loading data."""
    with open(path, 'rb') as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        n, d = int(header[0]), int(header[1])
    return n, d

def read_fbin_block(path: Path, start: int, count: int, dim: int) -> np.ndarray:
    """
    Load a block of FBIN vectors without reading the entire file.
    Args:
        path: fbin file
        start: starting row index (0-based)
        count: number of rows to load
        dim: vector dimensionality
    Returns: float32 array [count, dim]
    """
    if count <= 0:
        return np.empty((0, dim), dtype=np.float32)
    offset = 8 + start * dim * 4
    nbytes = count * dim * 4
    with open(path, 'rb') as f:
        f.seek(offset)
        buf = f.read(nbytes)
    # Guard against short reads at EOF or padding bytes; truncate to full rows
    total_bytes = len(buf)
    if total_bytes == 0:
        return np.empty((0, dim), dtype=np.float32)
    full_elems = (total_bytes // 4)
    full_rows = full_elems // dim
    if full_rows == 0:
        return np.empty((0, dim), dtype=np.float32)
    use_bytes = full_rows * dim * 4
    arr = np.frombuffer(memoryview(buf)[:use_bytes], dtype=np.float32)
    return arr.reshape((full_rows, dim))

def read_ibin_header(path: Path) -> Tuple[int, int]:
    """Read [n_queries, k] from IBIN header without loading data."""
    with open(path, 'rb') as f:
        header = np.frombuffer(f.read(8), dtype=np.uint32)
        n, k = int(header[0]), int(header[1])
    return n, k

# ---------------------------
# Import functions from laion10m_cache.py
# ---------------------------

try:
    from laion10m_cache import topk_exact, l2n as l2_normalize
except ImportError:
    print("Warning: Could not import from laion10m_cache.py, using local implementations")
    # Fallback implementations if import fails
    def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """L2 normalize vectors."""
        n = np.linalg.norm(x, axis=1, keepdims=True) + eps
        return (x / n).astype(np.float32)
    
    @torch.no_grad()
    def topk_exact(Q: np.ndarray, X: np.ndarray, k: int, batch: int = 8192, index_batch: int = 50000, use_fp16: bool = False) -> np.ndarray:
        """Fallback implementation if import fails."""
        # [Implementation would go here - keeping it minimal for now]
        raise NotImplementedError("Import from laion10m_cache failed and fallback not implemented")

# ---------------------------
# Cache builder
# ---------------------------
def build_cache(root: Path, out_path: Path, k_exact: int, up_to: int, 
                query_batch: int = 4096, index_batch: int = 25000, use_fp16: bool = False) -> None:
    """Build cache from T2I-10M dataset files."""
    print("Building T2I-10M cache...")
    
    data_dir = root / "data"
    
    # File paths
    base_path = data_dir / "base.10M.fbin"
    train_path = data_dir / "query.train.10M.fbin"
    test_path = data_dir / "query.10k.fbin"
    
    # Determine how much data to use based on up_to
    if up_to <= 0:
        # Use all data
        n_images_use = 10_000_000
        n_text_use = 10_000_000
    else:
        # Use up_to million points: 1M images + 1M text queries
        n_images_use = up_to * 1_000_000
        n_text_use = up_to * 1_000_000
    
    print(f"Using {n_images_use:,} images and {n_text_use:,} text queries")
    
    # Load base vectors (images) - these are the index
    print("Loading image embeddings (index)...")
    n_images, d_images = read_fbin_header(base_path)
    n_images_real = min(n_images, n_images_use)
    X_images = read_fbin_block(base_path, 0, n_images_real, d_images)
    X_images = l2_normalize(X_images.astype(np.float32))
    
    # Load training text queries - these are the training queries
    print("Loading training text queries...")
    n_text, d_text = read_fbin_header(train_path)
    n_text_real = min(n_text, n_text_use)
    T_train = read_fbin_block(train_path, 0, n_text_real, d_text)
    T_train = l2_normalize(T_train.astype(np.float32))
    
    # Load test queries (validation set) - always use all 10k test queries
    print("Loading test queries...")
    n_test, d_test = read_fbin_header(test_path)
    Q_test = read_fbin_block(test_path, 0, n_test, d_test)
    Q_test = l2_normalize(Q_test.astype(np.float32))
    
    # Validate dimensions match
    assert d_images == d_text == d_test, f"Dimension mismatch: images={d_images}, text={d_text}, test={d_test}"
    D = d_images
    
    print(f"Data shapes:")
    print(f"  Images (index): {X_images.shape}")
    print(f"  Train text queries: {T_train.shape}")
    print(f"  Test queries: {Q_test.shape}")

    print("Computing exact@K for train and val sets:")
    
    # Compute exact@K for training set (text queries vs image index)
    print("  • train: queries = train texts, index = images")
    knn_train = topk_exact(T_train, X_images, k=k_exact, batch=query_batch, index_batch=index_batch, use_fp16=use_fp16)
    
    # Compute exact@K for validation set (test queries vs image index)
    print("  • val: queries = test queries, index = images")
    knn_val = topk_exact(Q_test, X_images, k=k_exact, batch=query_batch, index_batch=index_batch, use_fp16=use_fp16)
    
    # Optional: Validate against ground truth if available (only meaningful for full dataset)
    gt_path = data_dir / "gt.10k.ibin"
    if gt_path.exists() and k_exact <= 100 and up_to <= 0:  # Only validate for full dataset
        print("  • validating against ground truth (full dataset)...")
        try:
            n_gt, k_gt = read_ibin_header(gt_path)
            if n_gt >= Q_test.shape[0] and k_gt >= k_exact:
                # Load ground truth for first few queries
                from assert_t2i_data import load_ibin_rows
                gt_rows = load_ibin_rows(gt_path, np.arange(min(100, Q_test.shape[0]), dtype=np.int64))[:, :k_exact]
                
                # Compute recall on subset
                inter = 0
                for i in range(min(100, Q_test.shape[0])):
                    inter += len(set(knn_val[i].tolist()).intersection(set(gt_rows[i].tolist())))
                recall = inter / float(min(100, Q_test.shape[0]) * k_exact)
                print(f"  ✓ Recall@{k_exact} on 100 queries vs GT: {recall:.6f}")
                
                if recall >= 0.999:
                    print("  ✅ Ground truth validation PASSED")
                else:
                    print("  ⚠️ Ground truth validation: low recall (may be due to different k)")
        except Exception as e:
            print(f"  ⚠️ Ground truth validation failed: {e}")
    elif up_to > 0:
        print(f"  • skipping ground truth validation (using subset with up_to={up_to})")

    # assemble cache
    data = {
        "train": {
            "image_features": X_images,         # Image embeddings (index)
            "text_features":  T_train,          # Train text queries
            "knn_indices":    knn_train,        # exact@K: text queries vs images
        },
        "val": {
            "text_features":  Q_test,           # Test text queries
            "knn_indices":    knn_val,          # exact@K: test queries vs images
            "index_ref":      "train",          # explicit reference to train images
        },
        # Metadata breadcrumb for training code adaptation
        "_meta": {
            "index_bank_split": "train",
            "k_exact": k_exact,
            "dim": int(D),
            "normalized": True,
            "up_to": up_to,
            "n_images": int(X_images.shape[0]),
            "n_train_queries": int(T_train.shape[0]),
            "n_val_queries": int(Q_test.shape[0]),
            "dataset": "t2i-10M",
        }
    }

    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✓ Wrote {out_path}")
    print("Summary:")
    print("  train: images", data["train"]["image_features"].shape,
          "text_queries", data["train"]["text_features"].shape,
          "knn", data["train"]["knn_indices"].shape)
    print("  val:  text_queries", data["val"]["text_features"].shape,
          "knn", data["val"]["knn_indices"].shape,
          " (index_ref=train)")
    print("  metadata:", data["_meta"])

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build T2I-10M cache (downloads files if missing).")
    p.add_argument("--root", type=str, default=DEF_ROOT, help="Dataset root directory.")
    p.add_argument("--out",  type=str, default=None,  help="Output pickle filename. If not provided, uses default naming with up_to.")
    p.add_argument("--up_to", type=int, default=DEF_UP_TO, help="Use up_to million points from the dataset (0 = use all 10M).")
    p.add_argument("--k_exact",   type=int, default=DEF_KEX, help="Exact@K to store in cache.")
    p.add_argument("--query_batch", type=int, default=4096, help="Batch size for query processing.")
    p.add_argument("--index_batch", type=int, default=25000, help="Batch size for index processing.")
    p.add_argument("--use_fp16", action="store_true", help="Do matmul in FP16 (faster but not perfectly exact). Default: FP32 exact.")
    p.add_argument("--force", action="store_true", help="Overwrite existing output file if it exists.")
    return p.parse_args()

def main():
    args = parse_args()
    
    root = Path(args.root)
    
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
    print(f"Up to: {args.up_to}M points")
    print(f"Output: {out_path}")
    
    # Check if output file exists and handle accordingly
    if out_path.exists() and not args.force:
        print(f"Error: Output file {out_path} already exists.")
        print("Use --force to overwrite or choose a different output name with --out.")
        return 1
    
    print("Checking data files...")
    ensure_t2i_data(root)

    print("Building cache...")
    build_cache(root, out_path, k_exact=args.k_exact, 
                up_to=args.up_to, query_batch=args.query_batch, 
                index_batch=args.index_batch, use_fp16=args.use_fp16)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

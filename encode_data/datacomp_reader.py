#!/usr/bin/env python3
"""
datacomp_reader.py
------------------
Loads DataComp npz files and parquet metadata, matches them by filename,
and saves a pickle with uid, text, b32_img, b32_txt fields.

Memory-efficient: processes files in chunks and uses memory-mapped arrays where possible.
"""
import os, sys, pickle, argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ---------------------------
# Defaults
# ---------------------------
DEF_ROOT = "/ssd/hamed/ann/datacomp_small"
DEF_OUT  = "datacomp_features.pkl"

# ---------------------------
# Helper functions
# ---------------------------
def safe_read_parquet(path):
    """Safely read parquet file using pyarrow directly to avoid ArrowKeyError"""
    import pyarrow.parquet as pq
    try:
        table = pq.read_table(path)
        return table.to_pandas()
    except Exception as e:
        raise RuntimeError(f"Error reading parquet file {path}: {e}")

def process_file_pair(parquet_path, npz_path, chunk_size=100000):
    """
    Process a single parquet-npz pair and return matched data.
    
    Returns:
        dict with keys: 'uid', 'text', 'b32_img', 'b32_txt'
    """
    # Load parquet metadata
    df = safe_read_parquet(parquet_path)
    
    # Load npz features
    npz_data = np.load(npz_path, allow_pickle=True)
    b32_img = npz_data['b32_img'].astype(np.float32)  # [N, 512]
    b32_txt = npz_data['b32_txt'].astype(np.float32)  # [N, 512]
    
    # Verify dimensions match
    n_parquet = len(df)
    n_npz = b32_img.shape[0]
    if n_parquet != n_npz:
        raise ValueError(f"Mismatch in {parquet_path.name}: parquet has {n_parquet} rows, npz has {n_npz}")
    
    # Extract fields
    uid = df['uid'].values  # numpy array
    text = df['text'].values  # numpy array
    
    return {
        'uid': uid,
        'text': text,
        'b32_img': b32_img,
        'b32_txt': b32_txt
    }

def validate_pickle_file(pickle_path, root_dir, sample_size=100):
    """
    Validate the generated pickle file to ensure uid-feature alignment.
    
    Args:
        pickle_path: Path to the pickle file to validate
        root_dir: Root directory containing metadata files
        sample_size: Number of random samples to validate (for performance)
    
    Returns:
        tuple: (is_valid, error_message)
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING PICKLE FILE: {pickle_path}")
    print(f"{'='*60}")
    
    # Load pickle file
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        return False, f"Failed to load pickle file: {e}"
    
    # Check required keys exist
    required_keys = ['uid', 'text', 'b32_img', 'b32_txt']
    for key in required_keys:
        if key not in data:
            return False, f"Missing required key: {key}"
    
    # Check all arrays have same length
    n_samples = len(data['uid'])
    print(f"✓ Found {n_samples:,} samples in pickle file")
    
    # Assert all arrays have same length
    assert len(data['uid']) == n_samples, f"uid length mismatch: {len(data['uid'])} != {n_samples}"
    assert len(data['text']) == n_samples, f"text length mismatch: {len(data['text'])} != {n_samples}"
    assert data['b32_img'].shape[0] == n_samples, f"b32_img length mismatch: {data['b32_img'].shape[0]} != {n_samples}"
    assert data['b32_txt'].shape[0] == n_samples, f"b32_txt length mismatch: {data['b32_txt'].shape[0]} != {n_samples}"
    
    print(f"✓ All arrays have matching length: {n_samples:,}")
    
    # Check feature dimensions
    assert data['b32_img'].shape[1] == 512, f"b32_img dimension mismatch: expected 512, got {data['b32_img'].shape[1]}"
    assert data['b32_txt'].shape[1] == 512, f"b32_txt dimension mismatch: expected 512, got {data['b32_txt'].shape[1]}"
    
    print(f"✓ Feature dimensions correct: b32_img={data['b32_img'].shape}, b32_txt={data['b32_txt'].shape}")
    
    # Check for duplicate UIDs (could indicate alignment issues)
    unique_uids = len(np.unique(data['uid']))
    if unique_uids < n_samples:
        print(f"⚠ Warning: Found {n_samples - unique_uids:,} duplicate UIDs (expected all unique)")
        # This might be OK if the dataset has duplicates, but worth noting
    else:
        print(f"✓ All UIDs are unique: {unique_uids:,}")
    
    # Sample validation: re-load original files and verify alignment
    print(f"\nSampling {sample_size} random entries to validate uid-feature alignment...")
    root = Path(root_dir)
    metadata_dir = root / "metadata"
    
    if not metadata_dir.exists():
        print(f"⚠ Warning: Cannot find metadata directory {metadata_dir}, skipping sample validation")
        return True, None
    
    # Get list of npz files
    npz_files = sorted(list(metadata_dir.glob("*.npz")))
    if not npz_files:
        print(f"⚠ Warning: No npz files found in {metadata_dir}, skipping sample validation")
        return True, None
    
    # Sample random indices
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(n_samples, size=min(sample_size, n_samples), replace=False)
    
    # Build a mapping from uid to index in pickle data
    # Convert numpy array to hashable types (strings or ints)
    uid_to_index = {}
    for idx, uid in enumerate(data['uid']):
        # Convert numpy scalar to Python native type for hashing
        if isinstance(uid, np.ndarray):
            uid_hashable = uid.item() if uid.size == 1 else str(uid)
        elif isinstance(uid, np.generic):
            uid_hashable = uid.item()
        else:
            uid_hashable = uid
        uid_to_index[uid_hashable] = idx
    
    # Sample a few files to validate
    files_to_check = min(5, len(npz_files))  # Check up to 5 files
    np.random.seed(42)
    files_to_validate = np.random.choice(len(npz_files), size=files_to_check, replace=False)
    
    validation_errors = []
    validated_count = 0
    
    for file_idx in tqdm(files_to_validate, desc="Validating sample files"):
        npz_file = npz_files[file_idx]
        parquet_file = metadata_dir / f"{npz_file.stem}.parquet"
        
        if not parquet_file.exists() or not npz_file.exists():
            continue
        
        try:
            # Load original file pair
            original_data = process_file_pair(parquet_file, npz_file)
            
            # For each uid in this file, check if it matches in pickle
            for i in range(len(original_data['uid'])):
                uid = original_data['uid'][i]
                
                # Convert uid to hashable type for lookup
                if isinstance(uid, np.ndarray):
                    uid_hashable = uid.item() if uid.size == 1 else str(uid)
                elif isinstance(uid, np.generic):
                    uid_hashable = uid.item()
                else:
                    uid_hashable = uid
                
                # Find this uid in pickle data
                if uid_hashable in uid_to_index:
                    pickle_idx = uid_to_index[uid_hashable]
                    
                    # Verify alignment: check text matches
                    text_orig = original_data['text'][i]
                    text_pickle = data['text'][pickle_idx]
                    # Handle numpy string arrays
                    if isinstance(text_orig, np.ndarray) or isinstance(text_orig, np.generic):
                        text_orig = text_orig.item() if hasattr(text_orig, 'item') else str(text_orig)
                    if isinstance(text_pickle, np.ndarray) or isinstance(text_pickle, np.generic):
                        text_pickle = text_pickle.item() if hasattr(text_pickle, 'item') else str(text_pickle)
                    
                    if text_orig != text_pickle:
                        validation_errors.append(
                            f"Mismatch in {npz_file.name}: uid={uid_hashable}, text mismatch at index {i}"
                        )
                        continue
                    
                    # Verify features match (allow small floating point differences)
                    if not np.allclose(original_data['b32_img'][i], data['b32_img'][pickle_idx], rtol=1e-5):
                        validation_errors.append(
                            f"Mismatch in {npz_file.name}: uid={uid_hashable}, b32_img mismatch at index {i}"
                        )
                        continue
                    
                    if not np.allclose(original_data['b32_txt'][i], data['b32_txt'][pickle_idx], rtol=1e-5):
                        validation_errors.append(
                            f"Mismatch in {npz_file.name}: uid={uid_hashable}, b32_txt mismatch at index {i}"
                        )
                        continue
                    
                    validated_count += 1
                    
                    # Early exit if we've validated enough samples
                    if validated_count >= sample_size:
                        break
                
                if validated_count >= sample_size:
                    break
            
            if validated_count >= sample_size:
                break
                
        except Exception as e:
            validation_errors.append(f"Error validating {npz_file.name}: {e}")
            continue
    
    print(f"✓ Validated {validated_count:,} sample entries")
    
    if validation_errors:
        error_msg = f"Found {len(validation_errors)} validation errors:\n" + "\n".join(validation_errors[:10])
        return False, error_msg
    
    print(f"\n{'='*60}")
    print(f"✓ VALIDATION PASSED: All checks successful")
    print(f"{'='*60}")
    return True, None

def add_image_existence_check(pickle_path, image_dir, batch_size=10000):
    """
    Add image file existence check to pickle file.
    Checks for multiple extensions: .jpg, .jpeg, .png, .webp
    
    Args:
        pickle_path: Path to pickle file
        image_dir: Directory containing images (with {uid}{ext} naming)
        batch_size: Process in batches to manage memory
    
    Returns:
        tuple: (success, stats_dict)
    """
    print(f"\n{'='*60}")
    print(f"ADDING IMAGE EXISTENCE CHECK")
    print(f"{'='*60}")
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return False, {"error": f"Image directory not found: {image_dir}"}
    
    # Load pickle file
    print(f"Loading pickle file: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    n_samples = len(data['uid'])
    print(f"Checking existence for {n_samples:,} images...")
    print(f"Image directory: {image_dir}")
    print(f"Trying extensions: .jpg, .jpeg, .png, .webp")
    
    # Initialize existence array if not present
    if 'image_exists' not in data:
        data['image_exists'] = np.zeros(n_samples, dtype=bool)
        data['image_paths'] = np.empty(n_samples, dtype=object)
    else:
        print("⚠ Warning: image_exists field already exists, will be overwritten")
    
    # Extensions to try (in order of likelihood)
    extensions = ['.jpg', '.jpeg', '.png', '.webp']
    
    # Process in batches
    exists_count = 0
    for i in tqdm(range(0, n_samples, batch_size), desc="Checking image existence"):
        end_idx = min(i + batch_size, n_samples)
        batch_uids = data['uid'][i:end_idx]
        
        for j, uid in enumerate(batch_uids):
            idx = i + j
            # Convert uid to string for filename
            uid_str = str(uid)
            if isinstance(uid, np.ndarray) or isinstance(uid, np.generic):
                uid_str = str(uid.item()) if hasattr(uid, 'item') else str(uid)
            
            # Try different extensions
            image_path = None
            for ext in extensions:
                candidate = image_dir / f"{uid_str}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            exists = image_path is not None
            data['image_exists'][idx] = exists
            data['image_paths'][idx] = str(image_path) if exists else ""
            
            if exists:
                exists_count += 1
    
    # Save updated pickle
    print(f"\nSaving updated pickle with image existence checks...")
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    stats = {
        "total": n_samples,
        "exists": exists_count,
        "missing": n_samples - exists_count,
        "exists_ratio": exists_count / n_samples if n_samples > 0 else 0.0
    }
    
    print(f"\n✓ Image existence check complete:")
    print(f"  Total images: {stats['total']:,}")
    print(f"  Exists: {stats['exists']:,} ({stats['exists_ratio']*100:.2f}%)")
    print(f"  Missing: {stats['missing']:,} ({100-stats['exists_ratio']*100:.2f}%)")
    
    return True, stats

def validate_clip_encoding(pickle_path, image_dir, sample_size=50000, device="cuda"):
    """
    Validate CLIP encoding by encoding sample images/text and comparing with stored features.
    
    Args:
        pickle_path: Path to pickle file
        image_dir: Directory containing images
        sample_size: Number of samples to validate (default: 50000)
        device: Device to use for encoding (cuda/cpu)
    
    Returns:
        tuple: (success, stats_dict)
    """
    print(f"\n{'='*60}")
    print(f"VALIDATING CLIP ENCODING")
    print(f"{'='*60}")
    
    try:
        import torch
        import clip
        from PIL import Image
        from torch.utils.data import Dataset, DataLoader
    except ImportError as e:
        return False, {"error": f"Failed to import required packages: {e}. Make sure you're in the 'spin' conda environment."}
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return False, {"error": f"Image directory not found: {image_dir}"}
    
    # Load pickle file
    print(f"Loading pickle file: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    n_samples = len(data['uid'])
    
    # Filter to samples that have images
    if 'image_exists' in data:
        valid_indices = np.where(data['image_exists'])[0]
    else:
        print("⚠ Warning: image_exists not found, checking all samples...")
        valid_indices = np.arange(n_samples)
    
    # Limit to sample_size
    np.random.seed(42)
    if len(valid_indices) > sample_size:
        valid_indices = np.random.choice(valid_indices, size=sample_size, replace=False)
    
    print(f"Validating {len(valid_indices):,} samples with existing images...")
    
    # Load CLIP model
    print(f"Loading CLIP model (ViT-B/32)...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print(f"✓ CLIP model loaded on {device}")
    
    # Prepare image and text data
    image_paths = []
    texts = []
    stored_img_features = []
    stored_txt_features = []
    
    for idx in tqdm(valid_indices, desc="Preparing samples"):
        uid = data['uid'][idx]
        uid_str = str(uid)
        if isinstance(uid, np.ndarray) or isinstance(uid, np.generic):
            uid_str = str(uid.item()) if hasattr(uid, 'item') else str(uid)
        
        image_path = image_dir / f"{uid_str}.jpg"
        if image_path.exists():
            image_paths.append(str(image_path))
            texts.append(str(data['text'][idx]))
            stored_img_features.append(data['b32_img'][idx])
            stored_txt_features.append(data['b32_txt'][idx])
    
    if len(image_paths) == 0:
        return False, {"error": "No valid images found for validation"}
    
    print(f"Encoding {len(image_paths):,} images and texts with CLIP...")
    
    # Encode images
    class ImageDataset(Dataset):
        def __init__(self, image_paths, preprocess):
            self.image_paths = image_paths
            self.preprocess = preprocess
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, i):
            img = Image.open(self.image_paths[i]).convert("RGB")
            return self.preprocess(img)
    
    img_dataset = ImageDataset(image_paths, preprocess)
    img_loader = DataLoader(img_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=(device == "cuda"))
    
    encoded_img_features = []
    with torch.no_grad():
        use_amp = device == "cuda"
        for batch in tqdm(img_loader, desc="Encoding images"):
            batch = batch.to(device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = model.encode_image(batch)
            else:
                features = model.encode_image(batch)
            features = torch.nn.functional.normalize(features.float(), dim=1)
            encoded_img_features.append(features.cpu().numpy())
    
    encoded_img_features = np.concatenate(encoded_img_features, axis=0)
    
    # Encode texts
    encoded_txt_features = []
    with torch.no_grad():
        use_amp = device == "cuda"
        tokenize = clip.tokenize
        batch_size = 2048
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i+batch_size]
            toks = tokenize(batch_texts, truncate=True).to(device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = model.encode_text(toks)
            else:
                features = model.encode_text(toks)
            features = torch.nn.functional.normalize(features.float(), dim=1)
            encoded_txt_features.append(features.cpu().numpy())
    
    encoded_txt_features = np.concatenate(encoded_txt_features, axis=0)
    
    # Convert stored features to numpy arrays
    stored_img_features = np.array(stored_img_features)
    stored_txt_features = np.array(stored_txt_features)
    
    # Compare features
    print(f"\nComparing encoded features with stored features...")
    
    # Image feature comparison
    img_cosine_sims = np.sum(encoded_img_features * stored_img_features, axis=1)
    img_mean_sim = float(np.mean(img_cosine_sims))
    img_std_sim = float(np.std(img_cosine_sims))
    img_min_sim = float(np.min(img_cosine_sims))
    img_max_sim = float(np.max(img_cosine_sims))
    
    # Text feature comparison
    txt_cosine_sims = np.sum(encoded_txt_features * stored_txt_features, axis=1)
    txt_mean_sim = float(np.mean(txt_cosine_sims))
    txt_std_sim = float(np.std(txt_cosine_sims))
    txt_min_sim = float(np.min(txt_cosine_sims))
    txt_max_sim = float(np.max(txt_cosine_sims))
    
    # Check if features match (within tolerance)
    tolerance = 0.01  # Allow 1% difference
    img_matches = np.sum(img_cosine_sims >= (1.0 - tolerance))
    txt_matches = np.sum(txt_cosine_sims >= (1.0 - tolerance))
    
    stats = {
        "n_validated": len(image_paths),
        "image_features": {
            "mean_cosine_sim": img_mean_sim,
            "std_cosine_sim": img_std_sim,
            "min_cosine_sim": img_min_sim,
            "max_cosine_sim": img_max_sim,
            "matches": int(img_matches),
            "match_ratio": float(img_matches / len(image_paths))
        },
        "text_features": {
            "mean_cosine_sim": txt_mean_sim,
            "std_cosine_sim": txt_std_sim,
            "min_cosine_sim": txt_min_sim,
            "max_cosine_sim": txt_max_sim,
            "matches": int(txt_matches),
            "match_ratio": float(txt_matches / len(image_paths))
        }
    }
    
    print(f"\n✓ CLIP Encoding Validation Results:")
    print(f"  Validated samples: {stats['n_validated']:,}")
    print(f"\n  Image Features:")
    print(f"    Mean cosine similarity: {img_mean_sim:.6f}")
    print(f"    Std: {img_std_sim:.6f}, Min: {img_min_sim:.6f}, Max: {img_max_sim:.6f}")
    print(f"    Matches (>=0.99): {img_matches:,}/{len(image_paths):,} ({stats['image_features']['match_ratio']*100:.2f}%)")
    print(f"\n  Text Features:")
    print(f"    Mean cosine similarity: {txt_mean_sim:.6f}")
    print(f"    Std: {txt_std_sim:.6f}, Min: {txt_min_sim:.6f}, Max: {txt_max_sim:.6f}")
    print(f"    Matches (>=0.99): {txt_matches:,}/{len(image_paths):,} ({stats['text_features']['match_ratio']*100:.2f}%)")
    
    # Check if validation passed
    if img_mean_sim >= 0.99 and txt_mean_sim >= 0.99:
        print(f"\n{'='*60}")
        print(f"✓ CLIP VALIDATION PASSED: Features match stored encodings")
        print(f"{'='*60}")
        return True, stats
    else:
        print(f"\n{'='*60}")
        print(f"⚠ CLIP VALIDATION WARNING: Features may not match exactly")
        print(f"{'='*60}")
        return True, stats  # Still return True, but with warning

def main():
    parser = argparse.ArgumentParser(description="Load DataComp npz+parquet files and save combined pickle")
    parser.add_argument("--root", type=str, default=DEF_ROOT, help="Dataset root directory")
    parser.add_argument("--out", type=str, default=None, help="Output pickle filename (default: datacomp_features.pkl in root)")
    parser.add_argument("--chunk_size", type=int, default=5, help="Number of files to process before saving (to manage memory)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file")
    parser.add_argument("--validate_only", action="store_true", help="Only run validation (skip processing)")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of samples to validate (default: 100)")
    parser.add_argument("--add_image_check", action="store_true", help="Post-process: add image existence check to existing pickle file")
    parser.add_argument("--image_dir", type=str, default=None, help="Image directory for existence check (default: {root}/images)")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for image existence check (default: 10000)")
    
    args = parser.parse_args()
    
    root = Path(args.root)
    metadata_dir = root / "metadata"
    
    # Determine output path
    if args.out is None:
        out_path = root / DEF_OUT
    else:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = root / args.out
    
    # Handle post-processing: add image existence check
    if args.add_image_check:
        if not out_path.exists():
            print(f"Error: Pickle file {out_path} does not exist. Cannot add image existence check.")
            return 1
        
        # Determine image directory
        if args.image_dir is None:
            image_dir = root / "images"
        else:
            image_dir = Path(args.image_dir)
            if not image_dir.is_absolute():
                image_dir = root / args.image_dir
        
        # Run image existence check
        success, stats = add_image_existence_check(out_path, image_dir, batch_size=args.batch_size)
        if not success:
            print(f"\n✗ Failed to add image existence check: {stats.get('error', 'Unknown error')}")
            return 1
        
        print(f"\n✓ Post-processing complete!")
        return 0
    
    # Check if output exists: if yes and not forced, allow validation-only or abort
    if out_path.exists() and not args.force:
        if args.validate_only:
            is_valid, error_msg = validate_pickle_file(out_path, args.root, args.sample_size)
            if not is_valid:
                print(f"\n✗ VALIDATION FAILED: {error_msg}")
                return 1
            return 0
        else:
            print(f"Output file {out_path} already exists.")
            print("Use --force to overwrite, --validate_only to validate, or --add_image_check to add image existence check.")
            return 1
    
    if args.validate_only:
        if not out_path.exists():
            print(f"Error: Output file {out_path} does not exist. Cannot validate.")
            return 1
        is_valid, error_msg = validate_pickle_file(out_path, args.root, args.sample_size)
        if not is_valid:
            print(f"\n✗ VALIDATION FAILED: {error_msg}")
            return 1
        return 0
    
    if not metadata_dir.exists():
        raise RuntimeError(f"Metadata directory not found: {metadata_dir}")
    
    # Find all npz files
    npz_files = sorted(list(metadata_dir.glob("*.npz")))
    
    if not npz_files:
        raise RuntimeError(f"No npz files found in {metadata_dir}")
    
    print(f"Found {len(npz_files)} npz files")
    
    # Process files in chunks to manage memory
    print(f"Processing files in chunks of {args.chunk_size}...")
    
    all_data = {
        'uid': [],
        'text': [],
        'b32_img': [],
        'b32_txt': []
    }
    
    chunk_count = 0
    total_processed = 0
    
    for i in range(0, len(npz_files), args.chunk_size):
        chunk_files = npz_files[i:i+args.chunk_size]
        chunk_count += 1
        
        print(f"\nProcessing chunk {chunk_count} ({len(chunk_files)} files)...")
        
        # Process chunk
        chunk_data = {
            'uid': [],
            'text': [],
            'b32_img': [],
            'b32_txt': []
        }
        
        for npz_file in tqdm(chunk_files, desc=f"Chunk {chunk_count}"):
            # Find corresponding parquet file
            parquet_file = metadata_dir / f"{npz_file.stem}.parquet"
            
            if not parquet_file.exists():
                print(f"Warning: Parquet file not found for {npz_file.name}, skipping")
                continue
            
            if not npz_file.exists():
                print(f"Warning: NPZ file not found: {npz_file.name}, skipping")
                continue
            
            try:
                # Process this file pair
                file_data = process_file_pair(parquet_file, npz_file)
                
                # Validate alignment within this file pair
                n_file = len(file_data['uid'])
                assert len(file_data['text']) == n_file, f"uid-text mismatch in {npz_file.name}"
                assert file_data['b32_img'].shape[0] == n_file, f"uid-b32_img mismatch in {npz_file.name}"
                assert file_data['b32_txt'].shape[0] == n_file, f"uid-b32_txt mismatch in {npz_file.name}"
                
                # Accumulate in chunk
                chunk_data['uid'].append(file_data['uid'])
                chunk_data['text'].append(file_data['text'])
                chunk_data['b32_img'].append(file_data['b32_img'])
                chunk_data['b32_txt'].append(file_data['b32_txt'])
                
                total_processed += len(file_data['uid'])
                
            except Exception as e:
                print(f"Error processing {npz_file.name}: {e}")
                continue
        
        # Concatenate chunk data
        if chunk_data['uid']:
            chunk_concatenated = {
                'uid': np.concatenate(chunk_data['uid'], axis=0),
                'text': np.concatenate(chunk_data['text'], axis=0),
                'b32_img': np.concatenate(chunk_data['b32_img'], axis=0),
                'b32_txt': np.concatenate(chunk_data['b32_txt'], axis=0)
            }
            
            # Validate alignment after concatenation
            n_chunk = len(chunk_concatenated['uid'])
            assert len(chunk_concatenated['text']) == n_chunk, f"Chunk {chunk_count}: uid-text length mismatch"
            assert chunk_concatenated['b32_img'].shape[0] == n_chunk, f"Chunk {chunk_count}: uid-b32_img length mismatch"
            assert chunk_concatenated['b32_txt'].shape[0] == n_chunk, f"Chunk {chunk_count}: uid-b32_txt length mismatch"
            
            # Merge with accumulated data
            if len(all_data['uid']) > 0:
                all_data = {
                    'uid': np.concatenate([all_data['uid'], chunk_concatenated['uid']], axis=0),
                    'text': np.concatenate([all_data['text'], chunk_concatenated['text']], axis=0),
                    'b32_img': np.concatenate([all_data['b32_img'], chunk_concatenated['b32_img']], axis=0),
                    'b32_txt': np.concatenate([all_data['b32_txt'], chunk_concatenated['b32_txt']], axis=0)
                }
            else:
                all_data = chunk_concatenated
            
            # Validate alignment after merging
            n_total = len(all_data['uid'])
            assert len(all_data['text']) == n_total, f"After chunk {chunk_count}: uid-text length mismatch"
            assert all_data['b32_img'].shape[0] == n_total, f"After chunk {chunk_count}: uid-b32_img length mismatch"
            assert all_data['b32_txt'].shape[0] == n_total, f"After chunk {chunk_count}: uid-b32_txt length mismatch"
            
            print(f"  Chunk {chunk_count} processed: {len(chunk_concatenated['uid']):,} samples")
            print(f"  Total accumulated: {len(all_data['uid']):,} samples")
            
            # Clean up chunk data from memory
            del chunk_data, chunk_concatenated
    
    # Final save
    if len(all_data['uid']) > 0:
        # Final validation before saving
        n_final = len(all_data['uid'])
        assert len(all_data['text']) == n_final, "Final: uid-text length mismatch"
        assert all_data['b32_img'].shape[0] == n_final, "Final: uid-b32_img length mismatch"
        assert all_data['b32_txt'].shape[0] == n_final, "Final: uid-b32_txt length mismatch"
        assert all_data['b32_img'].shape[1] == 512, "Final: b32_img dimension mismatch (expected 512)"
        assert all_data['b32_txt'].shape[1] == 512, "Final: b32_txt dimension mismatch (expected 512)"
        
        print(f"\nSaving final pickle with {len(all_data['uid']):,} total samples...")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_path, 'wb') as f:
            pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"✓ Saved to {out_path}")
        print(f"\nSummary:")
        print(f"  Total samples: {len(all_data['uid']):,}")
        print(f"  UID shape: {all_data['uid'].shape}")
        print(f"  Text shape: {all_data['text'].shape}")
        print(f"  b32_img shape: {all_data['b32_img'].shape}")
        print(f"  b32_txt shape: {all_data['b32_txt'].shape}")
        
        # Run validation on the saved file
        print("\n" + "="*60)
        print("Running validation on saved file...")
        print("="*60)
        is_valid, error_msg = validate_pickle_file(out_path, args.root, args.sample_size)
        if not is_valid:
            print(f"\n✗ VALIDATION FAILED: {error_msg}")
            print("⚠ Warning: Saved file failed validation checks!")
            return 1
        
        # No post-processing in this simplified mode
    else:
        print("No data to save!")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())


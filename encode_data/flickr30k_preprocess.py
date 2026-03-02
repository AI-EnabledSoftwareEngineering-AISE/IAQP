#!/usr/bin/env python3
"""
Flickr30K preprocessing script for generating soft labels for training.
- Train: 29K images + 29K captions (1 caption per image from 5 available)
- Val: 5K captions from train images (unseen captions) vs train images
- Test: 2K captions from val+test images vs val+test images (test queries)
- Inbound: 2K captions from val+test images vs train images (inbound queries)
"""

import pickle
import random
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm

# Hardcoded paths and settings
INPUT_DIR = "/home/hamed/projects/SPIN/adapter/data/flickr30k_cache"
OUTPUT_DIR = "/home/hamed/projects/SPIN/adapter/data/flickr30k_cache"
K_NEIGHBORS = 100


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_flickr30k_single_captions_from_cache(split: str):
    """Load Flickr30K data with single random caption per image from existing cache"""
    if split not in ["train", "val", "test"]:
        raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")
    
    # Load existing encoded data
    cache_file = Path(INPUT_DIR) / f"flickr30k_{split}.pkl"
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}. Run flickr30k_encode.py first!")
    
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    
    # Select single random caption per image
    image_features = cached_data['image_features']
    text_features = cached_data['text_features']
    text_to_image = cached_data['text_to_image']
    
    # For each image, randomly select one caption
    unique_images = np.unique(text_to_image)
    selected_captions = []
    selected_caption_indices = []
    
    for img_idx in unique_images:
        # Find all captions for this image
        caption_indices = np.where(text_to_image == img_idx)[0]
        # Randomly select one caption
        selected_idx = random.choice(caption_indices)
        selected_captions.append(text_features[selected_idx])
        selected_caption_indices.append(selected_idx)
    
    return {
        'image_features': image_features,
        'text_features': np.array(selected_captions),
        'image_filenames': cached_data['image_filenames'],
        'selected_caption_indices': selected_caption_indices,
        'all_text_features': text_features,  # Keep all captions for val creation
        'all_text_to_image': text_to_image   # Keep mapping for val creation
    }


def create_train_val_test_inbound_splits():
    """Create train/val/test/inbound splits according to Flickr30K setup"""
    print("Creating Flickr30K train/val/test/inbound splits...")
    
    # Load train data (29K images, 145K captions -> select 29K captions)
    train_data = load_flickr30k_single_captions_from_cache("train")
    train_images = train_data['image_features']
    train_texts = train_data['text_features']
    train_filenames = train_data['image_filenames']
    
    # Load val data (1K images, 5K captions)
    val_data = load_flickr30k_single_captions_from_cache("val")
    val_images = val_data['image_features']
    val_texts = val_data['text_features']
    val_filenames = val_data['image_filenames']
    
    # Load test data (1K images, 5K captions)
    test_data = load_flickr30k_single_captions_from_cache("test")
    test_images = test_data['image_features']
    test_texts = test_data['text_features']
    test_filenames = test_data['image_filenames']
    
    print(f"Original train: {len(train_images)} images")
    print(f"Original val: {len(val_images)} images")
    print(f"Original test: {len(test_images)} images")
    
    # Train: 29K images + 29K captions (1 caption per image)
    train_split = {
        'images': train_images,
        'texts': train_texts,
        'filenames': train_filenames
    }
    print(f"✅ Train: {len(train_images)} images + {len(train_texts)} captions")
    
    # Val: 5K captions from train images but with UNSEEN captions
    # Select 5K random images from train, then get alternative captions
    val_size = min(5000, len(train_images))
    val_image_indices = np.random.choice(len(train_images), size=val_size, replace=False)
    
    # Create val split with alternative captions
    val_split = create_alternative_captions_split(train_data, val_image_indices)
    val_split['text_to_image'] = np.array(val_image_indices, dtype=np.int64)
    if 'filenames' in val_split:
        del val_split['filenames']
    print(f"✅ Val: {len(val_split['texts'])} captions from {val_size} train images (unseen captions)")
    
    # Combine val and test images for test/inbound queries (2K images total)
    combined_images = np.vstack([val_images, test_images])
    combined_filenames = val_filenames + test_filenames
    print(f"✅ Combined val+test: {len(combined_images)} images for test/inbound queries")
    
    # Test: 2K captions from val+test images (1 caption per image)
    test_size = min(2000, len(combined_images))
    test_indices = np.random.choice(len(combined_images), size=test_size, replace=False)
    test_split = {
        'images': combined_images[test_indices],
        'texts': np.vstack([val_texts, test_texts])[test_indices],  # Use single captions from val+test
        'filenames': [combined_filenames[i] for i in test_indices],
        'text_to_image': np.array(test_indices, dtype=np.int64)
    }
    print(f"✅ Test: {len(test_split['texts'])} captions from {test_size} val+test images")
    
    # Inbound: 2K captions from val+test images, query against train images
    inbound_size = min(2000, len(combined_images))
    inbound_indices = np.random.choice(len(combined_images), size=inbound_size, replace=False)
    inbound_split = {
        'images': combined_images[inbound_indices],  # val+test images (for reference)
        'texts': np.vstack([val_texts, test_texts])[inbound_indices],  # val+test captions
        'filenames': [combined_filenames[i] for i in inbound_indices]
    }
    print(f"✅ Inbound: {len(inbound_split['texts'])} val+test captions querying against train images")
    
    print(f"\nFinal splits:")
    print(f"  Train: {len(train_split['images'])} images")
    print(f"  Val: {len(val_split['texts'])} captions")
    print(f"  Test: {len(test_split['texts'])} captions")
    print(f"  Inbound: {len(inbound_split['texts'])} captions")
    
    # Sanity check: validate our evaluation setup
    train_filenames_set = set(train_split['filenames'])
    test_filenames_set = set(test_split['filenames'])
    inbound_filenames_set = set(inbound_split['filenames'])
    
    # Check for problematic overlaps
    train_test_overlap = train_filenames_set.intersection(test_filenames_set)
    train_inbound_overlap = train_filenames_set.intersection(inbound_filenames_set)
    
    print(f"\n🔍 Evaluation setup validation:")
    print(f"  Train-Test overlap: {len(train_test_overlap)} files (should be 0 - different image sets)")
    print(f"  Train-Inbound overlap: {len(train_inbound_overlap)} files (should be 0 - different image sets)")
    
    if len(train_test_overlap) > 0 or len(train_inbound_overlap) > 0:
        raise ValueError("Data leakage detected! Train and test/inbound should use different image sets.")
    
    print("✅ No data leakage detected!")
    
    return train_split, val_split, test_split, inbound_split, combined_images, combined_filenames


def create_alternative_captions_split(train_data, image_indices):
    """Create split with alternative captions for selected images"""
    print("Creating val split with alternative captions...")
    
    # Get the images for val split
    val_images = train_data['image_features'][image_indices]
    val_filenames = [train_data['image_filenames'][i] for i in image_indices]
    
    # For each image in val split, find alternative captions
    val_captions = []
    val_caption_indices = []
    
    all_text_features = train_data['all_text_features']
    all_text_to_image = train_data['all_text_to_image']
    
    for i, img_idx in enumerate(image_indices):
        # Find all captions for this image
        caption_indices = np.where(all_text_to_image == img_idx)[0]
        
        # Remove the caption that was used in training (selected_caption_indices)
        train_caption_idx = train_data['selected_caption_indices'][img_idx]
        alternative_indices = caption_indices[caption_indices != train_caption_idx]
        
        if len(alternative_indices) > 0:
            # Select a random alternative caption
            selected_alt_idx = random.choice(alternative_indices)
            val_captions.append(all_text_features[selected_alt_idx])
            val_caption_indices.append(selected_alt_idx)
        else:
            # Fallback: use the training caption if no alternatives
            print(f"Warning: No alternative caption for image {img_idx}, using training caption")
            val_captions.append(train_data['text_features'][img_idx])
            val_caption_indices.append(train_caption_idx)
    
    return {
        'images': val_images,
        'texts': np.array(val_captions),
        'filenames': val_filenames,
        'caption_indices': val_caption_indices
    }


@torch.no_grad()
def compute_brute_force_knn_gpu(Q, X, k=100, device="cuda", chunk_size=4096):
    """
    Compute brute force KNN using GPU for efficiency.
    Returns (indices, distances) both as numpy arrays.
    """
    print(f"Computing brute force KNN: {len(Q)} queries vs {len(X)} images, k={k}")
    
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    Q_tensor = torch.from_numpy(Q).to(dev)
    X_tensor = torch.from_numpy(X).to(dev)
    
    k_eff = min(k, X.shape[0])
    indices_list = []
    distances_list = []
    
    for i in tqdm(range(0, len(Q_tensor), chunk_size), desc="Computing KNN"):
        batch_end = min(len(Q_tensor), i + chunk_size)
        Q_batch = Q_tensor[i:batch_end]
        
        # Compute similarities
        S = Q_batch @ X_tensor.T  # [batch_size, num_images]
        
        # Get top-k
        distances, indices = torch.topk(S, k=k_eff, dim=1, largest=True, sorted=True)
        
        indices_list.append(indices.cpu().numpy().astype(np.int32))
        distances_list.append(distances.cpu().numpy().astype(np.float32))
    
    indices = np.vstack(indices_list)  # [num_queries, k]
    distances = np.vstack(distances_list)  # [num_queries, k]
    
    print(f"✅ KNN computed: indices {indices.shape}, distances {distances.shape}")
    return indices, distances


def save_ranking_data(train_split, val_split, test_split, inbound_split, train_knn, val_knn, test_knn, inbound_knn, combined_images, combined_filenames):
    """Save all splits in flickr30k_ranking.pkl"""
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ranking data with all splits
    ranking_data = {
        "train": {
            "image_features": train_split['images'].astype(np.float32),
            "text_features": train_split['texts'].astype(np.float32),
            "image_filenames": train_split['filenames'],
            "text_to_image": np.arange(len(train_split['texts']), dtype=np.int64),
            "knn_indices": train_knn['indices'].astype(np.int32),
            "knn_distances": train_knn['distances'].astype(np.float32),
        },
        "val": {
            "image_features": train_split['images'].astype(np.float32),  # Use train images as bank
            "text_features": val_split['texts'].astype(np.float32),
            "image_filenames": train_split['filenames'],  # Train image filenames
            "text_to_image": val_split['text_to_image'].astype(np.int64),
            "knn_indices": val_knn['indices'].astype(np.int32),
            "knn_distances": val_knn['distances'].astype(np.float32),
        },
        "test": {
            "image_features": combined_images.astype(np.float32),  # Use ALL val+test images as bank
            "text_features": test_split['texts'].astype(np.float32),
            "image_filenames": combined_filenames,  # ALL val+test image filenames
            "text_to_image": test_split['text_to_image'].astype(np.int64),
            "knn_indices": test_knn['indices'].astype(np.int32),
            "knn_distances": test_knn['distances'].astype(np.float32),
        },
        "inbound": {
            "image_features": train_split['images'].astype(np.float32),  # Use train images as bank
            "text_features": inbound_split['texts'].astype(np.float32),
            "image_filenames": train_split['filenames'],  # Train image filenames
            "text_to_image": np.full(len(inbound_split['texts']), -1, dtype=np.int64),  # No meaningful pair hit
            "knn_indices": inbound_knn['indices'].astype(np.int32),
            "knn_distances": inbound_knn['distances'].astype(np.float32),
        },
        "metadata": {
            "embedding_dim": train_split['images'].shape[1],
            "model_type": "openai-clip:ViT-B/32",
            "k_neighbors": K_NEIGHBORS,
            "single_caption_per_image": True,
            "train_size": len(train_split['images']),
            "val_size": len(val_split['texts']),
            "test_size": len(test_split['texts']),
            "inbound_size": len(inbound_split['texts']),
        }
    }
    
    # Save ranking file
    output_file = out_dir / "flickr30k_ranking.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(ranking_data, f)
    
    print(f"✅ Saved ranking data to {output_file}")
    return output_file


def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Input dir: {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"K neighbors: {K_NEIGHBORS}")
    
    # Debug mode check
    debug_mode = False  # Set to False for full processing
    if debug_mode:
        print(f"🔍 DEBUG MODE: Processing smaller subsets")
    
    # Create train/val/test/inbound splits
    train_split, val_split, test_split, inbound_split, combined_images, combined_filenames = create_train_val_test_inbound_splits()
    
    if debug_mode:
        # Reduce sizes for debug
        debug_size = 100
        train_split['images'] = train_split['images'][:debug_size]
        train_split['texts'] = train_split['texts'][:debug_size]
        val_split['texts'] = val_split['texts'][:debug_size]
        test_split['texts'] = test_split['texts'][:debug_size]
        inbound_split['texts'] = inbound_split['texts'][:debug_size]
        combined_images = combined_images[:debug_size]
        combined_filenames = combined_filenames[:debug_size]
        print(f"[DEBUG] Reduced to {debug_size} samples per split for testing")
    
    # Compute KNN for each split according to evaluation setup
    print("\nComputing KNN for train split...")
    # Train: query train texts against train images
    train_knn_indices, train_knn_distances = compute_brute_force_knn_gpu(
        train_split['texts'], train_split['images'], k=K_NEIGHBORS, device=device
    )
    
    print("Computing KNN for val split...")
    # Val: query val texts (unseen captions) against train images
    val_knn_indices, val_knn_distances = compute_brute_force_knn_gpu(
        val_split['texts'], train_split['images'], k=K_NEIGHBORS, device=device
    )
    
    print("Computing KNN for test split...")
    # Test: query test texts against ALL val+test images
    print(f"  Querying {len(test_split['texts'])} test captions against {len(combined_images)} val+test images")
    test_knn_indices, test_knn_distances = compute_brute_force_knn_gpu(
        test_split['texts'], combined_images, k=K_NEIGHBORS, device=device
    )
    
    print("Computing KNN for inbound split...")
    # Inbound: query val+test captions against train images (cross-domain)
    inbound_knn_indices, inbound_knn_distances = compute_brute_force_knn_gpu(
        inbound_split['texts'], train_split['images'], k=K_NEIGHBORS, device=device
    )
    
    # Prepare KNN data
    train_knn = {'indices': train_knn_indices, 'distances': train_knn_distances}
    val_knn = {'indices': val_knn_indices, 'distances': val_knn_distances}
    test_knn = {'indices': test_knn_indices, 'distances': test_knn_distances}
    inbound_knn = {'indices': inbound_knn_indices, 'distances': inbound_knn_distances}
    
    # Save all splits
    print("Saving ranking data...")
    output_file = save_ranking_data(train_split, val_split, test_split, inbound_split, train_knn, val_knn, test_knn, inbound_knn, combined_images, combined_filenames)
    
    # Quick sanity check
    print("\n🔍 Sanity checks:")
    print(f"Train: {len(train_split['images'])} images, {len(train_split['texts'])} texts")
    print(f"Val: {len(val_split['texts'])} texts")
    print(f"Test: {len(test_split['texts'])} texts")
    print(f"Inbound: {len(inbound_split['texts'])} texts")
    print(f"Train KNN shape: {train_knn_indices.shape}")
    print(f"Val KNN shape: {val_knn_indices.shape}")
    print(f"Test KNN shape: {test_knn_indices.shape}")
    print(f"Inbound KNN shape: {inbound_knn_indices.shape}")
    
    print(f"\n✅ Preprocessing complete! Saved to: {output_file}")
    print("📁 This file contains:")
    print("   - train/val/test/inbound splits with no data leakage")
    print("   - KNN indices and distances for each split")
    print("   - Val: unseen captions from train images vs train images")
    print("   - Test: val+test captions vs val+test images")
    print("   - Inbound: val+test captions vs train images")
    print("   - Ready for training with soft supervision!")


if __name__ == "__main__":
    main()

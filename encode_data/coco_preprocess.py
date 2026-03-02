#!/usr/bin/env python3
"""
COCO preprocessing script for generating soft labels for training.
- Randomly selects one caption per image (real-world scenario)
- Computes brute force KNN to create soft labels with K=100 neighbors
- Saves both vector IDs and distances for reuse in training and RoarGraph
"""

import json
import pickle
import random
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
import clip


# Hardcoded paths and settings
COCO_ROOT = "/ssd/hamed/coco"
ANNOTATIONS_DIR = "/ssd/hamed/coco/annotations"
OUTPUT_DIR = "/home/hamed/projects/SPIN/adapter/data/coco_cache"
MODEL_NAME = "ViT-B/32"
K_NEIGHBORS = 100
DEBUG_SIZE = 1000


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_coco_single_captions_from_cache(split: str):
    """Load COCO data with single random caption per image from existing cache"""
    if split not in ["train2014", "val2014"]:
        raise ValueError(f"Split must be 'train2014' or 'val2014', got {split}")
    
    # Load existing encoded data
    cache_file = Path(OUTPUT_DIR) / f"coco_{split}.pkl"
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_file}. Run encode_coco.py first!")
    
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    
    # Get original annotations to select single captions
    ann_file = Path(ANNOTATIONS_DIR) / f"captions_{split}.json"
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Group captions by image_id
    image_captions = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_captions:
            image_captions[img_id] = []
        image_captions[img_id].append(ann['caption'])
    
    # Create image_id to filename mapping
    images = {img['id']: img['file_name'] for img in data['images']}
    
    # Select single random caption per image and get corresponding indices
    selected_captions = []
    selected_caption_indices = []
    image_features = cached_data['image_features']
    text_features = cached_data['text_features']
    text_to_image = cached_data['text_to_image']
    
    # For each image, randomly select one caption
    unique_images = np.unique(text_to_image)
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
        'image_paths': cached_data.get('image_paths', [str(Path(COCO_ROOT) / split / fn) for fn in cached_data['image_filenames']]),
        'selected_caption_indices': selected_caption_indices,
        'all_text_features': text_features,  # Keep all captions for inbound creation
        'all_text_to_image': text_to_image   # Keep mapping for inbound creation
    }


def create_inbound_split(train_data, inbound_indices):
    """Create inbound split with different captions for same images"""
    print("Creating inbound split with alternative captions...")
    
    # Get the images for inbound split
    inbound_images = train_data['image_features'][inbound_indices]
    inbound_filenames = [train_data['image_filenames'][i] for i in inbound_indices]
    inbound_paths = [train_data['image_paths'][i] for i in inbound_indices]
    
    # For each image in inbound split, find alternative captions
    inbound_captions = []
    inbound_caption_indices = []
    
    all_text_features = train_data['all_text_features']
    all_text_to_image = train_data['all_text_to_image']
    
    for i, img_idx in enumerate(inbound_indices):
        # Find all captions for this image
        caption_indices = np.where(all_text_to_image == img_idx)[0]
        
        # Remove the caption that was used in training (selected_caption_indices)
        train_caption_idx = train_data['selected_caption_indices'][img_idx]
        alternative_indices = caption_indices[caption_indices != train_caption_idx]
        
        if len(alternative_indices) > 0:
            # Select a random alternative caption
            selected_alt_idx = random.choice(alternative_indices)
            inbound_captions.append(all_text_features[selected_alt_idx])
            inbound_caption_indices.append(selected_alt_idx)
        else:
            # Fallback: use the training caption if no alternatives
            print(f"Warning: No alternative caption for image {img_idx}, using training caption")
            inbound_captions.append(train_data['text_features'][img_idx])
            inbound_caption_indices.append(train_caption_idx)
    
    return {
        'images': inbound_images,
        'texts': np.array(inbound_captions),
        'filenames': inbound_filenames,
        'paths': inbound_paths,
        'caption_indices': inbound_caption_indices
    }


def create_train_val_test_splits():
    """Create train/val/test/inbound splits with proper evaluation setup"""
    print("Creating train/val/test/inbound splits...")
    
    # Load train2014 data (82,783 images)
    train_data = load_coco_single_captions_from_cache("train2014")
    train_images = train_data['image_features']
    train_texts = train_data['text_features']
    train_filenames = train_data['image_filenames']
    train_paths = train_data['image_paths']
    
    # Load val2014 data (40,504 images)
    val_data = load_coco_single_captions_from_cache("val2014")
    val_images = val_data['image_features']
    val_texts = val_data['text_features']
    val_filenames = val_data['image_filenames']
    val_paths = val_data['image_paths']
    
    print(f"Original train: {len(train_images)} images")
    print(f"Original val: {len(val_images)} images")
    
    # Train: All train2014 data (82,783 images + captions)
    train_split = {
        'images': train_images,
        'texts': train_texts,
        'filenames': train_filenames,
        'paths': train_paths,
    }
    print(f"✅ Train: {len(train_images)} images + {len(train_texts)} captions")
    
    # Val: 10K captions from train2014 images but with UNSEEN captions
    # Select 10K random images from train2014, then get alternative captions
    val_size = min(10000, len(train_images))
    val_image_indices = np.random.choice(len(train_images), size=val_size, replace=False)
    val_split = create_inbound_split(train_data, val_image_indices)
    # Add text_to_image mapping for pair hit evaluation
    val_split['text_to_image'] = np.array(val_image_indices, dtype=np.int64)
    # Remove filenames since val uses train images as bank
    if 'filenames' in val_split:
        del val_split['filenames']
    print(f"✅ Val: {len(val_split['texts'])} captions from {val_size} train2014 images (unseen captions)")
    
    # Test: 10K captions from val2014 images (unseen images + captions)
    test_size = min(10000, len(val_images))
    test_indices = np.random.choice(len(val_images), size=test_size, replace=False)
    test_split = {
        'images': val_images[test_indices],
        'texts': val_texts[test_indices],
        'filenames': [val_filenames[i] for i in test_indices],
        'paths': [val_paths[i] for i in test_indices],
        'text_to_image': np.array(test_indices, dtype=np.int64)  # Map to selected val2014 image indices
    }
    print(f"✅ Test: {len(test_split['texts'])} captions from {test_size} val2014 images (unseen images)")
    
    # Inbound: 10K captions from val2014 images, query against train2014 images
    # This tests cross-domain retrieval (val2014 captions vs train2014 images)
    inbound_size = min(10000, len(val_images))
    inbound_image_indices = np.random.choice(len(val_images), size=inbound_size, replace=False)
    inbound_split = {
        'images': val_images[inbound_image_indices],  # val2014 images (for reference)
        'texts': val_texts[inbound_image_indices],    # val2014 captions
        'filenames': [val_filenames[i] for i in inbound_image_indices],
        'paths': [val_paths[i] for i in inbound_image_indices]
    }
    print(f"✅ Inbound: {len(inbound_split['texts'])} val2014 captions querying against train2014 images")
    
    print(f"Final splits:")
    print(f"  Train: {len(train_split['images'])} images")
    print(f"  Val: {len(val_split['texts'])} captions")
    print(f"  Test: {len(test_split['texts'])} captions")
    print(f"  Inbound: {len(inbound_split['texts'])} captions")
    
    # Sanity check: validate our evaluation setup
    train_filenames_set = set(train_split['filenames'])
    test_filenames_set = set(test_split['filenames'])
    inbound_filenames_set = set(inbound_split['filenames'])
    
    # Check for problematic overlaps (train and test should use different image sets)
    train_test_overlap = train_filenames_set.intersection(test_filenames_set)
    train_inbound_overlap = train_filenames_set.intersection(inbound_filenames_set)
    
    print(f"\n🔍 Evaluation setup validation:")
    print(f"  Train-Test overlap: {len(train_test_overlap)} files (should be 0 - different image sets)")
    print(f"  Train-Inbound overlap: {len(train_inbound_overlap)} files (should be 0 - different image sets)")
    
    # Check for problematic overlaps
    if len(train_test_overlap) > 0:
        raise ValueError("Data leakage detected! Train and test should use different image sets.")
    
    if len(train_inbound_overlap) > 0:
        raise ValueError("Data leakage detected! Train and inbound should use different image sets.")
    
    print("✅ No data leakage detected!")
    
    # Return val_images, val_filenames, and val_paths for use in KNN computation
    return train_split, val_split, test_split, inbound_split, val_images, val_filenames, val_paths


# Removed encoding functions - we reuse existing encoded data


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


def save_ranking_data(train_split, val_split, test_split, inbound_split, train_knn, val_knn, test_knn, inbound_knn, val_images, val_filenames, val_paths):
    """Save all splits in coco_ranking.pkl"""
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ranking data with all splits
    ranking_data = {
        "train": {
            "image_features": train_split['images'].astype(np.float32),
            "text_features": train_split['texts'].astype(np.float32),
            "image_filenames": train_split['filenames'],
            "image_paths": train_split['paths'],
            "text_to_image": np.arange(len(train_split['texts']), dtype=np.int64),  # Each text maps to its corresponding image
            "knn_indices": train_knn['indices'].astype(np.int32),
            "knn_distances": train_knn['distances'].astype(np.float32),
        },
        "val": {
            "image_features": train_split['images'].astype(np.float32),  # Use train images as bank
            "text_features": val_split['texts'].astype(np.float32),
            "image_filenames": train_split['filenames'],  # Train image filenames
            "image_paths": train_split['paths'],  # Train image paths
            "text_to_image": val_split['text_to_image'].astype(np.int64),  # Map to train images
            "knn_indices": val_knn['indices'].astype(np.int32),
            "knn_distances": val_knn['distances'].astype(np.float32),
        },
        "test": {
            "image_features": val_images.astype(np.float32),  # Use ALL val2014 images as bank (40,504)
            "text_features": test_split['texts'].astype(np.float32),
            "image_filenames": val_filenames,  # ALL val2014 image filenames
            "image_paths": val_paths,  # ALL val2014 image paths
            "text_to_image": test_split['text_to_image'].astype(np.int64),  # Map to selected val2014 images
            "knn_indices": test_knn['indices'].astype(np.int32),
            "knn_distances": test_knn['distances'].astype(np.float32),
        },
        "inbound": {
            "image_features": train_split['images'].astype(np.float32),  # Use train images as bank
            "text_features": inbound_split['texts'].astype(np.float32),
            "image_filenames": train_split['filenames'],  # Train image filenames
            "image_paths": train_split['paths'],  # Train image paths
            "text_to_image": np.full(len(inbound_split['texts']), -1, dtype=np.int64),  # No meaningful pair hit
            "knn_indices": inbound_knn['indices'].astype(np.int32),
            "knn_distances": inbound_knn['distances'].astype(np.float32),
        },
        "metadata": {
            "embedding_dim": train_split['images'].shape[1],
            "model_type": f"openai-clip:{MODEL_NAME}",
            "k_neighbors": K_NEIGHBORS,
            "single_caption_per_image": True,
            "train_size": len(train_split['images']),
            "val_size": len(val_split['images']),
            "test_size": len(test_split['images']),
            "inbound_size": len(inbound_split['images']),
        }
    }
    
    # Save ranking file
    output_file = out_dir / "coco_ranking.pkl"
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
    print(f"COCO root: {COCO_ROOT}")
    print(f"Annotations: {ANNOTATIONS_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"K neighbors: {K_NEIGHBORS}")
    
    # Debug mode check
    debug_mode = False  # Set to False for full processing
    if debug_mode:
        print(f"🔍 DEBUG MODE: Processing 100 images per split")
    
    # Create train/val/test/inbound splits
    train_split, val_split, test_split, inbound_split, val_images, val_filenames, val_paths = create_train_val_test_splits()
    
    if debug_mode:
        # Reduce sizes for debug: 100 images per split
        debug_size = 100
        train_split['images'] = train_split['images'][:debug_size]
        train_split['texts'] = train_split['texts'][:debug_size]
        val_split['images'] = val_split['images'][:debug_size]
        val_split['texts'] = val_split['texts'][:debug_size]
        test_split['images'] = test_split['images'][:debug_size]
        test_split['texts'] = test_split['texts'][:debug_size]
        inbound_split['images'] = inbound_split['images'][:debug_size]
        inbound_split['texts'] = inbound_split['texts'][:debug_size]
        print(f"[DEBUG] Reduced to {debug_size} images per split for testing")
    
    # Compute KNN for each split according to evaluation setup
    print("\nComputing KNN for train split...")
    # Train: query train texts against train images
    train_knn_indices, train_knn_distances = compute_brute_force_knn_gpu(
        train_split['texts'], train_split['images'], k=K_NEIGHBORS, device=device
    )
    
    print("Computing KNN for val split...")
    # Val: query val texts (unseen captions) against train images (same as training)
    val_knn_indices, val_knn_distances = compute_brute_force_knn_gpu(
        val_split['texts'], train_split['images'], k=K_NEIGHBORS, device=device
    )
    
    if len(test_split['texts']) > 0:
        print("Computing KNN for test split...")
        # Test: query test texts against ALL val2014 images (40,504), not just selected 10K
        print(f"  Querying {len(test_split['texts'])} test captions against {len(val_images)} val2014 images")
        test_knn_indices, test_knn_distances = compute_brute_force_knn_gpu(
            test_split['texts'], val_images, k=K_NEIGHBORS, device=device
        )
    else:
        print("Skipping test split KNN (no test samples)")
        test_knn_indices = np.array([]).reshape(0, K_NEIGHBORS)
        test_knn_distances = np.array([]).reshape(0, K_NEIGHBORS)
    
    print("Computing KNN for inbound split...")
    # Inbound: query val2014 captions against train2014 images (cross-domain)
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
    output_file = save_ranking_data(
        train_split,
        val_split,
        test_split,
        inbound_split,
        train_knn,
        val_knn,
        test_knn,
        inbound_knn,
        val_images,
        val_filenames,
        val_paths,
    )
    
    # Quick sanity check
    print("\n🔍 Sanity checks:")
    print(f"Train: {len(train_split['images'])} images, {len(train_split['texts'])} texts")
    print(f"Val: {len(val_split['images'])} images, {len(val_split['texts'])} texts")
    print(f"Test: {len(test_split['images'])} images, {len(test_split['texts'])} texts")
    print(f"Inbound: {len(inbound_split['images'])} images, {len(inbound_split['texts'])} texts")
    print(f"Train KNN shape: {train_knn_indices.shape}")
    print(f"Val KNN shape: {val_knn_indices.shape}")
    print(f"Test KNN shape: {test_knn_indices.shape}")
    print(f"Inbound KNN shape: {inbound_knn_indices.shape}")
    
    print(f"\n✅ Preprocessing complete! Saved to: {output_file}")
    print("📁 This file contains:")
    print("   - train/val/test/inbound splits with no data leakage")
    print("   - KNN indices and distances for each split")
    print("   - Inbound split: same images as train but different captions")
    print("   - Ready for training with soft supervision!")


if __name__ == "__main__":
    main()

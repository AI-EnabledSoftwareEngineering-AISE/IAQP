"""
COCO dataset loader for text-to-image retrieval training.
"""
import pickle
from typing import Dict, Tuple, Any
import numpy as np

from .base import BaseDatasetLoader
from ..utils import l2n_np, get_split, validate_cache


class CocoDatasetLoader(BaseDatasetLoader):
    """
    Dataset loader for COCO datasets.
    
    Handles loading and preprocessing of COCO text-to-image datasets
    with pre-computed CLIP features, exact KNN indices, and text-to-image mappings
    for pair hit evaluation.
    """
    
    def __init__(self, data_path: str, **kwargs):
        """
        Initialize COCO dataset loader.
        
        Args:
            data_path: Path to the COCO cache pickle file
            **kwargs: Additional parameters (currently unused)
        """
        super().__init__(data_path, **kwargs)
    
    def load_data(self) -> Dict[str, Any]:
        """
        Load COCO dataset from pickle cache.
        
        Returns:
            Dictionary containing train/val/test/inbound splits with:
            - image_features: CLIP image embeddings
            - text_features: CLIP text embeddings  
            - knn_indices: Exact KNN indices for each text query
            - text_to_image: Mapping from text to image for pair hit evaluation
        """
        print(f"Loading COCO data from {self.data_path}")
        print("This may take a while for large datasets...")
        
        import time
        start_time = time.time()
        
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)
        
        load_time = time.time() - start_time
        print(f"✅ Dataset loaded in {load_time:.2f} seconds")
        
        # Validate the loaded data
        self.validate_cache(data)
        
        return data
    
    def validate_cache(self, data: Dict[str, Any]) -> None:
        """
        Validate COCO dataset structure.
        
        Args:
            data: The loaded dataset dictionary
            
        Raises:
            AssertionError: If the dataset structure is invalid
        """
        # Check for required splits
        required_splits = ['train', 'val']
        for split in required_splits:
            assert split in data, f"Missing required split: {split}"
        
        # Check for required fields in each split
        required_fields = ['image_features', 'text_features', 'knn_indices', 'text_to_image']
        for split in required_splits:
            if split in data:  # Only check if split exists
                for field in required_fields:
                    assert field in data[split], f"Missing field '{field}' in {split} split"
        
        print("✅ COCO dataset structure validation passed")
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get COCO training data.
        
        Returns:
            Tuple of (image_features, text_features, knn_indices)
            All features are L2-normalized
        """
        if self.data is None:
            self.data = self.load_data()
        
        tr = self.data["train"]
        X_train = l2n_np(tr["image_features"].astype(np.float32))
        T_train = l2n_np(tr["text_features"].astype(np.float32))
        exact_idx_train = tr["knn_indices"].astype(np.int64)
        
        return X_train, T_train, exact_idx_train
    
    def get_split_data(self, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get COCO data for a specific split (val/test/inbound).
        
        COCO-specific handling:
        - val: Query against train2014 images (82,783)
        - test: Query against val2014 images (40,504) 
        - inbound: Query against train2014 images (82,783)
        
        Args:
            split: Split name ('val', 'test', 'inbound')
            
        Returns:
            Tuple of (image_bank, text_features, knn_indices, gt_img_idx)
            All features are L2-normalized
        """
        if self.data is None:
            self.data = self.load_data()
        
        if split not in self.data:
            raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(self.data.keys())}")
        
        split_data = self.data[split]
        
        # Get split-specific data
        T_split = l2n_np(split_data["text_features"].astype(np.float32))
        exact_ref = split_data["knn_indices"].astype(np.int64)
        gt_img_idx = split_data["text_to_image"].astype(np.int64)
        
        # Use the correct image bank for each split
        # The image_features in split_data contains the correct bank for evaluation
        X_bank = l2n_np(split_data["image_features"].astype(np.float32))
        
        return X_bank, T_split, exact_ref, gt_img_idx
    
    def get_feature_dim(self) -> int:
        """
        Get the COCO feature dimension.
        
        Returns:
            CLIP feature dimension (typically 512)
        """
        if self.data is None:
            self.data = self.load_data()
        
        return self.data["train"]["text_features"].shape[1]
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get COCO dataset information.
        
        Returns:
            Dictionary with COCO dataset metadata
        """
        info = super().get_dataset_info()
        
        # Add COCO-specific information
        info['dataset_type'] = 'COCO'
        info['feature_type'] = 'CLIP'
        
        # Add metadata if available
        if 'metadata' in self.data:
            info.update(self.data['metadata'])
        
        return info
    
    def get_recommended_config(self) -> Dict[str, Any]:
        """
        Get COCO-specific recommended configuration parameters.
        
        Returns:
            Dictionary with recommended configuration for COCO dataset
        """
        if self.data is None:
            self.data = self.load_data()
        
        train_size = len(self.data["train"]["image_features"])
        
        # COCO-specific recommendations based on dataset size
        if train_size > 50000:  # Full COCO dataset
            recommended_config = {
                'ivf_nlist': 2048,      # ~4*sqrt(82783) ≈ 1156, use 2048 for good performance
                'M_cells': 2048,        # Match ivf_nlist for consistency
                'hnsw_M': 64,           # Higher connectivity for large dataset
                'hnsw_efC': 400,        # Higher construction effort for better quality
                'batch_size': 1024,     # Large batch for efficiency
                'C_pack': 256,          # Larger pack size for better supervision
            }
        elif train_size > 10000:  # Medium dataset
            recommended_config = {
                'ivf_nlist': 1024,
                'M_cells': 1024,
                'hnsw_M': 32,
                'hnsw_efC': 200,
                'batch_size': 512,
                'C_pack': 128,
            }
        else:  # Small dataset (debug mode)
            recommended_config = {
                'ivf_nlist': 64,
                'M_cells': 64,
                'hnsw_M': 16,
                'hnsw_efC': 100,
                'batch_size': 32,
                'C_pack': 32,
            }
        
        recommended_config['dataset_size'] = train_size
        recommended_config['dataset_type'] = 'COCO'
        
        return recommended_config
    
    def get_available_splits(self) -> list:
        """
        Get list of available splits in the dataset.
        
        Returns:
            List of split names
        """
        if self.data is None:
            self.data = self.load_data()
        
        return [k for k in self.data.keys() if k != 'metadata']
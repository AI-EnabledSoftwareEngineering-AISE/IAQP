"""
LAION dataset loader for text-to-image retrieval training.
"""
import pickle
from typing import Dict, Tuple, Any
import numpy as np

from .base import BaseDatasetLoader
from ..utils import l2n_np, get_split, validate_cache


class LaionDatasetLoader(BaseDatasetLoader):
    """
    Dataset loader for LAION datasets.
    
    Handles loading and preprocessing of LAION text-to-image datasets
    with pre-computed CLIP features and exact KNN indices.
    """
    
    def __init__(self, data_path: str, **kwargs):
        """
        Initialize LAION dataset loader.
        
        Args:
            data_path: Path to the LAION cache pickle file
            **kwargs: Additional parameters (currently unused)
        """
        super().__init__(data_path, **kwargs)
    
    def load_data(self) -> Dict[str, Any]:
        """
        Load LAION dataset from pickle cache.
        
        Returns:
            Dictionary containing train/val/test splits with:
            - image_features: CLIP image embeddings
            - text_features: CLIP text embeddings  
            - knn_indices: Exact KNN indices for each text query
        """
        print(f"Loading LAION data from {self.data_path}")
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
        Validate LAION dataset structure.
        
        Args:
            data: The loaded dataset dictionary
            
        Raises:
            AssertionError: If the dataset structure is invalid
        """
        # Use the existing validation function from utils
        validate_cache(data)
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get LAION training data.
        
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
        Get LAION data for a specific split (val/test).
        
        Args:
            split: Split name ('val' or 'test')
            
        Returns:
            Tuple of (image_bank, text_features, knn_indices, gt_img_idx)
            All features are L2-normalized
        """
        if self.data is None:
            self.data = self.load_data()
        
        # Use the existing get_split function from utils
        return get_split(self.data, split)
    
    def get_feature_dim(self) -> int:
        """
        Get the LAION feature dimension.
        
        Returns:
            CLIP feature dimension (typically 512)
        """
        if self.data is None:
            self.data = self.load_data()
        
        return self.data["train"]["text_features"].shape[1]
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get LAION dataset information.
        
        Returns:
            Dictionary with LAION dataset metadata
        """
        info = super().get_dataset_info()
        
        # Add LAION-specific information
        info['dataset_type'] = 'LAION'
        info['feature_type'] = 'CLIP'
        
        return info

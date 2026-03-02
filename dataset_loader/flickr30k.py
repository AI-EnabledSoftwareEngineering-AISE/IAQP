"""
Flickr30K dataset loader for the projector training system.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from .base import BaseDatasetLoader


class Flickr30KDatasetLoader(BaseDatasetLoader):
    """Dataset loader for Flickr30K dataset."""
    
    def __init__(self, data_path: str):
        """
        Initialize Flickr30K dataset loader.
        
        Args:
            data_path: Path to the flickr30k_ranking.pkl file
        """
        self.data_path = Path(data_path)
        self.data = None
        self._validate_cache()
    
    def _validate_cache(self) -> None:
        """Validate that the cache file exists and has the expected structure."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Flickr30K cache file not found: {self.data_path}")
        
        print(f"Loading Flickr30K data from {self.data_path}")
        print("This may take a while for large datasets...")
        
        import time
        start_time = time.time()
        
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        load_time = time.time() - start_time
        print(f"✅ Dataset loaded in {load_time:.2f} seconds")
        
        # Validate structure
        required_splits = ['train', 'val', 'test', 'inbound']
        for split in required_splits:
            if split not in self.data:
                raise ValueError(f"Missing split '{split}' in Flickr30K data")
            
            split_data = self.data[split]
            required_keys = ['image_features', 'text_features', 'knn_indices', 'knn_distances']
            for key in required_keys:
                if key not in split_data:
                    raise ValueError(f"Missing key '{key}' in Flickr30K {split} split")
        
        # Print dataset info
        train_data = self.data['train']
        val_data = self.data['val']
        test_data = self.data['test']
        inbound_data = self.data['inbound']
        
        print(f"Train data: {len(train_data['image_features'])} images, {len(train_data['text_features'])} texts")
        print(f"Val data: {len(val_data['text_features'])} texts")
        print(f"Test data: {len(test_data['text_features'])} texts")
        print(f"Inbound data: {len(inbound_data['text_features'])} texts")
        print(f"Feature dim: {train_data['image_features'].shape[1]}")
        print(f"KNN K: {train_data['knn_indices'].shape[1]}")
    
    def load_data(self) -> Dict[str, Any]:
        """Load the full Flickr30K dataset."""
        if self.data is None:
            self._validate_cache()
        return self.data
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get training data.
        
        Returns:
            Tuple of (image_features, text_features, knn_indices)
        """
        if self.data is None:
            self._validate_cache()
        
        train_data = self.data['train']
        return (
            train_data['image_features'],
            train_data['text_features'], 
            train_data['knn_indices']
        )
    
    def get_split_data(self, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for a specific split.
        
        Args:
            split: Split name ('val', 'test', 'inbound')
            
        Returns:
            Tuple of (image_features, text_features, knn_indices, text_to_image)
        """
        if self.data is None:
            self._validate_cache()
        
        if split not in self.data:
            raise ValueError(f"Split '{split}' not found in Flickr30K data. Available: {list(self.data.keys())}")
        
        split_data = self.data[split]
        
        # For Flickr30K, we need to handle the different image banks
        # Val and Inbound use train images as bank, Test uses val+test images as bank
        image_features = split_data['image_features']  # Already set correctly in preprocessing
        text_features = split_data['text_features']
        knn_indices = split_data['knn_indices']
        text_to_image = split_data.get('text_to_image', np.arange(len(text_features), dtype=np.int64))
        
        return image_features, text_features, knn_indices, text_to_image
    
    def get_feature_dim(self) -> int:
        """Get the feature dimension."""
        if self.data is None:
            self._validate_cache()
        return self.data['train']['image_features'].shape[1]
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        if self.data is None:
            self._validate_cache()
        
        # Use the base class method to get the standard format
        info = super().get_dataset_info()
        
        # Add Flickr30K-specific information
        info['dataset_type'] = 'Flickr30K'
        info['feature_type'] = 'CLIP'
        info['inbound_texts'] = len(self.data['inbound']['text_features'])
        
        return info
    
    def validate_cache(self) -> bool:
        """Validate that the cache is properly formatted."""
        try:
            self._validate_cache()
            return True
        except Exception as e:
            print(f"Cache validation failed: {e}")
            return False

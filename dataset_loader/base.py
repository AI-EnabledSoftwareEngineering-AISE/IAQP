"""
Base dataset loader interface for text-to-image retrieval training.
"""
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch


class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    
    This defines the interface that all dataset loaders must implement
    to be compatible with the projector training pipeline.
    """
    
    def __init__(self, data_path: str, **kwargs):
        """
        Initialize the dataset loader.
        
        Args:
            data_path: Path to the dataset cache file
            **kwargs: Additional dataset-specific parameters
        """
        self.data_path = data_path
        self.data = None
        self._validate_path()
    
    def _validate_path(self):
        """Validate that the data path exists."""
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
    
    @abstractmethod
    def load_data(self) -> Dict[str, Any]:
        """
        Load the dataset from the cache file.
        
        Returns:
            Dictionary containing the loaded dataset with required keys:
            - 'train': Training data with 'image_features', 'text_features', 'knn_indices'
            - 'val': Validation data with 'text_features', 'knn_indices'
            - 'test': Test data with 'text_features', 'knn_indices' (optional)
        """
        pass
    
    @abstractmethod
    def validate_cache(self, data: Dict[str, Any]) -> None:
        """
        Validate the loaded dataset structure.
        
        Args:
            data: The loaded dataset dictionary
            
        Raises:
            AssertionError: If the dataset structure is invalid
        """
        pass
    
    @abstractmethod
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get training data.
        
        Returns:
            Tuple of (image_features, text_features, knn_indices)
        """
        pass
    
    @abstractmethod
    def get_split_data(self, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for a specific split (val/test).
        
        Args:
            split: Split name ('val' or 'test')
            
        Returns:
            Tuple of (image_bank, text_features, knn_indices, gt_img_idx)
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """
        Get the feature dimension.
        
        Returns:
            Feature dimension
        """
        pass
    
    def get_num_samples(self, split: str = 'train') -> int:
        """
        Get the number of samples in a split.
        
        Args:
            split: Split name
            
        Returns:
            Number of samples
        """
        if self.data is None:
            self.data = self.load_data()
        
        if split == 'train':
            return len(self.data['train']['text_features'])
        else:
            return len(self.data[split]['text_features'])
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get dataset information.
        
        Returns:
            Dictionary with dataset metadata
        """
        if self.data is None:
            self.data = self.load_data()
        
        info = {}
        for split in ['train', 'val', 'test']:
            if split in self.data:
                split_data = self.data[split]
                info[split] = {
                    'num_texts': len(split_data['text_features']),
                    'feature_dim': split_data['text_features'].shape[1]
                }
                if split == 'train':
                    info[split]['num_images'] = len(split_data['image_features'])
                    info[split]['knn_k'] = split_data['knn_indices'].shape[1]
        
        return info



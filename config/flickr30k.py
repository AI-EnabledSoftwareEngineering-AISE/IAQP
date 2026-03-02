"""
Flickr30K dataset-specific configuration.
"""

from dataclasses import dataclass
from typing import Dict, Any
from .base import BaseConfig


@dataclass
class Flickr30KConfig(BaseConfig):
    """Flickr30K-specific configuration with optimized defaults."""
    
    # Dataset-specific parameters
    dataset_name: str = "flickr30k"
    
    # Optimized for smaller dataset size (29K images vs 900K for LAION)
    batch_size: int = 512
    ivf_nlist: int = 128
    hnsw_M: int = 32
    hnsw_efC: int = 200
    M_cells: int = 128
    C_pack: int = 64
    
    # Learning parameters - adjusted for smaller dataset
    lr: float = 2e-3  # Slightly higher LR for faster convergence on smaller dataset
    epochs: int = 10
    weight_decay: float = 1e-4
    
    # Loss weights - similar to COCO but adjusted
    w_kld: float = 1.0
    w_id: float = 0.01
    w_gap: float = 0.1
    w_cell: float = 0.2
    
    # Evaluation parameters
    eval_split: str = "val"
    eval_topk: int = 10
    eval_ph: bool = True
    eval_r5: bool = True
    
    # Other parameters
    backend: str = "ivf"
    use_smart_identity: bool = True
    identity_mode: str = "cone"
    em_recompute_every: int = 0
    
    def get_recommended_config(self, dataset_size: int = 29000) -> Dict[str, Any]:
        """
        Get recommended configuration based on dataset size.
        
        Args:
            dataset_size: Number of images in the dataset
            
        Returns:
            Dictionary with recommended parameters
        """
        # Flickr30K specific recommendations based on 29K images
        recommendations = {
            'batch_size': 512,
            'ivf_nlist': 128,
            'hnsw_M': 32,
            'hnsw_efC': 200,
            'M_cells': 128,
            'C_pack': 64,
            'lr': 2e-3,
            'epochs': 10,
        }
        
        # Adjust based on actual dataset size if different
        if dataset_size < 10000:
            # Very small dataset
            recommendations.update({
                'batch_size': 256,
                'ivf_nlist': 64,
                'hnsw_M': 16,
                'hnsw_efC': 100,
                'M_cells': 64,
                'C_pack': 32,
                'lr': 5e-3,
                'epochs': 15,
            })
        elif dataset_size > 50000:
            # Larger dataset
            recommendations.update({
                'batch_size': 1024,
                'ivf_nlist': 256,
                'hnsw_M': 64,
                'hnsw_efC': 400,
                'M_cells': 256,
                'C_pack': 128,
                'lr': 1e-3,
                'epochs': 5,
            })
        
        return recommendations

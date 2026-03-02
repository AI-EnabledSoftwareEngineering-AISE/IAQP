"""
COCO-specific configuration for text-to-image retrieval training.
"""
from dataclasses import dataclass
from .base import BaseConfig


@dataclass
class CocoConfig(BaseConfig):
    """COCO dataset-specific configuration."""
    
    # Override dataset type
    dataset: str = "coco"
    
    # COCO-specific defaults
    data_path: str = "data/coco_cache/coco_ranking.pkl"
    
    # COCO-optimized parameters (for full dataset: 82,783 images)
    batch_size: int = 1024
    ivf_nlist: int = 2048      # ~4*sqrt(82783) ≈ 1156, use 2048 for good performance
    M_cells: int = 2048        # Match ivf_nlist for consistency
    hnsw_M: int = 64           # Higher connectivity for large dataset
    hnsw_efC: int = 400        # Higher construction effort for better quality
    C_pack: int = 256          # Larger pack size for better supervision
    
    # COCO-specific training settings
    epochs: int = 5
    lr: float = 1e-3
    
    @classmethod
    def get_recommended_config(cls, dataset_size: int) -> dict:
        """Get COCO-specific recommended configuration based on dataset size."""
        if dataset_size > 50000:  # Full COCO dataset (82,783 images)
            return {
                'ivf_nlist': 2048,      # ~4*sqrt(82783) ≈ 1156, use 2048 for good performance
                'M_cells': 2048,        # Match ivf_nlist for consistency
                'hnsw_M': 64,           # Higher connectivity for large dataset
                'hnsw_efC': 400,        # Higher construction effort for better quality
                'batch_size': 1024,     # Large batch for efficiency
                'C_pack': 256,          # Larger pack size for better supervision
            }
        elif dataset_size > 10000:  # Medium COCO dataset
            return {
                'ivf_nlist': 1024,
                'M_cells': 1024,
                'hnsw_M': 32,
                'hnsw_efC': 200,
                'batch_size': 512,
                'C_pack': 128,
            }
        else:  # Small COCO dataset (debug mode)
            return {
                'ivf_nlist': 64,
                'M_cells': 64,
                'hnsw_M': 16,
                'hnsw_efC': 100,
                'batch_size': 32,
                'C_pack': 32,
            }
    
    @classmethod
    def get_evaluation_config(cls) -> dict:
        """Get COCO-specific evaluation configuration."""
        return {
            'eval_split': 'val',        # Default to val split
            'eval_topk': 10,            # Standard evaluation K
            'eval_ph': True,            # Enable pair hit evaluation
            'eval_r5': False,           # Disable R@5 by default
            'num_threads': 16,          # Use multiple threads for evaluation
        }

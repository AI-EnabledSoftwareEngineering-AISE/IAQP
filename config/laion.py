"""
LAION-specific configuration for text-to-image retrieval training.
"""
from dataclasses import dataclass
from .base import BaseConfig


@dataclass
class LaionConfig(BaseConfig):
    """LAION dataset-specific configuration."""
    
    # Override dataset type
    dataset: str = "laion"
    
    # LAION-specific defaults
    data_path: str = "/home/hamed/projects/SPIN/adapter/data/laion-10M/laion1m_cache.pkl"
    
    # LAION-optimized parameters
    batch_size: int = 1024
    ivf_nlist: int = 2048
    M_cells: int = 2048
    hnsw_M: int = 32
    hnsw_efC: int = 200
    C_pack: int = 256
    
    # LAION-specific training settings
    epochs: int = 5
    lr: float = 1e-3
    
    @classmethod
    def get_recommended_config(cls, dataset_size: int) -> dict:
        """Get LAION-specific recommended configuration based on dataset size."""
        if dataset_size > 500000:  # Large LAION dataset
            return {
                'ivf_nlist': 2048,
                'M_cells': 2048,
                'hnsw_M': 32,
                'hnsw_efC': 200,
                'batch_size': 1024,
                'C_pack': 256,
            }
        elif dataset_size > 100000:  # Medium LAION dataset
            return {
                'ivf_nlist': 1024,
                'M_cells': 1024,
                'hnsw_M': 24,
                'hnsw_efC': 150,
                'batch_size': 512,
                'C_pack': 128,
            }
        else:  # Small LAION dataset
            return {
                'ivf_nlist': 512,
                'M_cells': 512,
                'hnsw_M': 16,
                'hnsw_efC': 100,
                'batch_size': 256,
                'C_pack': 64,
            }

"""
T2I-10M dataset configuration.
"""
from .base import BaseConfig


class T2IConfig(BaseConfig):
    """
    Configuration for T2I-10M dataset.
    
    T2I-10M is a text-to-image retrieval dataset with:
    - 10M image embeddings (200-dim CLIP features)
    - 10M text embeddings (200-dim CLIP features) 
    - 10k test queries
    - Exact KNN ground truth
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Dataset-specific settings
        self.dataset = "t2i"  # Override dataset type
        self.dataset_name = "t2i-10M"
        self.feature_dim = 200  # T2I-10M uses 200-dim CLIP features
        
        # T2I-10M specific recommendations
        self._apply_t2i_recommendations()
    
    def _apply_t2i_recommendations(self):
        """Apply T2I-10M specific parameter recommendations."""
        
        # ANN index parameters (optimized for 10M dataset)
        self.hnsw_M = 32                    # Good balance for 10M dataset
        self.hnsw_efC = 200                 # Sufficient for 10M dataset
        self.ivf_nlist = 4096               # Good clustering for 10M dataset
        
        # Training parameters (optimized for T2I-10M)
        self.batch_size = 8192              # Large batch size for efficiency
        self.epochs = 8                     # Sufficient for convergence
        self.lr = 1e-3                      # Standard learning rate
        
        # Budget settings (optimized for T2I-10M scale)
        self.budgets = tuple(range(10, 101, 10))  # 10, 20, ..., 100
        
        # Pack and evaluation settings
        self.C_pack = 256                   # Good pack size for T2I-10M
        self.k_eval_inside_pack = 10        # Standard evaluation K
        
        # Loss weights (balanced for T2I-10M)
        self.w_kld = 1.0                    # Main ranking loss
        self.w_id = 1e-3                    # Identity preservation
        self.w_gap = 0.05                   # Ranking stability
        self.w_cell = 0.3                   # Cell routing efficiency
        
        # Smart identity loss (good for T2I-10M)
        self.use_smart_identity = True
        self.identity_mode = "cone"         # Adaptive threshold
        
        # Cell parameters (optimized for 10M dataset)
        self.M_cells = 4096                 # Good number of cells for 10M
        
        # Memory management (important for large dataset)
        self.pack_chunk_size = 8192         # Efficient chunk size
        self.pack_x_batch_size = 25000      # Good batch size for streaming
        
        # Evaluation settings
        self.eval_split = "val"             # Use validation split
        self.eval_topk = 10                 # Standard recall@10
        self.skip_eval = False              # Always evaluate
        
        # ANN cache (helpful for large dataset)
        self.ann_cache_enable = True
        self.ann_khead = 256                # Good cache size
        
        # EM recomputation (disabled for large dataset)
        self.em_recompute_every = 0         # Disable EM recompute for efficiency
        
        print(f"✅ Applied T2I-10M dataset recommendations")
        print(f"   Feature dim: {self.feature_dim}")
        print(f"   HNSW M: {self.hnsw_M}, efC: {self.hnsw_efC}")
        print(f"   IVF nlist: {self.ivf_nlist}")
        print(f"   Batch size: {self.batch_size}, Epochs: {self.epochs}")
        print(f"   Budgets: {self.budgets}")
        print(f"   Pack size: {self.C_pack}, Cells: {self.M_cells}")
    
    def get_dataset_info(self):
        """Get T2I-10M dataset information."""
        return {
            'name': self.dataset_name,
            'description': 'T2I-10M text-to-image retrieval dataset',
            'feature_dim': self.feature_dim,
            'train_size': 10_000_000,      # 10M images + 10M text queries
            'val_size': 10_000,            # 10k test queries
            'total_size': 20_010_000,      # Total embeddings
            'feature_type': 'CLIP',
            'normalized': True,
            'has_exact_gt': True,
            'recommended_backend': 'hnsw',  # HNSW works well for T2I-10M
            'recommended_budgets': self.budgets,
            'memory_requirements_gb': 80,   # Approximate memory needed
        }
    
    def get_default_data_path(self):
        """Get default data path for T2I-10M dataset."""
        return "/ssd/hamed/ann/t2i-10M/precompute/t2i_cache_up_to_0.pkl"
    
    def validate_config(self):
        """Validate T2I-10M specific configuration."""
        super().validate_config()
        
        # T2I-10M specific validations
        if self.feature_dim != 200:
            print(f"⚠️ Warning: T2I-10M typically uses 200-dim features, got {self.feature_dim}")
        
        if self.M_cells < 1024:
            print(f"⚠️ Warning: T2I-10M benefits from more cells, consider M_cells >= 1024")
        
        if self.batch_size < 4096:
            print(f"⚠️ Warning: T2I-10M benefits from larger batch sizes, consider batch_size >= 4096")
        
        if self.C_pack < 128:
            print(f"⚠️ Warning: T2I-10M benefits from larger pack sizes, consider C_pack >= 128")
        
        print("✅ T2I-10M configuration validation passed")
    
    @classmethod
    def get_recommended_config(cls, dataset_size: int) -> dict:
        """Get T2I-10M-specific recommended configuration based on dataset size."""
        # T2I-10M is always large (10M+), so we return the same recommendations
        return {
            'ivf_nlist': 4096,
            'M_cells': 4096,
            'hnsw_M': 32,
            'hnsw_efC': 200,
            'batch_size': 8192,
            'C_pack': 256,
            'epochs': 8,
            'lr': 1e-3,
            'budgets': tuple(range(10, 101, 10)),
        }

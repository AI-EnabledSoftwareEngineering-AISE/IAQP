"""
Base configuration for text-to-image retrieval training.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BaseConfig:
    """Base configuration class with common parameters."""
    
    # Data
    data_path: str = "data/cache.pkl"
    indices_dir: str = "indices/"
    
    # Model
    alpha: float = 0.25
    hidden: Optional[int] = None  # default = dim
    
    # Training
    device: str = "cuda"
    epochs: int = 8
    batch_size: int = 8192
    lr: float = 1e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    print_every: int = 1
    
    # Candidate pack
    C_pack: int = 256
    K_exact_in_pickle: int = 100
    
    # Teachers
    kl_tau: float = 0.07
    ann_tau: float = 0.07
    
    # Losses (4 core losses only)
    w_kld: float = 1.0
    w_id: float = 1e-3
    w_gap: float = 0.05
    w_cell: float = 0.3
    k_eval_inside_pack: int = 10
    gap_margin: float = 0.03
    
    # Smart identity loss configuration
    use_smart_identity: bool = True
    identity_mode: str = "cone"  # "cone", "legacy", "barycentric"
    
    # New heads and losses
    M_cells: int = 4096
    enable_quantity_head: bool = False
    
    # EM recomputation control
    em_recompute_every: int = 0
    
    # Budgets
    budgets: Tuple[int, ...] = tuple(range(10, 101, 10))
    
    # ANN backend choice for training
    backend: str = "hnsw"  # 'hnsw' or 'ivf'
    
    # HNSW build
    hnsw_M: int = 32
    hnsw_efC: int = 200
    
    # IVF build
    ivf_nlist: int = 4096
    
    # Evaluation
    eval_split: str = "val"
    eval_topk: int = 10
    eval_ph: bool = False
    eval_r5: bool = False
    
    # Misc
    seed: int = 42
    save_path: Optional[str] = "outputs/projector.pt"
    
    # ANN cache
    ann_cache_enable: bool = True
    ann_khead: int = 256
    
    # Test-only evaluation
    test_only: bool = False
    model_path: Optional[str] = None
    
    # Memory management
    pack_chunk_size: int = 8192
    pack_x_batch_size: int = 25000
    force_cpu_eval: bool = False
    eval_timeout: int = 300
    
    # CPU threading
    num_threads: int = 0
    
    # Loss analysis
    loss_analysis: bool = False
    loss_analysis_file: str = "loss_analysis.json"
    
    # Evaluation control
    skip_eval: bool = False
    
    # Dataset type (will be set by dataset-specific configs)
    dataset: str = "laion"



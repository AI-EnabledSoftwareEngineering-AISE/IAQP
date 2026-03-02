"""
Configuration factory for creating dataset-specific configurations.
"""
from typing import Dict, Any
from .base import BaseConfig
from .laion import LaionConfig
from .coco import CocoConfig
from .flickr30k import Flickr30KConfig
from .t2i import T2IConfig


def create_config(dataset_type: str, **kwargs) -> BaseConfig:
    """
    Create a dataset-specific configuration.
    
    Args:
        dataset_type: Type of dataset ('laion', 'coco', etc.)
        **kwargs: Additional configuration parameters to override
        
    Returns:
        Dataset-specific configuration instance
        
    Raises:
        ValueError: If dataset_type is not supported
    """
    config_classes = {
        'laion': LaionConfig,
        'coco': CocoConfig,
        'flickr30k': Flickr30KConfig,
        't2i': T2IConfig,
    }
    
    if dataset_type.lower() not in config_classes:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                        f"Supported types: {list(config_classes.keys())}")
    
    config_class = config_classes[dataset_type.lower()]
    
    # Create configuration instance
    config = config_class(**kwargs)
    
    return config


def apply_dataset_recommendations(config: BaseConfig, dataset_size: int) -> BaseConfig:
    """
    Apply dataset-specific recommendations to the configuration.
    
    Args:
        config: Base configuration instance
        dataset_size: Size of the dataset
        
    Returns:
        Updated configuration with dataset-specific recommendations
    """
    if config.dataset == "coco":
        recommendations = CocoConfig.get_recommended_config(dataset_size)
        eval_config = CocoConfig.get_evaluation_config()
        recommendations.update(eval_config)
    elif config.dataset == "laion":
        recommendations = LaionConfig.get_recommended_config(dataset_size)
    elif config.dataset == "flickr30k":
        recommendations = Flickr30KConfig.get_recommended_config(dataset_size)
    elif config.dataset == "t2i":
        recommendations = T2IConfig.get_recommended_config(dataset_size)
    else:
        return config
    
    # Apply recommendations (only if not explicitly set by user)
    for key, value in recommendations.items():
        if hasattr(config, key):
            # Only update if it's a default value (not explicitly set by user)
            current_value = getattr(config, key)
            default_value = getattr(config.__class__, key)
            if current_value == default_value:
                setattr(config, key, value)
    
    return config

# Dataset Loader System

This directory contains the dataset loader system for text-to-image retrieval training. The system is designed to be modular and extensible, making it easy to add support for new datasets.

## Architecture

### Base Classes

- **`BaseDatasetLoader`**: Abstract base class that defines the interface all dataset loaders must implement
- **`LaionDatasetLoader`**: Implementation for LAION datasets
- **`CocoDatasetLoader`**: Example implementation for COCO datasets

### Key Methods

All dataset loaders must implement these methods from `BaseDatasetLoader`:

- `load_data()`: Load dataset from cache file
- `validate_cache()`: Validate dataset structure
- `get_train_data()`: Get training data (images, texts, KNN indices)
- `get_split_data()`: Get validation/test data
- `get_feature_dim()`: Get feature dimension
- `get_dataset_info()`: Get dataset metadata

## Usage

### Using Existing Dataset Loaders

```python
from projector.dataset_loader import LaionDatasetLoader

# Load LAION dataset
loader = LaionDatasetLoader("/path/to/laion_cache.pkl")
X_train, T_train, knn_indices = loader.get_train_data()
dim = loader.get_feature_dim()
info = loader.get_dataset_info()
```

### Adding a New Dataset

1. **Create a new loader class** that inherits from `BaseDatasetLoader`:

```python
from .base import BaseDatasetLoader

class MyDatasetLoader(BaseDatasetLoader):
    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path, **kwargs)
        # Add dataset-specific initialization
    
    def load_data(self):
        # Implement data loading logic
        pass
    
    def validate_cache(self, data):
        # Implement validation logic
        pass
    
    # ... implement other required methods
```

2. **Add the loader to `__init__.py`**:

```python
from .my_dataset import MyDatasetLoader
__all__ = ['BaseDatasetLoader', 'LaionDatasetLoader', 'MyDatasetLoader']
```

3. **Update the trainer** to use your dataset loader (if needed):

```python
from .dataset_loader import MyDatasetLoader

# In train_budget_aware function:
dataset_loader = MyDatasetLoader(cfg.data_path)
```

## Dataset Format

All datasets should follow this structure in the pickle cache:

```python
{
    'train': {
        'image_features': np.ndarray,  # [N_images, D] CLIP image features
        'text_features': np.ndarray,   # [N_texts, D] CLIP text features  
        'knn_indices': np.ndarray,     # [N_texts, K] exact KNN indices
    },
    'val': {
        'text_features': np.ndarray,   # [N_val_texts, D] CLIP text features
        'knn_indices': np.ndarray,     # [N_val_texts, K] exact KNN indices
        'text_to_image': np.ndarray,   # [N_val_texts] optional image mapping
    },
    'test': {
        # Same structure as val
    }
}
```

## Features

- **Modular Design**: Easy to add new datasets without modifying existing code
- **Type Safety**: Abstract base class ensures consistent interface
- **Validation**: Built-in dataset structure validation
- **Metadata**: Rich dataset information and statistics
- **Extensibility**: Support for dataset-specific parameters and preprocessing

## Examples

### LAION Dataset
```python
loader = LaionDatasetLoader("/path/to/laion_cache.pkl")
```

### COCO Dataset (with custom parameters)
```python
loader = CocoDatasetLoader(
    "/path/to/coco_cache.pkl",
    image_size=224,
    text_preprocessing="clip"
)
```

## Future Extensions

The system is designed to support:
- Different feature extractors (not just CLIP)
- Custom preprocessing pipelines
- Streaming data loading for very large datasets
- Multi-modal datasets with additional modalities
- Dataset-specific augmentation strategies



# Configuration package for text-to-image retrieval training
from .base import BaseConfig
from .laion import LaionConfig
from .coco import CocoConfig

__all__ = ['BaseConfig', 'LaionConfig', 'CocoConfig']



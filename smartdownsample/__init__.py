"""
SmartDownsample: Intelligent image downsampling for camera traps and large datasets.

Smart downsampling library that selects the most diverse images from large datasets,
perfect for camera trap data and machine learning workflows.
"""

from .core import sample_diverse

__version__ = "1.0.1"
__author__ = "Smart Image Downsampler"
__email__ = "your.email@example.com"

__all__ = ["sample_diverse"]
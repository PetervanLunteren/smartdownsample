"""
SmartDownsample: Intelligent image downsampling for camera traps and large datasets.

Smart downsampling library that selects the most diverse images from large datasets,
perfect for camera trap data and machine learning workflows.
"""

from .core import sample_diverse

__version__ = "1.9.2"
__author__ = "Peter van Lunteren"

__all__ = ["sample_diverse"]
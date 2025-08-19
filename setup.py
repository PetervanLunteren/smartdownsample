"""
Setup script for smartdownsample package.
This is a fallback for environments that don't support pyproject.toml.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="smartdownsample",
    version="0.1.0",
    author="Smart Image Downsampler",
    author_email="your.email@example.com",
    description="Intelligent image downsampling for camera traps and large datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PetervanLunteren/smartdownsample",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "imagehash>=4.2.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "natsort>=8.0.0",
        "matplotlib>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
    },
    keywords="image downsampling camera-trap machine-learning computer-vision diversity deduplication",
)
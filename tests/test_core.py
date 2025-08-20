#!/usr/bin/env python3
"""
Tests for smartdownsample core functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np

from smartdownsample import sample_diverse


class TestSmartDownsample:
    """Test suite for smart downsampling functionality."""
    
    @pytest.fixture
    def temp_images(self):
        """Create temporary test images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_paths = []
            
            # Create 10 test images with different patterns
            for i in range(10):
                img_path = Path(temp_dir) / f"test_image_{i:02d}.jpg"
                
                # Create simple test image with different colors
                img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                # Make each image slightly different
                img_array[i*10:(i+1)*10, :, :] = 255  # White stripe at different positions
                
                img = Image.fromarray(img_array)
                img.save(img_path)
                image_paths.append(str(img_path))
            
            yield image_paths
    
    def test_select_distinct_basic(self, temp_images):
        """Test basic functionality."""
        target_count = 5
        selected = sample_diverse(
            image_paths=temp_images,
            target_count=target_count,
            show_progress=False
        )
        
        assert len(selected) == target_count
        assert all(isinstance(path, str) for path in selected)
        assert all(Path(path).exists() for path in selected)
    
    def test_select_distinct_exact_count(self, temp_images):
        """Test that exact count is always returned."""
        for target in [1, 3, 5, 8, 10]:
            selected = sample_diverse(
                image_paths=temp_images,
                target_count=target,
                show_progress=False
            )
            assert len(selected) == target
    
    def test_select_distinct_target_larger_than_input(self, temp_images):
        """Test behavior when target is larger than input."""
        target_count = 20  # More than the 10 available images
        selected = sample_diverse(
            image_paths=temp_images,
            target_count=target_count,
            show_progress=False
        )
        
        # Should return all available images
        assert len(selected) == len(temp_images)
    
    def test_select_distinct_hash_size(self, temp_images):
        """Test different hash sizes."""
        selected = sample_diverse(
            image_paths=temp_images,
            target_count=5,
            hash_size=4,
            show_progress=False
        )
        
        assert len(selected) == 5
    
    
    def test_select_distinct_reproducible(self, temp_images):
        """Test that results are reproducible with same seed."""
        target_count = 5
        seed = 42
        
        selected1 = sample_diverse(
            image_paths=temp_images,
            target_count=target_count,
            random_seed=seed,
            show_progress=False
        )
        
        selected2 = sample_diverse(
            image_paths=temp_images,
            target_count=target_count,
            random_seed=seed,
            show_progress=False
        )
        
        # Results should be identical with same seed
        assert selected1 == selected2
    
    def test_select_distinct_different_seeds(self, temp_images):
        """Test that different seeds give different results."""
        target_count = 5
        
        selected1 = sample_diverse(
            image_paths=temp_images,
            target_count=target_count,
            random_seed=42,
            show_progress=False
        )
        
        selected2 = sample_diverse(
            image_paths=temp_images,
            target_count=target_count,
            random_seed=99,
            show_progress=False
        )
        
        # Results should be different with different seeds
        # (Note: there's a small chance they could be the same, but very unlikely)
        assert selected1 != selected2
    
    
    def test_select_distinct_empty_list(self):
        """Test behavior with empty image list."""
        selected = sample_diverse(
            image_paths=[],
            target_count=5,
            show_progress=False
        )
        
        assert len(selected) == 0
    
    def test_select_distinct_single_image(self, temp_images):
        """Test behavior with single image."""
        single_image = temp_images[:1]
        
        selected = sample_diverse(
            image_paths=single_image,
            target_count=1,
            show_progress=False
        )
        
        assert len(selected) == 1
        assert selected[0] == single_image[0]
    
    def test_select_distinct_path_objects(self, temp_images):
        """Test that Path objects work as input."""
        path_objects = [Path(p) for p in temp_images]
        
        selected = sample_diverse(
            image_paths=path_objects,
            target_count=5,
            show_progress=False
        )
        
        assert len(selected) == 5
        assert all(isinstance(path, str) for path in selected)  # Should return strings
    
    def test_select_distinct_hash_sizes(self, temp_images):
        """Test different hash sizes."""
        for hash_size in [4, 6, 8, 10]:
            selected = sample_diverse(
                image_paths=temp_images,
                target_count=5,
                hash_size=hash_size,
                show_progress=False
            )
            
            assert len(selected) == 5
    
    def test_select_distinct_n_workers(self, temp_images):
        """Test that n_workers parameter works without errors."""
        selected = sample_diverse(
            image_paths=temp_images,
            target_count=5,
            n_workers=2,
            show_progress=False
        )
        
        assert len(selected) == 5


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
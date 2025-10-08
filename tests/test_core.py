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
import io
import sys
import shutil

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
    
    def test_invalid_image_paths_skip_mode(self, temp_images, capsys):
        """Test error reporting for invalid image paths with skip mode."""
        # Add some invalid paths to the list
        invalid_paths = temp_images + [
            "/nonexistent/path/image1.jpg",
            "/nonexistent/path/image2.jpg",
            "invalid_file.jpg"
        ]
        
        selected = sample_diverse(
            image_paths=invalid_paths,
            target_count=5,
            image_loading_errors="skip",  # Use skip mode
            show_progress=True  # Enable progress to see detailed warnings
        )
        
        # Should still work with valid images
        assert len(selected) == 5
        
        # Check that warnings were printed
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "could not be loaded" in captured.out
        assert "File not found" in captured.out
    
    def test_invalid_image_paths_raise_mode(self, temp_images):
        """Test that errors are raised in raise mode."""
        # Add an invalid path to the list
        invalid_paths = temp_images + [
            "/nonexistent/path/image1.jpg"
        ]
        
        # Should raise an error in raise mode (default)
        with pytest.raises(FileNotFoundError, match="File not found.*Use image_loading_errors='skip'"):
            selected = sample_diverse(
                image_paths=invalid_paths,
                target_count=5,
                image_loading_errors="raise",  # Explicitly set raise mode
                show_progress=False
            )
    
    def test_corrupt_image_handling_skip(self, capsys):
        """Test handling of corrupt image files in skip mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid image
            valid_img_path = Path(temp_dir) / "valid.jpg"
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(valid_img_path)
            
            # Create corrupt image file (not actually an image)
            corrupt_img_path = Path(temp_dir) / "corrupt.jpg"
            with open(corrupt_img_path, 'w') as f:
                f.write("This is not an image file")
            
            image_paths = [str(valid_img_path), str(corrupt_img_path)]
            
            selected = sample_diverse(
                image_paths=image_paths,
                target_count=1,
                image_loading_errors="skip",  # Skip mode for corrupt images
                show_progress=True
            )
            
            # Should return the one valid image
            assert len(selected) == 1
            assert selected[0] == str(valid_img_path)
            
            # Check that error was reported
            captured = capsys.readouterr()
            assert "Warning" in captured.out
            assert "Image processing failed" in captured.out
    
    def test_all_invalid_images_skip(self, capsys):
        """Test behavior when all images are invalid in skip mode."""
        invalid_paths = [
            "/nonexistent/path/image1.jpg",
            "/nonexistent/path/image2.jpg",
            "/nonexistent/path/image3.jpg"
        ]
        
        selected = sample_diverse(
            image_paths=invalid_paths,
            target_count=2,
            image_loading_errors="skip",  # Skip invalid images
            show_progress=True
        )
        
        # Should return empty list when no valid images
        assert len(selected) == 0
        
        # Check that warnings were printed
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "could not be loaded" in captured.out
    
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
    
    def test_image_loading_errors_invalid_value(self, temp_images):
        """Test that invalid image_loading_errors value raises ValueError."""
        with pytest.raises(ValueError, match="image_loading_errors must be 'raise' or 'skip'"):
            sample_diverse(
                image_paths=temp_images,
                target_count=5,
                image_loading_errors="invalid_value",
                show_progress=False
            )
    
    def test_image_loading_errors_default_is_raise(self, temp_images):
        """Test that default behavior is to raise errors."""
        invalid_paths = temp_images + ["/nonexistent/path.jpg"]
        
        # Default should raise
        with pytest.raises(FileNotFoundError):
            sample_diverse(
                image_paths=invalid_paths,
                target_count=5,
                show_progress=False
                # image_loading_errors not specified, should default to "raise"
            )
    
    def test_permission_error_handling(self):
        """Test handling of permission errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid image
            valid_img_path = Path(temp_dir) / "valid.jpg"
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(valid_img_path)
            
            # Create a file with no read permissions (simulate permission error)
            no_perm_path = Path(temp_dir) / "no_permission.jpg"
            img.save(no_perm_path)
            os.chmod(no_perm_path, 0o000)  # Remove all permissions
            
            try:
                # Test raise mode
                with pytest.raises((PermissionError, OSError), match=".*Use image_loading_errors='skip'.*"):
                    sample_diverse(
                        image_paths=[str(valid_img_path), str(no_perm_path)],
                        target_count=1,
                        image_loading_errors="raise",
                        show_progress=False
                    )
                
                # Test skip mode
                selected = sample_diverse(
                    image_paths=[str(valid_img_path), str(no_perm_path)],
                    target_count=1,
                    image_loading_errors="skip",
                    show_progress=False
                )
                assert len(selected) == 1
                assert selected[0] == str(valid_img_path)
            finally:
                # Restore permissions for cleanup
                try:
                    os.chmod(no_perm_path, 0o644)
                except:
                    pass
    
    def test_compensate_for_skipped_images(self, temp_images, capsys):
        """Test that target_count is still met when images are skipped."""
        # Mix valid and invalid paths
        invalid_paths = [
            "/nonexistent/img1.jpg",
            "/nonexistent/img2.jpg",
            "/nonexistent/img3.jpg"
        ]
        all_paths = temp_images + invalid_paths
        
        # Request 8 images, should get 8 from the 10 valid ones
        selected = sample_diverse(
            image_paths=all_paths,
            target_count=8,
            image_loading_errors="skip",
            show_progress=True
        )
        
        # Should still return requested count from valid images
        assert len(selected) == 8
        assert all(path in temp_images for path in selected)
        
        # Check warnings were shown
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "3 image(s) could not be loaded" in captured.out
    
    def test_save_distribution_valid_path(self, temp_images):
        """Test saving distribution chart to valid PNG path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_distribution.png"
            
            selected = sample_diverse(
                image_paths=temp_images,
                target_count=5,
                save_distribution=str(output_path),
                show_progress=False
            )
            
            assert len(selected) == 5
            assert output_path.exists()
            assert output_path.stat().st_size > 0  # File has content
    
    def test_save_thumbnails_valid_path(self, temp_images):
        """Test saving thumbnail grids to valid PNG path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_thumbnails.png"
            
            selected = sample_diverse(
                image_paths=temp_images,
                target_count=5,
                save_thumbnails=str(output_path),
                show_progress=False
            )
            
            assert len(selected) == 5
            assert output_path.exists()
            assert output_path.stat().st_size > 0  # File has content
    
    def test_save_with_nested_directories(self, temp_images):
        """Test that nested directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "dirs" / "chart.png"
            
            selected = sample_diverse(
                image_paths=temp_images,
                target_count=5,
                save_distribution=str(output_path),
                show_progress=False
            )
            
            assert len(selected) == 5
            assert output_path.exists()
            assert output_path.parent.exists()
    
    def test_save_invalid_extension(self, temp_images):
        """Test that non-PNG extensions raise error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test .jpg extension
            with pytest.raises(ValueError, match="save_distribution must be a .png file"):
                sample_diverse(
                    image_paths=temp_images,
                    target_count=5,
                    save_distribution=str(Path(temp_dir) / "test.jpg"),
                    show_progress=False
                )
            
            # Test no extension
            with pytest.raises(ValueError, match="save_distribution must be a .png file"):
                sample_diverse(
                    image_paths=temp_images,
                    target_count=5,
                    save_distribution=str(Path(temp_dir) / "test"),
                    show_progress=False
                )
    
    def test_save_file_already_exists(self, temp_images):
        """Test that existing files raise error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "existing.png"
            # Create existing file
            output_path.write_text("existing content")
            
            with pytest.raises(ValueError, match="File already exists"):
                sample_diverse(
                    image_paths=temp_images,
                    target_count=5,
                    save_distribution=str(output_path),
                    show_progress=False
                )
    
    def test_save_both_charts(self, temp_images):
        """Test saving both distribution and thumbnails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dist_path = Path(temp_dir) / "distribution.png"
            thumb_path = Path(temp_dir) / "thumbnails.png"
            
            selected = sample_diverse(
                image_paths=temp_images,
                target_count=5,
                save_distribution=str(dist_path),
                save_thumbnails=str(thumb_path),
                show_progress=False
            )
            
            assert len(selected) == 5
            assert dist_path.exists()
            assert thumb_path.exists()
            # Thumbnails should be larger than distribution chart
            assert thumb_path.stat().st_size > dist_path.stat().st_size
    
    def test_boolean_value_error(self, temp_images):
        """Test that boolean values raise helpful error (for users migrating from old API)."""
        # Test True for save_distribution
        with pytest.raises(ValueError, match="must be a file path string or None, not a boolean"):
            sample_diverse(
                image_paths=temp_images,
                target_count=5,
                save_distribution=True,  # Old API style
                show_progress=False
            )
        
        # Test False for save_thumbnails
        with pytest.raises(ValueError, match="must be a file path string or None, not a boolean"):
            sample_diverse(
                image_paths=temp_images,
                target_count=5,
                save_thumbnails=False,  # Old API style
                show_progress=False
            )
    
    def test_return_indices_basic(self, temp_images):
        """Test basic functionality of return_indices parameter."""
        target_count = 5
        
        # Get paths (default behavior)
        selected_paths = sample_diverse(
            image_paths=temp_images,
            target_count=target_count,
            return_indices=False,
            show_progress=False
        )
        
        # Get indices
        selected_indices = sample_diverse(
            image_paths=temp_images,
            target_count=target_count,
            return_indices=True,
            show_progress=False
        )
        
        assert len(selected_paths) == target_count
        assert len(selected_indices) == target_count
        assert all(isinstance(idx, int) for idx in selected_indices)
        assert all(0 <= idx < len(temp_images) for idx in selected_indices)
        
        # Verify indices correspond to correct paths
        for path, idx in zip(selected_paths, selected_indices):
            assert temp_images[idx] == path
    
    def test_return_indices_all_images(self, temp_images):
        """Test return_indices when target_count >= available images."""
        target_count = 20  # More than the 10 available images
        
        selected_indices = sample_diverse(
            image_paths=temp_images,
            target_count=target_count,
            return_indices=True,
            show_progress=False
        )
        
        # Should return all indices in order [0, 1, 2, ..., 9]
        assert selected_indices == list(range(len(temp_images)))
    
    def test_return_indices_empty_target(self, temp_images):
        """Test return_indices with target_count=0."""
        selected_indices = sample_diverse(
            image_paths=temp_images,
            target_count=0,
            return_indices=True,
            show_progress=False
        )
        
        assert selected_indices == []
    
    def test_return_indices_with_skipped_images(self, temp_images):
        """Test return_indices when some images are skipped."""
        # Mix valid and invalid paths
        invalid_paths = [
            "/nonexistent/img1.jpg",
            "/nonexistent/img2.jpg"
        ]
        mixed_paths = temp_images[:5] + invalid_paths + temp_images[5:]
        
        # Get indices with skip mode
        selected_indices = sample_diverse(
            image_paths=mixed_paths,
            target_count=5,
            return_indices=True,
            image_loading_errors="skip",
            show_progress=False
        )
        
        assert len(selected_indices) == 5
        assert all(isinstance(idx, int) for idx in selected_indices)
        
        # All returned indices should refer to valid images in original list
        for idx in selected_indices:
            assert 0 <= idx < len(mixed_paths)
            # Check that the path at this index is one of the valid ones
            assert mixed_paths[idx] in temp_images
    
    def test_return_indices_reproducible(self, temp_images):
        """Test that return_indices gives reproducible results."""
        target_count = 5
        seed = 42
        
        indices1 = sample_diverse(
            image_paths=temp_images,
            target_count=target_count,
            return_indices=True,
            random_seed=seed,
            show_progress=False
        )
        
        indices2 = sample_diverse(
            image_paths=temp_images,
            target_count=target_count,
            return_indices=True,
            random_seed=seed,
            show_progress=False
        )
        
        # Results should be identical with same seed
        assert indices1 == indices2


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
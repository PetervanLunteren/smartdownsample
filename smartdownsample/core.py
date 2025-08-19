"""
Core functionality for smart image downsampling.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union, Optional
import imagehash
import random
from tqdm import tqdm
import warnings
from natsort import natsorted
from PIL import Image

warnings.filterwarnings('ignore')


def select_distinct(
    image_paths: List[Union[str, Path]], 
    target_count: int,
    window_size: int = 100,
    random_seed: int = 42,
    show_progress: bool = True,
    show_verification: bool = False
) -> List[str]:
    """
    Select the most diverse/distinct images from a large dataset.
    
    Perfect for camera trap data, eliminating duplicate poses and similar images
    while maintaining temporal and visual diversity.
    
    Args:
        image_paths: List of paths to images (str or Path objects)
        target_count: Exact number of images to return
        window_size: Rolling window size for diversity comparison
        random_seed: Random seed for reproducible results
        show_progress: Whether to show progress bars
        show_verification: Whether to show visual verification of excluded images
        
    Returns:
        List of exactly target_count selected image paths as strings
        
    Examples:
        >>> from smartdownsample import select_distinct
        >>> 
        >>> # Basic usage - select 100 most diverse images
        >>> selected = select_distinct(image_paths, target_count=100)
        >>> 
        >>> # For large datasets (100k+ images) - adjust window size as needed
        >>> selected = select_distinct(
        ...     large_dataset_paths, 
        ...     target_count=1000,
        ...     window_size=100
        ... )
        >>> 
        >>> # With visual verification of excluded images
        >>> selected = select_distinct(
        ...     image_paths,
        ...     target_count=100,
        ...     show_verification=True
        ... )
    """
    
    if target_count >= len(image_paths):
        if show_progress:
            print(f"Target count ({target_count}) >= input size ({len(image_paths)}). Returning all images.")
        return [str(p) for p in image_paths]
    
    return _select_distinct_rolling_window(
        image_paths, target_count, window_size, random_seed, show_progress, show_verification
    )


def _select_distinct_rolling_window(
    image_paths: List[Union[str, Path]], 
    target_count: int,
    window_size: int,
    random_seed: int,
    show_progress: bool,
    show_verification: bool
) -> List[str]:
    """Rolling window approach - scales to 100k+ images."""
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    if len(image_paths) <= target_count:
        if show_progress:
            print(f"Input has {len(image_paths)} images, target is {target_count}. Returning all images.")
        return [str(p) for p in image_paths]
    
    if show_progress:
        print(f"Selecting {target_count} most diverse images from {len(image_paths)} using rolling window (size: {window_size})...")
    
    # Sort paths by directory structure for logical ordering
    if show_progress:
        print("Sorting images by directory structure...")
    sorted_paths = _sort_paths_by_directory(image_paths)
    
    # Calculate perceptual hashes
    if show_progress:
        print("Calculating perceptual hashes...")
    hashes, valid_paths = _calculate_hashes(sorted_paths, show_progress)
    
    if len(valid_paths) <= target_count:
        if show_progress:
            print(f"Only {len(valid_paths)} valid images found. Returning all.")
        return valid_paths
    
    # Convert to binary arrays
    hash_arrays = np.array([_hash_to_binary_array(h) for h in hashes])
    
    # Rolling window selection
    if show_progress:
        print("Selecting most diverse images with rolling window...")
    selected_indices = _rolling_window_selection(hash_arrays, target_count, window_size, show_progress)
    
    selected_paths = [valid_paths[i] for i in selected_indices]
    
    if show_progress:
        print(f"Selected exactly {len(selected_paths)} most diverse images")
    
    # Show verification plot if requested
    if show_verification and len(valid_paths) > target_count:
        from .visualization import create_verification_image
        create_verification_image(valid_paths, selected_indices, random_seed)
    
    return selected_paths




def _sort_paths_by_directory(image_paths: List[Union[str, Path]]) -> List[str]:
    """Sort image paths by directory structure and filename using natural sorting.
    
    This keeps images in the same directory together and sorts them naturally
    (e.g., img1.jpg, img2.jpg, img10.jpg instead of img1.jpg, img10.jpg, img2.jpg).
    """
    path_objects = [Path(p) for p in image_paths]
    
    # Group by directory, then natural sort within each directory
    paths_by_dir = {}
    for path in path_objects:
        dir_key = str(path.parent)
        if dir_key not in paths_by_dir:
            paths_by_dir[dir_key] = []
        paths_by_dir[dir_key].append(path)
    
    # Natural sort directories and files within each directory
    sorted_paths = []
    for directory in natsorted(paths_by_dir.keys()):
        dir_paths = paths_by_dir[directory]
        # Natural sort files within directory
        sorted_dir_paths = natsorted(dir_paths, key=lambda p: p.name)
        sorted_paths.extend(sorted_dir_paths)
    
    return [str(p) for p in sorted_paths]


def _calculate_hashes(image_paths: List[Union[str, Path]], show_progress: bool) -> tuple:
    """Calculate perceptual hashes for all images."""
    hashes = []
    valid_paths = []
    
    iterator = tqdm(image_paths, desc="Computing hashes") if show_progress else image_paths
    
    for path in iterator:
        try:
            with Image.open(path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                hash_val = imagehash.phash(img, hash_size=16)
                hashes.append(hash_val)
                valid_paths.append(str(path))
        except Exception as e:
            if show_progress:
                print(f"Error processing {path}: {e}")
            continue
    
    if show_progress:
        print(f"Successfully processed {len(valid_paths)} images")
    
    return hashes, valid_paths


def _hash_to_binary_array(hash_obj) -> np.ndarray:
    """Convert ImageHash object to binary numpy array."""
    hex_str = str(hash_obj)
    binary_array = []
    for hex_char in hex_str:
        binary_bits = format(int(hex_char, 16), '04b')
        binary_array.extend([int(bit) for bit in binary_bits])
    return np.array(binary_array).astype(np.uint8)


def _rolling_window_selection(hash_arrays: np.ndarray, target_count: int, window_size: int, show_progress: bool) -> List[int]:
    """Rolling window algorithm for large datasets."""
    n_images = len(hash_arrays)
    
    if target_count >= n_images:
        return list(range(n_images))
    
    selected_indices = []
    remaining_indices = set(range(n_images))
    
    # Start with random image
    first_idx = random.choice(list(remaining_indices))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Rolling window selection
    iterations_needed = max(0, target_count - 1)
    iterator = tqdm(total=iterations_needed, desc="Rolling window selection") if show_progress else range(iterations_needed)
    
    for _ in iterator:
        if len(remaining_indices) == 0:
            break
            
        # Define rolling window
        window_start = max(0, len(selected_indices) - window_size)
        window_indices = selected_indices[window_start:]
        
        max_min_distance = -1
        best_candidate = None
        
        # Find most distant candidate from window
        for candidate_idx in remaining_indices:
            min_distance_to_window = float('inf')
            
            for window_idx in window_indices:
                distance = _hamming_distance(hash_arrays[candidate_idx], hash_arrays[window_idx])
                min_distance_to_window = min(min_distance_to_window, distance)
            
            if min_distance_to_window > max_min_distance:
                max_min_distance = min_distance_to_window
                best_candidate = candidate_idx
        
        if best_candidate is not None:
            selected_indices.append(best_candidate)
            remaining_indices.remove(best_candidate)
        
        if show_progress and hasattr(iterator, 'update'):
            iterator.update(1)
    
    return selected_indices





def _hamming_distance(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Calculate Hamming distance between two binary arrays."""
    return np.sum(arr1 != arr2) / len(arr1)
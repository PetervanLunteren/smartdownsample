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

warnings.filterwarnings('ignore')


def select_distinct(
    image_paths: List[Union[str, Path]], 
    target_count: int,
    method: str = "rolling_window",
    window_size: int = 100,
    similarity_threshold: float = 0.85,
    random_seed: int = 42,
    show_progress: bool = True
) -> List[str]:
    """
    Select the most diverse/distinct images from a large dataset.
    
    Perfect for camera trap data, eliminating duplicate poses and similar images
    while maintaining temporal and visual diversity.
    
    Args:
        image_paths: List of paths to images (str or Path objects)
        target_count: Exact number of images to return
        method: Algorithm to use ("rolling_window" for large datasets, "exact" for smaller ones)
        window_size: Rolling window size for diversity comparison (only for rolling_window method)
        similarity_threshold: Similarity threshold for clustering (only for exact method)
        random_seed: Random seed for reproducible results
        show_progress: Whether to show progress bars
        
    Returns:
        List of exactly target_count selected image paths as strings
        
    Examples:
        >>> from smartdownsample import select_distinct
        >>> 
        >>> # Basic usage - select 100 most diverse images
        >>> selected = select_distinct(image_paths, target_count=100)
        >>> 
        >>> # For large datasets (100k+ images) - use rolling window (default)
        >>> selected = select_distinct(
        ...     large_dataset_paths, 
        ...     target_count=1000,
        ...     method="rolling_window",
        ...     window_size=100
        ... )
        >>> 
        >>> # For smaller datasets - use exact algorithm for optimal results  
        >>> selected = select_distinct(
        ...     small_dataset_paths,
        ...     target_count=50, 
        ...     method="exact"
        ... )
    """
    
    if method == "rolling_window":
        return _select_distinct_rolling_window(
            image_paths, target_count, window_size, random_seed, show_progress
        )
    elif method == "exact":
        return _select_distinct_exact(
            image_paths, target_count, similarity_threshold, random_seed, show_progress
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'rolling_window' or 'exact'")


def _select_distinct_rolling_window(
    image_paths: List[Union[str, Path]], 
    target_count: int,
    window_size: int,
    random_seed: int,
    show_progress: bool
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
    
    return selected_paths


def _select_distinct_exact(
    image_paths: List[Union[str, Path]], 
    target_count: int,
    similarity_threshold: float,
    random_seed: int,
    show_progress: bool
) -> List[str]:
    """Exact greedy approach - optimal for smaller datasets."""
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    if len(image_paths) <= target_count:
        if show_progress:
            print(f"Input has {len(image_paths)} images, target is {target_count}. Returning all images.")
        return [str(p) for p in image_paths]
    
    if show_progress:
        print(f"Selecting exactly {target_count} most diverse images from {len(image_paths)}...")
    
    # Calculate hashes
    if show_progress:
        print("Calculating perceptual hashes...")
    hashes, valid_paths = _calculate_hashes(image_paths, show_progress)
    
    if len(valid_paths) <= target_count:
        if show_progress:
            print(f"Only {len(valid_paths)} valid images found. Returning all.")
        return valid_paths
    
    # Convert to binary arrays
    hash_arrays = np.array([_hash_to_binary_array(h) for h in hashes])
    
    # Exact greedy selection
    if show_progress:
        print("Selecting most diverse images...")
    selected_indices = _exact_greedy_selection(hash_arrays, target_count, show_progress)
    
    selected_paths = [valid_paths[i] for i in selected_indices]
    
    if show_progress:
        print(f"Selected exactly {len(selected_paths)} most diverse images")
    
    return selected_paths


def _sort_paths_by_directory(image_paths: List[Union[str, Path]]) -> List[str]:
    """Sort image paths by directory structure and filename."""
    path_objects = [Path(p) for p in image_paths]
    sorted_paths = sorted(path_objects, key=lambda p: (str(p.parent), p.name))
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
    iterator = tqdm(total=target_count-1, desc="Rolling window selection") if show_progress else range(target_count-1)
    
    for _ in iterator:
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
        
        selected_indices.append(best_candidate)
        remaining_indices.remove(best_candidate)
        
        if show_progress and hasattr(iterator, 'update'):
            iterator.update(1)
    
    return selected_indices


def _exact_greedy_selection(hash_arrays: np.ndarray, target_count: int, show_progress: bool) -> List[int]:
    """Exact greedy algorithm for optimal results."""
    n_images = len(hash_arrays)
    
    if target_count >= n_images:
        return list(range(n_images))
    
    selected_indices = []
    remaining_indices = set(range(n_images))
    
    # Start with random image
    first_idx = random.choice(list(remaining_indices))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Greedy selection
    iterator = tqdm(total=target_count-1, desc="Selecting diverse images") if show_progress else range(target_count-1)
    
    for _ in iterator:
        max_min_distance = -1
        best_candidate = None
        
        for candidate_idx in remaining_indices:
            min_distance_to_selected = float('inf')
            
            for selected_idx in selected_indices:
                distance = _hamming_distance(hash_arrays[candidate_idx], hash_arrays[selected_idx])
                min_distance_to_selected = min(min_distance_to_selected, distance)
            
            if min_distance_to_selected > max_min_distance:
                max_min_distance = min_distance_to_selected
                best_candidate = candidate_idx
        
        selected_indices.append(best_candidate)
        remaining_indices.remove(best_candidate)
        
        if show_progress and hasattr(iterator, 'update'):
            iterator.update(1)
    
    return selected_indices


def _hamming_distance(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Calculate Hamming distance between two binary arrays."""
    return np.sum(arr1 != arr2) / len(arr1)
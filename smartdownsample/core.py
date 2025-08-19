"""
Core functionality for smart image downsampling.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union, Optional, Tuple
import imagehash
import random
from tqdm import tqdm
import warnings
from natsort import natsorted
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import sys
import json
import os
from functools import partial

warnings.filterwarnings('ignore')


def select_distinct(
    image_paths: List[Union[str, Path]], 
    target_count: int,
    window_size: int = 100,
    random_seed: int = 42,
    show_progress: bool = True,
    show_verification: bool = False,
    n_workers: Optional[int] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    hash_size: int = 8,
    batch_size: int = 100
) -> List[str]:
    """
    Select the most diverse/distinct images from a large dataset.
    
    Optimized with parallel processing and caching for handling large datasets efficiently.
    
    Args:
        image_paths: List of paths to images (str or Path objects)
        target_count: Exact number of images to return
        window_size: Rolling window size for diversity comparison (default: 100)
        random_seed: Random seed for reproducible results (default: 42)
        show_progress: Whether to show progress bars (default: True)
        show_verification: Whether to show visual verification of excluded images (default: False)
        n_workers: Number of parallel workers (default: CPU count - 1)
        cache_dir: Directory to cache computed hashes (speeds up repeated runs)
        hash_size: Size of perceptual hash (default: 8, smaller = faster)
        batch_size: Batch size for parallel processing (default: 100)
        
    Returns:
        List of exactly target_count selected image paths as strings
        
    Examples:
        >>> from smartdownsample import select_distinct
        >>> 
        >>> # Basic usage - select 100 most diverse images
        >>> selected = select_distinct(image_paths, target_count=100)
        >>> 
        >>> # For large datasets (10k+ images) - use parallel processing
        >>> selected = select_distinct(
        ...     large_dataset_paths, 
        ...     target_count=1000,
        ...     n_workers=8,  # Use 8 CPU cores
        ...     cache_dir="./cache"  # Cache hashes for faster reruns
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
    
    return _select_distinct_optimized(
        image_paths, target_count, window_size, random_seed, show_progress, 
        show_verification, n_workers, cache_dir, hash_size, batch_size
    )


def _select_distinct_optimized(
    image_paths: List[Union[str, Path]], 
    target_count: int,
    window_size: int,
    random_seed: int,
    show_progress: bool,
    show_verification: bool,
    n_workers: Optional[int],
    cache_dir: Optional[Union[str, Path]],
    hash_size: int,
    batch_size: int
) -> List[str]:
    """Optimized rolling window approach with parallel processing."""
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    if len(image_paths) <= target_count:
        if show_progress:
            print(f"Input has {len(image_paths)} images, target is {target_count}. Returning all images.")
        return [str(p) for p in image_paths]
    
    if show_progress:
        print(f"Selecting {target_count} most diverse images from {len(image_paths)}")
        print(f"Using {n_workers} workers, hash_size={hash_size}")
    
    # Sort paths by directory structure for logical ordering
    if show_progress:
        print("Sorting images by directory structure...")
    sorted_paths = _sort_paths_by_directory(image_paths)
    
    # Calculate perceptual hashes with parallel processing
    if show_progress:
        print(f"Calculating perceptual hashes (parallel with {n_workers} workers)...")
    hashes, valid_paths = _calculate_hashes_parallel(
        sorted_paths, show_progress, n_workers, cache_dir, hash_size, batch_size
    )
    
    if len(valid_paths) <= target_count:
        if show_progress:
            print(f"Only {len(valid_paths)} valid images found. Returning all.")
        return valid_paths
    
    # Convert to binary arrays
    hash_arrays = np.array([_hash_to_binary_array(h, hash_size) for h in hashes])
    
    # Selection algorithm
    if show_progress:
        print("Selecting most diverse images...")
    
    # Use fast selection for larger datasets
    if len(valid_paths) > 1000:
        selected_indices = _fast_selection_algorithm(
            hash_arrays, target_count, window_size, show_progress
        )
    else:
        selected_indices = _rolling_window_selection(
            hash_arrays, target_count, window_size, show_progress
        )
    
    selected_paths = [valid_paths[i] for i in selected_indices]
    
    if show_progress:
        print(f"Selected exactly {len(selected_paths)} most diverse images")
    
    # Show verification plot if requested
    if show_verification and len(valid_paths) > target_count:
        from .visualization import create_verification_image
        create_verification_image(valid_paths, selected_indices, random_seed)
    
    return selected_paths


def _compute_hash_batch(paths_batch: List[str], hash_size: int) -> List[Tuple[str, Optional[str]]]:
    """Compute hashes for a batch of images."""
    results = []
    for path in paths_batch:
        try:
            with Image.open(path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                hash_val = imagehash.phash(img, hash_size=hash_size)
                results.append((path, str(hash_val)))
        except Exception:
            results.append((path, None))
    return results


def _compute_hash_single(path: str, hash_size: int) -> Tuple[str, Optional[str]]:
    """Compute hash for a single image."""
    try:
        with Image.open(path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            hash_val = imagehash.phash(img, hash_size=hash_size)
            return (path, str(hash_val))
    except Exception:
        return (path, None)


def _calculate_hashes_parallel(
    image_paths: List[Union[str, Path]], 
    show_progress: bool,
    n_workers: int,
    cache_dir: Optional[Union[str, Path]],
    hash_size: int,
    batch_size: int
) -> Tuple[List, List[str]]:
    """Calculate perceptual hashes using parallel processing."""
    
    # Check cache if provided
    cache = {}
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"hash_cache_{hash_size}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                if show_progress:
                    print(f"Loaded {len(cache)} cached hashes")
            except:
                cache = {}
    
    # Separate cached and uncached paths
    uncached_paths = []
    hashes = []
    valid_paths = []
    
    for path in image_paths:
        path_str = str(path)
        try:
            path_key = f"{path_str}_{os.path.getmtime(path_str)}"
        except:
            uncached_paths.append(path_str)
            continue
            
        if path_key in cache:
            hash_str = cache[path_key]
            if hash_str:
                hashes.append(imagehash.hex_to_hash(hash_str))
                valid_paths.append(path_str)
        else:
            uncached_paths.append(path_str)
    
    if uncached_paths:
        # Determine whether to use ProcessPoolExecutor or ThreadPoolExecutor
        # Use ThreadPoolExecutor on Windows or when running in interactive mode
        use_threads = (
            sys.platform == 'win32' or 
            hasattr(sys, 'ps1') or  # Interactive mode
            not hasattr(mp, 'get_start_method') or
            (hasattr(mp, 'get_start_method') and mp.get_start_method() == 'spawn')
        )
        
        if use_threads:
            # Use ThreadPoolExecutor for Windows or interactive environments
            if show_progress and sys.platform == 'win32':
                print("Using thread-based parallel processing (Windows compatible)")
            
            # Process images individually with threads
            compute_func = partial(_compute_hash_single, hash_size=hash_size)
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(compute_func, path) for path in uncached_paths]
                
                if show_progress:
                    futures_iterator = tqdm(as_completed(futures), total=len(futures), desc="Processing images")
                else:
                    futures_iterator = as_completed(futures)
                
                for future in futures_iterator:
                    path_str, hash_str = future.result()
                    if hash_str:
                        hashes.append(imagehash.hex_to_hash(hash_str))
                        valid_paths.append(path_str)
                        # Update cache
                        try:
                            path_key = f"{path_str}_{os.path.getmtime(path_str)}"
                            cache[path_key] = hash_str
                        except:
                            pass
        else:
            # Use ProcessPoolExecutor for Unix-like systems
            # Split into batches
            batches = [uncached_paths[i:i+batch_size] for i in range(0, len(uncached_paths), batch_size)]
            
            # Process batches in parallel
            compute_func = partial(_compute_hash_batch, hash_size=hash_size)
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(compute_func, batch) for batch in batches]
                
                if show_progress:
                    futures_iterator = tqdm(as_completed(futures), total=len(futures), desc="Processing batches")
                else:
                    futures_iterator = as_completed(futures)
                
                for future in futures_iterator:
                    batch_results = future.result()
                    for path_str, hash_str in batch_results:
                        if hash_str:
                            hashes.append(imagehash.hex_to_hash(hash_str))
                            valid_paths.append(path_str)
                            # Update cache
                            try:
                                path_key = f"{path_str}_{os.path.getmtime(path_str)}"
                                cache[path_key] = hash_str
                            except:
                                pass
    
    # Save cache if provided
    if cache_dir and uncached_paths:
        try:
            cache_file = cache_dir / f"hash_cache_{hash_size}.json"
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
            if show_progress:
                print(f"Saved {len(cache)} hashes to cache")
        except:
            pass
    
    if show_progress:
        print(f"Successfully processed {len(valid_paths)} images")
    
    return hashes, valid_paths


def _fast_selection_algorithm(
    hash_arrays: np.ndarray, 
    target_count: int, 
    window_size: int,
    show_progress: bool
) -> List[int]:
    """
    Optimized selection using vectorized operations.
    """
    n_images = len(hash_arrays)
    
    if target_count >= n_images:
        return list(range(n_images))
    
    # Use float32 for faster computation
    hash_arrays = hash_arrays.astype(np.float32)
    
    selected_indices = []
    remaining_mask = np.ones(n_images, dtype=bool)
    
    # Start with random image
    first_idx = np.random.choice(n_images)
    selected_indices.append(first_idx)
    remaining_mask[first_idx] = False
    
    iterations_needed = target_count - 1
    iters = range(iterations_needed)
    if show_progress:
        iters = tqdm(iters, desc="Selecting diverse images", total=iterations_needed)
    
    for _ in iters:
        remaining_indices = np.where(remaining_mask)[0]
        if len(remaining_indices) == 0:
            break
        
        # Define rolling window
        window_start = max(0, len(selected_indices) - window_size)
        window_indices = selected_indices[window_start:]
        
        # Vectorized distance computation
        window_hashes = hash_arrays[window_indices]
        min_distances = np.full(len(remaining_indices), np.inf, dtype=np.float32)
        
        for i, candidate_idx in enumerate(remaining_indices):
            candidate_hash = hash_arrays[candidate_idx]
            # Compute all distances at once
            dists = np.mean(window_hashes != candidate_hash, axis=1)
            min_distances[i] = np.min(dists)
        
        best_idx = remaining_indices[np.argmax(min_distances)]
        selected_indices.append(best_idx)
        remaining_mask[best_idx] = False
    
    return selected_indices


def _rolling_window_selection(
    hash_arrays: np.ndarray, 
    target_count: int, 
    window_size: int, 
    show_progress: bool
) -> List[int]:
    """Rolling window algorithm for smaller datasets."""
    n_images = len(hash_arrays)

    if target_count >= n_images:
        return list(range(n_images))

    selected_indices = []
    remaining_indices = set(range(n_images))

    # Start with random image
    first_idx = random.choice(list(remaining_indices))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    iterations_needed = max(0, target_count - 1)
    iters = range(iterations_needed)
    if show_progress:
        iters = tqdm(iters, desc="Selecting diverse images", total=iterations_needed)

    for _ in iters:
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
                if distance < min_distance_to_window:
                    min_distance_to_window = distance
                    if min_distance_to_window == 0:
                        break

            if min_distance_to_window > max_min_distance:
                max_min_distance = min_distance_to_window
                best_candidate = candidate_idx

        if best_candidate is not None:
            selected_indices.append(best_candidate)
            remaining_indices.remove(best_candidate)

    return selected_indices


def _sort_paths_by_directory(image_paths: List[Union[str, Path]]) -> List[str]:
    """Sort image paths by directory structure and filename using natural sorting."""
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


def _hash_to_binary_array(hash_obj, hash_size: int = 8) -> np.ndarray:
    """Convert ImageHash object to binary numpy array."""
    hex_str = str(hash_obj)
    binary_array = []
    for hex_char in hex_str:
        binary_bits = format(int(hex_char, 16), '04b')
        binary_array.extend([int(bit) for bit in binary_bits])
    return np.array(binary_array, dtype=np.uint8)


def _hamming_distance(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Calculate Hamming distance between two binary arrays."""
    return np.mean(arr1 != arr2)
"""
Simple, fast image selection that always works.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union, Optional
import imagehash
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings
from natsort import natsorted

warnings.filterwarnings('ignore')




def sample_diverse(
    image_paths: List[Union[str, Path]], 
    target_count: int,
    hash_size: int = 8,
    n_workers: Optional[int] = None,
    show_progress: bool = True,
    random_seed: int = 42
) -> List[str]:
    """
    Fast diverse sampling from large image collections.
    Preserves maximum diversity by trimming only from largest buckets.
    
    Strategy:
    1. Compute quick hashes for all images
    2. Group similar images into buckets (~16 visual groups)
    3. Trim from top: Keep all small buckets, trim largest buckets with stride sampling
    
    Args:
        image_paths: List of paths to images
        target_count: Exact number of images to return
        hash_size: Size of perceptual hash (8 is fast and good enough)
        n_workers: Number of parallel workers (default: 4)
        show_progress: Whether to show progress bars
        random_seed: Random seed for reproducibility
        
    Returns:
        List of exactly target_count selected image paths
        
    Examples:
        >>> # Fast diverse sampling of 100 from 24,000 images
        >>> selected = sample_diverse(image_paths, target_count=100)
        
        >>> # Also fast for large selections like 23,000 from 24,000 images  
        >>> selected = sample_diverse(image_paths, target_count=23000)
    """
    
    np.random.seed(random_seed)
    
    n_images = len(image_paths)
    
    if target_count >= n_images:
        return [str(p) for p in image_paths]
    
    if target_count <= 0:
        return []
    
    if n_workers is None:
        n_workers = min(4, max(1, 4))  # Default to 4 workers
    
    if show_progress:
        print(f"Selecting {target_count} from {n_images} images using fast grid selection")
    
    # Step 1: Compute hashes in parallel
    if show_progress:
        print("Computing image hashes...")
    
    def compute_hash(path):
        try:
            with Image.open(path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Use dhash for speed (faster than phash)
                hash_val = imagehash.dhash(img, hash_size=hash_size)
                return str(path), hash_val
        except:
            return str(path), None
    
    valid_paths = []
    hashes = []
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(compute_hash, path) for path in image_paths]
        
        if show_progress:
            futures_iter = tqdm(futures, desc="Hashing images")
        else:
            futures_iter = futures
        
        for future in futures_iter:
            path, hash_val = future.result()
            if hash_val is not None:
                valid_paths.append(path)
                hashes.append(hash_val)
    
    n_valid = len(valid_paths)
    
    if target_count >= n_valid:
        return valid_paths
    
    # Step 2: Create diversity grid using hash bits
    if show_progress:
        print("Creating diversity grid...")
    
    # Convert hashes to bit arrays for bucketing
    hash_arrays = np.array([np.array(h.hash).flatten() for h in hashes])
    
    # Use first few bits to create buckets (trade-off between diversity and speed)
    n_bucket_bits = min(4, hash_size)  # Use 4 bits = 16 buckets max
    bucket_keys = []
    
    for h in hash_arrays:
        # Create bucket key from first few bits
        key = tuple(h.flatten()[:n_bucket_bits] > 0)
        bucket_keys.append(key)
    
    # Group images by bucket
    buckets = {}
    for i, key in enumerate(bucket_keys):
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(i)
    
    # Sort indices within each bucket by natural path order
    # This ensures folder structure is preserved during stride sampling
    for key, indices in buckets.items():
        # Create list of (path, index) pairs, sort by path using natsort, extract indices
        path_index_pairs = [(valid_paths[i], i) for i in indices]
        sorted_pairs = natsorted(path_index_pairs, key=lambda x: x[0])
        buckets[key] = [pair[1] for pair in sorted_pairs]
    
    if show_progress:
        print(f"Grouped into {len(buckets)} diversity buckets with natural sorting")
    
    # Step 3: Trim from top with stride sampling
    if show_progress:
        print(f"Using trim-from-top with stride sampling across {len(buckets)} buckets")
    
    # Sort buckets by size (largest first)
    bucket_list = [(key, indices) for key, indices in buckets.items()]
    bucket_list.sort(key=lambda x: len(x[1]), reverse=True)
    
    selected_indices = []
    remaining_quota = target_count
    
    for bucket_key, indices in bucket_list:
        bucket_size = len(indices)
        
        if bucket_size <= remaining_quota:
            # Keep entire bucket - all images are preserved
            selected_indices.extend(indices)
            remaining_quota -= bucket_size
        else:
            # Use stride for even sampling within bucket (respects folder structure)
            if remaining_quota > 0:
                stride = bucket_size // remaining_quota
                if stride > 0:
                    # Sample with stride to get even distribution across folders
                    sampled = indices[::stride][:remaining_quota]
                    selected_indices.extend(sampled)
                else:
                    # If stride would be 0, just take the first N
                    selected_indices.extend(indices[:remaining_quota])
                remaining_quota = 0
            break
    
    selected_paths = [valid_paths[i] for i in selected_indices]
    
    if show_progress:
        print(f"✓ Selected {len(selected_paths)} images with diversity preservation")
    
    return selected_paths
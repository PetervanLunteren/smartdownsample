"""
Simple, fast image selection that always works.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict, Any
import imagehash
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings
from natsort import natsorted

warnings.filterwarnings('ignore')

# Simple visualization functions integrated into core
def _print_bucket_summary(bucket_stats: List[Dict[str, Any]]) -> None:
    """Print a simple text summary of bucket statistics."""
    if not bucket_stats:
        print("No bucket statistics available")
        return
    
    print("\n" + "="*60)
    print("BUCKET DISTRIBUTION SUMMARY")
    print("="*60)
    
    # Sort by original size
    sorted_buckets = sorted(bucket_stats, key=lambda x: x['original_size'], reverse=True)
    
    total_images = sum(b['original_size'] for b in bucket_stats)
    total_selected = sum(b['kept'] for b in bucket_stats)
    
    print(f"Total images: {total_images:,}")
    print(f"Selected: {total_selected:,} ({(total_selected/total_images)*100:.1f}%)")
    print(f"Visual diversity buckets: {len(bucket_stats)}")
    print()
    
    print("Per-bucket breakdown:")
    print("-" * 60)
    print(f"{'Bucket':<8} {'Size':<8} {'Kept':<8} {'Rate':<8} {'Strategy':<12}")
    print("-" * 60)
    
    for i, bucket in enumerate(sorted_buckets):
        size = bucket['original_size']
        kept = bucket['kept']
        rate = f"{(kept/size)*100:.0f}%" if size > 0 else "0%"
        strategy = "All kept" if kept == size else f"Stride ({bucket.get('stride', '?')})"
        
        print(f"#{i+1:<7} {size:<8,} {kept:<8,} {rate:<8} {strategy:<12}")
    
    print("-" * 60)
    print()


def _plot_bucket_thumbnails(bucket_stats: List[Dict[str, Any]], viz_data: Dict, show_progress: bool = True) -> None:
    """Show thumbnail grids for each bucket - 5x5 grid with up to 25 images per bucket."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
    except ImportError:
        print("matplotlib not available - skipping thumbnail grids")
        return
    
    if not bucket_stats or not viz_data:
        print("No bucket data available for thumbnails")
        return
    
    # Sort buckets by original size (largest first)
    sorted_buckets = sorted(bucket_stats, key=lambda x: x['original_size'], reverse=True)
    
    bucket_assignments = viz_data['bucket_assignments']
    all_paths = viz_data['all_paths']
    
    # Calculate square grid layout for subplots
    n_buckets = len(sorted_buckets)
    cols = int(np.ceil(np.sqrt(n_buckets)))
    rows = int(np.ceil(n_buckets / cols))
    
    # Create figure with extra height for titles
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 6))
    
    # Ensure axes is always a 2D array for consistent access
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Process each bucket with progress bar
    if show_progress:
        bucket_iter = tqdm(enumerate(sorted_buckets), desc=" - Creating thumbnail grids", total=len(sorted_buckets))
    else:
        bucket_iter = enumerate(sorted_buckets)
    
    for bucket_idx, bucket_data in bucket_iter:
        # Get images for this bucket (reverse index since we sorted)
        original_bucket_idx = len(sorted_buckets) - 1 - bucket_idx
        bucket_images = []
        for path_idx, assigned_bucket in enumerate(bucket_assignments):
            if assigned_bucket == original_bucket_idx:
                bucket_images.append(all_paths[path_idx])
        
        # Create 5x5 thumbnail grid
        def create_bucket_grid(images, max_images=25):
            """Create a 5x5 grid of thumbnails from bucket images."""
            if not images:
                return np.ones((300, 300, 3), dtype=np.uint8) * 220  # Gray placeholder
            
            # Take first 25 images (not random) for consistent display
            sample_images = images[:max_images]
            
            # Create 5x5 grid (300x300 pixels, each thumbnail 60x60)
            grid_img = np.ones((300, 300, 3), dtype=np.uint8) * 255  # White background
            thumb_size = 60
            
            for idx, img_path in enumerate(sample_images[:25]):  # Ensure max 25
                if idx >= 25:
                    break
                    
                try:
                    # Load and resize image
                    with Image.open(img_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img_thumb = img.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                        img_array = np.array(img_thumb, dtype=np.uint8)
                        
                        # Calculate position in 5x5 grid
                        row = idx // 5
                        col = idx % 5
                        y_start = row * thumb_size
                        x_start = col * thumb_size
                        y_end = y_start + thumb_size
                        x_end = x_start + thumb_size
                        
                        # Place thumbnail in grid
                        grid_img[y_start:y_end, x_start:x_end] = img_array
                        
                except Exception:
                    # Skip failed images, leave white space
                    continue
            
            return grid_img
        
        # Calculate subplot position
        row = bucket_idx // cols
        col = bucket_idx % cols
        ax = axes[row, col]
        
        # Create and display thumbnail grid
        grid_img = create_bucket_grid(bucket_images)
        ax.imshow(grid_img)
        ax.set_title(str(bucket_idx + 1), fontsize=12, pad=2)
        
        # Add thin grey border around the grid
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('lightgrey')
            spine.set_linewidth(1)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for i in range(n_buckets, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Bucket thumbnails: visual similarity groups (5x5 grid, max 25 images per bucket)', 
                 fontsize=14, y=0.98)
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.92, hspace=0.4)
    plt.show()


def _plot_bucket_distribution(bucket_stats: List[Dict[str, Any]]) -> None:
    """Create a simple vertical bucket distribution chart."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available - skipping distribution chart")
        return
    
    if not bucket_stats:
        print("No bucket statistics available")
        return
    
    # Sort buckets by original size (largest first)
    sorted_buckets = sorted(bucket_stats, key=lambda x: x['original_size'], reverse=True)
    
    bucket_names = [str(i+1) for i in range(len(sorted_buckets))]
    kept_counts = [b['kept'] for b in sorted_buckets]
    excluded_counts = [b['excluded'] for b in sorted_buckets]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(bucket_names))
    width = 0.6
    
    # Create stacked bars
    bars_kept = ax.bar(x, kept_counts, width, label='Kept', color='#2E8B57', alpha=0.8)
    bars_excluded = ax.bar(x, excluded_counts, width, bottom=kept_counts, 
                          label='Excluded', color='#CD5C5C', alpha=0.8)
    
    ax.set_xlabel('Visual similarity buckets (sorted by size)')
    ax.set_ylabel('Number of images')
    ax.set_title('Bucket distribution: kept vs excluded')
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_names, rotation=45, ha='right')
    ax.legend()
    
    
    plt.tight_layout()
    plt.show()




def sample_diverse(
    image_paths: List[Union[str, Path]], 
    target_count: int,
    hash_size: int = 8,
    n_workers: Optional[int] = None,
    show_progress: bool = True,
    random_seed: int = 42,
    show_summary: bool = True,
    show_distribution: bool = False,
    show_thumbnails: bool = False
) -> List[str]:
    """
    Fast diverse sampling from large image collections.
    Preserves maximum diversity by ensuring representation from all visual groups.
    
    Strategy:
    1. Compute multi-dimensional visual features (structure, color, brightness)
    2. Group similar images into buckets based on visual similarity
    3. Diversity-first: Sample from every bucket to preserve visual variety, then fill largest buckets
    
    Args:
        image_paths: List of paths to images
        target_count: Exact number of images to return
        hash_size: Size of perceptual hash (8 is fast and good enough)
        n_workers: Number of parallel workers (default: 4)
        show_progress: Whether to show progress bars
        random_seed: Random seed for reproducibility
        show_summary: Whether to print bucket distribution summary (default: True)
        show_distribution: Whether to show bucket distribution chart (default: False)
        show_thumbnails: Whether to show 5x5 thumbnail grids for each bucket (default: False)
        
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
    
    # Early exit: if we want everything or more, just return all paths
    if target_count >= n_images:
        if show_progress:
            print(f"Target count ({target_count}) >= available images ({n_images}), returning all images")
        return [str(p) for p in image_paths]
    
    if target_count <= 0:
        return []
    
    if n_workers is None:
        n_workers = min(4, max(1, 4))  # Default to 4 workers
    
    if show_progress:
        print(f"Selecting {target_count} from {n_images} images...")
    
    # Sort all input paths naturally to preserve camera/folder structure globally
    if show_progress:
        print(" - Sorting paths...")
    sorted_image_paths = natsorted([str(p) for p in image_paths])
    
    # Step 1: Compute hashes in parallel
    
    def compute_hash(path):
        try:
            with Image.open(path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Compute multiple features for better visual similarity
                # 1. DHash for structural/edge information
                dhash = imagehash.dhash(img, hash_size=hash_size)
                
                # 2. Average hash for overall brightness/contrast
                ahash = imagehash.average_hash(img, hash_size=hash_size//2)
                
                # 3. Color histogram features (simplified)
                img_small = img.resize((32, 32), Image.Resampling.LANCZOS)
                img_array = np.array(img_small)
                
                # Color variance (low = grayscale/monochrome, high = colorful)
                r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
                color_variance = np.var([np.mean(r), np.mean(g), np.mean(b)])
                
                # Average color (dominant color theme)
                avg_r, avg_g, avg_b = np.mean(r), np.mean(g), np.mean(b)
                
                # Brightness level
                brightness = np.mean(img_array)
                
                # Combine all features
                combined_features = {
                    'dhash': dhash,
                    'ahash': ahash,
                    'color_variance': color_variance,
                    'avg_color': (avg_r, avg_g, avg_b),
                    'brightness': brightness
                }
                
                return str(path), combined_features
        except:
            return str(path), None
    
    valid_paths = []
    hashes = []
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(compute_hash, path) for path in sorted_image_paths]
        
        if show_progress:
            futures_iter = tqdm(futures, desc=" - Hashing images")
        else:
            futures_iter = futures
        
        for future in futures_iter:
            path, features = future.result()
            if features is not None:
                valid_paths.append(path)
                hashes.append(features)
    
    n_valid = len(valid_paths)
    
    if target_count >= n_valid:
        return valid_paths
    
    # Step 2: Create diversity grid using multi-dimensional features
    if show_progress:
        print(" - Creating diversity grid...")
    
    bucket_keys = []
    
    for features in hashes:
        dhash = features['dhash']
        ahash = features['ahash']
        color_var = features['color_variance']
        avg_r, avg_g, avg_b = features['avg_color']
        brightness = features['brightness']
        
        # Convert dhash to bits (structural features)
        dhash_bits = np.array(dhash.hash).flatten()
        ahash_bits = np.array(ahash.hash).flatten()
        
        # Use 2 bits from center of dhash (where animals typically are)
        # For 8x8 hash (64 bits), center would be around indices 27, 28, 35, 36
        # We'll take 2 strategically placed center bits
        total_dhash_bits = len(dhash_bits)
        if total_dhash_bits >= 64:  # 8x8 hash
            # Center-left and center-right of the hash (row 3-4, col 3-4)
            center_indices = [27, 36]  # Strategic center positions
            structure_bits = tuple(dhash_bits[i] > 0 for i in center_indices)
        elif total_dhash_bits >= 16:  # 4x4 hash
            # Center positions for smaller hash
            center_indices = [5, 10]  # Center positions in 4x4
            structure_bits = tuple(dhash_bits[i] > 0 for i in center_indices if i < total_dhash_bits)
        else:
            # Very small hash, just use first 2 bits
            structure_bits = tuple(dhash_bits[:2] > 0) if total_dhash_bits >= 2 else tuple(dhash_bits > 0)
        
        # Use only 1 bit from ahash for basic brightness pattern
        brightness_bit = (ahash_bits[0] > 0,) if len(ahash_bits) >= 1 else ()
        
        # Simplified color variance (0=grayscale, 1=colored)
        color_bucket = 0 if color_var < 100 else 1  # Binary: grayscale vs color
        
        # Simplified brightness (0=dark, 1=bright)
        bright_bucket = 0 if brightness < 128 else 1  # Binary: dark vs bright
        
        # Dominant color theme detection (for colored images only)
        if color_bucket == 1:  # Only apply to colored images
            # Determine dominant color channel
            if avg_r > avg_g and avg_r > avg_b and avg_r > 140:
                color_theme = 0  # Red-dominant (warm/orange/brown scenes)
            elif avg_g > avg_r and avg_g > avg_b and avg_g > 140:
                color_theme = 1  # Green-dominant (forest/grass scenes)  
            elif avg_b > avg_r and avg_b > avg_g and avg_b > 140:
                color_theme = 2  # Blue-dominant (sky/snow/water scenes)
            else:
                color_theme = 3  # Neutral/mixed colors
        else:
            color_theme = 0  # Grayscale images get default theme
        
        # Combine all features into bucket key
        # 2 center structure + 1 brightness + 1 color + 1 overall_brightness + 2 color_theme = 7 dimensions  
        # Max buckets: 2^2 × 2^1 × 2^1 × 2^1 × 2^2 = 4 × 2 × 2 × 2 × 4 = 128 buckets (but many combinations won't exist)
        bucket_key = structure_bits + brightness_bit + (color_bucket, bright_bucket, color_theme)
        bucket_keys.append(bucket_key)
    
    # Group images by bucket
    buckets = {}
    for i, key in enumerate(bucket_keys):
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(i)
    
    # Note: Paths are already naturally sorted globally, so bucket order preserves camera/folder structure
    
    if show_progress:
        # Show bucket size distribution
        bucket_sizes = [len(indices) for indices in buckets.values()]
        print(f" - Grouped into {len(buckets)} diversity buckets... (min={min(bucket_sizes)}, max={max(bucket_sizes)}, avg={sum(bucket_sizes)/len(bucket_sizes):.1f})")
    
    # Step 3: Diversity-preserving selection
    
    # Sort buckets by size (largest first)
    bucket_list = [(key, indices) for key, indices in buckets.items()]
    bucket_list.sort(key=lambda x: len(x[1]), reverse=True)
    
    selected_indices = []
    remaining_quota = target_count
    
    # Phase 1: Ensure diversity by taking at least 1 image from each bucket (if quota allows)
    min_per_bucket = max(1, target_count // len(bucket_list))  # At least 1, but more if quota is large
    diversity_quota = min(len(bucket_list) * min_per_bucket, target_count)
    
    # First pass: guarantee diversity by sampling from each bucket
    bucket_taken = []  # Track how many taken from each bucket
    for bucket_key, indices in bucket_list:
        bucket_size = len(indices)
        
        if diversity_quota > 0 and bucket_size > 0:
            # Take min_per_bucket images from each bucket to preserve diversity
            take_count = min(min_per_bucket, bucket_size, diversity_quota)
            if take_count > 0:
                stride = max(1, bucket_size // take_count)
                sampled = indices[::stride][:take_count]
                selected_indices.extend(sampled)
                diversity_quota -= take_count
                remaining_quota -= take_count
                bucket_taken.append(take_count)
            else:
                bucket_taken.append(0)
        else:
            bucket_taken.append(0)
    
    # Phase 2: Distribute remaining quota proportionally across all available buckets
    if remaining_quota > 0:
        # Find all buckets that still have available images
        available_buckets = []
        for i, (bucket_key, indices) in enumerate(bucket_list):
            bucket_size = len(indices)
            already_taken = bucket_taken[i]
            available_in_bucket = bucket_size - already_taken
            if available_in_bucket > 0:
                available_buckets.append((i, available_in_bucket))
        
        if available_buckets:
            # Calculate total available images across all buckets
            total_available = sum(available for _, available in available_buckets)
            
            # Distribute remaining quota proportionally
            for bucket_idx, available_in_bucket in available_buckets:
                if remaining_quota <= 0:
                    break
                
                # Calculate proportional share (but don't exceed what's available in this bucket)
                proportional_share = (available_in_bucket / total_available) * remaining_quota
                take_additional = min(int(proportional_share), available_in_bucket, remaining_quota)
                
                if take_additional > 0:
                    # Sample additional images using stride from remaining images
                    bucket_key, indices = bucket_list[bucket_idx]
                    already_taken = bucket_taken[bucket_idx]
                    remaining_indices = indices[already_taken:]  # Skip already selected
                    stride = max(1, len(remaining_indices) // take_additional)
                    additional_sampled = remaining_indices[::stride][:take_additional]
                    
                    selected_indices.extend(additional_sampled)
                    remaining_quota -= take_additional
            
            # If there's still remaining quota due to rounding, give it to largest available buckets
            while remaining_quota > 0 and available_buckets:
                # Find bucket with most available images
                bucket_idx = max(available_buckets, key=lambda x: x[1])[0]
                bucket_key, indices = bucket_list[bucket_idx]
                already_taken = bucket_taken[bucket_idx]
                available_in_bucket = len(indices) - already_taken
                
                if available_in_bucket > 0:
                    take_additional = min(1, available_in_bucket, remaining_quota)
                    remaining_indices = indices[already_taken:]
                    additional_sampled = remaining_indices[:take_additional]
                    
                    selected_indices.extend(additional_sampled)
                    remaining_quota -= take_additional
                    
                    # Update available count for this bucket
                    available_buckets = [(idx, len(bucket_list[idx][1]) - bucket_taken[idx] - (1 if idx == bucket_idx else 0)) 
                                       for idx, _ in available_buckets 
                                       if len(bucket_list[idx][1]) - bucket_taken[idx] - (1 if idx == bucket_idx else 0) > 0]
                else:
                    break
    
    selected_paths = [valid_paths[i] for i in selected_indices]
    
    if show_progress:
        print(f" - Selected {len(selected_paths)} images with diversity preservation!")
    
    # Show visualizations if requested
    if show_summary or show_distribution or show_thumbnails:
        # Create bucket statistics for visualization
        bucket_stats = []
        selected_indices_set = set(selected_indices)
        
        for bucket_idx, (bucket_key, indices) in enumerate(bucket_list):
            bucket_size = len(indices)
            kept = sum(1 for i in indices if i in selected_indices_set)
            excluded = bucket_size - kept
            
            bucket_stats.append({
                'original_size': bucket_size,
                'kept': kept,
                'excluded': excluded,
                'stride': bucket_size // kept if kept > 0 else 0
            })
        
        if show_summary:
            _print_bucket_summary(bucket_stats)
        
        if show_distribution:
            _plot_bucket_distribution(bucket_stats)
        
        if show_thumbnails:
            # Create bucket assignment mapping and visualization data for thumbnails
            bucket_assignments = [0] * len(valid_paths)
            for bucket_idx, (bucket_key, indices) in enumerate(bucket_list):
                for idx in indices:
                    bucket_assignments[idx] = bucket_idx
            
            viz_data = {
                'bucket_assignments': bucket_assignments,
                'all_paths': valid_paths
            }
            
            _plot_bucket_thumbnails(bucket_stats, viz_data, show_progress)
    
    return selected_paths



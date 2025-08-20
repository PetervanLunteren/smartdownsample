#!/usr/bin/env python3
"""
Example script showing how to visualize smartdownsample selection patterns.

Usage:
    pip install smartdownsample  # Includes visualization by default
    python visualize_selection.py /path/to/image/folder
"""

import sys
from pathlib import Path
from smartdownsample import sample_diverse_with_stats
from smartdownsample import plot_bucket_distribution, plot_hash_similarity_scatter, print_bucket_summary


def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_selection.py /path/to/image/folder")
        sys.exit(1)
    
    image_folder = Path(sys.argv[1])
    
    if not image_folder.exists():
        print(f"Error: Folder {image_folder} does not exist")
        sys.exit(1)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(image_folder.glob(f"**/*{ext}"))
        image_paths.extend(image_folder.glob(f"**/*{ext.upper()}"))
    
    if not image_paths:
        print(f"No image files found in {image_folder}")
        sys.exit(1)
    
    print(f"Found {len(image_paths)} image files")
    
    # Select diverse images and get visualization data
    target_count = min(1000, len(image_paths) // 4)  # Select 25% or max 1000 images
    print(f"Selecting {target_count} diverse images...")
    
    selected_paths, viz_data = sample_diverse_with_stats(
        image_paths=image_paths,
        target_count=target_count,
        show_progress=True
    )
    
    print(f"\n✓ Selected {len(selected_paths)} images")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Text summary
    print_bucket_summary(viz_data['bucket_stats'])
    
    # 2. Bucket distribution bar chart (matplotlib)
    plot_bucket_distribution(
        viz_data['bucket_stats'],
        title=f"Selection Pattern: {len(selected_paths):,} from {len(image_paths):,} images",
        save_path="bucket_distribution.png"
    )
    print("✓ Bucket distribution chart saved as 'bucket_distribution.png'")
    
    # 3. Interactive scatter plot (plotly)
    plot_hash_similarity_scatter(
        viz_data,
        title=f"Visual Similarity: Selected vs Excluded Images",
        save_path="similarity_scatter.html",
        show_browser=True
    )
    print("✓ Interactive scatter plot saved as 'similarity_scatter.html'")
    
    print("\nVisualization complete!")
    print(f"Selected images are saved in the returned list (length: {len(selected_paths)})")


if __name__ == "__main__":
    main()
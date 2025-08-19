#!/usr/bin/env python3
"""
Basic usage example for SmartDownsample.
"""

from pathlib import Path
from smartdownsample import select_distinct


def basic_example():
    """Basic usage example."""
    print("SmartDownsample Basic Usage Example")
    print("=" * 40)
    
    # Example 1: Basic selection
    print("\n1. Basic Selection")
    print("-" * 20)
    
    # Simulate image paths (replace with your actual paths)
    image_paths = [
        "image_001.jpg", "image_002.jpg", "image_003.jpg",
        "image_004.jpg", "image_005.jpg", "image_006.jpg",
        "image_007.jpg", "image_008.jpg", "image_009.jpg",
        "image_010.jpg"
    ]
    
    print(f"Total images: {len(image_paths)}")
    
    # Select 5 most diverse images
    selected = select_distinct(
        image_paths=image_paths,
        target_count=5,
        show_progress=False  # Disable progress bars for cleaner output
    )
    
    print(f"Selected {len(selected)} images:")
    for img in selected:
        print(f"  - {img}")


def camera_trap_example():
    """Camera trap workflow example."""
    print("\n\n2. Camera Trap Workflow")
    print("-" * 25)
    
    # Example folder structure for camera trap data
    base_folder = Path("camera_trap_data")
    species_folders = ["deer", "rabbit", "squirrel", "bird"]
    
    print("Processing camera trap data by species:")
    
    for species in species_folders:
        species_folder = base_folder / species
        print(f"\nProcessing {species}:")
        
        # Simulate finding images (replace with actual glob)
        # image_paths = list(species_folder.glob("*.jpg"))
        
        # For demo, simulate different numbers of images per species
        image_counts = {"deer": 500, "rabbit": 300, "squirrel": 150, "bird": 800}
        num_images = image_counts.get(species, 100)
        
        # Simulate image paths
        image_paths = [f"{species}_{i:04d}.jpg" for i in range(1, num_images + 1)]
        
        print(f"  Found: {len(image_paths)} images")
        
        # Select diverse subset for training
        target_count = min(100, len(image_paths))  # Max 100 per species
        
        selected = select_distinct(
            image_paths=image_paths,
            target_count=target_count,
            method="rolling_window",
            window_size=50,
            show_progress=False
        )
        
        print(f"  Selected: {len(selected)} diverse images for training")


def performance_comparison():
    """Show performance comparison between methods."""
    print("\n\n3. Performance Comparison")
    print("-" * 28)
    
    import time
    
    # Simulate medium-sized dataset
    image_paths = [f"large_dataset_{i:05d}.jpg" for i in range(1000)]
    target_count = 100
    
    print(f"Dataset: {len(image_paths)} images → {target_count} selected")
    
    # Test rolling window method
    print("\nTesting rolling_window method...")
    start_time = time.time()
    
    selected_rolling = select_distinct(
        image_paths=image_paths,
        target_count=target_count,
        method="rolling_window",
        window_size=50,
        show_progress=False
    )
    
    rolling_time = time.time() - start_time
    print(f"  Time: {rolling_time:.2f} seconds")
    print(f"  Selected: {len(selected_rolling)} images")
    
    # Test exact method (only for smaller datasets)
    if len(image_paths) <= 200:  # Only test exact on small datasets
        print("\nTesting exact method...")
        start_time = time.time()
        
        selected_exact = select_distinct(
            image_paths=image_paths[:200],  # Use smaller subset
            target_count=min(target_count, 50),
            method="exact",
            show_progress=False
        )
        
        exact_time = time.time() - start_time
        print(f"  Time: {exact_time:.2f} seconds")
        print(f"  Selected: {len(selected_exact)} images")
        
        speedup = exact_time / rolling_time if rolling_time > 0 else 0
        print(f"\nSpeedup: {speedup:.1f}x faster with rolling_window")
    else:
        print("\nExact method skipped (dataset too large for demo)")


if __name__ == "__main__":
    # Run all examples
    basic_example()
    camera_trap_example()
    performance_comparison()
    
    print("\n" + "=" * 50)
    print("Example completed!")
    print("\nTo use with real images:")
    print("1. Replace simulated paths with actual image file paths")
    print("2. Use Path().glob('*.jpg') to find real images")
    print("3. Adjust target_count and parameters as needed")
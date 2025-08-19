#!/usr/bin/env python3
"""
Test script for Windows compatibility.
This demonstrates the proper way to use multiprocessing on Windows.
"""

def main():
    """Main function wrapped for Windows multiprocessing safety."""
    from smartdownsample import select_distinct
    from pathlib import Path
    
    print("Testing SmartDownsample on Windows")
    print("=" * 50)
    
    # Example image paths (replace with actual paths)
    image_paths = [
        "image_001.jpg", "image_002.jpg", "image_003.jpg",
        "image_004.jpg", "image_005.jpg", "image_006.jpg",
        "image_007.jpg", "image_008.jpg", "image_009.jpg",
        "image_010.jpg"
    ]
    
    # For real usage, find actual images:
    # image_folder = Path("C:/Users/YourName/Pictures")
    # image_paths = list(image_folder.glob("*.jpg"))
    
    print(f"Processing {len(image_paths)} images...")
    
    # Select diverse images
    selected = select_distinct(
        image_paths=image_paths,
        target_count=min(5, len(image_paths)),
        n_workers=4,  # Use 4 workers
        show_progress=True
    )
    
    print(f"\nSelected {len(selected)} diverse images")
    for img in selected[:5]:
        print(f"  - {img}")
    
    print("\n✅ Windows compatibility test complete!")


if __name__ == "__main__":
    # This guard is REQUIRED for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()  # Needed for Windows executables
    main()
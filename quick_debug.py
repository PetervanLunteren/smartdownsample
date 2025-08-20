#!/usr/bin/env python3

# Quick debug to see what's happening with your data
from smartdownsample import sample_diverse_with_stats, print_bucket_summary

# Test with just a few real images from your folder
print("=== QUICK BUCKET DEBUG ===")
print("This will help us see what's actually happening...")

# You can modify this path to point to a small subset of your images
test_folder = input("Enter path to a folder with a few test images (or press Enter to skip): ").strip()

if test_folder:
    from pathlib import Path
    test_path = Path(test_folder)
    
    if test_path.exists():
        # Find just the first 20 images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(test_path.glob(f"*{ext}"))[:5])
            image_paths.extend(list(test_path.glob(f"*{ext.upper()}"))[:5])
            if len(image_paths) >= 20:
                break
        
        image_paths = image_paths[:20]
        
        if image_paths:
            print(f"Found {len(image_paths)} test images")
            print("Testing with 5 selected...")
            
            selected, viz_data = sample_diverse_with_stats(
                image_paths=image_paths,
                target_count=5,
                show_progress=True
            )
            
            print(f"\nResults:")
            print(f"- Selected: {len(selected)}")
            print(f"- Total buckets: {len(viz_data['bucket_stats'])}")
            
            total_in_buckets = sum(b['original_size'] for b in viz_data['bucket_stats'])
            total_kept = sum(b['kept'] for b in viz_data['bucket_stats'])
            
            print(f"- Images processed: {total_in_buckets}")
            print(f"- Images kept: {total_kept}")
            
            print("\nDetailed bucket info:")
            for i, bucket in enumerate(viz_data['bucket_stats']):
                pct = (bucket['kept']/bucket['original_size'])*100 if bucket['original_size'] > 0 else 0
                print(f"  Bucket {i+1}: {bucket['original_size']} total → {bucket['kept']} kept ({pct:.0f}%)")
            
            if len(viz_data['bucket_stats']) == 1 and viz_data['bucket_stats'][0]['kept'] < viz_data['bucket_stats'][0]['original_size']:
                print("\n⚠️  FOUND THE ISSUE!")
                print("Only one bucket shows up, and it's partially selected.")
                print("This suggests all your images have very similar hashes.")
        else:
            print("No images found in that folder")
    else:
        print("Folder not found")
else:
    print("Skipping real image test")

print("\n" + "="*50)
print("DIAGNOSIS:")
print("If you see only 1 bucket with partial selection, that means:")
print("1. All your images are getting grouped into the same visual similarity bucket")
print("2. The algorithm is doing stride sampling on that one huge bucket")
print("3. This is actually correct behavior - your images are very visually similar!")
print("\nTry with more diverse images to see multiple buckets.")
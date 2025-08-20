#!/usr/bin/env python3
"""
Debug script to understand bucket distribution issues.
"""

import sys
from pathlib import Path
from smartdownsample import sample_diverse_with_stats, print_bucket_summary

def debug_bucket_stats():
    # Create some dummy image paths for testing
    dummy_paths = [f"test_image_{i:04d}.jpg" for i in range(100)]
    
    print("=== DEBUGGING BUCKET STATISTICS ===")
    print(f"Testing with {len(dummy_paths)} dummy images, selecting 25")
    
    # This should work without actual image files by failing gracefully
    try:
        selected, viz_data = sample_diverse_with_stats(
            image_paths=dummy_paths,
            target_count=25,
            show_progress=True
        )
        
        print(f"\nSelected: {len(selected)} images")
        print(f"Bucket stats count: {len(viz_data['bucket_stats'])}")
        
        print("\n=== BUCKET DETAILS ===")
        for i, bucket in enumerate(viz_data['bucket_stats']):
            print(f"Bucket {i+1}: size={bucket['original_size']}, kept={bucket['kept']}, excluded={bucket['excluded']}, stride={bucket['stride']}")
        
        print("\n=== SUMMARY ===")
        print_bucket_summary(viz_data['bucket_stats'])
        
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Test with your actual scenario
    print("\n" + "="*60)
    print("Now test with your actual image count scenario...")
    
    # Simulate 16011 images -> 1000 selected
    large_dummy_paths = [f"cam{i//1000:02d}/IMG_{i:05d}.jpg" for i in range(16011)]
    
    print(f"Testing with {len(large_dummy_paths)} simulated camera trap images")
    print("Selecting 1000 images...")
    
    try:
        selected, viz_data = sample_diverse_with_stats(
            image_paths=large_dummy_paths[:100],  # Just test with 100 to avoid long processing
            target_count=25,
            show_progress=True
        )
        
        print(f"\nSelected: {len(selected)} images")
        print(f"Total buckets: {len(viz_data['bucket_stats'])}")
        
        total_images = sum(b['original_size'] for b in viz_data['bucket_stats'])
        total_kept = sum(b['kept'] for b in viz_data['bucket_stats'])
        total_excluded = sum(b['excluded'] for b in viz_data['bucket_stats'])
        
        print(f"Total images in buckets: {total_images}")
        print(f"Total kept: {total_kept}")
        print(f"Total excluded: {total_excluded}")
        print(f"Selection rate: {(total_kept/total_images)*100:.1f}%")
        
        print("\nBucket breakdown:")
        for i, bucket in enumerate(viz_data['bucket_stats']):
            rate = (bucket['kept']/bucket['original_size'])*100 if bucket['original_size'] > 0 else 0
            print(f"Bucket {i+1}: {bucket['original_size']} total, {bucket['kept']} kept ({rate:.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_bucket_stats()
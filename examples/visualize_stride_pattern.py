#!/usr/bin/env python3
"""
Example script showing stride pattern visualizations.

Usage:
    python visualize_stride_pattern.py
"""

from smartdownsample import (
    sample_diverse_with_stats,
    plot_stride_pattern,
    plot_all_bucket_strides,
    print_bucket_summary
)

# Example usage with your image data
def demonstrate_stride_visualization():
    print("=== STRIDE PATTERN VISUALIZATION DEMO ===")
    print()
    
    # Replace this with your actual image paths
    print("1. First, get your selection data:")
    print("   selected, viz_data = sample_diverse_with_stats(my_image_list, target_count=1000)")
    print()
    
    print("2. Then visualize stride patterns:")
    print()
    
    print("# Single bucket detailed view - shows stride pattern in largest bucket")
    print("plot_stride_pattern(viz_data, bucket_id=0, save_path='bucket_0_stride.png')")
    print()
    
    print("# Multi-bucket overview - shows stride patterns across top 6 buckets")  
    print("plot_all_bucket_strides(viz_data, max_buckets=6, save_path='all_strides.png')")
    print()
    
    print("=== WHAT YOU'LL SEE ===")
    print()
    print("📊 Single Bucket View (plot_stride_pattern):")
    print("   • Top plot: Timeline with green dots (selected) and red dots (excluded)")
    print("   • Green dashed lines show stride intervals")
    print("   • Average stride calculation (e.g., 'Average stride: 3.6')")
    print("   • Bottom plot: Selection density across the timeline")
    print()
    
    print("📊 Multi-Bucket Overview (plot_all_bucket_strides):")
    print("   • Grid showing stride patterns for multiple buckets")
    print("   • Each subplot shows the selection pattern for one bucket")
    print("   • Easy comparison of stride patterns across different visual groups")
    print()
    
    print("🎯 What this reveals:")
    print("   • How evenly images are distributed across time/folders")
    print("   • Whether stride sampling is working as expected")
    print("   • Differences in sampling density between buckets")
    print("   • Visual confirmation of the proportional distribution algorithm")

if __name__ == "__main__":
    demonstrate_stride_visualization()
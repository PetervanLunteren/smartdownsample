# smartdownsample

**Fast, simple image downsampling that just works**

SmartDownsample samples diverse images from large collections in seconds, not hours. One simple function that works equally fast whether you're sampling 100 or 23,000 images from 24,000.

## Installation

```bash
pip install smartdownsample
```

## Features

- ⚡ **Always fast** - Seconds for any selection ratio
- 🎯 **Smart bucketing** - Better than random, faster than optimal algorithms
- 📊 **Scales linearly** - 24k images? No problem
- 🔧 **Dead simple** - One function, always works  
- 🎲 **Reproducible** - Set seed for consistent results
- ⚖️ **Honest trade-offs** - Speed over perfection, good enough for most use cases

## Usage

```python
from smartdownsample import sample_diverse

# Sample 100 diverse images from 24,000 - takes seconds
selected = sample_diverse(
    image_paths=my_24k_images,
    target_count=100
)

# Sample 23,000 images from 24,000 - also takes seconds!
selected = sample_diverse(
    image_paths=my_24k_images,
    target_count=23000
)

# It's that simple.
print(f"Sampled {len(selected)} diverse images")
```

## How It Works

Simple "trim from top" algorithm that maximizes diversity while being blazing fast:

### 1. **Hash Images** (Fast)
```
Image → 64-bit fingerprint in ~0.01 seconds
Uses DHash with 4 parallel workers
```

### 2. **Group Into Buckets** (O(n))
```
Use first 4 hash bits to create ~16 visual groups:
Bucket A: [landscape1.jpg, landscape2.jpg, ...]     # 45 images
Bucket B: [portrait1.jpg, portrait2.jpg, ...]       # 12 images  
Bucket C: [closeup1.jpg, closeup2.jpg, ...]         # 890 images
```

### 3. **Trim from Top** (Ultra Fast)
```
Sort buckets by size (largest first)
Natural sort images within each bucket (cam01/IMG_1.jpg → cam01/IMG_10.jpg → cam02/IMG_1.jpg)
Keep ALL small buckets intact
Trim only from largest buckets using stride sampling

Example: Want 500 from 1,390 camera trap images
• 50 small buckets (1 each): Keep all = 50 images ✓
• 30 medium buckets (5 each): Keep all = 150 images ✓  
• 19 large buckets (10 each): Keep all = 190 images ✓
• 1 huge bucket (1000): Keep every 9th with camera/folder respect = 110 images ✓
Total: 500 images with maximum diversity + camera location preservation
```

### Why It's Fast

**Algorithm advantages:**
- ✅ **O(n) complexity** - Just sort buckets once
- ✅ **Stride sampling** - Array slicing, not random selection
- ✅ **No complex math** - Simple bucket trimming
- ✅ **Maximum diversity** - Small buckets always preserved
- ✅ **Smart folder ordering** - Natural sorting preserves camera/folder structure

**What you get:**
- ✅ Fastest possible while maintaining quality
- ✅ Preserves rare/unique images (small buckets)
- ✅ Even sampling across camera locations and time sequences  
- ✅ Natural file ordering (IMG_1.jpg → IMG_2.jpg → IMG_10.jpg)

**Result:** Optimal speed + maximum diversity preservation + smart camera trap ordering.

## Algorithm Comparison

| Approach | Speed | Diversity | Camera/Folder Aware | Use Case |
|----------|-------|-----------|---------------------|----------|
| **Random sampling** | Fastest | Poor | No | Quick tests only |
| **smartdownsample** | Ultra Fast | Excellent | Yes | Camera trap data |
| **Complex diversity** | Very Slow | Perfect | No | Research only |

### Real Example: 24,000 camera trap images → 1,000 selected
- **Random**: 1 second, poor diversity, ignores folder structure
- **smartdownsample**: 20 seconds, excellent diversity + respects camera locations
- **Complex**: 2+ hours, mathematically perfect but breaks folder grouping

**Sweet spot:** Maximum diversity preservation with camera-aware sampling in minimal time.

## Performance

| Task | Time |
|------|------|
| 100 from 1,000 | <5 sec |
| 900 from 1,000 | <5 sec |
| 1,000 from 24,000 | ~30 sec |
| 23,000 from 24,000 | ~30 sec |
| Any ratio | Fast ✓ |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_paths` | Required | List of image file paths (str or Path objects) |
| `target_count` | Required | Exact number of images to select |
| `n_workers` | `4` | Number of parallel workers (4 is optimal) |
| `hash_size` | `8` | Hash size (8 is fast and good enough) |
| `random_seed` | `42` | Random seed for reproducible results |
| `show_progress` | `True` | Whether to display progress bars |

## Why It's Fast

- **Fixed algorithm** - No switching between methods
- **Simple hashing** - DHash is faster than PHash
- **Smart bucketing** - O(n) grouping instead of O(n²) comparisons
- **Natural sorting** - Built-in natsort handles camera trap folder structures efficiently
- **Parallel processing** - But capped at 4 workers (diminishing returns above that)

## License

MIT License – see LICENSE file.
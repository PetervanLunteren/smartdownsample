# smartdownsample

**Fast, good-enough image downsampling designed for camera trap animal crops**

SmartDownsample is specifically designed for camera trap images of animals, particularly cropped animal images where the subject is centered. It uses multi-dimensional visual features (DHash, AHash, color analysis) optimized for animal detection and camera trap scenarios.

**Honest trade-offs**: This tool prioritizes speed over perfection. It will do a pretty good job in minutes on 100k+ image datasets, rather than perfect results in hours or days. If you need mathematically optimal diversity, use specialized research tools. If you need fast, good-enough sampling for camera trap workflows, this is for you.

**Other use cases**: While designed for camera trap animal crops, it may work reasonably well for other centered-subject image collections (portraits, product photos, etc.).

## Installation

```bash
pip install smartdownsample
```

## Features

- 🐾 **Camera trap focused** - Designed for animal crops with center-focused detection
- 🎯 **Multi-dimensional features** - DHash (structure), AHash (brightness), color variance, color themes
- 🎨 **Environment aware** - Separates blue snow, green forest, brown desert scenes
- 💡 **Lighting distinction** - Groups grayscale IR vs color daylight images
- ⚡ **Fast at scale** - Minutes for 100k+ images, not hours
- 📊 **Smart bucketing** - 16-128 meaningful groups based on actual visual content
- 📁 **Camera trap friendly** - Natural sorting preserves folder structure and time sequences
- 📈 **Built-in visualization** - 5x5 thumbnail grids and distribution charts
- 🎲 **Reproducible** - Set seed for consistent results

## Usage

```python
from smartdownsample import sample_diverse

# Basic usage - intelligent visual diversity
selected = sample_diverse(
    image_paths=my_image_list,
    target_count=1000
)

# Full feature usage with visualization
selected = sample_diverse(
    image_paths=my_camera_trap_images,
    target_count=1000,
    hash_size=8,                # Perceptual hash size (8 recommended)
    n_workers=4,                # Parallel workers
    show_progress=True,         # Progress bars
    random_seed=42,             # Reproducible results
    show_summary=True,          # Text statistics  
    show_distribution=True,     # Bucket distribution chart
    show_thumbnails=True        # 10x10 thumbnail grids per bucket
)

print(f"Selected {len(selected)} images from {len(buckets)} visual similarity groups")
```

## Visualization Options

The algorithm includes three built-in visualization modes to understand bucket quality:

```python
# 1. Text summary (show_summary=True) - Default
selected = sample_diverse(paths, target_count=1000, show_summary=True)
# Prints: bucket sizes, distribution stats, diversity metrics

# 2. Distribution chart (show_distribution=True) 
selected = sample_diverse(paths, target_count=1000, show_distribution=True)  
# Shows: vertical bar chart of kept vs excluded per bucket

# 3. Thumbnail grids (show_thumbnails=True)
selected = sample_diverse(paths, target_count=1000, show_thumbnails=True)
# Shows: 10x10 grids of first 100 images from each bucket in square layout

# All visualizations together
selected = sample_diverse(paths, target_count=1000, 
                         show_summary=True, 
                         show_distribution=True, 
                         show_thumbnails=True)
```

## How It Works

Multi-dimensional visual similarity algorithm optimized for camera trap data:

### 1. **Multi-Feature Extraction** 
Each image is analyzed using 4 complementary visual features:

```python
# For each image, compute:
1. DHash (8x8) → Structural patterns, edges, shapes
2. AHash (4x4) → Brightness distribution, contrast  
3. Color Variance → Separates grayscale from colorful images
4. Overall Brightness → Separates dark from bright scenes
```

### 2. **Center-Focused Animal Detection**
For camera trap data where animals are typically centered:

```python
# From 8x8 DHash (64 bits), strategically sample center positions:
center_indices = [27, 36]  # Center-left and center-right positions
# Bit 27: Detects vertical edges (animal body/legs)  
# Bit 36: Detects horizontal edges (animal head/back)
```

### 3. **Smart Bucket Key Creation** 
Combine features into meaningful visual groups (max 32 buckets):

```python
bucket_key = (
    structure_bit_27,     # Center-left animal features (0 or 1)
    structure_bit_36,     # Center-right animal features (0 or 1)  
    brightness_pattern,   # AHash brightness pattern (0 or 1)
    color_type,          # Grayscale=0, Color=1
    brightness_level     # Dark=0, Bright=1
)
# Results in 2×2×2×2×2 = 32 maximum buckets
```

### 4. **Diversity-Preserving Selection**
```python
# Phase 1: Ensure diversity - sample from every bucket
# Phase 2: Fill remaining quota proportionally from largest buckets
# Within buckets: Natural sort preserves camera/folder structure

Example output buckets for camera trap data:
• Bucket 1: Dark grayscale deer (vertical edges)
• Bucket 2: Bright color birds (horizontal patterns) 
• Bucket 3: Grayscale empty frames (low structure)
• Bucket 4: Color daytime mammals (mixed patterns)
```

## Algorithm Benefits

### Visual Similarity Improvements
- **Better separation**: Color vs grayscale images grouped separately
- **Animal-focused**: Center-positioned features detect different animal poses/species  
- **Brightness aware**: Day vs night scenes properly distinguished
- **Structure sensitive**: Different animal orientations and camera angles detected
- **Manageable buckets**: 16-32 meaningful groups instead of random mixing

### Performance Characteristics
- **Still fast**: Multi-feature extraction adds minimal overhead (~20% slower)
- **Linear scaling**: O(n) complexity maintained across all features
- **Memory efficient**: Features computed on-the-fly, not stored
- **Parallel processing**: Hash computation parallelized across workers
- **Smart bucket counts**: Never creates excessive micro-buckets

### Camera Trap Optimizations
- **Natural sorting**: Preserves camera/folder structure (CAM01_IMG_001.jpg → CAM01_IMG_010.jpg)
- **Center detection**: Focus on image center where animals appear
- **Scene variety**: Separates empty frames, single animals, multiple animals
- **Lighting diversity**: Day/night scenes properly represented
- **Color preservation**: IR grayscale vs color daylight images distinguished

## Comparison with Other Methods

| Method | Bucket Quality | Speed | Animal Detection | Color Separation |
|--------|---------------|-------|------------------|------------------|
| **Random sampling** | None | Fastest | No | No |
| **Single DHash** | Poor mixing | Fast | No | No |  
| **smartdownsample v1.6+** | Excellent | Fast+ | Yes | Yes |
| **Complex ML clustering** | Perfect | Very Slow | Depends | Yes |

### Real Results: Camera Trap Dataset
**Before (v1.5)**: 495 buckets, color/grayscale mixed randomly in each bucket  
**After (v1.6+)**: 32 buckets, clear separation:
- Bucket 1: Grayscale deer images (IR night camera)
- Bucket 2: Color bird images (daylight camera)  
- Bucket 3: Dark empty frames (nighttime)
- Bucket 4: Bright color mammals (sunny daytime)

**Performance**: Only ~20% slower than single-hash method, dramatically better visual grouping.

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
| `hash_size` | `8` | Perceptual hash size - 8 recommended for good speed/quality balance |
| `n_workers` | `4` | Number of parallel workers for hash computation |
| `show_progress` | `True` | Display progress bars during processing |
| `random_seed` | `42` | Random seed for reproducible bucket selection |
| `show_summary` | `True` | Print bucket statistics and distribution summary |
| `show_distribution` | `False` | Show bucket distribution bar chart (requires matplotlib) |
| `show_thumbnails` | `False` | Show 10x10 thumbnail grids for each bucket (requires matplotlib) |

### Parameter Recommendations

**For camera trap animal crops (recommended use case):**
- `hash_size=8`: Optimal balance of speed and center-focused animal detection
- `show_thumbnails=True`: Essential for validating visual similarity quality
- `show_summary=True`: Understand bucket distribution and diversity

**For other centered-subject images:**
- `hash_size=8`: Still works well for portraits, product photos, etc.
- May work less effectively for landscapes, random compositions, or non-centered subjects

**Performance tuning:**
- `hash_size=6`: Faster processing, may reduce detection quality
- `hash_size=10`: Slower but more detailed structural analysis

## Technical Details

### Hash Features Explained

**DHash (Difference Hash)**
- Detects structural patterns, edges, object boundaries
- 8×8 hash = 64 bits representing horizontal gradients
- Center bits (positions 27, 36) focus on animal detection
- Fast computation: resize → grayscale → compare adjacent pixels

**AHash (Average Hash)** 
- Detects brightness patterns and contrast distribution
- 4×4 hash = 16 bits representing above/below average brightness
- Used for distinguishing lighting conditions
- Complements DHash with tonal information

**Color Variance**
- Separates grayscale (IR cameras) from color (daylight cameras)  
- Computed as variance of RGB channel means
- Threshold: variance < 100 = grayscale, ≥ 100 = color

**Overall Brightness**
- Separates dark (nighttime) from bright (daytime) scenes
- Computed as mean pixel value across all channels
- Threshold: brightness < 128 = dark, ≥ 128 = bright

### Performance Notes
- Multi-feature extraction adds ~20% processing time vs single hash
- Parallel hash computation scales linearly with worker count (up to CPU cores)
- Memory usage remains O(1) - features computed on-demand
- Bucket creation is O(n) - no expensive similarity comparisons

## License

MIT License – see LICENSE file.
# smartdownsample

**Fast and smart(_-ish_) image downsampling for large datasets**

`smartdownsample` helps select representative subsets of camera trap images. In many machine learning workflows, majority classes can contain hundreds of thousands of images. These often need to be downsampled for processing efficiency or dataset balance, but without losing too much valuable variation.  

An ideal solution would keep only truly distinct images and exclude near-duplicates, but that is very computationally expensive for large datasets. This package provides a practical compromise: fast downsampling that preserves diversity with minimal computations, reducing processing time from hours to minutes.  

If you need mathematically perfect results, this isn’t the tool. But if you want a smart, lightweight alternative that does a lot better than random sampling → `smartdownsample`.

## Installation

```bash
pip install smartdownsample
```

## Usage

```python
from smartdownsample import sample_diverse

# List of image paths
my_image_list = [
    "path/to/img1.jpg",
    "path/to/img2.jpg",
    "path/to/img3.jpg",
    # ...
]

# Basic usage
selected = sample_diverse(
    image_paths=my_image_list,
    target_count=50000
)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_paths` | Required | List of image file paths (str or Path objects) |
| `target_count` | Required | Exact number of images to select |
| `hash_size` | `8` | Perceptual hash size (8 recommended) |
| `n_workers` | `4` | Number of parallel workers for hash computation |
| `show_progress` | `True` | Display progress bars during processing |
| `random_seed` | `42` | Random seed for reproducible bucket selection |
| `show_summary` | `True` | Print bucket statistics and distribution summary |
| `show_distribution` | `False` | Show bucket distribution bar chart |
| `show_thumbnails` | `False` | Show 5x5 thumbnail grids for each bucket |

## How it works

The algorithm balances speed and diversity in four steps:

1. **Feature extraction**  
   Each image is reduced to a compact set of visual features:  
   - DHash (`2 bits`) → structure/edges  
   - AHash (`1 bit`) → brightness/contrast  
   - Color variance (`1 bit`) → grayscale vs. color  
   - Overall brightness (`1 bit`) → dark vs. bright  
   - Average color (`1 bit`) → dominant scene color (red/green/blue/neutral)  

2. **Bucket grouping**  
   Images are sorted into "similarity buckets" based on the visual features extracted at step 1.  
   - At most 128 buckets are possible (2×2×2×2×2×4 feature splits).  
   - In practice, most datasets produce only a few dozen buckets, depending on their diversity.  

3. **Selection across buckets**  
   - Ensure at least one image per bucket (diversity first)  
   - Fill the remaining quota proportionally from larger buckets  

4. **Within-bucket selection**  
   - Buckets are kept in their natural folder order  
   - This preserves any inherent structure in the dataset (e.g., locations, events, sequences, etc)  
   - Images are then sampled at regular intervals (every stride-th image) until the target count is reached, ensuring a systematic spread across the bucket  

5. **Optionally show distribution chart**  
   - Vertical bar chart of kept vs. excluded images per bucket  
<img src="https://github.com/PetervanLunteren/EcoAssist-metadata/blob/main/smartdown-sample/bar.png" width="80%">


6. **Optionally show thumbnail grids**  
   - 5×5 grids from each bucket, for quick visual review  
<img src="https://github.com/PetervanLunteren/EcoAssist-metadata/blob/main/smartdown-sample/grid.png" width="80%">


## License

MIT License → see [LICENSE file](https://github.com/PetervanLunteren/smartdownsample/blob/main/LICENSE).

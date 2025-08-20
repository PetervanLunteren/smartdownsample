# smartdownsample

**Fast image downsampling for large camera trap animal crop datasets**

`smartdownsample` helps select representative subsets of camera trap images, particularly centered animal crops. In many machine learning workflows, majority classes may contain hundreds of thousands of images. These often need to be downsampled for processing efficiency or dataset balance, but without losing valuable variation.  

An ideal solution would retain only truly distinct images and exclude near-duplicates, but that is computationally expensive for large datasets. This package provides a practical compromise: fast downsampling that preserves diversity with minimal computations, reducing processing time from hours or days to just minutes.  

If you need mathematically optimal results, this tool is not the right fit. If you want a reasonably intelligent selection that is more effective than random sampling, `smartdownsample` is designed for you.

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
    target_count=1000
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

   Maximum: 128 theoretical buckets (2×2×2×2×2×4)  
   Typical: 16–80 buckets, depending on dataset diversity  

   Examples of resulting groups:  
   - Dark grayscale (night IR)  
   - Bright blue snow scenes  
   - Color forest images with mixed poses  

2. **Bucket grouping**  
   Images are assigned to similarity buckets based on these features.  

3. **Selection across buckets**  
   - Ensure at least one image per bucket (diversity first)  
   - Fill the remaining quota proportionally from larger buckets  

4. **Within-bucket selection**  
   - Buckets are naturally sorted by folder structure
   - Locations, deployments, and sequences stay together in order  
   - Take every stride-th image until quota is met, ensuring a systematical sample across time and space

5. **Optionally show distribution chart**  
   - Vertical bar chart of kept vs. excluded images per bucket  
<img src="https://github.com/PetervanLunteren/EcoAssist-metadata/blob/main/smartdown-sample/bar.png" width="50%">


6. **Optionally show thumbnail grids**  
   - 5×5 grids of the first 25 images from each bucket, for quick visual review  
<img src="https://github.com/PetervanLunteren/EcoAssist-metadata/blob/main/smartdown-sample/grid.png" width="50%">


## License

MIT License – see LICENSE file.

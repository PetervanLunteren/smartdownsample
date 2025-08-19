# smartdownsample

**Efficient downsampling for image classification datasets**

SmartDownsample selects the most diverse images from large collections, ideal for reducing dataset size while preserving visual variability.

## Installation

```bash
pip install smartdownsample
```

## Usage

```python
from smartdownsample import select_distinct

# Example list of image paths
my_image_list = [
    "path/to/img1.jpg",
    "path/to/img2.jpg",
    "path/to/img3.jpg",
    "path/to/img4.jpg"
]

# Simple selection - get 100 most diverse images
selected = select_distinct(
    image_paths=my_image_list,
    target_count=100
)

# With visual verification to see excluded images in context
selected = select_distinct(
    image_paths=my_image_list,
    target_count=100,
    show_verification=True
)

print(f"Selected {len(selected)} images")

```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_paths` | Required | List of image file paths (str or Path objects) |
| `target_count` | Required | Exact number of images to select |
| `window_size` | `100` | Rolling window size (larger = better quality, slower) |
| `random_seed` | `42` | Random seed for reproducible results |
| `show_progress` | `True` | Whether to display progress bars |
| `show_verification` | `False` | Show visual verification comparing excluded vs included images |

## Step by Step

1. **Sort paths** by directory. Within each folder, files are naturally ordered (e.g., `img1.jpg`, `img2.jpg`, `img10.jpg`) so related images remain grouped.  
2. **Compute perceptual hashes** for all valid image paths.  
3. **Apply rolling window selection** on the hash array to choose indices of the most diverse images. This runs in O(n) time, scales to large classes of 100k+ images, and compares each candidate only to a sliding window of recent selections.  
4. **Return results** as `[valid_paths[i] for i in selected_indices]`.  
5. **Optional verification plot**: If `show_verification=True`, the algorithm displays a visual check of 18 randomly selected excluded images and their included counterpart. The visualization opens automatically in your default image viewer without saving files to disk.

<p align="center">
  <img src="https://raw.githubusercontent.com/PetervanLunteren/EcoAssist-metadata/b72573d4cad68301602ca4aceab1bdc7b62d95df/downsample-ex.png" alt="Downsample example" width="500"/>
</p>


## License

MIT License – see LICENSE file.

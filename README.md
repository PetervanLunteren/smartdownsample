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

## Step by step

1. Images are processed in folder-major order so files from the same folder stay together. Within each folder, paths are naturally sorted (e.g., `img1.jpg`, `img2.jpg`, `img10.jpg`) to keep related images grouped during processing.

2. Images are compared using a rolling window. This runs in O(n) time and scales to large lists images by comparing each candidate only to a sliding window of recent selections. The method computes perceptual hashes and applies greedy selection to maximize diversity while maintaining efficiency.

## License

MIT License – see LICENSE file.

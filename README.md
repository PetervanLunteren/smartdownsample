# SmartDownsample

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

## Algorithm

**Rolling Window:** O(n) time complexity, scalable to 100k+ images. Compares candidates to recent selections using a sliding window approach. Computes perceptual hashes and uses greedy selection to maximize diversity while maintaining efficiency.

Images are automatically sorted by folder structure with natural ordering (img1, img2, img10) to keep related images together during processing.

## License

MIT License – see LICENSE.md file.

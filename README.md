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

## How it works

1. **Natural sorting**: Images are processed in folder-major order with natural filename sorting (e.g., `img1.jpg`, `img2.jpg`, `img10.jpg`) to keep related images grouped.

2. **Rolling window algorithm**: Efficient O(n) algorithm that scales to 100k+ images. Compares each candidate only to a sliding window of recent selections rather than all previous selections.

3. **Diversity maximization**: Selects images that are maximally different from already-chosen images, eliminating duplicates and similar poses while preserving visual variety.

## Visual Verification

Set `show_verification=True` to see a comprehensive visualization showing up to 18 excluded images alongside their most similar included images. Features:

- **Side-by-side comparisons**: Red-bordered excluded images next to green-bordered included images
- **Professional layout**: Clean 3×6 grid showing multiple examples 
- **Algorithm validation**: Verify that excluded images are genuinely similar to included ones
- **Trust building**: Visual proof that the algorithm makes sensible decisions

The visualization opens automatically in your default image viewer without saving files to disk.

## License

MIT License – see LICENSE file.

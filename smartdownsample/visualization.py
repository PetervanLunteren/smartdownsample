"""
Visual verification module for SmartDownsample.

Creates side-by-side comparison images showing excluded images next to their
most similar selected images to help verify algorithm performance.
"""

import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List

from .core import _calculate_hashes, _hash_to_binary_array, _hamming_distance


def create_verification_image(valid_paths: List[str], selected_indices: List[int], random_seed: int = 42) -> None:
    """
    Create and display a visual verification image showing excluded images alongside 
    their most similar selected images.
    
    Args:
        valid_paths: List of all valid image file paths
        selected_indices: Indices of selected (included) images  
        random_seed: Random seed for reproducible visualization
    """
    
    random.seed(random_seed)
    
    # Find excluded indices
    selected_set = set(selected_indices)
    excluded_indices = [i for i in range(len(valid_paths)) if i not in selected_set]
    
    if len(excluded_indices) == 0:
        print("No excluded images to visualize.")
        return
    
    # Randomly select up to 18 excluded images for 18 pairs (3 per row, 6 rows)
    num_comparisons = min(18, len(excluded_indices))
    selected_exclusions = random.sample(excluded_indices, num_comparisons)
    
    print(f"Finding most similar selected images for {num_comparisons} excluded images...")
    
    # Calculate hashes for comparison
    print("Calculating perceptual hashes for comparison...")
    hashes, _ = _calculate_hashes(valid_paths, False)
    hash_arrays = np.array([_hash_to_binary_array(h) for h in hashes])
    
    print("Creating side-by-side comparison image...")
    
    # Create the visualization
    canvas = _create_comparison_grid(
        valid_paths=valid_paths,
        selected_exclusions=selected_exclusions,
        selected_indices=selected_indices,
        hash_arrays=hash_arrays
    )
    
    # Display the image directly without saving to disk
    try:
        canvas.show()
        print("Verification image displayed.")
    except:
        print("Could not display image automatically. Saving as fallback...")
        output_path = "verification_comparison.png"
        canvas.save(output_path)
        print(f"Verification image saved as: {output_path}")


def _create_comparison_grid(valid_paths: List[str], selected_exclusions: List[int], 
                           selected_indices: List[int], hash_arrays: np.ndarray) -> Image.Image:
    """
    Create the comparison grid image showing excluded vs selected pairs.
    
    Returns:
        PIL Image object containing the visualization
    """
    
    # Layout configuration
    img_size = 180  # Image dimensions
    border_size = 20  # Space around images within pair borders
    pair_gap = 70  # Space between pairs (horizontal)
    image_gap = 15  # Space between images within a pair
    row_gap = 8  # Vertical space between pair rows
    
    # Grid layout: 3 pairs per row, 6 rows
    pairs_per_row = 3
    rows = 6
    
    # Calculate canvas size
    canvas_width = pairs_per_row * (2 * img_size + image_gap + pair_gap) + pair_gap
    canvas_height = rows * (img_size + border_size * 2) + (rows - 1) * row_gap + border_size + 50
    
    # Create white canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Load fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)  # Large title
    except:
        title_font = ImageFont.load_default()
    
    # Draw title with border explanation
    title_text = "Why these images were excluded: red border = excluded, green border = included"
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((canvas_width - title_width) // 2, 10), title_text, fill='black', font=title_font)
    
    y_offset = 80  # Start below larger title
    
    # Create each comparison pair
    for pair_idx, excluded_idx in enumerate(selected_exclusions):
        _draw_comparison_pair(
            draw=draw,
            canvas=canvas,
            pair_idx=pair_idx,
            excluded_idx=excluded_idx,
            valid_paths=valid_paths,
            selected_indices=selected_indices,
            hash_arrays=hash_arrays,
            img_size=img_size,
            border_size=border_size,
            pair_gap=pair_gap,
            image_gap=image_gap,
            row_gap=row_gap,
            y_offset=y_offset,
            pairs_per_row=pairs_per_row
        )
    
    return canvas


def _draw_comparison_pair(draw: ImageDraw.Draw, canvas: Image.Image, pair_idx: int, 
                         excluded_idx: int, valid_paths: List[str], selected_indices: List[int],
                         hash_arrays: np.ndarray, img_size: int, border_size: int,
                         pair_gap: int, image_gap: int, row_gap: int, y_offset: int,
                         pairs_per_row: int) -> None:
    """
    Draw a single comparison pair (excluded image + most similar selected image).
    """
    
    # Find most similar selected image
    excluded_hash = hash_arrays[excluded_idx]
    min_distance = float('inf')
    most_similar_selected = None
    
    for selected_idx in selected_indices:
        distance = _hamming_distance(excluded_hash, hash_arrays[selected_idx])
        if distance < min_distance:
            min_distance = distance
            most_similar_selected = selected_idx
    
    # Calculate position in grid
    row = pair_idx // pairs_per_row
    col = pair_idx % pairs_per_row
    
    x_start = col * (2 * img_size + pair_gap) + pair_gap
    y_start = y_offset + row * (img_size + border_size * 2 + row_gap)
    
    # Load and resize images
    try:
        excluded_img = Image.open(valid_paths[excluded_idx]).convert('RGB').resize((img_size, img_size))
    except:
        excluded_img = Image.new('RGB', (img_size, img_size), (200, 200, 200))
    
    if most_similar_selected is not None:
        try:
            selected_img = Image.open(valid_paths[most_similar_selected]).convert('RGB').resize((img_size, img_size))
        except:
            selected_img = Image.new('RGB', (img_size, img_size), (200, 200, 200))
    else:
        selected_img = Image.new('RGB', (img_size, img_size), (200, 200, 200))
    
    # Draw background rectangle for the pair - light grey
    bg_color = (248, 248, 248)
    draw.rectangle([x_start - border_size, y_start - border_size, 
                   x_start + 2 * img_size + image_gap + border_size, y_start + img_size + border_size], 
                  fill=bg_color, outline='black', width=2)
    
    # Draw excluded image with red border
    draw.rectangle([x_start - 3, y_start - 3, x_start + img_size + 3, y_start + img_size + 3], 
                  fill='red', width=3)
    canvas.paste(excluded_img, (x_start, y_start))
    
    # Draw selected image with green border (with gap between images)
    selected_x = x_start + img_size + image_gap
    draw.rectangle([selected_x - 3, y_start - 3, 
                   selected_x + img_size + 3, y_start + img_size + 3], 
                  fill='green', width=3)
    canvas.paste(selected_img, (selected_x, y_start))
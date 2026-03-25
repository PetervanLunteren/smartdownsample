"""
Embedding-based diverse image sampling using DINOv2 and divide-and-conquer clustering.

Adapted from Dante Wasmuht's sentinel-pipeline-clustering (Conservation X Labs).
"""

import os
import re
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings
from natsort import natsorted

import torch
import torchvision.transforms as T
from sklearn.cluster import AgglomerativeClustering

warnings.filterwarnings('ignore')

# --- Constants ---

MODEL_NAME = 'dinov2_vits14'
EMBEDDING_DIM = 384
INFERENCE_BATCH_SIZE = 64
CHUNK_SIZE = 2000
N_REP = 5
DISTANCE_THRESHOLD = 0.5
SEED = 42

# --- Model cache ---

_model = None
_device = None
_transform = None


def _patch_dinov2_for_older_python():
    """
    Patch cached DINOv2 source files for Python < 3.10 compatibility.

    The latest DINOv2 code uses PEP 604 union syntax (float | None) which requires
    Python 3.10+. This adds 'from __future__ import annotations' to affected files.
    """
    hub_dir = torch.hub.get_dir()
    dinov2_dir = os.path.join(hub_dir, 'facebookresearch_dinov2_main')
    if not os.path.isdir(dinov2_dir):
        return

    for root, _dirs, files in os.walk(dinov2_dir):
        for filename in files:
            if not filename.endswith('.py'):
                continue
            filepath = os.path.join(root, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            if 'from __future__ import annotations' in content:
                continue
            if re.search(r':\s*\w+\s*\|\s*\w+', content) or re.search(r'->\s*\w+\s*\|\s*\w+', content):
                content = 'from __future__ import annotations\n' + content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)


def _get_model():
    """Load DINOv2 ViT-S/14 on first use, cache for subsequent calls."""
    global _model, _device, _transform
    if _model is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            _model = torch.hub.load('facebookresearch/dinov2', MODEL_NAME)
        except TypeError:
            # PEP 604 syntax error on Python < 3.10 -- patch and retry
            _patch_dinov2_for_older_python()
            _model = torch.hub.load('facebookresearch/dinov2', MODEL_NAME)
        _model = _model.to(_device).eval()
        _transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _model, _device, _transform


# --- Helpers ---

def _hierarchical_natsort(paths: List[Union[str, Path]]) -> List[str]:
    """
    Sort paths hierarchically with natural ordering.

    Files from different directories are not interleaved. Within each directory,
    files are sorted naturally, then subdirectories are processed recursively.
    """
    path_objects = [Path(p) for p in paths]
    sorted_paths = natsorted(path_objects, key=lambda p: p.parts)
    return [str(p) for p in sorted_paths]


def _validate_png_path(path: Optional[str], param_name: str) -> Optional[Path]:
    """Validate and prepare PNG file path."""
    if path is None:
        return None

    if isinstance(path, bool):
        raise ValueError(f"{param_name} must be a file path string or None, not a boolean. "
                        f"The API has changed: use {param_name}='path/to/file.png' to save, or None to skip.")

    path_obj = Path(path)

    if path_obj.suffix.lower() != '.png':
        raise ValueError(f"{param_name} must be a .png file path, got: {path}")

    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


# --- Embedding computation ---

def _load_image(path: str, transform, image_loading_errors: str):
    """Load and transform a single image. Returns (path, tensor, error_msg)."""
    try:
        with Image.open(path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            tensor = transform(img)
            return str(path), tensor, None
    except (FileNotFoundError, PermissionError, IOError, OSError) as e:
        error_type = type(e).__name__
        if error_type == "FileNotFoundError":
            error_msg = f"File not found: {path}"
        elif error_type == "PermissionError":
            error_msg = f"Permission denied: {path}"
        else:
            error_msg = f"Cannot read image file: {path}"

        if image_loading_errors == "raise":
            raise type(e)(f"{error_msg}. Use image_loading_errors='skip' to continue with remaining images") from e
        else:
            return str(path), None, error_msg
    except Exception as e:
        error_type = type(e).__name__
        error_msg = f"Image processing failed ({error_type}): {path}"
        if image_loading_errors == "raise":
            raise
        return str(path), None, error_msg


def _compute_embeddings(image_paths, n_workers, show_progress, image_loading_errors):
    """
    Compute DINOv2 embeddings for a list of image paths.

    Returns:
        valid_paths: List of paths that were successfully processed
        embeddings: numpy array of shape (N, 384), L2-normalized
        failed_paths: List of (path, error_msg) tuples
    """
    model, device, transform = _get_model()

    # Load and transform images in parallel
    valid_paths = []
    tensors = []
    failed_paths = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_load_image, path, transform, image_loading_errors)
                   for path in image_paths]

        if show_progress:
            futures_iter = tqdm(futures, desc=" - Loading images")
        else:
            futures_iter = futures

        for future in futures_iter:
            path, tensor, error_msg = future.result()
            if tensor is not None:
                valid_paths.append(path)
                tensors.append(tensor)
            else:
                failed_paths.append((path, error_msg))

    if not tensors:
        return valid_paths, np.empty((0, EMBEDDING_DIM)), failed_paths

    # Batch inference
    all_embeddings = []

    if show_progress:
        n_batches = (len(tensors) + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE
        batch_iter = tqdm(range(0, len(tensors), INFERENCE_BATCH_SIZE),
                         desc=" - Computing embeddings", total=n_batches)
    else:
        batch_iter = range(0, len(tensors), INFERENCE_BATCH_SIZE)

    with torch.no_grad():
        for start in batch_iter:
            batch = torch.stack(tensors[start:start + INFERENCE_BATCH_SIZE]).to(device)
            features = model(batch)
            all_embeddings.append(features.cpu().numpy())

    embeddings = np.vstack(all_embeddings)

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    embeddings = embeddings / norms

    return valid_paths, embeddings, failed_paths


# --- Divide-and-conquer clustering (adapted from Dante Wasmuht / Conservation X Labs) ---

def _select_medoids(embeddings, labels, n_rep=N_REP):
    """
    Select representative points from each cluster.

    For each cluster, picks up to n_rep points closest to the cluster centroid.
    Uses O(N) centroid distance instead of O(N^2) pairwise distances.

    Returns:
        rep_embeddings: numpy array of representative embeddings
        rep_indices: list of original indices for each representative
        cluster_membership: dict mapping cluster_label -> list of original indices
    """
    unique_labels = np.unique(labels)
    rep_embeddings = []
    rep_indices = []
    cluster_membership = {}

    for label in unique_labels:
        mask = labels == label
        cluster_indices = np.where(mask)[0]
        cluster_pts = embeddings[mask]
        cluster_membership[label] = cluster_indices.tolist()

        if len(cluster_pts) <= n_rep:
            rep_embeddings.append(cluster_pts)
            rep_indices.extend(cluster_indices.tolist())
        else:
            centroid = cluster_pts.mean(axis=0, keepdims=True)
            dists = np.linalg.norm(cluster_pts - centroid, axis=1)
            sorted_idx = np.argsort(dists)[:n_rep]
            rep_embeddings.append(cluster_pts[sorted_idx])
            rep_indices.extend(cluster_indices[sorted_idx].tolist())

    return np.vstack(rep_embeddings), rep_indices, cluster_membership


def _cluster_with_threshold(embeddings, threshold):
    """Run AgglomerativeClustering with the given cosine distance threshold."""
    agg = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, metric='cosine', linkage='average'
    )
    return agg.fit_predict(embeddings)


def _divide_and_conquer_cluster(embeddings, distance_threshold=DISTANCE_THRESHOLD, show_progress=True):
    """
    Cluster embeddings using divide-and-conquer for scalability.

    Uses a cosine distance threshold so the number of clusters reflects the
    natural visual structure of the data. The sampling phase then handles
    budget allocation across clusters.

    For small datasets (<= CHUNK_SIZE), clusters directly.
    For larger datasets, divides into chunks, clusters each, selects medoid
    representatives, re-clusters the representatives, and propagates labels.

    Returns:
        cluster_labels: numpy array of shape (N,) with cluster IDs
    """
    n = len(embeddings)
    threshold = distance_threshold

    # Direct clustering for small datasets
    if n <= CHUNK_SIZE:
        if show_progress:
            print(f" - Clustering {n:,} embeddings (threshold={threshold})...")
        labels = _cluster_with_threshold(embeddings, threshold)
        n_clusters = len(np.unique(labels))
        if show_progress:
            print(f" - Found {n_clusters} clusters")
        return labels

    # Divide-and-conquer for large datasets
    n_chunks = max(1, n // CHUNK_SIZE)
    if show_progress:
        print(f" - Divide-and-conquer: splitting {n:,} embeddings into {n_chunks} chunks (threshold={threshold})...")

    # Shuffle indices deterministically
    rng = np.random.RandomState(SEED)
    indices = np.arange(n)
    rng.shuffle(indices)
    chunk_indices = np.array_split(indices, n_chunks)

    # Cluster each chunk
    chunk_labels = []
    chunk_memberships = []
    all_rep_embeddings = []
    all_rep_global_indices = []

    if show_progress:
        chunk_iter = tqdm(enumerate(chunk_indices), desc=" - Clustering chunks",
                         total=len(chunk_indices))
    else:
        chunk_iter = enumerate(chunk_indices)

    for chunk_id, idx in chunk_iter:
        chunk_emb = embeddings[idx]
        labels = _cluster_with_threshold(chunk_emb, threshold)

        # Select medoids from this chunk
        rep_emb, rep_local_indices, membership = _select_medoids(chunk_emb, labels)

        # Map local indices back to global indices
        rep_global = [idx[i] for i in rep_local_indices]
        all_rep_embeddings.append(rep_emb)
        all_rep_global_indices.extend(rep_global)

        # Store chunk info for label propagation
        chunk_labels.append(labels)
        chunk_memberships.append((idx, membership))

    # Re-cluster all representative points
    rep_embeddings = np.vstack(all_rep_embeddings)
    if show_progress:
        print(f" - Re-clustering {len(rep_embeddings):,} representative points...")
    rep_labels = _cluster_with_threshold(rep_embeddings, threshold)

    # Build mapping: global_index -> final_cluster_id for representatives
    rep_label_map = {}
    for global_idx, final_label in zip(all_rep_global_indices, rep_labels):
        rep_label_map[global_idx] = int(final_label)

    # Propagate final labels to all points
    final_labels = np.full(n, -1, dtype=np.int32)

    for chunk_id, (idx, membership) in enumerate(chunk_memberships):
        for local_label, local_indices in membership.items():
            global_indices = [idx[i] for i in local_indices]

            # Find any representative from this cluster to get the final label
            medoid_final_label = None
            for gi in global_indices:
                if gi in rep_label_map:
                    medoid_final_label = rep_label_map[gi]
                    break

            if medoid_final_label is not None:
                for gi in global_indices:
                    final_labels[gi] = medoid_final_label

    # Handle any unlabeled points (shouldn't happen, but be safe)
    unlabeled = final_labels == -1
    if np.any(unlabeled):
        max_label = final_labels.max() + 1
        final_labels[unlabeled] = max_label

    n_clusters = len(np.unique(final_labels))
    if show_progress:
        print(f" - Found {n_clusters} clusters")

    return final_labels


# --- Cluster-aware sampling ---

def _cluster_aware_sample(embeddings, cluster_labels, target_count):
    """
    Select target_count indices, maximizing cluster diversity.

    Phase 1: Take the most central image (medoid) from each cluster.
    Phase 2: Distribute remaining quota using largest-remainder allocation
             for fair proportional representation.

    Returns:
        List of selected indices (into the embeddings array)
    """
    unique_labels = np.unique(cluster_labels)

    # Build cluster info: indices and centrality ranking
    cluster_info = {}
    for label in unique_labels:
        mask = cluster_labels == label
        indices = np.where(mask)[0]
        cluster_pts = embeddings[indices]

        # Compute centrality: distance to centroid
        centroid = cluster_pts.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(cluster_pts - centroid, axis=1)
        ranked = indices[np.argsort(dists)]  # Most central first

        cluster_info[label] = {
            'ranked_indices': ranked,
            'size': len(indices),
            'taken': 0,
        }

    selected = []

    # Phase 1: Take 1 medoid from each cluster (diversity guarantee)
    # Sort clusters by size (largest first) so if we can't cover all, we cover the biggest
    sorted_labels = sorted(cluster_info.keys(),
                           key=lambda l: cluster_info[l]['size'], reverse=True)

    for label in sorted_labels:
        if len(selected) >= target_count:
            break
        info = cluster_info[label]
        selected.append(info['ranked_indices'][0])
        info['taken'] = 1

    remaining = target_count - len(selected)

    # Phase 2: Largest-remainder allocation for fair proportional distribution
    if remaining > 0:
        # Compute fractional shares for each cluster
        available_clusters = []
        for label in sorted_labels:
            info = cluster_info[label]
            avail = info['size'] - info['taken']
            if avail > 0:
                available_clusters.append((label, avail))

        total_available = sum(a for _, a in available_clusters)

        # Calculate floor allocation and fractional remainders
        allocations = []
        for label, avail in available_clusters:
            exact_share = (avail / total_available) * remaining
            floor_share = int(exact_share)
            fraction = exact_share - floor_share
            # Don't allocate more than available
            floor_share = min(floor_share, avail)
            allocations.append((label, floor_share, fraction, avail))

        # Give everyone their floor allocation
        allocated = sum(a[1] for a in allocations)
        leftover = remaining - allocated

        # Distribute leftover slots one-by-one to clusters with largest fractional remainders
        if leftover > 0:
            by_remainder = sorted(allocations, key=lambda a: a[2], reverse=True)
            for i in range(min(leftover, len(by_remainder))):
                label, floor_share, fraction, avail = by_remainder[i]
                if floor_share < avail:
                    by_remainder[i] = (label, floor_share + 1, fraction, avail)

            allocations = by_remainder

        # Apply allocations: select images by centrality rank
        for label, take, _, _ in allocations:
            if take > 0:
                info = cluster_info[label]
                start = info['taken']
                new_indices = info['ranked_indices'][start:start + take]
                selected.extend(new_indices.tolist())
                info['taken'] += take

    return selected


# --- Visualization (unchanged from v1) ---

def _print_cluster_summary(cluster_stats: List[Dict[str, Any]]) -> None:
    """Print a simple text summary of cluster statistics."""
    if not cluster_stats:
        print("No cluster statistics available")
        return

    print("\n" + "="*60)
    print("CLUSTER DISTRIBUTION SUMMARY")
    print("="*60)

    sorted_clusters = sorted(cluster_stats, key=lambda x: x['original_size'], reverse=True)

    total_images = sum(b['original_size'] for b in cluster_stats)
    total_selected = sum(b['kept'] for b in cluster_stats)

    print(f"Total images: {total_images:,}")
    print(f"Selected: {total_selected:,} ({(total_selected/total_images)*100:.1f}%)")
    print(f"Clusters: {len(cluster_stats)}")
    print()

    print("Per-cluster breakdown:")
    print("-" * 60)
    print(f"{'Cluster':<10} {'Size':<8} {'Kept':<8} {'Rate':<8}")
    print("-" * 60)

    for i, cluster in enumerate(sorted_clusters):
        size = cluster['original_size']
        kept = cluster['kept']
        rate = f"{(kept/size)*100:.0f}%" if size > 0 else "0%"
        print(f"#{i+1:<9} {size:<8,} {kept:<8,} {rate:<8}")

    print("-" * 60)
    print()


def _plot_cluster_thumbnails(cluster_stats: List[Dict[str, Any]], viz_data: Dict,
                             save_path: Optional[Path] = None, show_progress: bool = True) -> None:
    """Create and optionally save thumbnail grids for each cluster."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if save_path:
            raise ImportError("matplotlib is required to save thumbnail grids. Install it with: pip install matplotlib")
        print("matplotlib not available - skipping thumbnail grids")
        return

    if not cluster_stats or not viz_data:
        if save_path:
            raise ValueError("No cluster data available for thumbnails")
        print("No cluster data available for thumbnails")
        return

    sorted_clusters = sorted(cluster_stats, key=lambda x: x['original_size'], reverse=True)

    cluster_assignments = viz_data['cluster_assignments']
    all_paths = viz_data['all_paths']

    n_clusters = len(sorted_clusters)
    cols = int(np.ceil(np.sqrt(n_clusters)))
    rows = int(np.ceil(n_clusters / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 6))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    if show_progress:
        cluster_iter = tqdm(enumerate(sorted_clusters), desc=" - Creating thumbnail grids",
                           total=len(sorted_clusters))
    else:
        cluster_iter = enumerate(sorted_clusters)

    def create_grid(images, max_images=25):
        if not images:
            return np.ones((300, 300, 3), dtype=np.uint8) * 220
        if len(images) > max_images:
            rng = np.random.RandomState(SEED)
            indices = rng.choice(len(images), max_images, replace=False)
            sample_images = [images[i] for i in sorted(indices)]
        else:
            sample_images = images
        grid_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
        thumb_size = 60
        for idx, img_path in enumerate(sample_images[:25]):
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_thumb = img.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                    img_array = np.array(img_thumb, dtype=np.uint8)
                    row_pos = idx // 5
                    col_pos = idx % 5
                    y_s = row_pos * thumb_size
                    x_s = col_pos * thumb_size
                    grid_img[y_s:y_s + thumb_size, x_s:x_s + thumb_size] = img_array
            except Exception:
                continue
        return grid_img

    for cluster_idx, cluster_data in cluster_iter:
        original_cluster_idx = len(sorted_clusters) - 1 - cluster_idx
        cluster_images = []
        for path_idx, assigned_cluster in enumerate(cluster_assignments):
            if assigned_cluster == original_cluster_idx:
                cluster_images.append(all_paths[path_idx])

        row = cluster_idx // cols
        col = cluster_idx % cols
        ax = axes[row, col]
        grid_img = create_grid(cluster_images)
        ax.imshow(grid_img)
        ax.set_title(str(cluster_idx + 1), fontsize=12, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('lightgrey')
            spine.set_linewidth(1)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_clusters, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.suptitle('Cluster thumbnails: visual similarity groups (5x5 grid, max 25 images per cluster)',
                 fontsize=14, y=0.98)
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.92, hspace=0.4)

    if save_path:
        try:
            plt.savefig(save_path, dpi=72, bbox_inches='tight', format='png')
            if show_progress:
                print(f" - Saved thumbnail grids to: {save_path}")
        except Exception as e:
            raise IOError(f"Failed to save thumbnail grids: {e}")
        finally:
            plt.close()
    else:
        plt.show()


def _plot_cluster_distribution(cluster_stats: List[Dict[str, Any]],
                               save_path: Optional[Path] = None, show_progress: bool = True) -> None:
    """Create and optionally save a vertical cluster distribution chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if save_path:
            raise ImportError("matplotlib is required to save distribution charts. Install it with: pip install matplotlib")
        print("matplotlib not available - skipping distribution chart")
        return

    if not cluster_stats:
        if save_path:
            raise ValueError("No cluster statistics available for distribution chart")
        print("No cluster statistics available")
        return

    sorted_clusters = sorted(cluster_stats, key=lambda x: x['original_size'], reverse=True)

    cluster_names = [str(i+1) for i in range(len(sorted_clusters))]
    kept_counts = [b['kept'] for b in sorted_clusters]
    excluded_counts = [b['excluded'] for b in sorted_clusters]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(cluster_names))
    width = 0.6

    ax.bar(x, kept_counts, width, label='Kept', color='#2E8B57', alpha=0.8)
    ax.bar(x, excluded_counts, width, bottom=kept_counts,
           label='Excluded', color='#CD5C5C', alpha=0.8)

    ax.set_xlabel('Visual similarity clusters (sorted by size)')
    ax.set_ylabel('Number of images')
    ax.set_title('Cluster distribution: kept vs excluded')
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_names, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=100, bbox_inches='tight', format='png')
            if show_progress:
                print(f" - Saved distribution chart to: {save_path}")
        except Exception as e:
            raise IOError(f"Failed to save distribution chart: {e}")
        finally:
            plt.close()
    else:
        plt.show()


# --- Main API ---

def sample_diverse(
    image_paths: List[Union[str, Path]],
    target_count: int,
    distance_threshold: float = DISTANCE_THRESHOLD,
    n_workers: Optional[int] = None,
    show_progress: bool = True,
    show_summary: bool = True,
    save_distribution: Optional[str] = None,
    save_thumbnails: Optional[str] = None,
    image_loading_errors: str = "raise",
    return_indices: bool = False
) -> Union[List[str], List[int]]:
    """
    Diverse sampling from large image collections using DINOv2 embeddings.

    Uses divide-and-conquer agglomerative clustering to group visually similar
    images, then samples to maximize diversity across clusters.

    Args:
        image_paths: List of paths to images
        target_count: Exact number of images to return
        distance_threshold: Cosine distance threshold for clustering (default: 0.5).
            Lower values create more clusters (stricter similarity).
            Higher values create fewer clusters (more lenient).
        n_workers: Number of parallel workers for image loading (default: 4)
        show_progress: Whether to show progress bars
        show_summary: Whether to print cluster distribution summary
        save_distribution: Path to save distribution chart as PNG (default: None)
        save_thumbnails: Path to save thumbnail grids as PNG (default: None)
        image_loading_errors: How to handle image loading errors - "raise" or "skip"
        return_indices: Return 0-based indices instead of paths

    Returns:
        List of exactly target_count selected image paths (if return_indices=False)
        or 0-based indices referring to original input list (if return_indices=True)
    """

    # Validate inputs
    if image_loading_errors not in ["raise", "skip"]:
        raise ValueError(f"image_loading_errors must be 'raise' or 'skip', got '{image_loading_errors}'")

    save_distribution_path = _validate_png_path(save_distribution, "save_distribution")
    save_thumbnails_path = _validate_png_path(save_thumbnails, "save_thumbnails")

    if return_indices:
        path_to_index = {str(path): idx for idx, path in enumerate(image_paths)}

    n_images = len(image_paths)

    # Early exit: target >= available
    if target_count >= n_images:
        if show_progress:
            print(f"Target count ({target_count}) >= available images ({n_images}), returning all images")
        if return_indices:
            return list(range(n_images))
        else:
            return [str(p) for p in image_paths]

    if target_count <= 0:
        return []

    if n_workers is None:
        n_workers = 4

    if show_progress:
        print(f"Selecting {target_count} from {n_images} images...")

    # Sort paths hierarchically
    if show_progress:
        print(" - Sorting paths...")
    sorted_image_paths = _hierarchical_natsort(image_paths)

    # Step 1: Compute embeddings
    valid_paths, embeddings, failed_paths = _compute_embeddings(
        sorted_image_paths, n_workers, show_progress, image_loading_errors
    )

    # Report failed images
    if failed_paths and image_loading_errors == "skip":
        n_failed = len(failed_paths)
        if show_progress:
            print(f" - Warning: {n_failed} image(s) could not be loaded and were skipped:")
            for path, error_msg in failed_paths[:10]:
                print(f"   x {error_msg}")
            if n_failed > 10:
                print(f"   ... and {n_failed - 10} more errors")
        else:
            print(f"Warning: {n_failed} image(s) could not be loaded. Use show_progress=True for details.")

    n_valid = len(valid_paths)

    if target_count >= n_valid:
        if return_indices:
            return [path_to_index[path] for path in valid_paths]
        else:
            return valid_paths

    # Step 2: Cluster
    cluster_labels = _divide_and_conquer_cluster(embeddings, distance_threshold=distance_threshold, show_progress=show_progress)

    # Step 3: Sample with cluster diversity
    if show_progress:
        print(" - Sampling with cluster diversity...")
    selected_indices = _cluster_aware_sample(embeddings, cluster_labels, target_count)
    selected_paths = [valid_paths[i] for i in selected_indices]

    if show_progress:
        print(f" - Selected {len(selected_paths)} images with diversity preservation!")

    # Visualizations
    if show_summary or save_distribution_path or save_thumbnails_path:
        # Build cluster stats
        unique_labels = np.unique(cluster_labels)
        selected_set = set(selected_indices)

        # Sort clusters by size for consistent ordering
        cluster_list = []
        for label in unique_labels:
            indices = np.where(cluster_labels == label)[0]
            cluster_list.append((label, indices.tolist()))
        cluster_list.sort(key=lambda x: len(x[1]), reverse=True)

        cluster_stats = []
        for label, indices in cluster_list:
            kept = sum(1 for i in indices if i in selected_set)
            cluster_stats.append({
                'original_size': len(indices),
                'kept': kept,
                'excluded': len(indices) - kept,
            })

        if show_summary:
            _print_cluster_summary(cluster_stats)

        if save_distribution_path:
            _plot_cluster_distribution(cluster_stats, save_distribution_path, show_progress)

        if save_thumbnails_path:
            # Build cluster assignment mapping for thumbnails
            cluster_assignments = np.zeros(len(valid_paths), dtype=int)
            for cluster_idx, (label, indices) in enumerate(cluster_list):
                for idx in indices:
                    cluster_assignments[idx] = cluster_idx

            viz_data = {
                'cluster_assignments': cluster_assignments.tolist(),
                'all_paths': valid_paths,
            }
            _plot_cluster_thumbnails(cluster_stats, viz_data, save_thumbnails_path, show_progress)

    # Return indices or paths
    if return_indices:
        return [path_to_index[path] for path in selected_paths]
    else:
        return selected_paths

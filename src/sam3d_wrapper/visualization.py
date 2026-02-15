"""
Visualization helpers that wrap the upstream notebook/utils.py functions.

These provide convenient access to the upstream visualization utilities
without requiring users to manage sys.path manually.
"""

import numpy as np

from sam3d_wrapper.repo import activate_imports


def render_mesh_overlay(
    image_bgr: np.ndarray,
    persons: list[dict],
    faces: np.ndarray,
) -> np.ndarray:
    """
    Render all detected persons' meshes overlaid on the original image.

    Returns a concatenated image: [original | keypoints | mesh overlay | side view].

    Args:
        image_bgr: Original image in BGR format (HxWx3 uint8).
        persons: List of raw person output dicts from the estimator.
        faces: Mesh face indices array.

    Returns:
        Concatenated visualization as uint8 numpy array.
    """
    activate_imports()
    from tools.vis_utils import visualize_sample_together

    vis = visualize_sample_together(image_bgr, persons, faces)
    return vis.astype(np.uint8)


def render_per_person(
    image_bgr: np.ndarray,
    persons: list[dict],
    faces: np.ndarray,
) -> list[np.ndarray]:
    """
    Render separate visualizations for each detected person.

    Returns a list of concatenated images, one per person:
    [original | keypoints+bbox | mesh overlay | side view].

    Args:
        image_bgr: Original image in BGR format.
        persons: List of raw person output dicts.
        faces: Mesh face indices array.

    Returns:
        List of visualization images, one per person.
    """
    activate_imports()
    from tools.vis_utils import visualize_sample

    results = visualize_sample(image_bgr, persons, faces)
    return [r.astype(np.uint8) for r in results]


def render_3d_mesh_views(
    image_bgr: np.ndarray,
    persons: list[dict],
    faces: np.ndarray,
) -> list[np.ndarray]:
    """
    Render 3D mesh views (overlay + white background + side view) per person.

    Wraps notebook.utils.visualize_3d_mesh.

    Args:
        image_bgr: Original image in BGR format.
        persons: List of raw person output dicts.
        faces: Mesh face indices array.

    Returns:
        List of combined view images, one per person.
    """
    activate_imports()
    from notebook.utils import visualize_3d_mesh

    return visualize_3d_mesh(image_bgr, persons, faces)


def render_2d_keypoints(
    image_bgr: np.ndarray,
    persons: list[dict],
) -> list[np.ndarray]:
    """
    Render 2D keypoint skeletons and bounding boxes per person.

    Wraps notebook.utils.visualize_2d_results.

    Args:
        image_bgr: Original image in BGR format.
        persons: List of raw person output dicts.

    Returns:
        List of annotated images, one per person.
    """
    activate_imports()
    from notebook.utils import setup_visualizer, visualize_2d_results

    visualizer = setup_visualizer()
    return visualize_2d_results(image_bgr, persons, visualizer)


def save_meshes(
    image_bgr: np.ndarray,
    persons: list[dict],
    faces: np.ndarray,
    save_dir: str,
    image_name: str,
) -> list[str]:
    """
    Save .ply mesh files and overlay images for each detected person.

    Wraps notebook.utils.save_mesh_results.

    Args:
        image_bgr: Original image in BGR format.
        persons: List of raw person output dicts.
        faces: Mesh face indices array.
        save_dir: Directory to save results.
        image_name: Base name for output files.

    Returns:
        List of paths to saved .ply files.
    """
    activate_imports()
    from notebook.utils import save_mesh_results

    return save_mesh_results(image_bgr, persons, faces, save_dir, image_name)

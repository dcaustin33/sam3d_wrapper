"""
sam3d_wrapper - A convenient wrapper for Meta's SAM 3D Body.

Provides easy installation, checkpoint management, and batch inference
for single-image full-body 3D human mesh recovery.

Usage:
    from sam3d_wrapper import Sam3DBody

    model = Sam3DBody(model_variant="dinov3")
    results = model.predict("path/to/image.jpg")
    results = model.predict_batch("path/to/images/")
"""

__version__ = "0.1.0"

from sam3d_wrapper.repo import ensure_repo, get_repo_path
from sam3d_wrapper.inference import Sam3DBody

__all__ = [
    "Sam3DBody",
    "ensure_repo",
    "get_repo_path",
]

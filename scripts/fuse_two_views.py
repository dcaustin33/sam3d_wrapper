"""Thin wrapper around sam3d_wrapper.multiview_fusion:cli_fuse.

Registered as the ``sam3d-fuse`` entry point in pyproject.toml; this
script exists so ``python scripts/fuse_two_views.py ...`` also works.
"""

from sam3d_wrapper.multiview_fusion import cli_fuse

if __name__ == "__main__":
    raise SystemExit(cli_fuse())

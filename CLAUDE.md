# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A pip-installable Python wrapper around Meta's SAM 3D Body for single-image full-body 3D human mesh recovery. Manages upstream repo cloning, checkpoint downloading, and exposes a high-level `Sam3DBody` API and CLI tools.

## Build & Setup Commands

```bash
# Full one-command setup (install deps, clone upstream, install detectron2, download checkpoints)
bash scripts/setup.sh

# Step-by-step:
uv sync                                    # Install package and dependencies
uv run sam3d-setup                         # Clone upstream repo + install detectron2
uv run sam3d-download --variant dinov3     # Download checkpoints (requires HF login)
```

## CLI Entry Points

```bash
uv run sam3d-setup       # Clone upstream repo, install detectron2
uv run sam3d-download    # Download model checkpoints from HuggingFace
PYOPENGL_PLATFORM=egl uv run sam3d-infer --images ./photos --output ./results
PYOPENGL_PLATFORM=egl uv run sam3d-infer --images ./photos --output ./results --shape-override shape.npy
```

## Running Tests

```bash
uv run pytest            # Requires dev dependencies: uv sync --extra dev
```

## Architecture

The package lives in `src/sam3d_wrapper/` with four modules:

- **repo.py** — Manages the upstream `sam-3d-body` git repo: cloning into `vendor/`, installing detectron2 (pinned commit), adding upstream to `sys.path` via `activate_imports()`. The `ensure_repo()` function is the key entry point that makes upstream modules importable. Uses `uv pip` when available, falls back to `python -m pip`.

- **download.py** — Downloads model checkpoints from HuggingFace (`facebook/sam-3d-body-dinov3` or `facebook/sam-3d-body-vith`) into `checkpoints/` using `huggingface_hub.snapshot_download`. Verifies downloads by checking for `model.ckpt` and `assets/mhr_model.pt`.

- **inference.py** — Core API. `Sam3DBody` class wraps upstream's `setup_sam_3d_body()` and exposes `predict()` (single image) and `predict_batch()` (directory). Returns structured `ImageResult`/`PersonResult` dataclasses with vertices, keypoints, camera params, and body/hand/shape parameters. Supports `shape_override` (45D numpy array) on both `predict()` and `predict_batch()` to enforce a fixed body shape across frames. Use `Sam3DBody.average_shape(results)` to compute a mean shape from prior results, or `recompute_with_shape(result, shape)` to update an existing result in-place.

- **visualization.py** — Thin wrappers around upstream's `tools.vis_utils` and `notebook.utils` for mesh overlay rendering, per-person visualization, 3D mesh views, 2D keypoints, and .ply export.

## Key Design Patterns

- **Lazy upstream imports**: All upstream code access goes through `activate_imports()` which calls `ensure_repo()` and injects the vendor path into `sys.path`. Upstream modules (`notebook.utils`, `tools.vis_utils`) are imported inside functions, not at module level.

- **Environment variable overrides**: `SAM3D_VENDOR_DIR` and `SAM3D_CHECKPOINT_DIR` override default `./vendor/` and `./checkpoints/` directories.

- **Two model variants**: `dinov3` (DINOv3-H+, 840M params) and `vith` (ViT-H, 631M params). The variant name maps to HuggingFace repo IDs in `download.py:HF_REPOS`.

## Setup Gotchas

- **detectron2 lives outside uv's lock file**: It is installed via `uv pip install` with `--no-build-isolation --no-deps` from a pinned git commit. Running `uv sync` will **remove** detectron2. Reinstall with: `uv pip install "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9" --no-build-isolation --no-deps`

- **Headless rendering needs `PYOPENGL_PLATFORM=egl`**: Always set this env var when running inference with visualization on servers without a display. The EGL backend also requires the user to be in the `video` and `render` groups for `/dev/dri` access.

- **System packages for rendering**: `libegl1-mesa` and `libosmesa6` must be installed via apt for pyrender's offscreen rendering to work.

- **setuptools pinned <81**: detectron2 uses `pkg_resources` which was removed in setuptools 81+. The project pins `setuptools<81` and also includes `cloudpickle` as a direct dependency since detectron2 is installed with `--no-deps`.

## Dependencies

Uses `uv` as package manager with `hatchling` build backend. PyTorch and TorchVision are pinned to CUDA 12.4 index via `tool.uv.sources`. The `allow-direct-references = true` hatch setting supports the `moge` git dependency for FOV estimation.

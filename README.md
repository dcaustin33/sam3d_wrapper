# sam3d_wrapper

A pip-installable wrapper around [Meta's SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) for easy single-image full-body 3D human mesh recovery.

## What this does

- Installs all dependencies via `uv` / `pip`
- Clones and configures the upstream `sam-3d-body` repository automatically
- Provides scripts to download model checkpoints from Hugging Face
- Exposes a simple Python API (`Sam3DBody`) for single-image and batch inference
- Wraps the upstream `notebook/utils.py` visualization helpers
- Includes CLI tools for setup, download, and inference

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- A GPU with CUDA support (CPU works but is slow)
- Hugging Face account with access approved for the model checkpoints
- System packages for headless rendering: `libegl1-mesa`, `libosmesa6` (installed automatically by `setup.sh`)

### Request checkpoint access

Before downloading checkpoints, you need to request access on Hugging Face:

1. Go to [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) and/or [facebook/sam-3d-body-vith](https://huggingface.co/facebook/sam-3d-body-vith)
2. Click "Request Access"
3. Authenticate locally: `huggingface-cli login`

## Quick start

### One-command setup

```bash
# Clone this repo
git clone <this-repo-url>
cd sam3d_wrapper

# Full setup: install deps, clone upstream repo, install detectron2, download checkpoints
bash scripts/setup.sh
```

### Step-by-step setup

```bash
# 1. Install the package
uv sync

# 2. Clone upstream repo + install detectron2
uv run sam3d-setup

# 3. Download checkpoints (requires HF login)
uv run sam3d-download --variant dinov3
```

## Setup gotchas

### detectron2 is installed outside uv's lock file

detectron2 must be installed with `--no-build-isolation --no-deps` from a pinned git commit. Because it is not in `pyproject.toml`, **running `uv sync` will remove it**. If that happens, reinstall with:

```bash
uv pip install "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9" --no-build-isolation --no-deps
```

### Headless rendering requires PYOPENGL_PLATFORM=egl

On headless servers (no display), pyrender needs the EGL backend. Set this environment variable before running inference with visualization:

```bash
export PYOPENGL_PLATFORM=egl
```

The EGL backend also requires the current user to be in the `video` and `render` groups for `/dev/dri` access. The setup script handles this, but you may need to log out and back in for group changes to take effect.

### setuptools must be pinned below v81

detectron2 uses `pkg_resources` which was removed in setuptools 81+. The package pins `setuptools<81` to ensure compatibility.

## Usage

### Python API

```python
from sam3d_wrapper import Sam3DBody

# Initialize (loads model, detector, FOV estimator)
model = Sam3DBody(model_variant="dinov3")

# Single image inference
result = model.predict("path/to/image.jpg")
for person in result.persons:
    print(f"Person {person.person_id}:")
    print(f"  Vertices shape: {person.pred_vertices.shape}")
    print(f"  2D keypoints: {person.pred_keypoints_2d.shape}")
    print(f"  Focal length: {person.focal_length}")

# Batch inference over a directory
results = model.predict_batch(
    "path/to/images/",
    output_dir="path/to/output/",
    save_meshes=True,           # Save .ply mesh files
    save_visualizations=True,   # Save overlay images
)
```

### Using upstream notebook_utils directly

```python
from sam3d_wrapper.repo import ensure_repo

# This clones the repo (if needed) and adds it to sys.path
ensure_repo()

# Now you can import upstream modules directly
from notebook.utils import setup_sam_3d_body, visualize_3d_mesh, save_mesh_results
from tools.vis_utils import visualize_sample_together

estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")
# ... use exactly as shown in Meta's demo notebook
```

### Visualization helpers

```python
from sam3d_wrapper.visualization import (
    render_mesh_overlay,    # Full scene with all persons
    render_per_person,      # Separate vis per person
    render_3d_mesh_views,   # Multi-view 3D mesh
    render_2d_keypoints,    # 2D skeleton overlay
    save_meshes,            # Export .ply files
)
```

### CLI inference

```bash
# Basic inference
uv run sam3d-infer --images ./my_photos --output ./results

# With options
uv run sam3d-infer \
    --images ./my_photos \
    --output ./results \
    --variant dinov3 \
    --bbox-threshold 0.5 \
    --save-meshes \
    --use-mask
```

## CLI commands

| Command | Description |
|---------|-------------|
| `sam3d-setup` | Clone upstream repo, install detectron2 |
| `sam3d-download` | Download model checkpoints from Hugging Face |
| `sam3d-infer` | Run inference on images |

### sam3d-setup options

```
--vendor-dir PATH     Where to clone repos (default: ./vendor/)
--skip-detectron2     Skip detectron2 installation
--with-sam3           Also install SAM3 for segmentation support
```

### sam3d-download options

```
--variant {dinov3,vith}   Model variant (default: dinov3)
--checkpoint-dir PATH     Where to store checkpoints (default: ./checkpoints/)
--all                     Download all variants
```

## Model variants

| Variant | Backbone | Params | HF Repo |
|---------|----------|--------|---------|
| `dinov3` | DINOv3-H+ | 840M | `facebook/sam-3d-body-dinov3` |
| `vith` | ViT-H | 631M | `facebook/sam-3d-body-vith` |

## Environment variables

| Variable | Description |
|----------|-------------|
| `SAM3D_VENDOR_DIR` | Override vendor directory for cloned repos |
| `SAM3D_CHECKPOINT_DIR` | Override checkpoint storage directory |

## Project structure

```
sam3d_wrapper/
├── pyproject.toml              # Package config + all dependencies
├── scripts/
│   ├── setup.sh                # One-command full setup
│   └── download_checkpoints.sh # Checkpoint download helper
├── src/sam3d_wrapper/
│   ├── __init__.py             # Main exports: Sam3DBody
│   ├── repo.py                 # Upstream repo management
│   ├── download.py             # Checkpoint download from HF
│   ├── inference.py            # Sam3DBody class + batch inference
│   └── visualization.py        # Visualization wrappers
├── vendor/                     # (gitignored) Cloned upstream repos
│   └── sam-3d-body/
└── checkpoints/                # (gitignored) Downloaded model weights
    └── sam-3d-body-dinov3/
```

## License

This wrapper is provided as-is. The upstream SAM 3D Body model and code are licensed under the [SAM License](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE) by Meta Platforms, Inc.

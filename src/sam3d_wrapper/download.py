"""
Download model checkpoints from Hugging Face.

Checkpoints require access approval at:
  - https://huggingface.co/facebook/sam-3d-body-dinov3
  - https://huggingface.co/facebook/sam-3d-body-vith

You must first:
  1. Request access on the Hugging Face repo page
  2. Authenticate via `huggingface-cli login`
"""

import argparse
import os
from pathlib import Path

HF_REPOS = {
    "dinov3": "facebook/sam-3d-body-dinov3",
    "vith": "facebook/sam-3d-body-vith",
}

_DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"


def get_checkpoint_dir() -> Path:
    """Return the checkpoint directory, respecting SAM3D_CHECKPOINT_DIR env var."""
    env = os.environ.get("SAM3D_CHECKPOINT_DIR")
    if env:
        return Path(env)
    return _DEFAULT_CHECKPOINT_DIR


def get_checkpoint_path(variant: str = "dinov3") -> Path:
    """Return the expected checkpoint directory for a given model variant."""
    hf_repo = HF_REPOS.get(variant)
    if hf_repo is None:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(HF_REPOS.keys())}")
    repo_name = hf_repo.split("/")[-1]
    return get_checkpoint_dir() / repo_name


def download_checkpoint(variant: str = "dinov3", checkpoint_dir: Path | None = None) -> Path:
    """
    Download a SAM 3D Body checkpoint from Hugging Face.

    Args:
        variant: Model variant - "dinov3" (840M params) or "vith" (631M params).
        checkpoint_dir: Override directory for storing checkpoints.

    Returns:
        Path to the downloaded checkpoint directory.
    """
    from huggingface_hub import snapshot_download

    hf_repo = HF_REPOS.get(variant)
    if hf_repo is None:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(HF_REPOS.keys())}")

    checkpoint_dir = checkpoint_dir or get_checkpoint_dir()
    repo_name = hf_repo.split("/")[-1]
    local_dir = checkpoint_dir / repo_name

    print(f"Downloading {hf_repo} to {local_dir} ...")
    print("(You must have requested access and be logged in via `huggingface-cli login`)")

    snapshot_download(
        repo_id=hf_repo,
        local_dir=str(local_dir),
    )

    print(f"Download complete: {local_dir}")
    _print_checkpoint_contents(local_dir)
    return local_dir


def _print_checkpoint_contents(path: Path) -> None:
    """Print a summary of downloaded checkpoint files."""
    print("\nDownloaded files:")
    for f in sorted(path.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            rel = f.relative_to(path)
            print(f"  {rel} ({size_mb:.1f} MB)")


def verify_checkpoint(variant: str = "dinov3") -> bool:
    """Check whether a checkpoint has been downloaded for the given variant."""
    ckpt_dir = get_checkpoint_path(variant)
    model_ckpt = ckpt_dir / "model.ckpt"
    mhr_model = ckpt_dir / "assets" / "mhr_model.pt"
    return model_ckpt.exists() and mhr_model.exists()


def cli_download() -> None:
    """CLI entry point: sam3d-download"""
    parser = argparse.ArgumentParser(
        description="Download SAM 3D Body checkpoints from Hugging Face."
    )
    parser.add_argument(
        "--variant",
        choices=list(HF_REPOS.keys()),
        default="dinov3",
        help="Model variant to download (default: dinov3). "
             "dinov3 = DINOv3-H+ backbone (840M params), "
             "vith = ViT-H backbone (631M params).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory to store checkpoints (default: <package_root>/checkpoints/)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="download_all",
        help="Download all available model variants",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SAM 3D Body - Checkpoint Download")
    print("=" * 60)

    if args.download_all:
        for variant in HF_REPOS:
            print(f"\n--- Downloading variant: {variant} ---")
            download_checkpoint(variant, args.checkpoint_dir)
    else:
        download_checkpoint(args.variant, args.checkpoint_dir)

    print("\n" + "=" * 60)
    print("Done! You can now run inference:")
    print(f"  sam3d-infer --images ./my_images --output ./results")
    print("=" * 60)

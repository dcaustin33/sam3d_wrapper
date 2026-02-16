"""
Manages the upstream sam-3d-body repository.

Since the upstream repo is not pip-installable, we clone it into a vendor
directory and add it to sys.path at runtime so that `sam_3d_body`, `tools`,
and `notebook` imports work transparently.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/facebookresearch/sam-3d-body.git"
DETECTRON2_URL = "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9"
SAM3_URL = "https://github.com/facebookresearch/sam3.git"

_DEFAULT_VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor"


def get_vendor_dir() -> Path:
    """Return the vendor directory, respecting SAM3D_VENDOR_DIR env var."""
    env = os.environ.get("SAM3D_VENDOR_DIR")
    if env:
        return Path(env)
    return _DEFAULT_VENDOR_DIR


def get_repo_path() -> Path:
    """Return the path to the cloned sam-3d-body repo."""
    return get_vendor_dir() / "sam-3d-body"


def _run(cmd: list[str], **kwargs) -> None:
    print(f"  Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, **kwargs)


def clone_repo(vendor_dir: Path | None = None) -> Path:
    """Clone the sam-3d-body repo into the vendor directory."""
    vendor_dir = vendor_dir or get_vendor_dir()
    repo_dir = vendor_dir / "sam-3d-body"

    if repo_dir.exists() and (repo_dir / "sam_3d_body" / "__init__.py").exists():
        print(f"sam-3d-body repo already exists at {repo_dir}")
        return repo_dir

    vendor_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cloning sam-3d-body into {repo_dir} ...")
    _run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)])
    print("Clone complete.")
    return repo_dir


def _pip_install(*args: str) -> None:
    """Install packages using uv pip (preferred) or pip as fallback."""
    import shutil

    uv = shutil.which("uv")
    if uv:
        _run([uv, "pip", "install", *args])
    else:
        _run([sys.executable, "-m", "pip", "install", *args])


def install_detectron2() -> None:
    """Install detectron2 from the specific commit used by sam-3d-body."""
    print("Installing detectron2 ...")
    _pip_install(DETECTRON2_URL, "--no-build-isolation", "--no-deps")
    print("detectron2 installed.")


def install_sam3(vendor_dir: Path | None = None) -> None:
    """Clone and install SAM3 for segmentation support (optional)."""
    vendor_dir = vendor_dir or get_vendor_dir()
    sam3_dir = vendor_dir / "sam3"

    if not sam3_dir.exists():
        print(f"Cloning SAM3 into {sam3_dir} ...")
        _run(["git", "clone", "--depth", "1", SAM3_URL, str(sam3_dir)])

    print("Installing SAM3 ...")
    _pip_install("-e", str(sam3_dir))
    _pip_install("decord", "psutil")
    print("SAM3 installed.")


def ensure_detectron2() -> None:
    """Ensure detectron2 is installed. Installs automatically if missing."""
    try:
        import detectron2  # noqa: F401
    except ImportError:
        print("detectron2 not found, installing automatically ...")
        install_detectron2()


def ensure_repo() -> Path:
    """
    Ensure the sam-3d-body repo is cloned and on sys.path.

    Returns the repo path. Safe to call multiple times.
    """
    repo_path = get_repo_path()
    if not repo_path.exists():
        repo_path = clone_repo()

    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    return repo_path


def activate_imports() -> None:
    """Add the repo to sys.path so upstream modules can be imported."""
    repo_path = get_repo_path()
    if not repo_path.exists():
        raise RuntimeError(
            f"sam-3d-body repo not found at {repo_path}. "
            "Run `sam3d-setup` or call sam3d_wrapper.ensure_repo() first."
        )
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def cli_setup() -> None:
    """CLI entry point: sam3d-setup"""
    parser = argparse.ArgumentParser(
        description="Set up sam-3d-body: clone repo, install detectron2, and optionally install SAM3."
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=None,
        help="Directory to clone repos into (default: <package_root>/vendor/)",
    )
    parser.add_argument(
        "--skip-detectron2",
        action="store_true",
        help="Skip installing detectron2",
    )
    parser.add_argument(
        "--with-sam3",
        action="store_true",
        help="Also clone and install SAM3 for segmentation",
    )
    args = parser.parse_args()

    vendor_dir = args.vendor_dir or get_vendor_dir()

    print("=" * 60)
    print("SAM 3D Body - Setup")
    print("=" * 60)

    # 1. Clone the repo
    print("\n[1/3] Cloning sam-3d-body repository ...")
    clone_repo(vendor_dir)

    # 2. Install detectron2
    if not args.skip_detectron2:
        print("\n[2/3] Installing detectron2 ...")
        install_detectron2()
    else:
        print("\n[2/3] Skipping detectron2 installation")

    # 3. Optionally install SAM3
    if args.with_sam3:
        print("\n[3/3] Installing SAM3 ...")
        install_sam3(vendor_dir)
    else:
        print("\n[3/3] Skipping SAM3 (use --with-sam3 to install)")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print(f"Repo cloned to: {vendor_dir / 'sam-3d-body'}")
    print()
    print("Next steps:")
    print("  1. Download checkpoints:  sam3d-download --variant dinov3")
    print("  2. Run inference:         sam3d-infer --images ./my_images --output ./results")
    print("=" * 60)

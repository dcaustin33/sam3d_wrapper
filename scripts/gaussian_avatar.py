#!/usr/bin/env python3
"""Run SAM 3D Body on a single-person image directory with consistent shape.

Randomly samples 10 images to estimate the person's body shape, then runs
inference on all images using that fixed shape. Saves mesh overlay images
and raw per-frame outputs (.npz) with the same name as the input image.

Usage:
    PYOPENGL_PLATFORM=egl uv run python scripts/gaussian_avatar.py \
        --images /path/to/images --output /path/to/output

    # Use more samples for a better shape estimate
    PYOPENGL_PLATFORM=egl uv run python scripts/gaussian_avatar.py \
        --images /path/to/images --output /path/to/output --n-shape-samples 20
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from sam3d_wrapper.inference import Sam3DBody, _collect_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM 3D inference with consistent shape for gaussian avatar pipelines."
    )
    parser.add_argument(
        "--images", type=str, required=True,
        help="Directory of images of a single person",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for overlays and raw results",
    )
    parser.add_argument(
        "--variant", choices=["dinov3", "vith"], default="dinov3",
        help="Model variant (default: dinov3)",
    )
    parser.add_argument(
        "--n-shape-samples", type=int, default=10,
        help="Number of images to sample for shape estimation (default: 10)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: cuda if available)",
    )
    parser.add_argument(
        "--bbox-threshold", type=float, default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shape sample selection (default: 42)",
    )
    parser.add_argument(
        "--no-vis", action="store_true",
        help="Skip saving overlay visualizations",
    )
    parser.add_argument(
        "--no-detector", action="store_true",
        help="Disable human detection (use full image)",
    )
    parser.add_argument(
        "--no-fov", action="store_true",
        help="Disable FOV estimation",
    )
    args = parser.parse_args()

    image_files = _collect_images(args.images)
    if not image_files:
        print(f"No images found in {args.images}")
        sys.exit(1)

    out_dir = Path(args.output)
    raw_dir = out_dir / "raw"
    vis_dir = out_dir / "vis"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(image_files)} images.")

    # --- Load model ---
    model = Sam3DBody(
        model_variant=args.variant,
        device=args.device,
        use_detector=not args.no_detector,
        use_fov=not args.no_fov,
        bbox_threshold=args.bbox_threshold,
    )

    # --- Phase 1: estimate shape from random samples ---
    n_samples = min(args.n_shape_samples, len(image_files))
    random.seed(args.seed)
    sample_files = random.sample(image_files, n_samples)

    print(f"\nPhase 1: Estimating shape from {n_samples} random images ...")
    sample_results = []
    for img_path in tqdm(sample_files, desc="Shape estimation"):
        result = model.predict(img_path, bbox_threshold=args.bbox_threshold)
        if result.num_persons > 0:
            sample_results.append(result)

    if not sample_results:
        print("ERROR: No persons detected in any of the sampled images.")
        sys.exit(1)

    avg_shape = Sam3DBody.average_shape(sample_results)
    np.save(out_dir / "shape.npy", avg_shape)
    print(f"Mean shape saved to {out_dir / 'shape.npy'}")

    # --- Phase 2: run inference on all images with fixed shape ---
    print(f"\nPhase 2: Running inference on {len(image_files)} images with fixed shape ...")
    faces = model.faces

    for img_path in tqdm(image_files, desc="Inference"):
        result = model.predict(
            img_path,
            bbox_threshold=args.bbox_threshold,
            shape_override=avg_shape,
        )

        stem = img_path.stem

        # Save raw output per person (one .npz per image)
        if result.num_persons > 0:
            person = result.persons[0]
            raw = person.raw
            save_dict = {}
            for k, v in raw.items():
                if isinstance(v, np.ndarray):
                    save_dict[k] = v
                elif isinstance(v, (int, float)):
                    save_dict[k] = np.array(v)
            np.savez(raw_dir / f"{stem}.npz", **save_dict)
        else:
            # Save empty marker so downstream knows this frame had no detection
            np.savez(raw_dir / f"{stem}.npz", no_detection=np.array(True))

        # Save overlay visualization
        if not args.no_vis and result.num_persons > 0:
            from sam3d_wrapper.repo import activate_imports
            activate_imports()
            from tools.vis_utils import visualize_sample_together

            raw_outputs = [p.raw for p in result.persons]
            vis = visualize_sample_together(result.image_bgr, raw_outputs, faces)
            cv2.imwrite(str(vis_dir / f"{stem}.jpg"), vis.astype(np.uint8))

    total_raw = len(list(raw_dir.glob("*.npz")))
    total_vis = len(list(vis_dir.glob("*.jpg"))) if not args.no_vis else 0
    print(f"\nDone! {total_raw} raw outputs, {total_vis} overlays saved to {out_dir}")


if __name__ == "__main__":
    main()

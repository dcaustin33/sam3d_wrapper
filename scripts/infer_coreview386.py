#!/usr/bin/env python3
"""Run SAM 3D Body inference on all cameras in CoreView_386.

Loads the model once, then iterates over all Camera_B* directories,
using the per-camera focal length from intri.yml.
"""

import re
import time
from pathlib import Path

import numpy as np

from sam3d_wrapper.inference import Sam3DBody

DATASET = Path("/workspace/gaussian_avatar/datasets/zju_mocap/CoreView_386")


def parse_focal_lengths(intri_path: Path) -> dict[str, float]:
    """Extract per-camera fx from intri.yml."""
    content = intri_path.read_text()
    result = {}
    for m in re.finditer(r"K_(Camera_B\d+).*?data:\s*\[(.*?)\]", content, re.DOTALL):
        cam = m.group(1)
        data = [float(x.strip()) for x in m.group(2).split(",")]
        result[cam] = data[0]  # fx
    return result


def main() -> None:
    focal_lengths = parse_focal_lengths(DATASET / "intri.yml")

    camera_names = [f"Camera_B{i}" for i in range(9, 24)]
    mask_dirs = [DATASET / f"{name}_masks" for name in camera_names]
    print(f"Will process {len(camera_names)} cameras: {', '.join(camera_names)}")

    model = Sam3DBody(
        model_variant="dinov3",
        use_detector=True,
        use_fov=False,
    )

    for cam_name, cam_dir in zip(camera_names, mask_dirs):
        # Wait for the directory to appear (created by another process)
        while not cam_dir.exists():
            print(f"Waiting for {cam_dir.name} to appear...")
            time.sleep(10)

        focal = focal_lengths.get(cam_name)
        if focal is None:
            print(f"WARNING: No focal length for {cam_name}, skipping")
            continue

        output_dir = cam_dir / "results"
        print(f"\n=== {cam_dir.name} (fx={focal:.1f}) ===")

        model.predict_batch(
            images_path=str(cam_dir),
            output_dir=str(output_dir),
            save_visualizations=False,
            shape_calibration_frames=20,
            focal_length=focal,
        )


if __name__ == "__main__":
    main()

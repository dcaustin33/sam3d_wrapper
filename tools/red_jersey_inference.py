"""SAM3D inference picking the person with the most red in their bbox.

For Chiefs footage (Mahomes in red jersey) where the default largest-bbox
selection would pick whichever player happens to be closest to camera.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from sam3d_wrapper import Sam3DBody
from sam3d_wrapper.inference import _collect_images


def red_fraction(img_bgr: np.ndarray, bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = img_bgr[y1:y2, x1:x2].astype(np.int16)
    b, g, r = crop[..., 0], crop[..., 1], crop[..., 2]
    red = ((r - g) > 40) & ((r - b) > 40) & (r > 80)
    return float(red.mean())


def bbox_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox[:4]
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox[:4]
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)


def _size_filtered_candidates(persons, min_area_ratio: float):
    if not persons:
        return []
    areas = [bbox_area(p.bbox) for p in persons]
    max_area = max(areas) if areas else 0.0
    threshold = max_area * min_area_ratio
    filtered = [p for p, a in zip(persons, areas) if a >= threshold]
    return filtered or persons


def pick_red_max(result, img_bgr, min_area_ratio: float = 0.3):
    """Keep only the person with the most red in their bbox, restricted to
    bboxes at least ``min_area_ratio`` of the largest bbox in the frame so
    a tiny red detection cannot outscore a large one."""
    if not result.persons:
        return
    candidates = _size_filtered_candidates(result.persons, min_area_ratio)
    scores = [red_fraction(img_bgr, p.bbox) for p in candidates]
    winner = candidates[int(np.argmax(scores))]
    result.persons = [winner]


def pick_nearest_red(
    result,
    img_bgr: np.ndarray,
    prev_center: np.ndarray,
    min_area_ratio: float = 0.3,
    min_red_fraction: float = 0.1,
):
    """Track identity across frames: among size-filtered candidates that also
    have enough red in their bbox, pick the one closest to ``prev_center``.
    Falls back to red-max if no candidate meets the red floor (so we never
    silently hand-off to a non-red player when Mahomes is lost)."""
    if not result.persons:
        return
    size_filtered = _size_filtered_candidates(result.persons, min_area_ratio)
    red_scores = [red_fraction(img_bgr, p.bbox) for p in size_filtered]
    red_filtered = [
        p for p, s in zip(size_filtered, red_scores) if s >= min_red_fraction
    ]
    if red_filtered:
        dists = [
            np.linalg.norm(bbox_center(p.bbox) - prev_center) for p in red_filtered
        ]
        winner = red_filtered[int(np.argmin(dists))]
    else:
        winner = size_filtered[int(np.argmax(red_scores))]
    result.persons = [winner]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cal-frames", type=int, default=15)
    ap.add_argument("--bbox-threshold", type=float, default=0.3)
    args = ap.parse_args()

    images = _collect_images(args.images)
    if not images:
        raise SystemExit(f"No images under {args.images}")
    out_dir = Path(args.output)
    raw_dir = out_dir / "raw"
    vis_dir = out_dir / "vis"
    raw_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = Sam3DBody(bbox_threshold=args.bbox_threshold)
    _ = model.faces  # trigger lazy load + vendor sys.path injection
    from tools.vis_utils import visualize_sample_together

    n_cal = min(args.cal_frames, len(images))
    cal_imgs = random.sample(images, n_cal)
    print(f"Calibrating shape from {n_cal} frames (red-max person)...")
    cal_cache: dict[str, object] = {}
    shapes = []
    for p in tqdm(cal_imgs, desc="Calibration"):
        img = cv2.imread(str(p))
        res = model.predict(img, bbox_threshold=args.bbox_threshold)
        res.image_bgr = img
        res.image_path = str(p)
        pick_red_max(res, img)
        if res.persons and res.persons[0].shape_params is not None:
            shapes.append(res.persons[0].shape_params)
        cal_cache[str(p)] = res
    if not shapes:
        raise RuntimeError("No shapes found in calibration frames")
    shape_override = np.mean(shapes, axis=0).astype(np.float32)
    np.save(str(out_dir / "shape.npy"), shape_override)
    print(f"Saved shape to {out_dir / 'shape.npy'}")

    print(f"Running inference on {len(images)} frames (red-max + tracking)...")
    n_detected = 0
    prev_center: np.ndarray | None = None
    for p in tqdm(images, desc="Inference"):
        img = cv2.imread(str(p))
        res = model.predict(img, bbox_threshold=args.bbox_threshold)
        res.image_bgr = img
        res.image_path = str(p)
        if res.persons:
            if prev_center is None:
                pick_red_max(res, img)
            else:
                pick_nearest_red(res, img, prev_center)
            model.recompute_with_shape(res, shape_override)
            prev_center = bbox_center(res.persons[0].bbox)
        if not res.persons:
            continue
        n_detected += 1
        stem = p.stem
        np.savez(str(raw_dir / f"{stem}.npz"), **res.persons[0].raw)
        vis = visualize_sample_together(
            res.image_bgr, [res.persons[0].raw], model.faces
        )
        cv2.imwrite(str(vis_dir / f"{stem}.jpg"), vis.astype(np.uint8))

    print(f"Done: {len(images)} images, {n_detected} persons selected.")


if __name__ == "__main__":
    main()

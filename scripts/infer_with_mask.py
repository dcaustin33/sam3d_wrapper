"""Run SAM 3D Body and SAM 3 person segmentation on a directory of images.

Produces two sibling output folders:

    <output>/
    ├── mhr/
    │   ├── <image_stem>.npz     # Per-person params from Sam3DBody.predict()
    │   └── <image_stem>.json    # {image_path, num_persons, per-person bbox/focal}
    └── mask/
        ├── <image_stem>.png     # Union mask (8-bit grayscale 0/255) for "a person"
        └── <image_stem>.json    # Per-instance scores + boxes (only when >1 mask)

The two pipelines run concurrently in separate threads, with SAM 3 running in
micro-batches for throughput. Disk I/O is offloaded to a small thread pool so
neither GPU thread blocks on PNG/NPZ encoding.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import numpy as np
from PIL import Image

from sam3d_wrapper.inference import IMAGE_EXTENSIONS, Sam3DBody

logger = logging.getLogger("infer_with_mask")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return [
            f for f in sorted(path.iterdir())
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]
    raise FileNotFoundError(f"Path does not exist: {path}")


def _save_mhr_outputs(result, mhr_dir: Path, stem: str) -> None:
    """Save per-person raw arrays as ``.npz``.

    Mirrors ``Sam3DBody._save_numpy_results``: one ``.npz`` per person,
    storing ``**p.raw``. Single-person frames go to ``<stem>.npz``;
    multi-person frames fall back to ``<stem>_person{i}.npz``.
    """
    if result.num_persons == 0:
        return
    for p in result.persons:
        suffix = "" if result.num_persons == 1 else f"_person{p.person_id}"
        np.savez(str(mhr_dir / f"{stem}{suffix}.npz"), **p.raw)


def _save_mask_outputs(seg_result, mask_dir: Path, stem: str) -> None:
    """Save union mask PNG and (when multi-instance) scores/boxes JSON."""
    png_path = mask_dir / f"{stem}.png"

    if not seg_result.masks:
        # Empty mask at a default size is useless; write a 1x1 zero PNG so
        # downstream code can rely on the file existing.
        Image.fromarray(np.zeros((1, 1), dtype=np.uint8), mode="L").save(png_path)
        return

    # Union all instance masks into a single 8-bit mask
    union = np.zeros_like(seg_result.masks[0], dtype=bool)
    for m in seg_result.masks:
        union |= m.astype(bool)
    mask_img = (union.astype(np.uint8) * 255)
    Image.fromarray(mask_img, mode="L").save(png_path)

    if len(seg_result.masks) > 1:
        meta = {
            "image_path": str(seg_result.image_path),
            "text_prompt": seg_result.text_prompt,
            "num_masks": int(seg_result.num_masks),
            "scores": [float(s) for s in seg_result.scores],
            "boxes": [np.asarray(b).tolist() for b in seg_result.boxes],
        }
        with open(mask_dir / f"{stem}.json", "w") as f:
            json.dump(meta, f, indent=2)


# -----------------------------------------------------------------------------
# Worker threads
# -----------------------------------------------------------------------------

class _WorkerThread(threading.Thread):
    """Thread that captures exceptions so the main thread can re-raise them."""

    def __init__(self, target, name):
        super().__init__(target=target, name=name, daemon=True)
        self.exc: BaseException | None = None

    def run(self) -> None:  # type: ignore[override]
        try:
            if self._target is not None:
                self._target()
        except BaseException as e:  # noqa: BLE001 - propagate to main
            self.exc = e
            logger.error("Worker %s failed: %s\n%s", self.name, e, traceback.format_exc())


def _run_sam3d(
    body_model: Sam3DBody,
    image_paths: list[Path],
    mhr_dir: Path,
    io_pool: ThreadPoolExecutor,
    bbox_threshold: float,
    progress: "Queue[str]",
    shape_override: np.ndarray | None = None,
) -> None:
    for img_path in image_paths:
        try:
            result = body_model.predict(
                img_path,
                bbox_threshold=bbox_threshold,
                shape_override=shape_override,
            )
            result.image_path = str(img_path)
        except Exception:  # noqa: BLE001 - log & continue per-image
            logger.exception("SAM3D predict failed for %s", img_path)
            progress.put(f"sam3d:skip:{img_path.name}")
            continue

        stem = img_path.stem
        # Detach from the GPU thread: snapshot numpy arrays are already on CPU
        io_pool.submit(_save_mhr_outputs, result, mhr_dir, stem)
        progress.put(f"sam3d:done:{img_path.name}")


def _run_sam3(
    segmenter,
    image_paths: list[Path],
    mask_dir: Path,
    io_pool: ThreadPoolExecutor,
    text: str,
    batch_size: int,
    threshold: float,
    mask_threshold: float,
    progress: "Queue[str]",
) -> None:
    for start in range(0, len(image_paths), batch_size):
        batch = image_paths[start:start + batch_size]
        try:
            if len(batch) == 1:
                batch_results = [segmenter.predict(
                    batch[0], text=text,
                    threshold=threshold, mask_threshold=mask_threshold,
                )]
            else:
                batch_results = segmenter.predict_batch(
                    batch, text=text,
                    threshold=threshold, mask_threshold=mask_threshold,
                )
        except Exception:  # noqa: BLE001
            logger.exception(
                "SAM3 predict_batch failed for batch starting at %s", batch[0]
            )
            for img_path in batch:
                progress.put(f"sam3:skip:{img_path.name}")
            continue

        for seg_result, img_path in zip(batch_results, batch):
            # Ensure image_path reflects the input path even if the segmenter
            # normalized it
            seg_result.image_path = str(img_path)
            io_pool.submit(_save_mask_outputs, seg_result, mask_dir, img_path.stem)
            progress.put(f"sam3:done:{img_path.name}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAM 3D Body + SAM 3 person masks on a folder of images."
    )
    parser.add_argument("--images", type=str, required=True,
                        help="Path to an image file or directory of images")
    parser.add_argument("--output", type=str, default="./sam3d_mask_results",
                        help="Output directory (default: ./sam3d_mask_results). "
                             "Ignored if both --mhr-dir and --mask-dir are set.")
    parser.add_argument("--mhr-dir", type=str, default=None,
                        help="Explicit dir for SAM3D .npz outputs "
                             "(default: <output>/mhr)")
    parser.add_argument("--mask-dir", type=str, default=None,
                        help="Explicit dir for SAM3 mask .png outputs "
                             "(default: <output>/mask)")
    parser.add_argument("--shape-override", type=str, default=None,
                        help="Path to a (45,) .npy shape vector applied to "
                             "every SAM3D prediction")
    parser.add_argument("--variant", choices=["dinov3", "vith"], default="dinov3",
                        help="SAM 3D Body model variant (default: dinov3)")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (default: cuda if available)")
    parser.add_argument("--bbox-threshold", type=float, default=0.5,
                        help="SAM 3D Body detection threshold (default: 0.5)")
    parser.add_argument("--text", type=str, default="a person",
                        help="SAM 3 text prompt (default: 'a person')")
    parser.add_argument("--mask-threshold", type=float, default=0.5,
                        help="SAM 3 mask binarization threshold (default: 0.5)")
    parser.add_argument("--seg-threshold", type=float, default=0.5,
                        help="SAM 3 detection confidence threshold (default: 0.5)")
    parser.add_argument("--mask-batch-size", type=int, default=8,
                        help="Images per SAM 3 forward pass (default: 8)")
    parser.add_argument("--io-workers", type=int, default=4,
                        help="Background threads for disk I/O (default: 4)")
    parser.add_argument("--sequential", action="store_true",
                        help="Run SAM3D then SAM3 serially (frees GPU between "
                             "models; use on memory-constrained devices).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        from sam3_wrapper import Sam3Segmenter
    except ImportError as e:
        sys.exit(
            f"Failed to import sam3_wrapper ({e}).\n"
            "Install the optional extra with:\n"
            "    bash scripts/setup_sam3.sh\n"
            "(or: uv sync --extra sam3 && uv run sam3-download)"
        )

    image_root = Path(args.images)
    image_paths = _collect_images(image_root)
    if not image_paths:
        sys.exit(f"No images found at {image_root}")

    output_dir = Path(args.output)
    mhr_dir = Path(args.mhr_dir) if args.mhr_dir else output_dir / "mhr"
    mask_dir = Path(args.mask_dir) if args.mask_dir else output_dir / "mask"
    mhr_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    shape_override: np.ndarray | None = None
    if args.shape_override is not None:
        shape_override = np.load(args.shape_override).astype(np.float32)
        if shape_override.shape != (45,):
            sys.exit(
                f"--shape-override must be a (45,) array, got {shape_override.shape}"
            )
        print(f"Using shape override from {args.shape_override} (shape={shape_override.shape})")

    print(f"Found {len(image_paths)} images. Loading models ...")

    progress: "Queue[str]" = Queue()
    io_pool = ThreadPoolExecutor(max_workers=args.io_workers, thread_name_prefix="io")

    if args.sequential:
        import gc

        import torch

        # --- SAM3D pass ---
        print("Sequential mode: running SAM3D first ...")
        body_model = Sam3DBody(
            model_variant=args.variant,
            device=args.device,
            bbox_threshold=args.bbox_threshold,
        )
        body_model._ensure_estimator()
        _run_sam3d(
            body_model, image_paths, mhr_dir, io_pool,
            args.bbox_threshold, progress,
            shape_override=shape_override,
        )
        sam3d_done = 0
        while sam3d_done < len(image_paths):
            progress.get()
            sam3d_done += 1
            if sam3d_done % 10 == 0 or sam3d_done == len(image_paths):
                print(f"  progress: sam3d {sam3d_done}/{len(image_paths)}")

        del body_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- SAM3 pass ---
        print("Sequential mode: running SAM3 masks ...")
        segmenter = Sam3Segmenter(
            device=args.device,
            threshold=args.seg_threshold,
            mask_threshold=args.mask_threshold,
        )
        _run_sam3(
            segmenter, image_paths, mask_dir, io_pool,
            args.text, args.mask_batch_size,
            args.seg_threshold, args.mask_threshold, progress,
        )
        sam3_done = 0
        while sam3_done < len(image_paths):
            progress.get()
            sam3_done += 1
            if sam3_done % 10 == 0 or sam3_done == len(image_paths):
                print(f"  progress: sam3 {sam3_done}/{len(image_paths)}")

        io_pool.shutdown(wait=True)
        print(f"Done. MHR outputs: {mhr_dir} | Mask outputs: {mask_dir}")
        return

    # --- Concurrent (default) mode ---
    body_model = Sam3DBody(
        model_variant=args.variant,
        device=args.device,
        bbox_threshold=args.bbox_threshold,
    )
    body_model._ensure_estimator()

    segmenter = Sam3Segmenter(
        device=args.device,
        threshold=args.seg_threshold,
        mask_threshold=args.mask_threshold,
    )

    print("Starting concurrent SAM3D + SAM3 pipelines ...")

    sam3d_thread = _WorkerThread(
        target=lambda: _run_sam3d(
            body_model, image_paths, mhr_dir, io_pool,
            args.bbox_threshold, progress,
            shape_override=shape_override,
        ),
        name="sam3d",
    )
    sam3_thread = _WorkerThread(
        target=lambda: _run_sam3(
            segmenter, image_paths, mask_dir, io_pool,
            args.text, args.mask_batch_size,
            args.seg_threshold, args.mask_threshold, progress,
        ),
        name="sam3",
    )

    sam3d_thread.start()
    sam3_thread.start()

    expected = 2 * len(image_paths)
    sam3d_done = sam3_done = 0
    while sam3d_done + sam3_done < expected:
        msg = progress.get()
        pipeline, status, _name = msg.split(":", 2)
        if pipeline == "sam3d":
            sam3d_done += 1
        else:
            sam3_done += 1
        if (sam3d_done + sam3_done) % 10 == 0 or (sam3d_done + sam3_done) == expected:
            print(
                f"  progress: sam3d {sam3d_done}/{len(image_paths)} | "
                f"sam3 {sam3_done}/{len(image_paths)}"
            )

    sam3d_thread.join()
    sam3_thread.join()

    io_pool.shutdown(wait=True)

    for t in (sam3d_thread, sam3_thread):
        if t.exc is not None:
            raise t.exc

    print(f"Done. MHR outputs: {mhr_dir} | Mask outputs: {mask_dir}")


if __name__ == "__main__":
    main()

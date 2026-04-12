"""
High-level inference API for SAM 3D Body.

Provides Sam3DBody class for single-image and batch inference,
wrapping the upstream notebook_utils and model loading.
"""

import argparse
import functools
import logging
import os
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any

import cv2
import numpy as np

from sam3d_wrapper.download import HF_REPOS, get_checkpoint_path, verify_checkpoint
from sam3d_wrapper.repo import activate_imports, ensure_detectron2, ensure_repo

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}

_PREFETCH_SENTINEL = object()  # signals end of prefetch queue

logger = logging.getLogger(__name__)


def _bg_save(fn):
    """Decorator: dispatch save method to background thread when enabled.

    When ``self._bg_saver`` is active, snapshots numpy array arguments
    (via copy) on the calling thread and queues the work. Returns None.
    When no saver is active, calls *fn* directly.
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if self._bg_saver is None:
            return fn(self, *args, **kwargs)
        # Snapshot numpy arrays so the caller can safely reuse buffers
        safe_args = tuple(
            a.copy() if isinstance(a, np.ndarray) else a for a in args
        )
        safe_kwargs = {
            k: v.copy() if isinstance(v, np.ndarray) else v
            for k, v in kwargs.items()
        }
        self._bg_saver.submit(lambda: fn(self, *safe_args, **safe_kwargs))
        return None

    return wrapper


class _BackgroundSaver:
    """Daemon thread + bounded queue for offloading I/O from the main loop.

    Modelled after gaussian_avatar's ImageSaver pattern.
    """

    def __init__(self, maxsize: int = 100) -> None:
        self._queue: Queue = Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            fn = self._queue.get()
            if fn is None:
                self._queue.task_done()
                break
            try:
                fn()
            except Exception:
                logger.exception("Background save failed")
            self._queue.task_done()

    def submit(self, fn) -> None:
        self._queue.put(fn)

    def flush(self) -> None:
        """Block until all queued work is done."""
        self._queue.join()

    def shutdown(self) -> None:
        """Stop the worker thread. Safe to call multiple times."""
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=60)
        self._thread = None


class _ImagePrefetcher:
    """Reads images from disk on a background thread, one step ahead of inference.

    Usage::

        prefetcher = _ImagePrefetcher(file_list, maxsize=2)
        for path, img_bgr in prefetcher:
            ...  # img_bgr is already loaded
    """

    def __init__(self, paths: list[Path], maxsize: int = 2) -> None:
        self._paths = paths
        self._queue: Queue = Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        for p in self._paths:
            img = cv2.imread(str(p))
            self._queue.put((p, img))
        self._queue.put(_PREFETCH_SENTINEL)

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is _PREFETCH_SENTINEL:
                break
            yield item


@dataclass
class PersonResult:
    """Result for a single detected person in an image."""

    person_id: int
    pred_vertices: np.ndarray
    pred_keypoints_3d: np.ndarray
    pred_keypoints_2d: np.ndarray
    pred_cam_t: np.ndarray
    focal_length: float
    bbox: np.ndarray
    body_pose_params: np.ndarray | None = None
    hand_pose_params: np.ndarray | None = None
    shape_params: np.ndarray | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class ImageResult:
    """Result for a single image containing zero or more detected people."""

    image_path: str
    persons: list[PersonResult]
    image_bgr: np.ndarray | None = field(default=None, repr=False)

    @property
    def num_persons(self) -> int:
        return len(self.persons)


def _collect_images(path: str | Path) -> list[Path]:
    """Collect image files from a path (file or directory)."""
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = []
        for f in sorted(path.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(f)
        return files
    raise FileNotFoundError(f"Path does not exist: {path}")


def _bbox_area(bbox: np.ndarray) -> float:
    """Area of an [x1, y1, x2, y2] bbox."""
    x1, y1, x2, y2 = bbox[:4]
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def _select_largest(persons: list[PersonResult], max_persons: int) -> list[PersonResult]:
    """Keep the ``max_persons`` largest bboxes, preserving original order."""
    if max_persons <= 0 or len(persons) <= max_persons:
        return persons
    ranked = sorted(persons, key=lambda p: _bbox_area(p.bbox), reverse=True)
    keep = {id(p) for p in ranked[:max_persons]}
    return [p for p in persons if id(p) in keep]


def _bbox_center(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox[:4]
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)


def _select_most_isolated(
    persons: list[PersonResult], max_persons: int
) -> list[PersonResult]:
    """Keep the ``max_persons`` boxes with the largest nearest-neighbor distance.

    Each box is scored by the Euclidean distance from its center to the
    center of its closest other box; the top-scoring boxes are kept. When
    there are at most ``max_persons`` boxes the list is returned as-is.
    """
    if max_persons <= 0 or len(persons) <= max_persons:
        return persons
    centers = np.stack([_bbox_center(p.bbox) for p in persons])
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    nn_dist = dists.min(axis=1)
    top_idx = set(np.argsort(-nn_dist)[:max_persons].tolist())
    return [p for i, p in enumerate(persons) if i in top_idx]


def _parse_outputs(raw_outputs: list[dict], person_offset: int = 0) -> list[PersonResult]:
    """Convert raw upstream output dicts to PersonResult objects."""
    results = []
    for i, out in enumerate(raw_outputs):
        results.append(PersonResult(
            person_id=person_offset + i,
            pred_vertices=out["pred_vertices"],
            pred_keypoints_3d=out.get("pred_keypoints_3d", np.array([])),
            pred_keypoints_2d=out["pred_keypoints_2d"],
            pred_cam_t=out["pred_cam_t"],
            focal_length=out["focal_length"],
            bbox=out["bbox"],
            body_pose_params=out.get("body_pose_params"),
            hand_pose_params=out.get("hand_pose_params"),
            shape_params=out.get("shape_params"),
            raw=out,
        ))
    return results


class Sam3DBody:
    """
    High-level wrapper for SAM 3D Body inference.

    Example:
        from sam3d_wrapper import Sam3DBody

        model = Sam3DBody(model_variant="dinov3")

        # Single image
        result = model.predict("photo.jpg")
        for person in result.persons:
            print(person.pred_vertices.shape)

        # Batch of images
        results = model.predict_batch("./images/", output_dir="./output/")

    Args:
        model_variant: "dinov3" (DINOv3-H+, 840M params) or "vith" (ViT-H, 631M params).
        hf_repo_id: Override the Hugging Face repo ID (uses variant default if None).
        device: PyTorch device string. Defaults to "cuda" if available, else "cpu".
        use_detector: Enable human detection (ViTDet). Requires detectron2.
        use_segmentor: Enable segmentation (SAM2). Requires SAM3 installed and path set.
        use_fov: Enable field-of-view estimation (MoGe2). Requires moge package.
        segmentor_path: Path to SAM2/SAM3 weights (required if use_segmentor=True).
        bbox_threshold: Detection confidence threshold (default: 0.5).
    """

    def __init__(
        self,
        model_variant: str = "dinov3",
        hf_repo_id: str | None = None,
        device: str | None = None,
        use_detector: bool = True,
        use_segmentor: bool = False,
        use_fov: bool = True,
        segmentor_path: str = "",
        bbox_threshold: float = 0.5,
    ):
        import torch

        self.model_variant = model_variant
        self.bbox_threshold = bbox_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve HF repo
        if hf_repo_id is None:
            hf_repo = HF_REPOS.get(model_variant)
            if hf_repo is None:
                raise ValueError(
                    f"Unknown model_variant '{model_variant}'. "
                    f"Choose from: {list(HF_REPOS.keys())}"
                )
            self.hf_repo_id = hf_repo
        else:
            self.hf_repo_id = hf_repo_id

        # Store init config for lazy loading
        self._use_detector = use_detector
        self._use_segmentor = use_segmentor
        self._use_fov = use_fov
        self._segmentor_path = segmentor_path

        # Estimator is loaded lazily on first predict() call
        self._estimator = None

        # Background saver for offloading I/O; set per predict_batch call
        self._bg_saver: _BackgroundSaver | None = None

    def _ensure_estimator(self) -> None:
        """Load the model on first use (lazy init)."""
        if self._estimator is not None:
            return

        ensure_repo()
        if self._use_detector:
            ensure_detectron2()
        activate_imports()

        from notebook.utils import setup_sam_3d_body

        self._estimator = setup_sam_3d_body(
            hf_repo_id=self.hf_repo_id,
            detector_name="vitdet" if self._use_detector else "",
            segmentor_name="sam2" if self._use_segmentor else "",
            fov_name="moge2" if self._use_fov else "",
            segmentor_path=self._segmentor_path,
            device=self.device,
        )

    @property
    def faces(self) -> np.ndarray:
        """Mesh face indices from the loaded model."""
        self._ensure_estimator()
        return self._estimator.faces

    @staticmethod
    def average_shape(results: list[ImageResult]) -> np.ndarray:
        """Compute the mean shape vector across all detected persons.

        Useful for getting a consistent body shape from a sequence of frames
        of the same person, then passing it as ``shape_override`` to
        ``predict`` or ``predict_batch``.

        Args:
            results: List of ImageResult from prior inference.

        Returns:
            Mean shape vector as a (45,) numpy array.
        """
        shapes = [
            p.shape_params
            for r in results
            for p in r.persons
            if p.shape_params is not None
        ]
        if not shapes:
            raise ValueError("No shape parameters found in results")
        return np.mean(shapes, axis=0)

    def recompute_with_shape(
        self, result: ImageResult, shape_override: np.ndarray
    ) -> ImageResult:
        """Recompute mesh vertices for all persons using a fixed shape vector.

        Replaces ``pred_vertices``, ``pred_keypoints_3d``, and
        ``shape_params`` in each person's result, keeping the original
        pose, scale, and camera parameters.

        When a frame contains multiple persons, their parameters are batched
        into a single forward pass for efficiency.

        Args:
            result: An ImageResult from a prior ``predict`` call.
            shape_override: A (45,) numpy array of shape PCA coefficients.

        Returns:
            The same ImageResult, mutated in-place with updated vertices.
        """
        if not result.persons:
            return result

        import torch

        self._ensure_estimator()
        head = self._estimator.model.head_pose
        device = next(head.parameters()).device

        n = len(result.persons)

        # Stack all person params into batched tensors (N, dim)
        def _stack(key):
            return torch.tensor(
                np.stack([p.raw[key] for p in result.persons]),
                dtype=torch.float32,
            ).to(device)

        global_rot = _stack("global_rot")
        body_pose = _stack("body_pose_params")
        hand_pose = _stack("hand_pose_params")
        scale = _stack("scale_params")
        expr = _stack("expr_params")
        shape_t = torch.tensor(
            shape_override, dtype=torch.float32
        ).unsqueeze(0).expand(n, -1).to(device)

        with torch.no_grad():
            verts, j3d, jcoords, model_params, joint_rots = head.mhr_forward(
                global_trans=torch.zeros_like(global_rot),
                global_rot=global_rot,
                body_pose_params=body_pose,
                hand_pose_params=hand_pose,
                scale_params=scale,
                shape_params=shape_t,
                expr_params=expr,
                return_keypoints=True,
                return_joint_coords=True,
                return_model_params=True,
                return_joint_rotations=True,
            )
            j3d = j3d[:, :70]
            verts[..., [1, 2]] *= -1
            j3d[..., [1, 2]] *= -1

        # Unpack batched results back to per-person
        verts_np = verts.cpu().numpy()
        j3d_np = j3d.cpu().numpy()
        jcoords_np = jcoords.cpu().numpy()
        model_params_np = model_params.cpu().numpy()
        joint_rots_np = joint_rots.cpu().numpy()

        for i, person in enumerate(result.persons):
            raw = person.raw
            person.pred_vertices = verts_np[i]
            person.pred_keypoints_3d = j3d_np[i]
            person.shape_params = shape_override.copy()
            raw["pred_vertices"] = verts_np[i]
            raw["pred_keypoints_3d"] = j3d_np[i]
            raw["shape_params"] = shape_override.copy()
            raw["pred_joint_coords"] = jcoords_np[i]
            raw["mhr_model_params"] = model_params_np[i]
            raw["pred_global_rots"] = joint_rots_np[i]

        return result

    @staticmethod
    def _build_cam_intrinsics(
        focal_length: float, image_hw: tuple[int, int]
    ) -> np.ndarray:
        """Build a (1, 3, 3) camera intrinsics matrix from a focal length."""
        h, w = image_hw
        return np.array(
            [[[focal_length, 0, w / 2.0],
              [0, focal_length, h / 2.0],
              [0, 0, 1]]],
            dtype=np.float32,
        )

    def predict(
        self,
        image: str | Path | np.ndarray,
        bbox_threshold: float | None = None,
        use_mask: bool = False,
        bboxes: np.ndarray | None = None,
        shape_override: np.ndarray | None = None,
        focal_length: float | None = None,
        max_persons: int | None = None,
        max_isolated_persons: int | None = None,
    ) -> ImageResult:
        """
        Run inference on a single image.

        Args:
            image: Path to an image file, or a BGR numpy array (HxWx3 uint8).
            bbox_threshold: Override the default detection threshold.
            use_mask: Enable mask-conditioned prediction.
            bboxes: Provide manual bounding boxes as Nx4 array [x1,y1,x2,y2].
            shape_override: Fixed (45,) shape vector applied after inference.
                Useful for enforcing consistent body shape across a sequence.
            focal_length: Fixed focal length in pixels. When provided, skips
                FOV estimation and uses this value for both fx and fy with the
                principal point at the image center.
            max_persons: If set, keep only the N persons with the largest
                bboxes (by area). Applied before ``shape_override``.
            max_isolated_persons: If set, keep only the N persons whose
                bbox center is farthest from its nearest neighbor. Applied
                after ``max_persons`` and before ``shape_override``.

        Returns:
            ImageResult with detected persons and their mesh data.
        """
        import torch

        self._ensure_estimator()

        thr = bbox_threshold if bbox_threshold is not None else self.bbox_threshold

        if isinstance(image, (str, Path)):
            image_path = str(image)
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
        else:
            image_path = "<array>"
            img_bgr = image

        kwargs: dict[str, Any] = {"bbox_thr": thr}
        if use_mask:
            kwargs["use_mask"] = True
        if bboxes is not None:
            kwargs["bboxes"] = bboxes
        if focal_length is not None:
            h, w = img_bgr.shape[:2]
            cam_int = self._build_cam_intrinsics(focal_length, (h, w))
            kwargs["cam_int"] = torch.tensor(cam_int)

        raw_outputs = self._estimator.process_one_image(
            img_bgr if image_path == "<array>" else image_path, **kwargs
        )
        persons = _parse_outputs(raw_outputs)

        if max_persons is not None:
            persons = _select_largest(persons, max_persons)
        if max_isolated_persons is not None:
            persons = _select_most_isolated(persons, max_isolated_persons)

        result = ImageResult(
            image_path=image_path,
            persons=persons,
            image_bgr=img_bgr,
        )

        if shape_override is not None:
            self.recompute_with_shape(result, shape_override)

        return result

    def predict_batch(
        self,
        images_path: str | Path,
        output_dir: str | Path | None = None,
        bbox_threshold: float | None = None,
        use_mask: bool = False,
        save_meshes: bool = False,
        save_visualizations: bool = True,
        shape_override: np.ndarray | None = None,
        shape_calibration_frames: int | None = None,
        focal_length: float | None = None,
        max_persons: int | None = None,
        max_isolated_persons: int | None = None,
    ) -> list[ImageResult]:
        """
        Run inference on a directory of images.

        Args:
            images_path: Directory or single file path.
            output_dir: If set, save visualizations/meshes here.
            bbox_threshold: Override detection threshold.
            use_mask: Enable mask-conditioned prediction.
            save_meshes: Save .ply mesh files per person.
            save_visualizations: Save rendered overlay images.
            shape_override: Fixed (45,) shape vector applied to every frame.
                Use ``average_shape()`` on a prior run's results to obtain one.
            shape_calibration_frames: If set, randomly sample this many frames,
                run inference on them to compute an average shape, then re-run
                all frames with that shape locked in. The computed shape is saved
                to ``output_dir/shape.npy``. Ignored when ``shape_override``
                is already provided.
            focal_length: Fixed focal length in pixels. When provided, skips
                FOV estimation and uses this value for every frame.
            max_persons: If set, keep only the N persons with the largest
                bboxes (by area) in each frame.
            max_isolated_persons: If set, keep only the N persons whose
                bbox center is farthest from its nearest neighbor. Applied
                after ``max_persons``.

        Returns:
            List of ImageResult, one per input image.
        """
        from tqdm import tqdm

        image_files = _collect_images(images_path)
        if not image_files:
            print(f"No images found at {images_path}")
            return []

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # --- Shape calibration phase (results are cached for reuse) ---
        cal_cache: dict[str, ImageResult] = {}
        if shape_calibration_frames is not None and shape_override is None:
            n = min(shape_calibration_frames, len(image_files))
            calibration_files = random.sample(image_files, n)
            print(
                f"Shape calibration: running {n} randomly sampled frames "
                f"out of {len(image_files)} ..."
            )
            cal_results = []
            for img_path in tqdm(calibration_files, desc="Calibration"):
                r = self.predict(
                    img_path,
                    bbox_threshold=bbox_threshold,
                    use_mask=use_mask,
                    focal_length=focal_length,
                    max_persons=max_persons,
                    max_isolated_persons=max_isolated_persons,
                )
                cal_results.append(r)
                cal_cache[str(img_path)] = r
            shape_override = self.average_shape(cal_results)
            print(f"Calibrated shape from {n} frames.")

        print(f"Processing {len(image_files)} images ...")
        if shape_override is not None:
            print("Using fixed shape override for all frames.")
            if output_dir is not None:
                shape_path = output_dir / "shape.npy"
                np.save(str(shape_path), shape_override)
                print(f"Saved shape to {shape_path}")

        # Start background saver so I/O doesn't block the inference loop
        if output_dir is not None:
            self._bg_saver = _BackgroundSaver()

        # Prefetch images from disk while GPU is busy
        prefetcher = _ImagePrefetcher(image_files)

        results = []
        try:
            for img_path, img_bgr in tqdm(prefetcher, total=len(image_files), desc="Inference"):
                # Reuse cached calibration result (just apply shape override)
                cached = cal_cache.pop(str(img_path), None)
                if cached is not None:
                    if shape_override is not None:
                        self.recompute_with_shape(cached, shape_override)
                    result = cached
                else:
                    if img_bgr is None:
                        logger.warning("Could not read image: %s (skipping)", img_path)
                        continue
                    result = self.predict(
                        img_bgr,
                        bbox_threshold=bbox_threshold,
                        use_mask=use_mask,
                        shape_override=shape_override,
                        focal_length=focal_length,
                        max_persons=max_persons,
                        max_isolated_persons=max_isolated_persons,
                    )
                    # Restore the real path (predict sets "<array>" for ndarray input)
                    result.image_path = str(img_path)

                results.append(result)

                if output_dir is not None and result.num_persons > 0:
                    image_name = img_path.stem

                    # Always save raw numpy results
                    self._save_numpy_results(result, output_dir, image_name)

                    if save_visualizations:
                        self._save_visualization(result, output_dir, image_name)

                    if save_meshes:
                        self._save_meshes(result, output_dir, image_name)
        finally:
            if self._bg_saver is not None:
                self._bg_saver.flush()
                self._bg_saver.shutdown()
                self._bg_saver = None

        total_persons = sum(r.num_persons for r in results)
        print(f"Done: {len(results)} images, {total_persons} persons detected.")
        return results

    @_bg_save
    def _save_numpy_results(
        self, result: ImageResult, output_dir: Path, image_name: str
    ) -> None:
        """Save all raw arrays per person as .npz files in raw/ subdir."""
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        for p in result.persons:
            suffix = "" if result.num_persons == 1 else f"_person{p.person_id}"
            out_path = raw_dir / f"{image_name}{suffix}.npz"
            np.savez(str(out_path), **p.raw)

    @_bg_save
    def _save_visualization(
        self, result: ImageResult, output_dir: Path, image_name: str
    ) -> None:
        """Save a visualization image with mesh overlays in vis/ subdir."""
        from tools.vis_utils import visualize_sample_together

        if result.image_bgr is None or result.num_persons == 0:
            return

        vis_dir = output_dir / "vis"
        vis_dir.mkdir(exist_ok=True)
        raw_outputs = [p.raw for p in result.persons]
        vis = visualize_sample_together(result.image_bgr, raw_outputs, self.faces)
        out_path = vis_dir / f"{image_name}.jpg"
        cv2.imwrite(str(out_path), vis.astype(np.uint8))

    @_bg_save
    def _save_meshes(
        self, result: ImageResult, output_dir: Path, image_name: str
    ) -> None:
        """Save .ply mesh files for each detected person."""
        from notebook.utils import save_mesh_results

        if result.image_bgr is None or result.num_persons == 0:
            return

        raw_outputs = [p.raw for p in result.persons]
        save_mesh_results(
            result.image_bgr,
            raw_outputs,
            self.faces,
            str(output_dir),
            image_name,
        )


def cli_infer() -> None:
    """CLI entry point: sam3d-infer"""
    parser = argparse.ArgumentParser(
        description="Run SAM 3D Body inference on images."
    )
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to an image file or directory of images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./sam3d_output",
        help="Output directory for results (default: ./sam3d_output)",
    )
    parser.add_argument(
        "--variant",
        choices=list(HF_REPOS.keys()),
        default="dinov3",
        help="Model variant (default: dinov3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda if available)",
    )
    parser.add_argument(
        "--bbox-threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--use-mask",
        action="store_true",
        help="Enable mask-conditioned prediction",
    )
    parser.add_argument(
        "--save-meshes",
        action="store_true",
        help="Save .ply mesh files",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Skip saving visualization images",
    )
    parser.add_argument(
        "--no-detector",
        action="store_true",
        help="Disable human detection (use full image)",
    )
    parser.add_argument(
        "--no-fov",
        action="store_true",
        help="Disable FOV estimation (use default FOV)",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=None,
        help="Fixed focal length in pixels (skips FOV estimation)",
    )
    parser.add_argument(
        "--shape-override",
        type=str,
        default=None,
        help="Path to a .npy file containing a (45,) shape vector to use for all frames",
    )
    parser.add_argument(
        "--max-persons",
        type=int,
        default=None,
        help="Keep only the N persons with the largest bboxes per frame.",
    )
    parser.add_argument(
        "--max-isolated-persons",
        type=int,
        default=None,
        help="Keep only the N persons whose bbox is farthest from its nearest neighbor.",
    )
    parser.add_argument(
        "--shape-calibration-frames",
        type=int,
        default=None,
        help="Randomly sample N frames to estimate an average shape, then use it for all frames. "
             "Ignored when --shape-override is provided.",
    )
    args = parser.parse_args()

    shape_override = None
    if args.shape_override:
        shape_override = np.load(args.shape_override)
        print(f"Loaded shape override from {args.shape_override}")

    model = Sam3DBody(
        model_variant=args.variant,
        device=args.device,
        use_detector=not args.no_detector,
        use_fov=not args.no_fov,
        bbox_threshold=args.bbox_threshold,
    )

    model.predict_batch(
        images_path=args.images,
        output_dir=args.output,
        bbox_threshold=args.bbox_threshold,
        use_mask=args.use_mask,
        save_meshes=args.save_meshes,
        save_visualizations=not args.no_vis,
        shape_override=shape_override,
        shape_calibration_frames=args.shape_calibration_frames,
        focal_length=args.focal_length,
        max_persons=args.max_persons,
        max_isolated_persons=args.max_isolated_persons,
    )

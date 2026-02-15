"""
High-level inference API for SAM 3D Body.

Provides Sam3DBody class for single-image and batch inference,
wrapping the upstream notebook_utils and model loading.
"""

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sam3d_wrapper.download import HF_REPOS, get_checkpoint_path, verify_checkpoint
from sam3d_wrapper.repo import activate_imports, ensure_repo

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}


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

        # Ensure the upstream repo is available
        ensure_repo()
        activate_imports()

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

        # Import upstream modules (now on sys.path)
        from notebook.utils import setup_sam_3d_body

        # Build estimator
        self._estimator = setup_sam_3d_body(
            hf_repo_id=self.hf_repo_id,
            detector_name="vitdet" if use_detector else "",
            segmentor_name="sam2" if use_segmentor else "",
            fov_name="moge2" if use_fov else "",
            segmentor_path=segmentor_path,
            device=self.device,
        )

    @property
    def faces(self) -> np.ndarray:
        """Mesh face indices from the loaded model."""
        return self._estimator.faces

    def predict(
        self,
        image: str | Path | np.ndarray,
        bbox_threshold: float | None = None,
        use_mask: bool = False,
        bboxes: np.ndarray | None = None,
    ) -> ImageResult:
        """
        Run inference on a single image.

        Args:
            image: Path to an image file, or a BGR numpy array (HxWx3 uint8).
            bbox_threshold: Override the default detection threshold.
            use_mask: Enable mask-conditioned prediction.
            bboxes: Provide manual bounding boxes as Nx4 array [x1,y1,x2,y2].

        Returns:
            ImageResult with detected persons and their mesh data.
        """
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

        raw_outputs = self._estimator.process_one_image(image_path, **kwargs)
        persons = _parse_outputs(raw_outputs)

        return ImageResult(
            image_path=image_path,
            persons=persons,
            image_bgr=img_bgr,
        )

    def predict_batch(
        self,
        images_path: str | Path,
        output_dir: str | Path | None = None,
        bbox_threshold: float | None = None,
        use_mask: bool = False,
        save_meshes: bool = False,
        save_visualizations: bool = True,
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

        Returns:
            List of ImageResult, one per input image.
        """
        from tqdm import tqdm

        image_files = _collect_images(images_path)
        if not image_files:
            print(f"No images found at {images_path}")
            return []

        print(f"Processing {len(image_files)} images ...")

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for img_path in tqdm(image_files, desc="Inference"):
            result = self.predict(
                img_path,
                bbox_threshold=bbox_threshold,
                use_mask=use_mask,
            )
            results.append(result)

            if output_dir is not None and result.num_persons > 0:
                image_name = img_path.stem

                if save_visualizations:
                    self._save_visualization(result, output_dir, image_name)

                if save_meshes:
                    self._save_meshes(result, output_dir, image_name)

        total_persons = sum(r.num_persons for r in results)
        print(f"Done: {len(results)} images, {total_persons} persons detected.")
        return results

    def _save_visualization(
        self, result: ImageResult, output_dir: Path, image_name: str
    ) -> None:
        """Save a visualization image with mesh overlays."""
        activate_imports()
        from tools.vis_utils import visualize_sample_together

        if result.image_bgr is None or result.num_persons == 0:
            return

        raw_outputs = [p.raw for p in result.persons]
        vis = visualize_sample_together(result.image_bgr, raw_outputs, self.faces)
        out_path = output_dir / f"{image_name}_vis.jpg"
        cv2.imwrite(str(out_path), vis.astype(np.uint8))

    def _save_meshes(
        self, result: ImageResult, output_dir: Path, image_name: str
    ) -> None:
        """Save .ply mesh files for each detected person."""
        activate_imports()
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
    args = parser.parse_args()

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
    )

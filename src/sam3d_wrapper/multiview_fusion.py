"""
Two-view mesh fusion for SAM 3D Body outputs.

Fuses two single-image ``PersonResult`` / ``.npz`` outputs of the same
person captured simultaneously from different cameras into a single mesh
in camera A's frame. Alignment uses the 70 MHR joints as known
correspondences (Umeyama scaled Procrustes); geometry is blended with
per-vertex visibility weights derived from surface normals.

The primary fused output (``fused.npz``) mirrors view A's
``predict_batch`` npz schema 1:1 so it is a drop-in replacement for any
downstream consumer. Only ``pred_vertices`` is replaced with the fused
mesh; all other keys (cam_t, focal, pose/shape params, ...) are inherited
from view A. Alignment diagnostics are written to a sidecar
(``fused.alignment.npz``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# MHR70 joint grouping (see vendor/sam-3d-body/sam_3d_body/metadata/mhr70.py).
# Indices 0-20   : body + feet
# Indices 21-62  : left hand + right hand (42 finger joints, often noisy)
# Indices 63-69  : olecranon / cubital fossa / acromion / neck
BODY_JOINT_INDICES: tuple[int, ...] = tuple(range(0, 21)) + tuple(range(63, 70))
HAND_JOINT_INDICES: tuple[int, ...] = tuple(range(21, 63))
NUM_JOINTS: int = 70


@dataclass
class AlignmentResult:
    """Result of aligning view B's joints into view A's camera frame."""

    s: float
    R: np.ndarray              # (3, 3)
    t: np.ndarray              # (3,)
    residuals: np.ndarray      # (K_used,) per-joint L2 after alignment
    median_residual: float
    joint_mask: np.ndarray     # (70,) bool — which joints entered the solve
    J_A: np.ndarray            # (K_used, 3) A's joints in A's frame
    J_B: np.ndarray            # (K_used, 3) B's joints in B's frame
    J_B_in_A: np.ndarray       # (K_used, 3) B's joints after alignment
    flag: str = "ok"           # "ok" | "bad_fit"


@dataclass
class FusionResult:
    """Result of fusing two meshes in view A's camera frame."""

    V_fused: np.ndarray        # (V, 3) fused mesh in A's frame (pre cam_t)
    faces: np.ndarray          # (F, 3)
    w_A: np.ndarray            # (V,) normalized weight used for view A
    w_B: np.ndarray            # (V,) normalized weight used for view B
    alignment: AlignmentResult
    V_A: np.ndarray            # (V, 3) A's original mesh in A's frame
    V_B_in_A: np.ndarray       # (V, 3) B's mesh transformed into A's frame


@dataclass
class _FusionInput:
    """Internal: everything fusion needs from one view, constructed via
    either ``from_npz`` or ``from_person_result``."""

    V: np.ndarray                          # (V, 3) mesh in camera frame
    J3d: np.ndarray                        # (70, 3) joints in camera frame
    J2d: np.ndarray                        # (70, 2) pixel coords
    cam_t: np.ndarray                      # (3,)
    focal_length: float
    image_wh: tuple[int, int] | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        image_wh: tuple[int, int] | None = None,
    ) -> "_FusionInput":
        """Load an npz produced by ``predict_batch`` (see inference.py:667)."""
        path = Path(path)
        with np.load(path, allow_pickle=True) as data:
            raw = {k: data[k] for k in data.files}
        # Unwrap 0-d scalar arrays to Python scalars where useful.
        focal = raw["focal_length"]
        focal_value = float(focal.item()) if focal.shape == () else float(focal[0])
        return cls(
            V=np.asarray(raw["pred_vertices"]),
            J3d=np.asarray(raw["pred_keypoints_3d"]),
            J2d=np.asarray(raw["pred_keypoints_2d"]),
            cam_t=np.asarray(raw["pred_cam_t"]).astype(np.float64),
            focal_length=focal_value,
            image_wh=image_wh,
            raw=raw,
        )

    @classmethod
    def from_person_result(
        cls,
        person,  # noqa: ANN001 — PersonResult (avoid hard import to keep this module lightweight)
        image_wh: tuple[int, int] | None = None,
    ) -> "_FusionInput":
        return cls(
            V=np.asarray(person.pred_vertices),
            J3d=np.asarray(person.pred_keypoints_3d),
            J2d=np.asarray(person.pred_keypoints_2d),
            cam_t=np.asarray(person.pred_cam_t).astype(np.float64),
            focal_length=float(person.focal_length),
            image_wh=image_wh,
            raw=dict(getattr(person, "raw", {})),
        )


# ----- math primitives -------------------------------------------------------

def umeyama(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve ``min ||s·R·src + t - dst||²`` in closed form.

    Args:
        src: (N, 3) source points.
        dst: (N, 3) target points.
        with_scale: if False, force s=1 (rigid Procrustes).

    Returns:
        (s, R, t) minimizing the squared error above. R is a proper
        rotation (det = +1) — the reflection case is handled by flipping
        the inner diagonal.
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"umeyama expects matching (N, 3) inputs; got {src.shape} and {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError(f"umeyama needs at least 3 points, got {n}")

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d

    H = sc.T @ dc / n
    U, D, Vt = np.linalg.svd(H)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = Vt.T @ S @ U.T

    if with_scale:
        var_s = (sc ** 2).sum() / n
        if var_s < 1e-20:
            s = 1.0
        else:
            s = float(np.trace(np.diag(D) @ S) / var_s)
    else:
        s = 1.0

    t = mu_d - s * R @ mu_s
    return s, R, t


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals.

    Face normals are kept at their cross-product magnitude (proportional
    to 2*area) so larger triangles dominate at shared vertices; the
    per-vertex accumulator is normalized at the end.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    N = np.zeros_like(V)
    tri = V[F]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for i in range(3):
        np.add.at(N, F[:, i], face_n)
    norms = np.linalg.norm(N, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return N / norms


def vis_weight(
    V: np.ndarray, normals: np.ndarray, cam_pos: np.ndarray, eps: float = 1e-3
) -> np.ndarray:
    """Per-vertex visibility weight in [eps, 1+eps].

    Higher when the vertex normal points toward the camera. The ``eps``
    floor keeps back-facing vertices from zero-weighting the blend when
    the other view also has low confidence.
    """
    V = np.asarray(V, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    d = cam_pos[None, :] - V
    d_norm = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.where(d_norm < 1e-12, 1.0, d_norm)
    w = np.einsum("ij,ij->i", normals, d)
    return np.clip(w, 0.0, 1.0) + eps


# ----- orchestration ---------------------------------------------------------

def _resolve_joint_mask(
    drop_joints: Iterable[int] | None, keep_hands: bool = False
) -> np.ndarray:
    """Boolean mask over the 70 MHR joints selecting the solve set."""
    mask = np.zeros(NUM_JOINTS, dtype=bool)
    mask[list(BODY_JOINT_INDICES)] = True
    if keep_hands:
        mask[list(HAND_JOINT_INDICES)] = True
    if drop_joints is not None:
        for idx in drop_joints:
            if not 0 <= int(idx) < NUM_JOINTS:
                raise ValueError(f"drop_joints index out of range: {idx}")
            mask[int(idx)] = False
    return mask


def align_views(
    a: _FusionInput,
    b: _FusionInput,
    *,
    drop_joints: Iterable[int] | None = None,
    min_joints: int = 20,
    use_scale: bool = True,
    keep_hands: bool = False,
    bad_fit_threshold: float | None = None,
) -> AlignmentResult:
    """Align view B's joints into view A's camera frame.

    The solve operates on camera-origin-relative coordinates
    ``J = J3d + cam_t`` for each view, so the transform maps between
    camera frames — not between the mesh canonical frames. Returns an
    ``AlignmentResult``; sets ``flag = "bad_fit"`` if the median residual
    exceeds ``bad_fit_threshold`` (meters).
    """
    if a.J3d.shape != (NUM_JOINTS, 3) or b.J3d.shape != (NUM_JOINTS, 3):
        raise ValueError(
            f"expected (70, 3) joint arrays; got {a.J3d.shape}, {b.J3d.shape}"
        )

    J_A_full = a.J3d + a.cam_t[None, :]
    J_B_full = b.J3d + b.cam_t[None, :]

    mask = _resolve_joint_mask(drop_joints, keep_hands=keep_hands)
    if mask.sum() < min_joints:
        raise ValueError(
            f"only {int(mask.sum())} joints in solve set after drops; need >= {min_joints}"
        )

    J_A = J_A_full[mask]
    J_B = J_B_full[mask]

    s, R, t = umeyama(J_B, J_A, with_scale=use_scale)
    J_B_in_A = s * (J_B @ R.T) + t
    residuals = np.linalg.norm(J_B_in_A - J_A, axis=1)
    median_residual = float(np.median(residuals))

    flag = "ok"
    if bad_fit_threshold is not None and median_residual > bad_fit_threshold:
        flag = "bad_fit"

    return AlignmentResult(
        s=float(s),
        R=R,
        t=t,
        residuals=residuals,
        median_residual=median_residual,
        joint_mask=mask,
        J_A=J_A,
        J_B=J_B,
        J_B_in_A=J_B_in_A,
        flag=flag,
    )


def _transform_points(
    P: np.ndarray, s: float, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """``P' = s·P·Rᵀ + t`` for (N, 3) point arrays."""
    return s * (P @ R.T) + t


def fuse_two_views(
    a: _FusionInput,
    b: _FusionInput,
    faces: np.ndarray,
    *,
    drop_joints: Iterable[int] | None = None,
    min_joints: int = 20,
    use_scale: bool = True,
    keep_hands: bool = False,
    bad_fit_threshold: float | None = None,
) -> FusionResult:
    """Fuse two views into a single mesh in camera A's frame.

    Vertices are expressed camera-origin-relative
    (``V + cam_t``) before transforming B into A's frame, then converted
    back to A's ``V = V_world - cam_t_A`` convention so the output mesh
    plugs straight into view A's existing ``pred_cam_t``. Blending uses
    per-view visibility weights from surface normals.
    """
    if a.V.shape != b.V.shape:
        raise ValueError(f"mesh vertex counts differ: {a.V.shape} vs {b.V.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3); got {faces.shape}")

    alignment = align_views(
        a,
        b,
        drop_joints=drop_joints,
        min_joints=min_joints,
        use_scale=use_scale,
        keep_hands=keep_hands,
        bad_fit_threshold=bad_fit_threshold,
    )

    # A's mesh in A's camera frame (origin-relative).
    V_A_world = a.V + a.cam_t[None, :]
    # B's mesh in A's frame after alignment.
    V_B_world = b.V + b.cam_t[None, :]
    V_B_in_A = _transform_points(V_B_world, alignment.s, alignment.R, alignment.t)

    # Per-vertex normals in each view's frame-of-expression.
    N_A = vertex_normals(V_A_world, faces)
    # For B's verts now in A's frame, rotate normals (scale shouldn't
    # affect normal direction); for weighting purposes, recompute from
    # the transformed vertex array to get correct area weighting.
    N_B_in_A = vertex_normals(V_B_in_A, faces)

    # Camera positions in A's frame.
    cam_A_pos = np.zeros(3, dtype=np.float64)
    cam_B_pos = alignment.t  # where B's camera origin lands after alignment

    w_A = vis_weight(V_A_world, N_A, cam_A_pos)
    w_B = vis_weight(V_B_in_A, N_B_in_A, cam_B_pos)

    w_sum = w_A + w_B
    # w_sum is always >= 2*eps > 0, but guard regardless.
    w_sum = np.where(w_sum < 1e-12, 1.0, w_sum)
    w_A_norm = w_A / w_sum
    w_B_norm = w_B / w_sum

    V_fused_world = (
        w_A_norm[:, None] * V_A_world + w_B_norm[:, None] * V_B_in_A
    )
    # Return to V = V_world - cam_t_A convention so downstream consumers
    # can use view A's cam_t unchanged.
    V_fused = V_fused_world - a.cam_t[None, :]
    V_A_out = V_A_world - a.cam_t[None, :]
    V_B_in_A_out = V_B_in_A - a.cam_t[None, :]

    return FusionResult(
        V_fused=V_fused,
        faces=np.asarray(faces),
        w_A=w_A_norm,
        w_B=w_B_norm,
        alignment=alignment,
        V_A=V_A_out,
        V_B_in_A=V_B_in_A_out,
    )


def _project_camera_to_pixel(
    P_cam: np.ndarray, focal: float, image_wh: tuple[int, int] | None
) -> np.ndarray:
    """Pinhole project (N, 3) camera-frame points → (N, 2) pixel coords.

    Assumes image-center principal point. ``image_wh=None`` falls back
    to ``(cx, cy) = (0, 0)`` — relative pixel error is still meaningful
    for comparing joints to each other within the same view.
    """
    cx, cy = (image_wh[0] / 2.0, image_wh[1] / 2.0) if image_wh else (0.0, 0.0)
    z = P_cam[:, 2]
    z = np.where(np.abs(z) < 1e-8, 1e-8, z)
    u = focal * P_cam[:, 0] / z + cx
    v = focal * P_cam[:, 1] / z + cy
    return np.stack([u, v], axis=1)


def validate_alignment(
    a: _FusionInput, b: _FusionInput, alignment: AlignmentResult
) -> dict[str, Any]:
    """Reprojection and mirror sanity checks for an alignment result.

    Returns:
        dict with keys ``reproj_err_A``, ``reproj_err_B`` (both ``(K_used, 2)``),
        ``median_px_A``, ``median_px_B``, ``mirror_ok`` (bool),
        ``shoulder_dot`` (signed cosine between A's and B's shoulder vectors
        after alignment).
    """
    mask = alignment.joint_mask

    # View A: project A's 3D joints (in A's camera frame, camera-origin
    # relative) and compare against A's 2D detections.
    J_A_cam = a.J3d + a.cam_t[None, :]
    J_A_cam_masked = J_A_cam[mask]
    proj_A = _project_camera_to_pixel(J_A_cam_masked, a.focal_length, a.image_wh)
    reproj_err_A = proj_A - a.J2d[mask]

    # View B: invert the alignment (A→B: p_B = (p_A - t) @ R / s) and
    # project into B's image via B's focal.
    inv = (alignment.J_B_in_A - alignment.t) @ alignment.R / alignment.s
    # Sanity: inv should equal alignment.J_B to numerical precision.
    proj_B = _project_camera_to_pixel(alignment.J_B, b.focal_length, b.image_wh)
    reproj_err_B = proj_B - b.J2d[mask]
    # Suppress "inv" unused-warning — it's the algebraic check that the
    # solve is self-consistent; compute its residual and expose it.
    inv_residual = float(np.linalg.norm(inv - alignment.J_B, axis=1).max())

    # Mirror check: shoulder vector (joint 5 left_shoulder → joint 6 right_shoulder).
    # Joints 5 and 6 are always in BODY_JOINT_INDICES, so they're in the solve set.
    idx_full_to_used = np.where(mask)[0]
    try:
        i5 = int(np.where(idx_full_to_used == 5)[0][0])
        i6 = int(np.where(idx_full_to_used == 6)[0][0])
        v_A = alignment.J_A[i5] - alignment.J_A[i6]
        v_B = alignment.J_B_in_A[i5] - alignment.J_B_in_A[i6]
        denom = np.linalg.norm(v_A) * np.linalg.norm(v_B)
        shoulder_dot = float(v_A @ v_B / denom) if denom > 1e-8 else 0.0
    except IndexError:
        shoulder_dot = 0.0
    mirror_ok = shoulder_dot > 0.0

    return {
        "reproj_err_A": reproj_err_A,
        "reproj_err_B": reproj_err_B,
        "median_px_A": float(np.median(np.linalg.norm(reproj_err_A, axis=1))),
        "median_px_B": float(np.median(np.linalg.norm(reproj_err_B, axis=1))),
        "mirror_ok": mirror_ok,
        "shoulder_dot": shoulder_dot,
        "inverse_residual": inv_residual,
    }


# ----- I/O -------------------------------------------------------------------

_FACES_CACHE = Path.home() / ".cache" / "sam3d_wrapper" / "faces.npy"


def load_faces(path: Path | None = None) -> np.ndarray:
    """Return the MHR mesh face indices.

    Resolution order:
      1. ``path`` argument, if given.
      2. ``~/.cache/sam3d_wrapper/faces.npy``, if it exists.
      3. Fallback: instantiate ``Sam3DBody`` once, read ``model.faces``,
         and write it to the cache for subsequent calls. This path is
         slow (loads the full detector+model pipeline).
    """
    if path is not None:
        return np.load(Path(path))
    if _FACES_CACHE.exists():
        return np.load(_FACES_CACHE)

    from sam3d_wrapper.inference import Sam3DBody

    model = Sam3DBody()
    faces = np.asarray(model.faces)
    _FACES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(_FACES_CACHE, faces)
    return faces


def _match_dtype(array: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Cast ``array`` to match ``template``'s dtype when safe."""
    if template.dtype == array.dtype:
        return array
    return array.astype(template.dtype)


def save_fused_person_npz(
    res: FusionResult, raw_a: dict[str, Any], path: Path
) -> None:
    """Write a drop-in replacement for view A's predict_batch npz.

    Every key from ``raw_a`` passes through with its original shape and
    dtype. ``pred_vertices`` is replaced with the fused mesh, cast to
    A's original vertex dtype so downstream consumers see identical
    layout.
    """
    out = dict(raw_a)
    original_verts = raw_a["pred_vertices"]
    if res.V_fused.shape != original_verts.shape:
        raise ValueError(
            f"fused vertex shape {res.V_fused.shape} != A's {original_verts.shape}"
        )
    out["pred_vertices"] = _match_dtype(res.V_fused, original_verts)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **out)


def save_fusion_alignment_npz(
    res: FusionResult, raw_b: dict[str, Any], path: Path
) -> None:
    """Write alignment diagnostics + B's originals to a sidecar npz."""
    out: dict[str, Any] = {
        "s": np.float64(res.alignment.s),
        "R": res.alignment.R.astype(np.float64),
        "t": res.alignment.t.astype(np.float64),
        "residuals": res.alignment.residuals.astype(np.float64),
        "median_residual": np.float64(res.alignment.median_residual),
        "joint_mask": res.alignment.joint_mask.astype(bool),
        "flag": np.array(res.alignment.flag),
        "w_A": res.w_A.astype(np.float64),
        "w_B": res.w_B.astype(np.float64),
        "V_A": res.V_A.astype(np.float32),
        "V_B_in_A": res.V_B_in_A.astype(np.float32),
        # B's originals for traceability.
        "b_pred_vertices": np.asarray(raw_b["pred_vertices"]),
        "b_pred_keypoints_3d": np.asarray(raw_b["pred_keypoints_3d"]),
        "b_pred_cam_t": np.asarray(raw_b["pred_cam_t"]),
        "b_focal_length": np.asarray(raw_b["focal_length"]),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **out)


def save_fusion_ply(res: FusionResult, path: Path) -> None:
    import trimesh

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(res.V_fused, dtype=np.float64),
        faces=np.asarray(res.faces, dtype=np.int64),
        process=False,
    )
    mesh.export(str(path))


# ----- debug -----------------------------------------------------------------

def save_residual_histogram(res: FusionResult, path: Path) -> None:
    """Write a histogram of per-joint post-alignment residuals (meters)."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    residuals = res.alignment.residuals
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=min(30, max(5, len(residuals) // 2)), color="#4477aa", edgecolor="black")
    ax.axvline(
        res.alignment.median_residual,
        color="red", linestyle="--",
        label=f"median = {res.alignment.median_residual * 1000:.1f} mm",
    )
    ax.set_xlabel("Per-joint residual (m)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Alignment residuals — s={res.alignment.s:.4f}, |t|={np.linalg.norm(res.alignment.t):.3f} m"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)


def save_joint_overlay(
    fi: _FusionInput,
    image_path: Path,
    out_path: Path,
    joints_3d_override: np.ndarray | None = None,
) -> None:
    """Draw 2D-detected and reprojected joints on top of the source image.

    Green dots = ``fi.J2d`` (detector output).
    Red dots   = pinhole projection of ``fi.J3d + fi.cam_t``.
    """
    import cv2

    image_path = Path(image_path)
    out_path = Path(out_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    h, w = img.shape[:2]
    wh = fi.image_wh if fi.image_wh is not None else (w, h)

    J3d = fi.J3d if joints_3d_override is None else joints_3d_override
    P_cam = J3d + fi.cam_t[None, :]
    proj = _project_camera_to_pixel(P_cam, fi.focal_length, wh)

    for (u, v) in fi.J2d:
        if np.isfinite(u) and np.isfinite(v):
            cv2.circle(img, (int(round(u)), int(round(v))), 3, (0, 255, 0), -1)
    for (u, v) in proj:
        if np.isfinite(u) and np.isfinite(v):
            cv2.circle(img, (int(round(u)), int(round(v))), 3, (0, 0, 255), 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# ----- CLI -------------------------------------------------------------------

def _parse_drop_joints(arg: str | None) -> list[int] | None:
    if arg is None or arg.strip() == "":
        return None
    out: list[int] = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _parse_image_wh(image_path: str | None) -> tuple[int, int] | None:
    if image_path is None:
        return None
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    return (w, h)


def _rotation_angle_degrees(R: np.ndarray) -> float:
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def cli_fuse(argv: list[str] | None = None) -> int:
    """CLI entry point: sam3d-fuse."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Fuse two SAM 3D Body views of the same person into a single "
            "mesh in view A's camera frame. The fused .npz mirrors view "
            "A's predict_batch schema (drop-in replacement)."
        )
    )
    parser.add_argument("--npz-a", required=True, help="Path to view A's npz (from predict_batch)")
    parser.add_argument("--npz-b", required=True, help="Path to view B's npz")
    parser.add_argument("--out-dir", required=True, help="Output directory for fused artifacts")
    parser.add_argument("--faces", default=None, help="Path to faces .npy (defaults to auto-cache)")
    parser.add_argument(
        "--keep-hands", action="store_true",
        help="Include MHR70 hand joints (21-62) in the solve. Default: body-only.",
    )
    parser.add_argument(
        "--drop-joints", default=None,
        help="Comma-separated joint indices to drop from the solve (e.g. '21,22,41').",
    )
    parser.add_argument(
        "--min-joints", type=int, default=20,
        help="Minimum joints required after masking (default: 20).",
    )
    parser.add_argument(
        "--no-scale", action="store_true",
        help="Force s=1 (use when both focal lengths are known and equal).",
    )
    parser.add_argument(
        "--bad-fit-threshold", type=float, default=None,
        help="Median-residual threshold (meters) above which alignment is flagged.",
    )
    parser.add_argument("--image-a", default=None, help="View A's source image (enables overlay).")
    parser.add_argument("--image-b", default=None, help="View B's source image (enables overlay).")
    parser.add_argument("--debug-dir", default=None, help="Directory for residual histogram + overlays.")
    parser.add_argument("--no-ply", action="store_true", help="Skip writing fused.ply.")

    args = parser.parse_args(argv)

    wh_a = _parse_image_wh(args.image_a)
    wh_b = _parse_image_wh(args.image_b)

    fi_a = _FusionInput.from_npz(args.npz_a, image_wh=wh_a)
    fi_b = _FusionInput.from_npz(args.npz_b, image_wh=wh_b)

    faces_path = Path(args.faces) if args.faces else None
    faces = load_faces(faces_path)

    drop = _parse_drop_joints(args.drop_joints)

    res = fuse_two_views(
        fi_a,
        fi_b,
        faces,
        drop_joints=drop,
        min_joints=args.min_joints,
        use_scale=not args.no_scale,
        keep_hands=args.keep_hands,
        bad_fit_threshold=args.bad_fit_threshold,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_fused_person_npz(res, fi_a.raw, out_dir / "fused.npz")
    save_fusion_alignment_npz(res, fi_b.raw, out_dir / "fused.alignment.npz")
    if not args.no_ply:
        save_fusion_ply(res, out_dir / "fused.ply")

    diag = validate_alignment(fi_a, fi_b, res.alignment)

    if args.debug_dir:
        dbg_dir = Path(args.debug_dir)
        save_residual_histogram(res, dbg_dir / "residuals.png")
        if args.image_a:
            save_joint_overlay(fi_a, Path(args.image_a), dbg_dir / "overlay_A.png")
        if args.image_b:
            save_joint_overlay(fi_b, Path(args.image_b), dbg_dir / "overlay_B.png")

    print(
        f"Fusion complete: s={res.alignment.s:.4f} "
        f"|t|={np.linalg.norm(res.alignment.t):.4f} m "
        f"R_angle={_rotation_angle_degrees(res.alignment.R):.2f}° "
        f"median_res={res.alignment.median_residual * 1000:.2f} mm "
        f"joints_used={int(res.alignment.joint_mask.sum())} "
        f"mirror_ok={diag['mirror_ok']} "
        f"flag={res.alignment.flag}"
    )

    return 1 if res.alignment.flag == "bad_fit" else 0

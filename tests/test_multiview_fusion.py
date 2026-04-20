"""Unit tests for sam3d_wrapper.multiview_fusion."""

from __future__ import annotations

import numpy as np
import pytest

from sam3d_wrapper import multiview_fusion as mvf


# ----- module smoke ---------------------------------------------------------


def test_module_imports() -> None:
    assert hasattr(mvf, "umeyama")
    assert hasattr(mvf, "vertex_normals")
    assert hasattr(mvf, "vis_weight")
    assert hasattr(mvf, "align_views")
    assert hasattr(mvf, "fuse_two_views")
    assert hasattr(mvf, "cli_fuse")


def test_joint_index_partition_is_complete() -> None:
    body = set(mvf.BODY_JOINT_INDICES)
    hand = set(mvf.HAND_JOINT_INDICES)
    assert body.isdisjoint(hand)
    assert body | hand == set(range(mvf.NUM_JOINTS))


# ----- umeyama --------------------------------------------------------------


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    # Random rotation via QR decomposition of a random matrix.
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


@pytest.mark.parametrize("with_scale, scale", [(True, 1.7), (True, 0.4), (False, 1.0)])
def test_umeyama_recovers_known_transform(with_scale: bool, scale: float) -> None:
    rng = np.random.default_rng(42)
    R_true = _random_rotation(rng)
    t_true = rng.standard_normal(3) * 5.0
    src = rng.standard_normal((70, 3)) * 2.0
    dst = scale * (src @ R_true.T) + t_true

    s, R, t = mvf.umeyama(src, dst, with_scale=with_scale)

    expected_s = scale if with_scale else 1.0
    assert np.isclose(s, expected_s, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(R, R_true, atol=1e-8)
    np.testing.assert_allclose(t, t_true, atol=1e-7)


def test_umeyama_returns_proper_rotation_on_reflection() -> None:
    # Reflect along x; umeyama should produce a proper rotation (det = +1),
    # not silently return a reflection.
    rng = np.random.default_rng(7)
    src = rng.standard_normal((70, 3))
    dst = src.copy()
    dst[:, 0] *= -1

    _, R, _ = mvf.umeyama(src, dst, with_scale=True)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)


def test_umeyama_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        mvf.umeyama(np.zeros((10, 2)), np.zeros((10, 2)))
    with pytest.raises(ValueError):
        mvf.umeyama(np.zeros((10, 3)), np.zeros((11, 3)))
    with pytest.raises(ValueError):
        mvf.umeyama(np.zeros((2, 3)), np.zeros((2, 3)))


# ----- vertex_normals -------------------------------------------------------


def _unit_cube() -> tuple[np.ndarray, np.ndarray]:
    V = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # z = 0 (bottom)
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # z = 1 (top)
        ],
        dtype=np.float64,
    )
    # Outward-facing winding (CCW when viewed from outside).
    F = np.array(
        [
            [0, 3, 2], [0, 2, 1],  # bottom (-z)
            [4, 5, 6], [4, 6, 7],  # top    (+z)
            [0, 1, 5], [0, 5, 4],  # -y
            [1, 2, 6], [1, 6, 5],  # +x
            [2, 3, 7], [2, 7, 6],  # +y
            [3, 0, 4], [3, 4, 7],  # -x
        ],
        dtype=np.int64,
    )
    return V, F


def test_vertex_normals_unit_cube_corners_point_outward() -> None:
    V, F = _unit_cube()
    N = mvf.vertex_normals(V, F)

    assert N.shape == V.shape
    # All normals unit-length.
    np.testing.assert_allclose(np.linalg.norm(N, axis=1), 1.0, atol=1e-10)
    # At each corner, the normal should point along the outward diagonal
    # (equal contributions from three adjacent faces).
    center = V.mean(axis=0)
    for i in range(8):
        outward = V[i] - center
        outward /= np.linalg.norm(outward)
        # Dot with normal should be positive and close to sqrt(1/3) * 3 = 1.
        assert N[i] @ outward > 0.9


# ----- vis_weight -----------------------------------------------------------


def test_vis_weight_front_vs_back() -> None:
    # Two vertices at origin, with opposite normals. Camera at +Z.
    V = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
    N = np.array([[0, 0, 1], [0, 0, -1]], dtype=np.float64)
    cam = np.array([0, 0, 10], dtype=np.float64)

    w = mvf.vis_weight(V, N, cam, eps=0.0)

    # Normal toward camera → weight 1.
    assert np.isclose(w[0], 1.0, atol=1e-10)
    # Normal away from camera → weight clipped to 0.
    assert np.isclose(w[1], 0.0, atol=1e-10)


def test_vis_weight_epsilon_floor() -> None:
    V = np.array([[0, 0, 0]], dtype=np.float64)
    N = np.array([[0, 0, -1]], dtype=np.float64)  # back-facing
    cam = np.array([0, 0, 10], dtype=np.float64)
    w = mvf.vis_weight(V, N, cam, eps=1e-3)
    assert np.isclose(w[0], 1e-3, atol=1e-12)


# ----- align_views ----------------------------------------------------------


def _make_fusion_input(
    V: np.ndarray, J3d: np.ndarray, cam_t: np.ndarray
) -> mvf._FusionInput:
    return mvf._FusionInput(
        V=V,
        J3d=J3d,
        J2d=np.zeros((mvf.NUM_JOINTS, 2), dtype=np.float32),
        cam_t=cam_t,
        focal_length=1000.0,
    )


def test_align_views_recovers_known_transform() -> None:
    rng = np.random.default_rng(123)
    # Ground-truth joints in world (A's camera) frame.
    joints_world = rng.standard_normal((mvf.NUM_JOINTS, 3)) * 0.5

    # A sees the joints directly: J3d_A + cam_t_A == joints_world.
    cam_t_A = np.array([0.1, -0.2, 2.0])
    J3d_A = joints_world - cam_t_A[None, :]

    # B's camera differs by a known rigid transform. Joints in B's camera
    # frame are: R @ (joints_world - cam_center_B), so in the
    # "J3d + cam_t" convention, cam_t_B plays the role of the negated
    # camera center in the rotated frame. We build J3d_B/cam_t_B so that
    # J3d_B + cam_t_B = R @ joints_world + t_offset.
    R_gt = _random_rotation(rng)
    t_offset = np.array([0.3, 0.5, -0.4])
    joints_B_world = (joints_world @ R_gt.T) + t_offset
    cam_t_B = np.array([-0.4, 0.2, 1.5])
    J3d_B = joints_B_world - cam_t_B[None, :]

    a = _make_fusion_input(np.zeros((4, 3)), J3d_A, cam_t_A)
    b = _make_fusion_input(np.zeros((4, 3)), J3d_B, cam_t_B)

    # use_scale=False (focal lengths are "known and equal" in this test).
    result = mvf.align_views(a, b, use_scale=False, min_joints=10)

    # We want (s, R, t) such that s·R·J_B_full + t ≈ J_A_full, with
    # J_X_full = J3d_X + cam_t_X. In our construction J_B_full = R_gt · J_A_full + t_offset,
    # so s=1, R_solve = R_gt.T, t_solve = -R_gt.T · t_offset.
    np.testing.assert_allclose(result.s, 1.0, atol=1e-10)
    np.testing.assert_allclose(result.R, R_gt.T, atol=1e-8)
    np.testing.assert_allclose(result.t, -R_gt.T @ t_offset, atol=1e-8)
    assert result.median_residual < 1e-8
    assert result.flag == "ok"


def test_align_views_respects_min_joints() -> None:
    # Drop all body joints — below the min_joints floor.
    rng = np.random.default_rng(0)
    J = rng.standard_normal((mvf.NUM_JOINTS, 3))
    a = _make_fusion_input(np.zeros((4, 3)), J, np.zeros(3))
    b = _make_fusion_input(np.zeros((4, 3)), J.copy(), np.zeros(3))
    with pytest.raises(ValueError, match="joints in solve set"):
        mvf.align_views(
            a, b, drop_joints=mvf.BODY_JOINT_INDICES, min_joints=20
        )


def test_align_views_flags_bad_fit() -> None:
    rng = np.random.default_rng(0)
    J = rng.standard_normal((mvf.NUM_JOINTS, 3))
    J_noisy = J + rng.standard_normal(J.shape) * 0.5
    a = _make_fusion_input(np.zeros((4, 3)), J, np.zeros(3))
    b = _make_fusion_input(np.zeros((4, 3)), J_noisy, np.zeros(3))
    result = mvf.align_views(a, b, use_scale=False, bad_fit_threshold=0.01)
    assert result.flag == "bad_fit"


# ----- fuse_two_views -------------------------------------------------------


def test_fuse_two_views_identity() -> None:
    # Same mesh + same joints in both views → fused should equal input.
    V, F = _unit_cube()
    rng = np.random.default_rng(5)
    J = rng.standard_normal((mvf.NUM_JOINTS, 3))
    cam_t = np.array([0.0, 0.0, 2.5])

    a = _make_fusion_input(V.copy(), J.copy(), cam_t)
    b = _make_fusion_input(V.copy(), J.copy(), cam_t)

    res = mvf.fuse_two_views(a, b, F, min_joints=10, use_scale=False)

    np.testing.assert_allclose(res.alignment.R, np.eye(3), atol=1e-8)
    np.testing.assert_allclose(res.alignment.t, np.zeros(3), atol=1e-8)
    np.testing.assert_allclose(res.V_fused, V, atol=1e-9)
    # Weights should sum to 1 per vertex.
    np.testing.assert_allclose(res.w_A + res.w_B, 1.0, atol=1e-12)


def test_fuse_two_views_recovers_under_known_camera_transform() -> None:
    # B is the same scene as A, observed by a camera with a known transform.
    # After alignment, fused mesh in A's frame should equal A's input mesh.
    rng = np.random.default_rng(11)
    V, F = _unit_cube()
    J_world = rng.standard_normal((mvf.NUM_JOINTS, 3)) * 0.5

    cam_t_A = np.array([0.0, 0.0, 3.0])
    V_A_local = V - cam_t_A  # A's mesh in A-local (pre cam_t) coords
    J_A_local = J_world - cam_t_A

    # Apply a rigid transform to "move" the camera.
    R_cam = _random_rotation(rng)
    t_cam = np.array([0.2, -0.1, 0.3])
    V_world = V  # mesh lives in A's camera frame == "world" here
    V_B_world = (V_world @ R_cam.T) + t_cam
    J_B_world = (J_world @ R_cam.T) + t_cam

    cam_t_B = np.array([-0.3, 0.2, 2.5])
    V_B_local = V_B_world - cam_t_B
    J_B_local = J_B_world - cam_t_B

    a = _make_fusion_input(V_A_local, J_A_local, cam_t_A)
    b = _make_fusion_input(V_B_local, J_B_local, cam_t_B)

    res = mvf.fuse_two_views(a, b, F, use_scale=False, min_joints=10)

    assert res.alignment.median_residual < 1e-8
    # V_fused is expressed in A's (V = V_world - cam_t_A) convention.
    np.testing.assert_allclose(res.V_fused + cam_t_A, V_world, atol=1e-7)


def test_fuse_two_views_rejects_mismatched_meshes() -> None:
    V_a = np.zeros((10, 3))
    V_b = np.zeros((9, 3))
    J = np.zeros((mvf.NUM_JOINTS, 3))
    a = _make_fusion_input(V_a, J, np.zeros(3))
    b = _make_fusion_input(V_b, J.copy(), np.zeros(3))
    F = np.zeros((1, 3), dtype=np.int64)
    with pytest.raises(ValueError, match="mesh vertex counts differ"):
        mvf.fuse_two_views(a, b, F, min_joints=10)


# ----- I/O ------------------------------------------------------------------


REAL_NPZ = "/home/derek/sam3d_wrapper/mcilroy_swing/mhr/raw/frame_0001.npz"


def _real_npz_available() -> bool:
    from pathlib import Path

    return Path(REAL_NPZ).exists()


@pytest.mark.skipif(not _real_npz_available(), reason="real npz not available")
def test_from_npz_loads_real_predict_batch_output() -> None:
    fi = mvf._FusionInput.from_npz(REAL_NPZ)
    assert fi.V.shape == (18439, 3)
    assert fi.J3d.shape == (mvf.NUM_JOINTS, 3)
    assert fi.J2d.shape == (mvf.NUM_JOINTS, 2)
    assert fi.cam_t.shape == (3,)
    assert np.isfinite(fi.focal_length)
    # raw dict must hold the full npz content.
    assert "pred_vertices" in fi.raw
    assert "mhr_model_params" in fi.raw


@pytest.mark.skipif(not _real_npz_available(), reason="real npz not available")
def test_fused_npz_schema_matches_predict_batch(tmp_path) -> None:
    # Guards the downstream-compatibility contract: fused.npz must be a
    # drop-in replacement for predict_batch output.
    fi_a = mvf._FusionInput.from_npz(REAL_NPZ)
    fi_b = mvf._FusionInput.from_npz(REAL_NPZ)

    # Fuse the frame with itself — arbitrary "fake" faces are fine since
    # fused output == input when inputs are identical.
    F_fake = np.array([[0, 1, 2]], dtype=np.int64)
    # Provide matching-sized fake faces so vertex_normals has *some* input
    # — use a minimal mesh; the geometry fidelity is tested elsewhere.
    F = np.stack(
        [
            np.arange(0, fi_a.V.shape[0] - 2),
            np.arange(1, fi_a.V.shape[0] - 1),
            np.arange(2, fi_a.V.shape[0]),
        ],
        axis=1,
    )

    res = mvf.fuse_two_views(fi_a, fi_b, F, min_joints=10, use_scale=False)

    out_path = tmp_path / "fused.npz"
    mvf.save_fused_person_npz(res, fi_a.raw, out_path)

    with np.load(REAL_NPZ, allow_pickle=True) as orig, np.load(
        out_path, allow_pickle=True
    ) as fused:
        assert set(orig.files) == set(fused.files), (
            f"fused npz key set differs: "
            f"missing={set(orig.files) - set(fused.files)}, "
            f"extra={set(fused.files) - set(orig.files)}"
        )
        for k in orig.files:
            o = orig[k]
            f = fused[k]
            assert o.shape == f.shape, f"shape mismatch on {k}: {o.shape} vs {f.shape}"
            assert o.dtype == f.dtype, f"dtype mismatch on {k}: {o.dtype} vs {f.dtype}"
        # Fused pred_vertices should equal res.V_fused (in A's original dtype).
        np.testing.assert_allclose(
            fused["pred_vertices"], res.V_fused.astype(fused["pred_vertices"].dtype)
        )


@pytest.mark.skipif(not _real_npz_available(), reason="real npz not available")
def test_alignment_sidecar_writes_expected_keys(tmp_path) -> None:
    fi_a = mvf._FusionInput.from_npz(REAL_NPZ)
    fi_b = mvf._FusionInput.from_npz(REAL_NPZ)
    F = np.stack(
        [
            np.arange(0, fi_a.V.shape[0] - 2),
            np.arange(1, fi_a.V.shape[0] - 1),
            np.arange(2, fi_a.V.shape[0]),
        ],
        axis=1,
    )
    res = mvf.fuse_two_views(fi_a, fi_b, F, min_joints=10, use_scale=False)

    out = tmp_path / "fused.alignment.npz"
    mvf.save_fusion_alignment_npz(res, fi_b.raw, out)

    with np.load(out, allow_pickle=True) as data:
        for k in [
            "s", "R", "t", "residuals", "median_residual",
            "joint_mask", "flag", "w_A", "w_B", "V_A", "V_B_in_A",
            "b_pred_vertices", "b_pred_keypoints_3d", "b_pred_cam_t",
            "b_focal_length",
        ]:
            assert k in data.files, f"missing sidecar key: {k}"


# ----- validate_alignment ---------------------------------------------------


def test_validate_alignment_identity_has_near_zero_residuals() -> None:
    rng = np.random.default_rng(3)
    V, F = _unit_cube()
    J = rng.standard_normal((mvf.NUM_JOINTS, 3)) * 0.3
    cam_t = np.array([0.0, 0.0, 2.5])

    # Build a consistent J2d from the 3D joints via the same projection.
    def project(J3d, cam_t, focal, wh):
        P_cam = J3d + cam_t[None, :]
        cx, cy = wh[0] / 2, wh[1] / 2
        z = P_cam[:, 2]
        return np.stack(
            [focal * P_cam[:, 0] / z + cx, focal * P_cam[:, 1] / z + cy], axis=1
        )

    focal = 1000.0
    wh = (1024, 768)
    J2d = project(J, cam_t, focal, wh)

    a = mvf._FusionInput(V=V, J3d=J, J2d=J2d, cam_t=cam_t, focal_length=focal, image_wh=wh)
    b = mvf._FusionInput(V=V, J3d=J.copy(), J2d=J2d.copy(), cam_t=cam_t, focal_length=focal, image_wh=wh)

    alignment = mvf.align_views(a, b, min_joints=10, use_scale=False)
    diag = mvf.validate_alignment(a, b, alignment)

    assert diag["median_px_A"] < 1e-6
    assert diag["median_px_B"] < 1e-6
    assert diag["mirror_ok"] is True
    assert diag["shoulder_dot"] > 0.99 or np.isnan(diag["shoulder_dot"])
    assert diag["inverse_residual"] < 1e-8


def _make_fusion_result() -> mvf.FusionResult:
    V, F = _unit_cube()
    return mvf.FusionResult(
        V_fused=V,
        faces=F,
        w_A=np.ones(8) * 0.5,
        w_B=np.ones(8) * 0.5,
        alignment=mvf.AlignmentResult(
            s=1.0, R=np.eye(3), t=np.zeros(3),
            residuals=np.array([0.001, 0.002, 0.004, 0.003, 0.001, 0.0025] * 5),
            median_residual=0.0025,
            joint_mask=np.ones(mvf.NUM_JOINTS, dtype=bool),
            J_A=np.zeros((10, 3)), J_B=np.zeros((10, 3)),
            J_B_in_A=np.zeros((10, 3)),
        ),
        V_A=V, V_B_in_A=V,
    )


def test_save_fusion_ply_writes_readable_mesh(tmp_path) -> None:
    import trimesh

    V, F = _unit_cube()
    res = _make_fusion_result()
    out = tmp_path / "fused.ply"
    mvf.save_fusion_ply(res, out)
    assert out.exists() and out.stat().st_size > 0

    mesh = trimesh.load(str(out), process=False)
    assert mesh.vertices.shape == V.shape
    assert mesh.faces.shape == F.shape


def test_save_residual_histogram_writes_file(tmp_path) -> None:
    res = _make_fusion_result()
    out = tmp_path / "residuals.png"
    mvf.save_residual_histogram(res, out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.skipif(not _real_npz_available(), reason="real npz not available")
def test_cli_fuse_end_to_end_smoke(tmp_path) -> None:
    # Build a synthetic but topology-valid faces array. The CLI doesn't
    # care about the true MHR topology for plumbing verification; this
    # exercises every code path without loading the full model.
    V_count = 18439
    # Fan triangulation: a valid 2-manifold-ish strip referencing all vertices.
    faces = np.stack(
        [
            np.arange(0, V_count - 2),
            np.arange(1, V_count - 1),
            np.arange(2, V_count),
        ],
        axis=1,
    ).astype(np.int64)
    faces_path = tmp_path / "faces.npy"
    np.save(faces_path, faces)

    out_dir = tmp_path / "fused_smoke"
    debug_dir = tmp_path / "debug"

    exit_code = mvf.cli_fuse(
        [
            "--npz-a", REAL_NPZ,
            "--npz-b", REAL_NPZ,
            "--out-dir", str(out_dir),
            "--faces", str(faces_path),
            "--no-scale",
            "--min-joints", "10",
            "--debug-dir", str(debug_dir),
        ]
    )
    assert exit_code == 0
    assert (out_dir / "fused.npz").exists()
    assert (out_dir / "fused.alignment.npz").exists()
    assert (out_dir / "fused.ply").exists()
    assert (debug_dir / "residuals.png").exists()

    # Drop-in compatibility: fused.npz reads back with the same schema
    # as view A's original.
    with np.load(REAL_NPZ, allow_pickle=True) as orig, np.load(
        out_dir / "fused.npz", allow_pickle=True
    ) as fused:
        assert set(orig.files) == set(fused.files)


def test_save_joint_overlay_writes_file(tmp_path) -> None:
    import cv2

    img_path = tmp_path / "input.jpg"
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    rng = np.random.default_rng(2)
    fi = mvf._FusionInput(
        V=np.zeros((4, 3)),
        J3d=rng.standard_normal((mvf.NUM_JOINTS, 3)) * 0.2,
        J2d=rng.uniform(0, 640, size=(mvf.NUM_JOINTS, 2)),
        cam_t=np.array([0.0, 0.0, 2.0]),
        focal_length=800.0,
        image_wh=(640, 480),
    )
    out = tmp_path / "overlay.jpg"
    mvf.save_joint_overlay(fi, img_path, out)
    assert out.exists() and out.stat().st_size > 0

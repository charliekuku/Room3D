import math

import numpy as np
import trimesh

from src.detection.model_fitting import _rotation_y, fit_uploaded_glb


def _asymmetric_mesh():
    body = trimesh.creation.box(extents=[1.0, 1.6, 0.5])
    arm = trimesh.creation.box(extents=[0.35, 0.4, 0.9])
    arm.apply_translation([0.3, 0.6, 0.2])
    return trimesh.util.concatenate([body, arm])


def test_point_guided_fit_recovers_scale_and_asset_yaw():
    mesh = _asymmetric_mesh()
    lo, hi = mesh.bounds
    offset = np.array([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    vertices = np.asarray(mesh.vertices) + offset
    target_size = np.ptp(_rotation_y(vertices * 1.2, math.pi / 2), axis=0)
    points, _ = trimesh.sample.sample_surface(mesh, 384, seed=11)
    points = _rotation_y((points + offset) * 1.2, math.pi / 2)

    result = fit_uploaded_glb(
        mesh.export(file_type="glb"), target_size.tolist(), points.tolist(),
        (target_size / (hi - lo)).tolist(), offset.tolist(),
    )

    assert result["fit"]["method"] == "hybrid-point-cloud-v1"
    assert abs(abs(result["model_yaw"]) - math.pi / 2) < 1e-6
    assert np.allclose(result["model_scale"], [1.2, 1.2, 1.2], atol=1e-4)
    assert result["fit"]["point_agreement"] > 0.8


def test_point_guided_fit_falls_back_when_scan_support_is_too_sparse():
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    result = fit_uploaded_glb(
        mesh.export(file_type="glb"), [2, 2, 2], [[0, 0, 0]] * 10,
        [2, 2, 2], [0, 0.5, 0],
    )

    assert result["fit"]["method"] == "bounding-box-fallback"
    assert result["fit"]["reason"] == "insufficient-scan-points"
    assert result["model_scale"] == [2.0, 2.0, 2.0]

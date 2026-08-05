"""Constrained fitting of an uploaded GLB to an object's scan support.

The scene editor's original replacement path stretched every GLB axis to the
detected box.  That guarantees matching bounds, but it can badly distort a
model and cannot correct a 90-degree (or otherwise rotated) asset.  This module
uses the object's small, local point sample to choose scale, yaw and a modest
horizontal translation while keeping the fit inside the detected placement.
"""
from __future__ import annotations

import io
import math

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _rotation_y(points: np.ndarray, yaw: float) -> np.ndarray:
    """Apply the same +Y rotation convention as THREE.Object3D.rotation.y."""
    c, s = math.cos(yaw), math.sin(yaw)
    result = np.empty_like(points)
    result[:, 0] = c * points[:, 0] + s * points[:, 2]
    result[:, 1] = points[:, 1]
    result[:, 2] = -s * points[:, 0] + c * points[:, 2]
    return result


def _fallback(scale: list[float], offset: list[float], reason: str) -> dict:
    return {
        "model_scale": [float(v) for v in scale],
        "model_offset": [float(v) for v in offset],
        "model_yaw": 0.0,
        "model_translation": [0.0, 0.0, 0.0],
        "fit": {
            "method": "bounding-box-fallback",
            "reason": reason,
        },
    }


def fit_uploaded_glb(
    glb_bytes: bytes,
    target_size: list[float],
    fit_points: list[list[float]] | None,
    fallback_scale: list[float],
    fallback_offset: list[float],
    *,
    max_axis_stretch: float = 1.35,
    surface_samples: int = 1400,
) -> dict:
    """Find a conservative transform for an uploaded model.

    ``fit_points`` are expected in the object's bottom-centred, yaw-removed
    local frame.  The returned transform is applied as
    ``translation + rotation_y(yaw) * scale * (vertex + offset)``.

    A one-sided scan-to-mesh distance is intentional: a scan only observes part
    of an object, while an uploaded asset is normally complete.  Bounding-box,
    overflow, anisotropy and translation penalties keep that partial evidence
    from pulling an arbitrary part of the model onto the scan.
    """
    target = np.asarray(target_size, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)) or np.any(target <= 1e-6):
        return _fallback(fallback_scale, fallback_offset, "invalid-target-size")

    try:
        loaded = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", process=False)
        mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
        if mesh is None or len(getattr(mesh, "faces", [])) == 0:
            return _fallback(fallback_scale, fallback_offset, "no-mesh-geometry")
        bounds = np.asarray(mesh.bounds, dtype=float)
        extents = bounds[1] - bounds[0]
        if not np.all(np.isfinite(extents)) or np.any(extents <= 1e-6):
            return _fallback(fallback_scale, fallback_offset, "degenerate-mesh-bounds")
        sampled, _ = trimesh.sample.sample_surface(
            mesh, min(max(int(surface_samples), 300), 3000), seed=0,
        )
    except Exception:
        return _fallback(fallback_scale, fallback_offset, "glb-load-failed")

    points = np.asarray(fit_points if fit_points is not None else [], dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,):
        return _fallback(fallback_scale, fallback_offset, "no-scan-points")
    points = points[np.isfinite(points).all(axis=1)]
    # Reject points well outside the placement. They are normally background or
    # supporting-furniture leakage from an imperfect segmentation mask.
    pad = np.array([0.15, 0.10, 0.15]) * target
    inside = (
        (np.abs(points[:, 0]) <= target[0] / 2 + pad[0])
        & (points[:, 1] >= -pad[1])
        & (points[:, 1] <= target[1] + pad[1])
        & (np.abs(points[:, 2]) <= target[2] / 2 + pad[2])
    )
    points = points[inside]
    if len(points) < 24:
        return _fallback(fallback_scale, fallback_offset, "insufficient-scan-points")

    center = (bounds[0] + bounds[1]) / 2.0
    offset = np.array([-center[0], -bounds[0, 1], -center[2]], dtype=float)
    samples = np.asarray(sampled, dtype=float) + offset
    diagonal = float(np.linalg.norm(target))
    sigma = max(diagonal * 0.055, 1e-4)

    best: dict | None = None
    yaw_values = np.radians(np.arange(-180, 180, 15, dtype=float))
    translations = [(0.0, 0.0)]
    for dx in (-0.04, 0.0, 0.04):
        for dz in (-0.04, 0.0, 0.04):
            if dx or dz:
                translations.append((dx * target[0], dz * target[2]))

    for yaw in yaw_values:
        ac, ass = abs(math.cos(yaw)), abs(math.sin(yaw))
        # Solve the rotated XZ bounding extents for source-axis scale. Near 45
        # degrees the system is singular, so least squares gives the stable,
        # minimum-norm answer.
        matrix = np.array([[ac * extents[0], ass * extents[2]],
                           [ass * extents[0], ac * extents[2]]], dtype=float)
        sx, sz = np.linalg.lstsq(matrix, target[[0, 2]], rcond=None)[0]
        exact = np.array([max(sx, 1e-6), target[1] / extents[1], max(sz, 1e-6)])
        uniform_contain = float(np.min(target / np.maximum(
            [ac * extents[0] + ass * extents[2], extents[1],
             ass * extents[0] + ac * extents[2]], 1e-6)))
        uniform_median = float(np.median(target / np.maximum(
            [ac * extents[0] + ass * extents[2], extents[1],
             ass * extents[0] + ac * extents[2]], 1e-6)))

        geometric = float(np.exp(np.mean(np.log(np.maximum(exact, 1e-9)))))
        bounded = np.clip(exact, geometric / max_axis_stretch,
                          geometric * max_axis_stretch)
        scale_candidates = [
            np.full(3, uniform_contain),
            np.full(3, uniform_median),
            bounded,
        ]

        for scale in scale_candidates:
            transformed_base = _rotation_y(samples * scale, float(yaw))
            model_extent = np.ptp(transformed_base, axis=0)
            anisotropy = float(np.log(max(scale) / max(min(scale), 1e-9)))
            distortion_penalty = anisotropy / max(math.log(max_axis_stretch), 1e-6)

            for tx, tz in translations:
                translation = np.array([tx, 0.0, tz])
                transformed = transformed_base + translation
                distances, _ = cKDTree(transformed).query(points, k=1)
                # Trim the worst 10% so a little mask leakage cannot dominate.
                cutoff = np.percentile(distances, 90)
                robust = distances[distances <= cutoff]
                alignment = float(np.mean(np.exp(-0.5 * (robust / sigma) ** 2)))
                fill = float(np.exp(-np.mean(np.abs(np.log(
                    np.maximum(model_extent, 1e-6) / target
                )))))

                overflow_xyz = np.column_stack([
                    np.maximum(np.abs(transformed[:, 0]) - target[0] / 2, 0) / target[0],
                    np.maximum(-transformed[:, 1], 0) / target[1]
                    + np.maximum(transformed[:, 1] - target[1], 0) / target[1],
                    np.maximum(np.abs(transformed[:, 2]) - target[2] / 2, 0) / target[2],
                ])
                overflow = float(np.mean(np.max(overflow_xyz, axis=1)))
                shift_penalty = float(np.linalg.norm(translation[[0, 2]]) /
                                      max(np.linalg.norm(target[[0, 2]]), 1e-6))
                score = (
                    0.72 * alignment + 0.18 * fill
                    - 0.08 * overflow - 0.06 * distortion_penalty
                    - 0.03 * shift_penalty
                )
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "alignment": alignment,
                        "fill": fill,
                        "overflow": overflow,
                        "scale": scale.copy(),
                        "yaw": float(yaw),
                        "translation": translation.copy(),
                        "anisotropy": float(max(scale) / max(min(scale), 1e-9)),
                    }

    if best is None or best["alignment"] < 0.08:
        return _fallback(fallback_scale, fallback_offset, "weak-point-agreement")

    return {
        "model_scale": best["scale"].tolist(),
        "model_offset": offset.tolist(),
        "model_yaw": best["yaw"],
        "model_translation": best["translation"].tolist(),
        "fit": {
            "method": "hybrid-point-cloud-v1",
            "score": round(float(best["score"]), 4),
            "point_agreement": round(float(best["alignment"]), 4),
            "box_fill": round(float(best["fill"]), 4),
            "overflow": round(float(best["overflow"]), 4),
            "axis_stretch_ratio": round(float(best["anisotropy"]), 4),
            "yaw_degrees": round(math.degrees(float(best["yaw"])), 2),
            "point_count": int(len(points)),
        },
    }

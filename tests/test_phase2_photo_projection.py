"""Unit tests for photo_projection (photo-textured objects, best-view recolor)
and scene_builder's photo-thumbnail helper.

Run:  python tests/test_phase2_photo_projection.py
(synthetic cameras/images only — no model downloads)
"""
import os
import sys
import tempfile

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detection.photo_projection import (
    apply_photo_texture,
    project_to_frame,
    recolor_best_view,
)
from src.detection import scene_builder as sb

# Simple synthetic pinhole: 64×64, f=100, principal point at (32, 32),
# camera at origin looking down +Z (identity world→cam extrinsic).
K = np.array([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]])
E = np.hstack([np.eye(3), np.zeros((3, 1))])
H = W = 64


def _facing_quad():
    return trimesh.Trimesh(
        vertices=[[-0.2, -0.2, 1], [0.2, -0.2, 1], [0.2, 0.2, 1], [-0.2, 0.2, 1]],
        faces=[[0, 1, 2], [0, 2, 3]], process=False)


def test_project_to_frame_pinhole():
    uv, z = project_to_frame(
        np.array([[0.0, 0.0, 1.0], [0.16, 0.0, 1.0], [0.0, -0.16, 2.0]]), E, K)
    assert np.allclose(uv[0], [32, 32]) and np.isclose(z[0], 1.0)
    assert np.allclose(uv[1], [48, 32])        # +x → +u
    assert np.allclose(uv[2], [32, 24])        # 100·(−0.16)/2 = −8 in v
    assert np.allclose(z, [1.0, 1.0, 2.0])


def test_apply_photo_texture_visible_quad():
    img = np.zeros((H, W, 3))
    img[:, :, 0] = np.linspace(0, 1, W)[None, :]
    quad = _facing_quad()
    ok = apply_photo_texture(quad, np.asarray(quad.vertices), img, E, K,
                             np.full((H, W), 1.0))
    assert ok
    # vertex 0 at (−0.2, −0.2, 1) → pixel (12, 12) → trimesh uv (12/63, 1−12/63)
    assert np.allclose(quad.visual.uv[0], [12 / 63, 1 - 12 / 63], atol=1e-6)
    # explicit PBR: no 40%-gray multiplier, and metallicFactor written as an
    # explicit 0 — an omitted one defaults to 1.0 (fully metallic) per spec,
    # which killed diffuse response and rendered everything dark.
    mat = quad.visual.material
    assert list(mat.baseColorFactor) == [255, 255, 255, 255]
    assert mat.metallicFactor == 0.0


def test_apply_photo_texture_rejects_occluded():
    quad = _facing_quad()
    before = quad.visual
    # depth map says the surface is at 0.5 — our verts at z=1 are hidden behind it
    ok = apply_photo_texture(quad, np.asarray(quad.vertices),
                             np.zeros((H, W, 3)), E, K, np.full((H, W), 0.5))
    assert not ok
    assert quad.visual is before, "failed texturing must leave visuals untouched"


def test_recolor_best_view_prefers_closest_frame():
    mesh = trimesh.Trimesh(vertices=[[0, 0, 1], [0, 0, 1.5], [5, 5, -3]],
                           faces=[[0, 1, 2]], process=False)
    mesh.visual.vertex_colors = np.tile([10, 10, 10, 255], (3, 1)).astype(np.uint8)
    img_red = np.zeros((3, H, W)); img_red[0] = 1.0
    img_green = np.zeros((3, H, W)); img_green[1] = 1.0
    # frame 1's camera sits 0.5 closer along +Z (p_cam = p + t, t_z = −0.5)
    E_closer = np.hstack([np.eye(3), np.array([[0.0], [0.0], [-0.5]])])
    preds = {
        "images": np.stack([img_red, img_green]),
        "depth": np.full((2, H, W, 1), 10.0),
        "extrinsic": np.stack([E, E_closer]),
        "intrinsic": np.stack([K, K]),
    }
    n = recolor_best_view(mesh, preds, label="test")
    cols = np.asarray(mesh.visual.vertex_colors)
    assert n == 2
    assert (cols[0][:3] == [0, 255, 0]).all(), "closer frame (green) should win"
    assert (cols[2][:3] == [10, 10, 10]).all(), "behind-camera vertex keeps TSDF color"


def test_dense_crop_inverse_transform_roundtrip():
    """The local→VGGT inverse used for object photo-texturing in build_scene
    must exactly undo _crop_dense_mesh's world→local mapping (T_total = I,
    scale = 1, offset = 0 → pl.position == world_center)."""
    yaw = 0.7
    full = trimesh.creation.box(extents=[2, 2, 2])
    for _ in range(5):
        full = full.subdivide()
    full.apply_translation([1.0, 1.0, -0.5])
    full.visual.vertex_colors = np.tile([100, 80, 60, 255],
                                        (len(full.vertices), 1)).astype(np.uint8)
    world_center = np.array([1.0, 0.0, -0.5])
    cropped = sb._crop_dense_mesh(full, world_center, yaw, [1.5, 1.5, 1.5])
    assert cropped is not None

    c, s = np.cos(-yaw), np.sin(-yaw)
    R_yaw = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    recovered = np.asarray(cropped.vertices) @ R_yaw + world_center

    from scipy.spatial import cKDTree
    d, _ = cKDTree(full.vertices).query(recovered)
    assert d.max() < 1e-9, f"roundtrip error {d.max()}"


def test_save_object_photo():
    import cv2
    tmp = tempfile.mkdtemp()
    img_path = os.path.join(tmp, "frame0.png")
    img = np.zeros((480, 640, 3), np.uint8)
    img[100:300, 200:400] = (0, 200, 255)
    cv2.imwrite(img_path, img)
    scene_dir = os.path.join(tmp, "scene")
    os.makedirs(os.path.join(scene_dir, "objects"))

    inst = {"best_frame": 0, "best_box": np.array([200, 100, 400, 300])}
    rel = sb._save_object_photo([img_path], inst, scene_dir, "obj_000")
    assert rel == "objects/obj_000.jpg"
    out = cv2.imread(os.path.join(scene_dir, rel))
    assert out is not None and max(out.shape[:2]) <= 480
    assert abs(out.shape[1] / out.shape[0] - 640 / 480) < 0.02
    mid = out[out.shape[0] // 2, out.shape[1] // 2].astype(int)
    assert (np.abs(mid - [0, 200, 255]) < 20).all(), f"frame content wrong: {mid}"

    # degrades to None instead of crashing
    assert sb._save_object_photo([img_path], {"best_frame": None, "best_box": None},
                                 scene_dir, "x") is None


def test_save_object_photo_preserves_portrait_frame_orientation():
    import cv2
    tmp = tempfile.mkdtemp()
    img_path = os.path.join(tmp, "portrait.png")
    cv2.imwrite(img_path, np.zeros((800, 450, 3), np.uint8))
    scene_dir = os.path.join(tmp, "scene")
    os.makedirs(os.path.join(scene_dir, "objects"))

    rel = sb._save_object_photo(
        [img_path], {"best_frame": 0, "best_box": np.array([50, 250, 400, 500])},
        scene_dir, "obj_000",
    )
    out = cv2.imread(os.path.join(scene_dir, rel))
    assert out.shape[0] > out.shape[1]
    assert abs(out.shape[1] / out.shape[0] - 450 / 800) < 0.02
    assert sb._save_object_photo([img_path], {"best_frame": 5,
                                              "best_box": np.array([0, 0, 9, 9])},
                                 scene_dir, "x") is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {fn.__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)

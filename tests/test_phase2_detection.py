"""Unit tests for scene_builder's placement / clustering / calibration math.

Run:  python tests/test_scene_builder.py
(no VGGT model or SAM download needed — synthetic point sets only)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vggt_omega_repo"))

from src.detection import scene_builder as sb


def _box_points(center, size, yaw=0.0, n=3000, seed=0):
    """Random points filling a box centred at `center` with extents `size`,
    rotated by `yaw` about +Y in the exporter's convention (x toward z)."""
    rng = np.random.default_rng(seed)
    pts = (rng.random((n, 3)) - 0.5) * np.asarray(size)
    x, z = pts[:, 0], pts[:, 2]
    pts = np.stack([np.cos(yaw) * x - np.sin(yaw) * z,
                    pts[:, 1],
                    np.sin(yaw) * x + np.cos(yaw) * z], axis=1)
    return pts + np.asarray(center)


def test_yaw_recovery():
    # Elongated footprint at 30° should be recovered (mod 180°)
    pts = _box_points([0, 1, 0], [2.0, 1.0, 0.5], yaw=np.radians(30))
    yaw, ratio = sb._yaw_of_points(pts[:, [0, 2]])
    assert ratio > 1.5, f"expected elongated footprint, ratio={ratio}"
    err = abs((yaw - np.radians(30) + np.pi / 2) % np.pi - np.pi / 2)
    assert err < np.radians(4), f"yaw error {np.degrees(err):.1f}°"


def test_snap_yaw():
    assert abs(sb._snap_yaw(np.radians(4), 0.0) - 0.0) < 1e-9
    assert abs(sb._snap_yaw(np.radians(87), 0.0) - np.pi / 2) < 1e-9
    # 30° off the grid → left alone
    assert abs(sb._snap_yaw(np.radians(30), 0.0) - np.radians(30)) < 1e-9
    # snapping is relative to the room's own orientation
    assert abs(sb._snap_yaw(np.radians(25), np.radians(20)) - np.radians(20)) < 1e-9


def test_cluster_merges_same_object_and_keeps_separate_ones():
    cols = np.zeros((3000, 3), dtype=np.uint8)
    obs = []
    # Rack A seen from 3 frames (slightly shifted centres)
    for f in range(3):
        pts = _box_points([0 + 0.03 * f, 1.0, 0], [0.6, 2.0, 1.0], seed=f)
        obs.append(sb._make_observation("server rack", 0.6, f, pts, cols))
    # Rack B 3 m away, 2 frames
    for f in range(2):
        pts = _box_points([3.0, 1.0, 0], [0.6, 2.0, 1.0], seed=10 + f)
        obs.append(sb._make_observation("server rack", 0.55, f, pts, cols))
    # Single-frame low-score ghost → should be dropped
    obs.append(sb._make_observation(
        "server rack", 0.3, 0, _box_points([8, 1, 8], [0.5, 0.5, 0.5]), cols))

    instances = sb.cluster_observations(obs)
    assert len(instances) == 2, f"expected 2 instances, got {len(instances)}"
    assert sorted(i["frames_seen"] for i in instances) == [2, 3]


def test_placement_accuracy():
    # Rack: bottom at y=0.05 (near floor at 0), centred at (1.5, ·, -2)
    pts = _box_points([1.5, 1.05, -2.0], [0.6, 2.0, 1.0], yaw=0.0, n=6000)
    inst = {"pts": pts}
    p = sb.place_instance(inst, floor_y=0.0, room_yaw=0.0, floor_snap_tol=0.3)
    assert abs(p["position"][0] - 1.5) < 0.05
    assert abs(p["position"][2] + 2.0) < 0.05
    assert p["position"][1] == 0.0, "bottom should snap to floor"
    # major axis (depth=1.0) along Z → local X extent should be ~1.0 after yaw fold
    dims = sorted(p["size"])
    assert abs(dims[-1] - 2.0) < 0.15         # height
    assert abs(p["size"][1] - 2.0) < 0.15     # size[1] is height


def test_rack_scale():
    placed = [
        {"label": "server rack", "placement": {"size": [0.3, 1.0, 0.5]}},
        {"label": "server rack", "placement": {"size": [0.3, 1.1, 0.5]}},
        {"label": "desk", "placement": {"size": [1.0, 0.4, 0.6]}},
    ]
    scale, calibrated = sb.rack_scale(placed)
    assert calibrated
    # median rack height 1.05 → scale = 2.0 / 1.05
    assert abs(scale - 2.0 / 1.05) < 1e-6

    scale, calibrated = sb.rack_scale([placed[2]])
    assert not calibrated and scale == 1.0


def test_level_transform():
    # Floor tilted 10° about X, plus wall points so the fit has to pick the band
    rng = np.random.default_rng(1)
    floor = np.stack([rng.uniform(-5, 5, 4000), np.zeros(4000), rng.uniform(-5, 5, 4000)], axis=1)
    wall = np.stack([rng.uniform(-5, 5, 2000), rng.uniform(0, 3, 2000), np.full(2000, 5.0)], axis=1)
    tilt = np.radians(10)
    R = np.array([[1, 0, 0],
                  [0, np.cos(tilt), -np.sin(tilt)],
                  [0, np.sin(tilt), np.cos(tilt)]])
    pts = np.vstack([floor, wall]) @ R.T
    T = sb._level_transform(pts)
    levelled = pts @ T[:3, :3].T
    floor_levelled = levelled[:4000]
    spread = np.percentile(floor_levelled[:, 1], 98) - np.percentile(floor_levelled[:, 1], 2)
    assert spread < 0.1, f"floor not level after correction, Y spread {spread:.3f}"


def test_level_transform_steep_tilt_with_prior():
    # A capture tilted 50° in the aligned frame (e.g. shot down an aisle):
    # the real floor is no longer the lowest-Y band and a wall dominates it,
    # so the legacy +Y assumption rotates the WALL flat and leaves a wall as
    # the ground. With a gravity prior the true floor levels regardless of tilt.
    rng = np.random.default_rng(2)
    floor = np.stack([rng.uniform(-5, 5, 4000), np.zeros(4000), rng.uniform(-5, 5, 4000)], axis=1)
    wall = np.stack([rng.uniform(-5, 5, 3000), rng.uniform(0, 3, 3000), np.full(3000, 5.0)], axis=1)
    tilt = np.radians(50)
    R = np.array([[1, 0, 0],
                  [0, np.cos(tilt), -np.sin(tilt)],
                  [0, np.sin(tilt), np.cos(tilt)]])
    pts = np.vstack([floor, wall]) @ R.T
    up_prior = R @ np.array([0.0, 1.0, 0.0])          # gravity in the aligned frame

    # Legacy assumption (no prior) fails: floor stays far from level.
    legacy = pts[:4000] @ sb._level_transform(pts)[:3, :3].T
    legacy_spread = np.percentile(legacy[:, 1], 98) - np.percentile(legacy[:, 1], 2)
    assert legacy_spread > 0.5, "expected the no-prior path to mishandle a 50° tilt"

    # With a trusted gravity prior the true floor levels (the 50° correction
    # exceeds the untrusted 30° cap, so trusted=True is required here).
    T = sb._level_transform(pts, up_prior=up_prior, trusted=True)
    floor_levelled = pts[:4000] @ T[:3, :3].T
    spread = np.percentile(floor_levelled[:, 1], 98) - np.percentile(floor_levelled[:, 1], 2)
    assert spread < 0.15, f"floor not level after prior-guided correction, Y spread {spread:.3f}"


def _orbit_extrinsics(n=8, radius=3.0, height=1.6, target=np.array([0, 0, 1.2])):
    """Cameras orbiting the room at constant height (real up = world +Z), each
    pitched slightly down toward a target — OpenCV convention (x right, y down,
    z forward)."""
    def look_at(pos, tgt, world_up=np.array([0, 0, 1.0])):
        f = tgt - pos; f /= np.linalg.norm(f)          # +z_cam (forward)
        r = np.cross(f, world_up); r /= np.linalg.norm(r)
        d = np.cross(f, r)                              # +y_cam (down)
        Rwc = np.stack([r, d, f], axis=0)              # world→cam
        return np.hstack([Rwc, (-Rwc @ pos)[:, None]])
    return np.array([
        look_at(np.array([radius * np.cos(a), radius * np.sin(a), height]), tgt=target)
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False)
    ])


def test_estimate_up_recovers_gravity():
    # A constant-height orbit gives the camera-path estimator a clean 2-D
    # horizontal patch, so it recovers the same up T_align maps world-up onto —
    # regardless of frame 0 — and reports it as trusted.
    exts = _orbit_extrinsics()
    T_align = sb._scene_transform(exts)
    up, trusted = sb._estimate_up(exts, T_align)
    gt = T_align[:3, :3] @ np.array([0, 0, 1.0]); gt /= np.linalg.norm(gt)
    assert float(up @ gt) > 0.98, f"estimated up off from gravity: dot={up @ gt:.3f}"
    assert trusted, "constant-height orbit should yield a trusted camera-path estimate"


def test_estimate_up_falls_back_untrusted_on_degenerate_path():
    # A straight-line walk: camera centres lie on a 1-D line, so height is not a
    # clear least-variance minimum (a horizontal axis is just as flat). The path
    # estimator must bail and _estimate_up fall back to an untrusted median.
    def look_at(pos, tgt, world_up=np.array([0, 0, 1.0])):
        f = tgt - pos; f /= np.linalg.norm(f)
        r = np.cross(f, world_up); r /= np.linalg.norm(r)
        d = np.cross(f, r)
        Rwc = np.stack([r, d, f], axis=0)
        return np.hstack([Rwc, (-Rwc @ pos)[:, None]])

    exts = np.array([
        look_at(np.array([x, 0.0, 1.6]), np.array([x, 5.0, 1.2]))   # walk along +X
        for x in np.linspace(-3, 3, 8)
    ])
    T_align = sb._scene_transform(exts)
    _, trusted = sb._estimate_up(exts, T_align)
    assert not trusted, "straight-line walk should not yield a trusted path estimate"


def test_floor_snap_is_label_aware():
    # A tabletop slab floating at 0.62 m — legs too thin to survive depth
    # filtering, the common real-scan case. Tables must still ground.
    rng = np.random.default_rng(0)
    pts = rng.random((3000, 3)) * [0.8, 0.15, 0.8] + [0, 0.62, 0]
    tol = 0.3

    table = sb.place_instance({"pts": pts, "label": "table"},
                              floor_y=0.0, room_yaw=0.0, floor_snap_tol=tol)
    assert table["position"][1] == 0.0, "floor-standing kind should snap"
    assert abs(table["size"][1] - 0.77) < 0.05, "snapped height spans floor→top"

    monitor = sb.place_instance({"pts": pts, "label": "monitor"},
                                floor_y=0.0, room_yaw=0.0, floor_snap_tol=tol)
    assert monitor["position"][1] > 0.5, "wall/stand-mounted kinds must not snap"

    unlabeled = sb.place_instance({"pts": pts},
                                  floor_y=0.0, room_yaw=0.0, floor_snap_tol=tol)
    assert unlabeled["position"][1] > 0.5, "unknown labels keep strict tolerance"


def test_region_restricted_tsdf_fusion():
    """include_mask + bounds fuse only the masked pixels into a small
    high-res grid — the basis of per-object re-fusion."""
    from src.reconstruction.tsdf_fusion import TSDFConfig, fuse_tsdf_raw_mesh

    H = W = 64
    f = 50.0
    K = np.array([[f, 0, 32], [0, f, 32], [0, 0, 1]])
    E = np.hstack([np.eye(3), np.zeros((3, 1))])
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    preds = {
        "depth": np.full((1, H, W, 1), 1.0, np.float32),
        "depth_conf": np.full((1, H, W), 10.0, np.float32),
        "images": np.full((1, 3, H, W), 0.5, np.float32),
        "extrinsic": E[None], "intrinsic": K[None],
        "world_points_from_depth":
            np.stack([(u - 32) / f, (v - 32) / f, np.ones_like(u, float)], -1)[None],
    }
    inc = np.zeros((1, H, W), bool)
    inc[0, 20:44, 20:44] = True
    bounds = (np.array([-0.4, -0.4, 0.7]), np.array([0.4, 0.4, 1.3]))
    cfg = TSDFConfig(conf_percentile=0, auto_resolution=96, max_dim=112,
                     filter_depth_edges=False)

    mesh = fuse_tsdf_raw_mesh(preds, cfg, include_mask=inc, bounds=bounds)
    vs = np.asarray(mesh.vertices)
    assert len(vs) > 50
    assert abs(np.median(vs[:, 2]) - 1.0) < 0.05, "plane should sit at z=1"
    # mask spans pixels 20..43 → xy within ±0.24 at z=1, plus truncation slack
    assert np.abs(vs[:, :2]).max() < 0.32, "geometry leaked outside the mask"

    # same bounds without the mask fill wider — the mask is what restricts
    full = fuse_tsdf_raw_mesh(preds, cfg, bounds=bounds)
    assert np.abs(np.asarray(full.vertices)[:, :2]).max() > 0.35


def test_prefabs_have_expected_bounds():
    for kind, dims in [("rack", (0.6, 2.0, 1.0)), ("desk", (1.4, 0.75, 0.7)),
                       ("monitor", (0.6, 0.4, 0.2)), ("chair", (0.5, 0.9, 0.5)),
                       ("cabinet", (0.8, 1.8, 0.5))]:
        mesh = sb._PREFAB_BUILDERS[kind](*dims, (120, 120, 120))
        lo, hi = mesh.bounds
        assert lo[1] > -1e-6, f"{kind} dips below floor: {lo[1]}"
        assert abs(hi[1] - dims[1]) < dims[1] * 0.1, f"{kind} height {hi[1]} vs {dims[1]}"
        for ax in (0, 2):
            assert hi[ax] - lo[ax] <= max(dims[0], dims[2]) * 1.15, f"{kind} axis {ax} too big"

    # rack with major axis on X gets rotated: extents must still match input
    mesh = sb._PREFAB_BUILDERS["rack"](1.0, 2.0, 0.6, (120, 120, 120))
    lo, hi = mesh.bounds
    assert abs((hi[0] - lo[0]) - 1.0) < 0.12
    assert abs((hi[2] - lo[2]) - 0.6) < 0.12


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

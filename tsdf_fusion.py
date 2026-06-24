"""
TSDF fusion + marching cubes meshing for Room3D.

Consumes the raw VGGT-Omega predictions dict (same format as visual_util.predictions_to_glb)
and returns a trimesh.Scene with a watertight mesh instead of a point cloud.

Data contract (squeezed batch dim, confirmed from app.py / vggt_omega_repo):
  depth        : (S, H, W, 1)  float32 — exp(logits), camera-space Z
  depth_conf   : (S, H, W)     float32 — values ≥ 1.0
  images       : (S, 3, H, W)  float32 — [0, 1]
  extrinsic    : (S, 3, 4)     float64 — world-to-camera (R | t)
  intrinsic    : (S, 3, 3)     float64 — pinhole, pixel-space at H×W
  world_points_from_depth : (S, H, W, 3)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


_PYMESHLAB_SCRIPT = """
import sys, pickle, numpy as np, pymeshlab
in_path, out_path = sys.argv[1], sys.argv[2]
with open(in_path, "rb") as f:
    pts, cols_u8, depth_param = pickle.load(f)
rgba = np.ones((len(cols_u8), 4), dtype=np.float32)
rgba[:, :3] = cols_u8.astype(np.float32) / 255.0
ms = pymeshlab.MeshSet()
ms.add_mesh(pymeshlab.Mesh(vertex_matrix=pts.astype(np.float64), v_color_matrix=rgba))
ms.compute_normal_for_point_clouds(k=30)
ms.generate_surface_reconstruction_screened_poisson(depth=depth_param, preclean=False)
out = ms.current_mesh()
with open(out_path, "wb") as f:
    pickle.dump((out.vertex_matrix(), out.face_matrix(), out.vertex_color_matrix()), f)
"""


def _run_poisson_subprocess(all_points: np.ndarray, all_colors_u8: np.ndarray, depth_param: int):
    """Run PyMeshLab Poisson in a clean subprocess to avoid OpenMP conflict with PyTorch."""
    import sys, pickle, subprocess, tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump((all_points, all_colors_u8, depth_param), f)
        in_path = f.name
    out_path = in_path + ".out.pkl"
    try:
        subprocess.run(
            [sys.executable, "-c", _PYMESHLAB_SCRIPT, in_path, out_path],
            check=True, timeout=300,
        )
        with open(out_path, "rb") as f:
            return pickle.load(f)
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


@dataclass
class TSDFConfig:
    conf_percentile: float = 20.0   # same semantics as the point-cloud conf_thres slider
    auto_resolution: int = 256      # target voxels along the scene's longest axis
    sdf_trunc_factor: float = 4.0   # sdf_trunc = factor × voxel_length
    min_cluster_faces: int = 500    # remove floater components smaller than this
    filter_depth_edges: bool = True
    depth_edge_rtol: float = 0.03


def predictions_to_mesh_glb(
    predictions: dict,
    conf_thres: float = 20.0,
    show_cam: bool = False,
    mesh_resolution: int = 256,
) -> trimesh.Scene:
    """Public entry point — mirrors predictions_to_glb's signature."""
    config = TSDFConfig(
        conf_percentile=conf_thres,
        auto_resolution=mesh_resolution,
    )
    return fuse_tsdf(predictions, config, show_cam=show_cam)


def fuse_tsdf(
    predictions: dict,
    config: TSDFConfig,
    show_cam: bool = False,
) -> trimesh.Scene:
    """TSDF fusion via numpy + skimage marching cubes (no open3d dependency)."""
    from skimage.measure import marching_cubes
    from visual_util import depth_edge, apply_scene_alignment, integrate_camera_into_scene

    depth_map  = predictions["depth"]                       # (S, H, W, 1)
    depth_conf = predictions["depth_conf"]                  # (S, H, W)
    images     = predictions["images"]                      # (S, 3, H, W)
    extrinsic  = predictions["extrinsic"]                   # (S, 3, 4)
    intrinsic  = predictions["intrinsic"]                   # (S, 3, 3)
    world_pts  = predictions["world_points_from_depth"]     # (S, H, W, 3)

    S, H, W = depth_conf.shape

    # ── Confidence mask ───────────────────────────────────────────────────────
    conf = depth_conf.copy()
    if config.filter_depth_edges:
        conf[depth_edge(depth_map[..., 0], rtol=config.depth_edge_rtol)] = 0.0

    valid = np.isfinite(conf) & (conf > 1e-5)
    conf_threshold = float(np.percentile(conf[valid], config.conf_percentile)) if valid.any() and config.conf_percentile > 0 else 0.0

    valid_mask = valid & (conf >= conf_threshold) & np.isfinite(world_pts).all(axis=-1)
    valid_pts  = world_pts[valid_mask]
    if len(valid_pts) < 100:
        raise ValueError("Too few valid points — try lowering the confidence threshold.")

    # ── Voxel grid geometry ───────────────────────────────────────────────────
    lower = np.percentile(valid_pts, 2,  axis=0)
    upper = np.percentile(valid_pts, 98, axis=0)
    pad   = (upper - lower) * 0.05
    lower -= pad;  upper += pad

    extent       = float(np.linalg.norm(upper - lower))
    voxel_length = extent / max(config.auto_resolution, 1)
    sdf_trunc    = config.sdf_trunc_factor * voxel_length

    dims = np.maximum(((upper - lower) / voxel_length).astype(int) + 2, 2)
    # Cap to avoid OOM on large scenes
    max_dim = 192
    if dims.max() > max_dim:
        dims = np.maximum((dims * max_dim / dims.max()).astype(int), 2)
        voxel_length = float(np.max((upper - lower) / (dims - 1)))
        sdf_trunc    = config.sdf_trunc_factor * voxel_length

    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    N = nx * ny * nz

    # World-space voxel centres, shape (N, 3)
    xs = np.linspace(lower[0], upper[0], nx, dtype=np.float32)
    ys = np.linspace(lower[1], upper[1], ny, dtype=np.float32)
    zs = np.linspace(lower[2], upper[2], nz, dtype=np.float32)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    vox_world = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)  # (N, 3)

    tsdf_w    = np.zeros(N, dtype=np.float32)   # weight sum
    tsdf_sum  = np.full(N, sdf_trunc, dtype=np.float32)   # weighted SDF sum (init = outside)
    color_sum = np.zeros((N, 3), dtype=np.float32)

    # ── Per-frame TSDF integration (vectorised over voxels) ───────────────────
    for i in range(S):
        K   = intrinsic[i]
        ext = extrinsic[i]                       # (3, 4) world-to-cam

        cam = (ext[:, :3] @ vox_world.T + ext[:, 3:]).T   # (N, 3) camera coords
        z   = cam[:, 2]

        u = K[0, 0] * cam[:, 0] / np.where(z > 0, z, 1) + K[0, 2]
        v = K[1, 1] * cam[:, 1] / np.where(z > 0, z, 1) + K[1, 2]
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)

        in_view = (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not in_view.any():
            continue

        ui_c = ui.clip(0, W - 1);  vi_c = vi.clip(0, H - 1)
        d_px   = depth_map[i, vi_c, ui_c, 0]
        c_px   = conf[i,      vi_c, ui_c]
        col_px = np.transpose(images[i], (1, 2, 0))[vi_c, ui_c]  # (N, 3)

        sdf = d_px - z
        ok  = in_view & (d_px > 0) & (sdf > -sdf_trunc) & (c_px >= conf_threshold)

        sdf_t  = np.clip(sdf, -sdf_trunc, sdf_trunc)
        weight = c_px * ok.astype(np.float32)

        # Accumulate — bincount is O(N) and much faster than np.add.at
        tsdf_sum  += np.bincount(np.arange(N), weights=sdf_t  * weight, minlength=N).astype(np.float32)
        tsdf_w    += np.bincount(np.arange(N), weights=weight,           minlength=N).astype(np.float32)
        for c in range(3):
            color_sum[:, c] += np.bincount(np.arange(N), weights=col_px[:, c] * weight, minlength=N).astype(np.float32)

    # ── Average and reshape ───────────────────────────────────────────────────
    has_w = tsdf_w > 0
    tsdf_vol  = np.where(has_w, tsdf_sum / np.where(has_w, tsdf_w, 1), sdf_trunc).reshape(nx, ny, nz)
    color_vol = np.where(has_w[:, None], color_sum / np.where(has_w[:, None], tsdf_w[:, None], 1), 0.5).reshape(nx, ny, nz, 3)

    # ── Marching cubes ────────────────────────────────────────────────────────
    try:
        verts_idx, faces, _, _ = marching_cubes(tsdf_vol, level=0.0, allow_degenerate=False)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Marching cubes failed — try lowering confidence threshold. ({exc})") from exc

    if len(verts_idx) == 0:
        raise ValueError("TSDF produced empty mesh — try lowering confidence threshold.")

    # Voxel indices → world coords
    scale = (upper - lower) / np.array([nx - 1, ny - 1, nz - 1], dtype=np.float64)
    verts_world = lower + verts_idx * scale

    # Interpolate vertex colors from the voxel grid
    vi_int = np.floor(verts_idx).astype(int).clip([[0, 0, 0]], [[nx - 2, ny - 2, nz - 2]])
    vert_colors = (color_vol[vi_int[:, 0], vi_int[:, 1], vi_int[:, 2]] * 255).clip(0, 255).astype(np.uint8)

    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, vertex_colors=vert_colors, process=False)

    # Remove floaters — keep only the largest component
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda m: len(m.faces))

    # ── Scene assembly + alignment ────────────────────────────────────────────
    extrinsics_4x4                  = np.zeros((S, 4, 4), dtype=np.float64)
    extrinsics_4x4[:, :3, :4]       = extrinsic
    extrinsics_4x4[:, 3, 3]         = 1.0

    scene_scale = float(np.linalg.norm(upper - lower))
    if scene_scale <= 0:
        scene_scale = 1.0

    scene = trimesh.Scene()
    scene.add_geometry(mesh)

    if show_cam:
        from matplotlib import colormaps
        colormap = colormaps.get_cmap("gist_rainbow")
        for i, world_to_cam in enumerate(extrinsics_4x4):
            cam_to_world = np.linalg.inv(world_to_cam)
            rgba  = colormap(i / max(S, 1))
            color = tuple(int(255 * x) for x in rgba[:3])
            integrate_camera_into_scene(scene, cam_to_world, color, scene_scale)

    return apply_scene_alignment(scene, extrinsics_4x4)


def predictions_to_mesh_poisson(
    predictions: dict,
    conf_thres: float = 20.0,
    show_cam: bool = False,
    mesh_resolution: int = 256,
) -> trimesh.Scene:
    """Poisson surface reconstruction — smoother and faster than TSDF."""
    from visual_util import depth_edge, apply_scene_alignment, integrate_camera_into_scene

    depth_map = predictions["depth"]
    depth_conf = predictions["depth_conf"]
    images = predictions["images"]
    extrinsic = predictions["extrinsic"]
    world_pts = predictions["world_points_from_depth"]

    S, H, W = depth_conf.shape

    # ── Build point cloud with colors ─────────────────────────────────────────
    conf = depth_conf.copy()
    conf[depth_edge(depth_map[..., 0], rtol=0.03)] = 0.0

    valid = np.isfinite(conf) & (conf > 1e-5)
    if valid.any() and conf_thres > 0:
        conf_threshold = float(np.percentile(conf[valid], conf_thres))
    else:
        conf_threshold = 0.0

    # Collect points and colors
    points_list = []
    colors_list = []

    for i in range(S):
        mask = (conf[i] >= conf_threshold) & np.isfinite(world_pts[i]).all(axis=-1)
        if mask.any():
            pts = world_pts[i][mask]
            # Colors from images (convert from [0,1] to [0,255])
            color_hwc = np.transpose(images[i], (1, 2, 0))
            cols = color_hwc[mask] * 255
            points_list.append(pts)
            colors_list.append(cols)

    if not points_list:
        raise ValueError(
            "No valid points for Poisson reconstruction — try lowering the confidence threshold."
        )

    all_points = np.vstack(points_list).astype(np.float64)
    all_colors = np.vstack(colors_list)  # [0, 255] uint8

    # ── Poisson surface reconstruction via PyMeshLab (subprocess) ────────────
    depth_param = max(6, min(12, int(np.log2(mesh_resolution / 32))))
    verts, faces, vert_colors_f = _run_poisson_subprocess(
        all_points, all_colors.astype(np.uint8), depth_param
    )

    if len(verts) == 0:
        raise ValueError(
            "Poisson reconstruction produced an empty mesh.  "
            "Try lowering the confidence threshold."
        )

    # ── Convert to trimesh ────────────────────────────────────────────────────
    tri_mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_colors=(
            (vert_colors_f[:, :3] * 255).clip(0, 255).astype(np.uint8)
            if len(vert_colors_f) > 0
            else None
        ),
        process=False,
    )

    # ── Scene assembly + alignment ────────────────────────────────────────────
    extrinsics_4x4 = np.zeros((S, 4, 4), dtype=np.float64)
    extrinsics_4x4[:, :3, :4] = extrinsic
    extrinsics_4x4[:, 3, 3] = 1.0

    scene_scale = float(np.linalg.norm(np.percentile(all_points, 95, axis=0) - np.percentile(all_points, 5, axis=0)))
    if scene_scale <= 0:
        scene_scale = 1.0

    scene = trimesh.Scene()
    scene.add_geometry(tri_mesh)

    if show_cam:
        from matplotlib import colormaps
        colormap = colormaps.get_cmap("gist_rainbow")
        for i, world_to_cam in enumerate(extrinsics_4x4):
            cam_to_world = np.linalg.inv(world_to_cam)
            rgba = colormap(i / max(S, 1))
            color = tuple(int(255 * x) for x in rgba[:3])
            integrate_camera_into_scene(scene, cam_to_world, color, scene_scale)

    return apply_scene_alignment(scene, extrinsics_4x4)

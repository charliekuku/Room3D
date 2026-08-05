"""
photo_projection.py — Put real photo pixels on reconstructed geometry.

The TSDF pipeline colors meshes from voxel-averaged vertex colors: every
frame's contribution gets blended at voxel resolution, which reads as blurred
clay. But the source frames, their depth maps, and their camera matrices are
all still available at build time — so instead:

  apply_photo_texture   UV-projects a mesh into its single best source frame
                        and attaches that frame as an actual texture. Used for
                        object meshes (small, seen well in one view).
  recolor_best_view     Re-samples each vertex's color from the one frame that
                        saw it closest (instead of the all-frames average).
                        Used for the room-scale meshes, where no single frame
                        covers everything so a texture atlas would be needed —
                        per-vertex best-view sampling removes the cross-view
                        ghosting/blur at a fraction of the complexity.

Both work in VGGT's original coordinate frame (the one `extrinsic`/`depth`
live in), so callers must project *before* applying scene alignment/scale
transforms, or hand in vertices mapped back into that frame.
"""
from __future__ import annotations

import numpy as np
import trimesh


def project_to_frame(pts: np.ndarray, extrinsic: np.ndarray,
                     intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World(VGGT)-space points → ((N,2) pixel coords, (N,) camera depth).

    extrinsic is VGGT's 3×4 world→camera [R|t] (p_cam = R p + t), matching
    app.py's convention (project_detections_to_3d inverts the same way).
    """
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    cam = pts @ R.T + t
    z = cam[:, 2]
    safe_z = np.where(np.abs(z) > 1e-9, z, 1e-9)
    u = intrinsic[0, 0] * cam[:, 0] / safe_z + intrinsic[0, 2]
    v = intrinsic[1, 1] * cam[:, 1] / safe_z + intrinsic[1, 2]
    return np.stack([u, v], axis=1), z


def _visible(uv: np.ndarray, z: np.ndarray, depth_frame: np.ndarray,
             rel_tol: float = 0.08, abs_tol: float = 0.03) -> np.ndarray:
    """True where a projected point is in front of the camera, inside the
    frame, and not occluded (its depth agrees with the frame's depth map —
    a point behind the recorded surface was hidden in this view). Tolerances
    are loose on purpose: TSDF surfaces deviate slightly from raw depth."""
    H, W = depth_frame.shape
    vis = np.zeros(len(z), dtype=bool)
    inb = (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) \
        & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
    if not inb.any():
        return vis
    ui = np.round(uv[inb, 0]).astype(np.int32)
    vi = np.round(uv[inb, 1]).astype(np.int32)
    d = depth_frame[vi, ui]
    ok = (d > 1e-6) & (z[inb] <= d * (1.0 + rel_tol) + abs_tol)
    vis[np.flatnonzero(inb)[ok]] = True
    return vis


def apply_photo_texture(mesh: trimesh.Trimesh, verts_vggt: np.ndarray,
                        image_hwc: np.ndarray, extrinsic: np.ndarray,
                        intrinsic: np.ndarray, depth_frame: np.ndarray,
                        min_visible: float = 0.6) -> bool:
    """Texture `mesh` with the actual source frame, UVs from projecting each
    vertex into that frame's camera. Replaces mesh.visual on success.

    Returns False (mesh untouched, keeps its baked vertex colors) when fewer
    than `min_visible` of the vertices project cleanly — e.g. a closed Poisson
    mesh whose back side this frame never saw would smear front pixels across
    it. Open dense-crop surfaces, which only exist where a camera saw them,
    normally clear the bar easily.
    """
    uv_px, z = project_to_frame(verts_vggt, extrinsic, intrinsic)
    vis = _visible(uv_px, z, depth_frame)
    if len(vis) == 0 or vis.mean() < min_visible:
        return False

    H, W = depth_frame.shape
    u = np.clip(uv_px[:, 0], 0, W - 1) / max(W - 1, 1)
    # trimesh UV convention: v=0 is the image's bottom row (its glTF exporter
    # flips to spec top-left origin) — verified against exported texcoords.
    v = 1.0 - np.clip(uv_px[:, 1], 0, H - 1) / max(H - 1, 1)

    from PIL import Image
    img = Image.fromarray((image_hwc * 255).clip(0, 255).astype(np.uint8))
    # Explicit PBR with metallicFactor 0: an omitted metallicFactor defaults
    # to 1.0 (fully metallic) per the glTF spec, which kills the diffuse
    # response and renders everything as dark rough metal — same bug
    # scene_builder._ensure_material documents. White base color so the
    # photo texture carries the color undiminished.
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=img,
        baseColorFactor=[255, 255, 255, 255],
        metallicFactor=0.0,
        roughnessFactor=0.95,
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.stack([u, v], axis=1), material=material)
    return True


def recolor_best_view(mesh: trimesh.Trimesh, predictions: dict,
                      label: str = "mesh") -> int:
    """Replace each vertex's TSDF-averaged color with the color sampled from
    the single frame that saw that vertex from closest up. Mesh must still be
    in VGGT coordinates. Vertices no frame saw cleanly keep their TSDF color.
    Returns the number of vertices recolored."""
    images = predictions["images"]          # (S, 3, H, W) float 0..1
    depth = predictions["depth"]            # (S, H, W, 1)
    extrinsic = predictions["extrinsic"]    # (S, 3, 4)
    intrinsic = predictions["intrinsic"]    # (S, 3, 3)
    S = images.shape[0]

    verts = np.asarray(mesh.vertices)
    n = len(verts)
    best_z = np.full(n, np.inf, dtype=np.float32)
    new_cols = np.zeros((n, 3), dtype=np.uint8)
    have = np.zeros(n, dtype=bool)

    for fi in range(S):
        uv, z = project_to_frame(verts, extrinsic[fi], intrinsic[fi])
        vis = _visible(uv, z, depth[fi, ..., 0])
        upd = vis & (z < best_z)
        if not upd.any():
            continue
        ui = np.round(uv[upd, 0]).astype(np.int32)
        vi = np.round(uv[upd, 1]).astype(np.int32)
        img_hwc = np.transpose(images[fi], (1, 2, 0))
        new_cols[upd] = (img_hwc[vi, ui] * 255).clip(0, 255).astype(np.uint8)
        best_z[upd] = z[upd].astype(np.float32)
        have |= upd

    if have.any():
        colors = np.asarray(mesh.visual.vertex_colors).copy()
        colors[have, :3] = new_cols[have]
        mesh.visual.vertex_colors = colors
    print(f"[PhotoProjection] {label}: best-view recolored "
          f"{int(have.sum())}/{n} vertices from {S} frames.")
    return int(have.sum())

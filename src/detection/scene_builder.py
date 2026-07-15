"""
scene_builder.py — Editable scene construction for Room3D.

Turns VGGT-Omega predictions + Grounding DINO detections into an editable scene:

  1. SAM (box-prompted) refines each 2-D detection into a pixel mask.
  2. Masked depth pixels are lifted to world space per frame (an "observation").
  3. Observations are clustered across frames into object instances — multi-frame
     agreement is what makes positions accurate; single-frame outliers are dropped.
  4. The scene is levelled (floor plane → horizontal) and metrically calibrated
     from detected server racks (42U rack ≈ 2.0 m tall).
  5. Each instance becomes a separate movable node: a parametric prefab for
     standard equipment (rack, desk, monitor, chair, cabinet), or for irregular
     objects, a piece cut directly out of a dense full-scene mesh (falling
     back to an isolated per-object Poisson reconstruction, then a plain box).
  6. The background is re-fused with object pixels carved out, plus a floor
     slab, a ceiling slab (only when real ceiling coverage was observed), and
     wall planes at the room's bounding rectangle.

Output layout (scenes/<name>/):
  scene.json        — object list with positions/yaw/size in metres
  background.glb    — carved room mesh
  objects/<id>.glb  — per-object geometry, origin at bottom-centre
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Callable

import cv2
import numpy as np
import trimesh

from src.detection.photo_projection import apply_photo_texture, recolor_best_view

# 42U server rack — the metric anchor for data-center scenes
RACK_HEIGHT_M = 2.0

# Labels that get parametric prefab geometry; everything else gets a scan cut-out
_PREFAB_KINDS = {
    "server rack": "rack", "rack": "rack", "server cabinet": "rack",
    "desk": "desk", "table": "desk",
    "monitor": "monitor", "screen": "monitor",
    "chair": "chair",
    "cabinet": "cabinet", "ups": "cabinet", "pdu": "cabinet",
    "network switch": "shelfbox", "server": "shelfbox", "patch panel": "shelfbox",
}

_MAX_DETS_PER_FRAME = 20
_MAX_PTS_PER_OBS = 4000


# ── SAM 2 segmentation ────────────────────────────────────────────────────────
# CUDA: segmentation runs after VGGT's forward pass has freed its activations,
# and hiera-tiny is ~0.1 GB — GPU is far faster per frame. MPS/CPU hosts stay
# on CPU: unified memory is the scarce resource there.

_sam_model = None
_sam_processor = None


def get_sam():
    global _sam_model, _sam_processor
    if _sam_model is None:
        import torch
        from transformers import Sam2Model, Sam2Processor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[SceneBuilder] Loading SAM 2 (facebook/sam2.1-hiera-tiny)…")
        _sam_processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-tiny")
        _sam_model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").eval().to(device)
        print(f"[SceneBuilder] SAM 2 ready ({device}).")
    return _sam_model, _sam_processor


def segment_detections(
    image_paths: list[str],
    dets_2d: list[dict],
    abort_check: Callable[[], None] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> list[list[np.ndarray | None]]:
    """Box-prompted SAM 2 masks. Returns per-frame lists of bool masks at original
    image resolution (None where SAM 2's IoU estimate says the mask is junk)."""
    import torch
    from PIL import Image as PILImage

    model, processor = get_sam()
    all_masks: list[list[np.ndarray | None]] = []
    n = len(dets_2d)

    for fi, frame in enumerate(dets_2d):
        if abort_check is not None:
            abort_check()
        if progress_cb is not None and (fi % 10 == 0 or fi == n - 1):
            progress_cb(0.5 * fi / max(n, 1), f"Segmenting detections: {fi}/{n} frames")
        boxes = np.asarray(frame["boxes"])[:_MAX_DETS_PER_FRAME]
        if len(boxes) == 0:
            all_masks.append([])
            continue

        img = PILImage.open(image_paths[fi]).convert("RGB")
        inputs = processor(img, input_boxes=[boxes.tolist()], return_tensors="pt")
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        # transformers ≥5.x dropped reshaped_input_sizes from post_process_masks
        # entirely (not renamed) — it's no longer in `inputs` at all.
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
        )[0]                                            # (N, 1, H, W) bool
        ious = outputs.iou_scores.cpu().numpy().reshape(len(boxes))

        frame_masks: list[np.ndarray | None] = []
        for bi in range(len(boxes)):
            m = masks[bi, 0].numpy().astype(bool)
            frame_masks.append(m if ious[bi] >= 0.5 and m.sum() >= 16 else None)
        all_masks.append(frame_masks)

    return all_masks


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _scene_transform(extrinsic: np.ndarray) -> np.ndarray:
    """Same alignment visual_util.apply_scene_alignment uses (first cam + OpenGL flip)."""
    opengl = np.eye(4); opengl[1, 1] = -1; opengl[2, 2] = -1
    E0 = np.eye(4); E0[:3, :4] = extrinsic[0]
    return np.linalg.inv(E0) @ opengl


def _apply44(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def _up_from_camera_path(extrinsic: np.ndarray, T_align: np.ndarray) -> np.ndarray | None:
    """Gravity from the camera *trajectory* — the strong, tilt-proof estimate.

    A person walks the room at roughly constant height, so the least-variance
    axis of the camera centres is gravity. Crucially this uses only where the
    cameras *are*, not where they *point*: it recovers a tilt held across the
    whole capture (every shot angled down an aisle), which camera-0 alignment
    and per-frame orientation both bake into +Y and cannot undo.

    Returns None when the path can't support the estimate — fewer than 4
    frames, or the height axis isn't a clear minimum (panning in place, or a
    straight-line walk, where a horizontal axis is just as flat as height).
    """
    E = np.asarray(extrinsic, dtype=np.float64)
    if E.shape[0] < 4:
        return None
    R = E[:, :3, :3]
    t = E[:, :3, 3]
    cams_w = -np.einsum("sij,sj->si", np.transpose(R, (0, 2, 1)), t)   # centres, world
    cams_a = _apply44(T_align, cams_w)
    centred = cams_a - cams_a.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    # The least-variance axis is gravity only when the path genuinely samples a
    # 2-D horizontal patch: two comparably large spreads (the walk) with height
    # a clear minimum. A straight-line walk fails s[1] >= 0.25·s[0] (its second
    # axis is near zero, so the "minimum" is a horizontal direction, not
    # height); an in-place pan or an ambiguous height fails s[2] <= 0.5·s[1].
    if s[0] < 1e-9 or s[1] < 0.25 * s[0] or s[2] > 0.5 * s[1]:
        return None
    up = vt[2]
    if up[1] < 0:                                     # toward camera-0 up (= +Y)
        up = -up
    n = np.linalg.norm(up)
    return up / n if n > 1e-9 else None


def _estimate_up(extrinsic: np.ndarray, T_align: np.ndarray) -> tuple[np.ndarray, bool]:
    """Gravity ("up") in the aligned frame, plus whether it can be trusted.

    T_align only re-expresses the scene in camera-0's frame — it does NOT
    gravity-align. If the first frame was tilted (shooting down a data-center
    aisle, angling down at a desk), +Y in the aligned frame is nowhere near
    real up, and the old "+Y is up" leveling assumption breaks: past ~30° tilt
    it gives up, past ~50° it rotates a wall flat and calls it the floor.

    Primary estimate is the camera path (see _up_from_camera_path), which is
    accurate to ~1° at any tilt and is returned as *trusted* — the plane fit is
    then free to apply an arbitrarily large correction. When the path is too
    degenerate for that, fall back to the median camera up-axis: VGGT uses the
    OpenCV convention (x right, y DOWN, z forward), so a camera's world-space up
    is -R[1], and the median across frames beats per-frame noise. That fallback
    is biased for varied-yaw captures and blind to systematic pitch, so it is
    returned *untrusted* — the plane fit keeps a conservative cap on how far it
    will rotate off it.
    """
    path_up = _up_from_camera_path(extrinsic, T_align)
    if path_up is not None:
        return path_up, True

    R = np.asarray(extrinsic)[:, :3, :3]              # (S,3,3) world→cam, OpenCV
    up_world = -np.median(R[:, 1, :], axis=0)         # world up ≈ -(median cam +Y)
    n = np.linalg.norm(up_world)
    if n < 1e-6:
        return np.array([0.0, 1.0, 0.0]), False
    up = T_align[:3, :3] @ (up_world / n)             # into the aligned frame
    n = np.linalg.norm(up)
    return (up / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])), False


def _level_transform(pts: np.ndarray, up_prior: np.ndarray | None = None,
                     trusted: bool = False, iters: int = 400) -> np.ndarray:
    """Rotation that makes the floor plane horizontal (+Y up).

    RANSAC plane fit over the floor side of the cloud, then least-squares
    refined on the inliers, then rotated so the floor normal lands on +Y.

    `up_prior` is a gravity estimate (see _estimate_up) used to (a) pick the
    floor *band* — the points lowest along gravity, not lowest in raw Y — and
    (b) constrain candidate normals to be floor-like *relative to gravity*, so
    walls and clutter can't win. Defaults to +Y (the legacy assumption) when
    no prior is given.

    `trusted` says whether the prior is the camera-path estimate (accurate at
    any tilt) or a weaker fallback. When trusted, the correction is uncapped —
    the prior handles gross tilt, so a floor pitched 40-60° in the aligned
    frame levels correctly. When not, the correction is capped at 30°: rotating
    a wall flat off a shaky prior is worse than leaving a mild tilt, so big
    corrections are refused.
    """
    up_prior = np.array([0.0, 1.0, 0.0]) if up_prior is None else np.asarray(up_prior, float)
    npr = np.linalg.norm(up_prior)
    up_prior = up_prior / npr if npr > 1e-9 else np.array([0.0, 1.0, 0.0])

    h = pts @ up_prior                                # height along gravity
    band = pts[h <= np.percentile(h, 30)]            # floor side of the room
    if len(band) < 300:
        return np.eye(4)
    if len(band) > 30_000:
        band = band[:: len(band) // 30_000 + 1]

    eps = 0.02 * max(float(np.percentile(h, 98) - np.percentile(h, 2)), 1e-6)
    rng = np.random.default_rng(0)
    best_inliers = None

    for _ in range(iters):
        p0, p1, p2 = band[rng.choice(len(band), 3, replace=False)]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if abs(n @ up_prior) < 0.7:              # not floor-like vs. gravity
            continue
        inl = np.abs((band - p0) @ n) < eps
        if best_inliers is None or inl.sum() > best_inliers.sum():
            best_inliers = inl

    if best_inliers is None or best_inliers.sum() < 0.25 * len(band):
        return np.eye(4)

    support = band[best_inliers]
    centred = support - support.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[2]
    if normal @ up_prior < 0:                     # orient toward gravity
        normal = -normal

    up = np.array([0.0, 1.0, 0.0])
    cos_a = float(np.clip(normal @ up, -1.0, 1.0))
    angle = np.arccos(cos_a)
    max_corr = np.radians(80.0) if trusted else np.radians(30.0)
    if angle < np.radians(1.0) or angle > max_corr:   # level, or too far to trust
        return np.eye(4)

    axis = np.cross(normal, up)
    na = np.linalg.norm(axis)
    if na < 1e-9:                                 # normal already ∥ +Y
        return np.eye(4)
    axis /= na
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - cos_a) * (K @ K)
    T = np.eye(4); T[:3, :3] = R
    return T


def _yaw_of_points(pts_xz: np.ndarray) -> tuple[float, float]:
    """Principal-axis yaw (radians about +Y) and elongation ratio of an XZ footprint."""
    centred = pts_xz - np.median(pts_xz, axis=0)
    cov = np.cov(centred.T)
    if not np.all(np.isfinite(cov)):
        return 0.0, 1.0
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, np.argmax(evals)]
    yaw = float(np.arctan2(major[1], major[0]))
    ratio = float(np.sqrt(max(evals) / max(min(evals), 1e-9)))
    # Fold into [-90°, 90°) — a footprint axis has no sign
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    return yaw, ratio


def _snap_yaw(yaw: float, room_yaw: float, tol_deg: float = 15.0) -> float:
    """Snap to the room's 90° grid when close — data centers are axis-aligned."""
    rel = yaw - room_yaw
    snapped = round(rel / (np.pi / 2)) * (np.pi / 2)
    if abs(rel - snapped) <= np.radians(tol_deg):
        return room_yaw + snapped
    return yaw


# ── Instance clustering ───────────────────────────────────────────────────────

def _make_observation(label: str, score: float, frame: int,
                      pts: np.ndarray, cols: np.ndarray,
                      box: np.ndarray | None = None,
                      det_idx: int | None = None) -> dict:
    n_raw = len(pts)          # pre-subsample count ≈ visible mask area
    if len(pts) > _MAX_PTS_PER_OBS:
        sel = np.random.default_rng(0).choice(len(pts), _MAX_PTS_PER_OBS, replace=False)
        pts, cols = pts[sel], cols[sel]
    lo = np.percentile(pts, 5, axis=0)
    hi = np.percentile(pts, 95, axis=0)
    return {
        "label": label, "score": score, "frame": frame,
        "pts": pts, "cols": cols, "box": box, "n_raw": n_raw,
        "det_idx": det_idx,
        "center": (lo + hi) / 2.0,
        "diag": float(np.linalg.norm(hi - lo)),
    }


def cluster_observations(observations: list[dict]) -> list[dict]:
    """Greedy same-label clustering by robust-centre proximity (scale-free).

    Instances seen in only one frame need a high detection score to survive —
    cross-frame agreement is the accuracy filter.
    """
    from collections import defaultdict

    by_label: dict[str, list[dict]] = defaultdict(list)
    for o in observations:
        by_label[o["label"]].append(o)

    instances = []
    for label, obs in by_label.items():
        obs = sorted(obs, key=lambda o: -o["score"])
        used = [False] * len(obs)
        for i, a in enumerate(obs):
            if used[i]:
                continue
            group = [a]; used[i] = True
            for j in range(i + 1, len(obs)):
                if used[j]:
                    continue
                b = obs[j]
                thresh = 0.6 * max(a["diag"], b["diag"], 1e-6)
                if np.linalg.norm(a["center"] - b["center"]) < thresh:
                    group.append(b); used[j] = True

            frames = {g["frame"] for g in group}
            best = max(g["score"] for g in group)
            if len(frames) < 2 and best < 0.45:
                continue        # single-frame, low-confidence → likely spurious

            # The observation with the most mask pixels is the frame where
            # this object was seen largest/most head-on — used for the photo
            # thumbnail and photo-projected texturing.
            best_obs = max(group, key=lambda g: g.get("n_raw", len(g["pts"])))
            instances.append({
                "label": label,
                "score": best,
                "frames_seen": len(frames),
                "best_frame": best_obs["frame"],
                "best_box": best_obs.get("box"),
                # (frame, detection index) per observation — lets build_scene
                # look this instance's SAM masks back up for per-object
                # high-resolution TSDF re-fusion.
                "obs_refs": [(g["frame"], g["det_idx"]) for g in group
                             if g.get("det_idx") is not None],
                "pts": np.vstack([g["pts"] for g in group]),
                "cols": np.vstack([g["cols"] for g in group]),
            })
    return instances


# Equipment that stands on the floor by nature. Their scan points often start
# well above it — thin legs/stands rarely survive depth-edge filtering, so a
# table is frequently seen only from its tabletop up. These kinds get a much
# larger floor-snap tolerance; wall/desk-mounted kinds (monitor, shelfbox) and
# unrecognized labels keep the strict one.
_FLOOR_STANDING_KINDS = {"rack", "desk", "chair", "cabinet"}


def place_instance(inst: dict, floor_y: float, room_yaw: float,
                   floor_snap_tol: float) -> dict:
    """Robust pose from the aggregated multi-frame point set."""
    pts = inst["pts"]
    y0 = float(np.percentile(pts[:, 1], 2))
    y1 = float(np.percentile(pts[:, 1], 98))

    kind = _PREFAB_KINDS.get(inst.get("label", "").lower().strip())
    if kind in _FLOOR_STANDING_KINDS:
        floor_snap_tol = floor_snap_tol * 3.5

    yaw, ratio = _yaw_of_points(pts[:, [0, 2]])
    yaw = _snap_yaw(yaw if ratio >= 1.2 else room_yaw, room_yaw)

    # Oriented footprint: rotate XZ into the yaw frame, take robust extents
    c, s = np.cos(-yaw), np.sin(-yaw)
    R2 = np.array([[c, -s], [s, c]])
    xz = pts[:, [0, 2]] @ R2.T
    lo = np.percentile(xz, 2, axis=0)
    hi = np.percentile(xz, 98, axis=0)
    cx_r, cz_r = (lo + hi) / 2.0
    w, d = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    cx, cz = np.array([cx_r, cz_r]) @ R2      # back to world XZ

    # Snap to floor when the object's base is near it (racks, desks, chairs…)
    if y0 - floor_y < floor_snap_tol:
        y0 = floor_y
    h = max(y1 - y0, 0.02)

    return {
        "position": [float(cx), float(y0), float(cz)],   # bottom-centre
        "yaw": float(yaw),
        "size": [max(w, 0.02), h, max(d, 0.02)],
    }


# ── Metric calibration ────────────────────────────────────────────────────────

def rack_scale(instances_placed: list[dict], rack_height_m: float = RACK_HEIGHT_M) -> tuple[float, bool]:
    """Scene-units → metres scale from detected rack heights. (scale, calibrated)."""
    heights = [
        p["placement"]["size"][1]
        for p in instances_placed
        if "rack" in p["label"].lower() and p["placement"]["size"][1] > 1e-3
    ]
    if not heights:
        return 1.0, False
    scale = rack_height_m / float(np.median(heights))
    return (scale, True) if 0.05 < scale < 100.0 else (1.0, False)


# ── Prefab geometry (Y-up, origin at bottom-centre, vertex-coloured) ─────────

def _box(extents, translate, color) -> trimesh.Trimesh:
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(translate)
    b.visual.vertex_colors = np.tile(np.array([*color, 255], np.uint8), (len(b.vertices), 1))
    return b


def _ensure_material(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Bake plain vertex colors into a proper glTF material with a
    baseColorTexture before export. A mesh with only vertex_colors and no
    material exports with every primitive's "material" left null — valid per
    the glTF spec's default-material fallback, but rendered flat/uncolored by
    the editor's actual GLTFLoader + WebGL pipeline instead of the intended
    color. ColorVisuals.to_texture() is trimesh's own standard fix: it bakes
    the vertex colors into a small texture image and emits an explicit
    material referencing it, which every glTF consumer renders consistently.

    to_texture()'s SimpleMaterial defaults diffuse/ambient/specular to a flat
    40% gray [102,102,102,255] (exports as baseColorFactor [0.4,0.4,0.4,1]) —
    that multiplies against the (correctly-colored) texture at render time,
    cutting brightness by ~60% scene-wide. Force it to white so the texture
    alone carries the color, undiminished.

    Exported as an explicit PBR material because SimpleMaterial's glTF export
    omits metallicFactor entirely, and the glTF spec defaults an unspecified
    metallicFactor to 1.0 — fully metallic. A metallic surface has almost no
    diffuse response, so every mesh rendered as dark rough metal regardless
    of its texture. metallicFactor must be written out as an explicit 0.
    """
    if mesh.visual.kind == "vertex":
        mesh.visual = mesh.visual.to_texture()
        pbr = mesh.visual.material.to_pbr()
        pbr.baseColorFactor = [255, 255, 255, 255]
        pbr.metallicFactor = 0.0
        pbr.roughnessFactor = 0.95
        mesh.visual.material = pbr
    return mesh


def _prefab_rack(w, h, d, tint) -> trimesh.Trimesh:
    # A rack's door sits on its narrow (600 mm) face. The local X axis is the
    # footprint's major axis, so when X is the longer side it is the rack's
    # depth: build the canonical rack and spin it 90°.
    if w > d:
        mesh = _prefab_rack(d, h, w, tint)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        return mesh
    body = np.array([38, 40, 44]) * 0.6 + np.asarray(tint) * 0.4
    frame_c = body * 0.7
    parts = [
        _box([w, h * 0.985, d], [0, h * 0.985 / 2 + h * 0.015, 0], body.astype(int)),
        _box([w, h * 0.015, d], [0, h * 0.015 / 2, 0], frame_c.astype(int)),        # plinth
        _box([w * 0.92, h * 0.9, d * 0.02], [0, h * 0.06 + h * 0.9 / 2, d / 2], (15, 16, 18)),  # front door
    ]
    # Vent slats on the front door
    n_slats = max(int(h / 0.12), 4)
    for i in range(n_slats):
        y = h * 0.08 + (h * 0.86) * (i + 0.5) / n_slats
        parts.append(_box([w * 0.8, h * 0.012, d * 0.005], [0, y, d / 2 * 1.02], (70, 74, 80)))
    return trimesh.util.concatenate(parts)


def _prefab_desk(w, h, d, tint) -> trimesh.Trimesh:
    top_t = min(0.04 * h / 0.75 if h > 0 else 0.03, h * 0.2)
    leg = min(w, d) * 0.06
    parts = [_box([w, top_t, d], [0, h - top_t / 2, 0], tint)]
    for sx in (-1, 1):
        for sz in (-1, 1):
            parts.append(_box([leg, h - top_t, leg],
                              [sx * (w / 2 - leg), (h - top_t) / 2, sz * (d / 2 - leg)],
                              (90, 90, 95)))
    return trimesh.util.concatenate(parts)


def _prefab_monitor(w, h, d, tint) -> trimesh.Trimesh:
    panel_h = h * 0.7
    parts = [
        _box([w, panel_h, max(d * 0.15, 0.01)], [0, h - panel_h / 2, 0], (18, 18, 22)),
        _box([w * 0.12, h * 0.25, max(d * 0.3, 0.02)], [0, h * 0.18, 0], (60, 60, 66)),
        _box([w * 0.45, h * 0.04, max(d * 0.8, 0.04)], [0, h * 0.02, 0], (60, 60, 66)),
    ]
    return trimesh.util.concatenate(parts)


def _prefab_chair(w, h, d, tint) -> trimesh.Trimesh:
    seat_y = h * 0.45
    parts = [
        _box([w, h * 0.06, d], [0, seat_y, 0], tint),                                   # seat
        _box([w, h * 0.5, d * 0.12], [0, seat_y + h * 0.28, -d / 2 + d * 0.06], tint),  # back
        _box([w * 0.1, seat_y, w * 0.1], [0, seat_y / 2, 0], (50, 50, 55)),             # column
        _box([w * 0.9, h * 0.03, d * 0.9], [0, h * 0.015, 0], (50, 50, 55)),            # base
    ]
    return trimesh.util.concatenate(parts)


def _prefab_cabinet(w, h, d, tint) -> trimesh.Trimesh:
    parts = [
        _box([w, h, d], [0, h / 2, 0], tint),
        _box([w * 0.9, h * 0.9, d * 0.02], [0, h / 2, d / 2], (np.asarray(tint) * 0.8).astype(int)),
    ]
    return trimesh.util.concatenate(parts)


_PREFAB_BUILDERS = {
    "rack": _prefab_rack,
    "desk": _prefab_desk,
    "monitor": _prefab_monitor,
    "chair": _prefab_chair,
    "cabinet": _prefab_cabinet,
    "shelfbox": _prefab_cabinet,
}


def build_object_mesh(label: str, size: list[float], pts: np.ndarray,
                      cols: np.ndarray, yaw: float, position: list[float],
                      scale: float, full_mesh: trimesh.Trimesh | None = None,
                      world_position: list[float] | None = None) -> tuple[trimesh.Trimesh, str]:
    """Real scanned geometry first — cut the object's region directly out of
    the dense full-scene mesh when available, falling back to an isolated
    per-object Poisson reconstruction, then a parametric prefab for
    recognized equipment types, then a plain box. Scan-derived geometry is
    preferred over prefabs whenever there's enough real data to build it —
    prefabs exist for when there isn't, not as the default for anything
    recognizable.

    Returns (mesh, source) where source is one of scan-dense/scan-poisson/prefab/box.
    """
    w, h, d = size
    tint = np.median(cols, axis=0).clip(0, 255).astype(int) if len(cols) else np.array([128, 128, 128])

    mesh, source = None, None
    if full_mesh is not None and world_position is not None:
        mesh = _crop_dense_mesh(full_mesh, np.asarray(world_position), yaw, size)
        source = "scan-dense"
    if mesh is None:
        mesh = _scan_cutout_mesh(pts, cols, yaw, position, scale, size)
        source = "scan-poisson"
    if mesh is None:
        kind = _PREFAB_KINDS.get(label.lower().strip())
        if kind is not None:
            mesh, source = _PREFAB_BUILDERS[kind](w, h, d, tuple(tint)), "prefab"
    if mesh is None:
        mesh, source = _box([w, h, d], [0, h / 2, 0], tuple(tint)), "box"

    return _ensure_material(mesh), source


def _crop_to_local_box(mesh: trimesh.Trimesh, size: list[float],
                       pad: float = 1.15) -> trimesh.Trimesh | None:
    """Given a mesh already in the object's local frame (yaw-derotated,
    bottom-centre at origin), crop to its padded bounding box and keep the
    largest surviving connected component. Returns None if too little of the
    mesh survives — e.g. a Poisson hallucination bubble or a dense-mesh crop
    that missed the object entirely."""
    w, h, d = size
    keep = trimesh.bounds.contains(
        np.array([[-w / 2 * pad, -h * 0.1, -d / 2 * pad],
                  [w / 2 * pad, h * pad, d / 2 * pad]]),
        mesh.vertices,
    )
    faces_keep = keep[mesh.faces].all(axis=1)
    if not faces_keep.any():
        return None
    mesh = mesh.copy()
    mesh.update_faces(faces_keep)
    mesh.remove_unreferenced_vertices()
    # Low bar deliberately: real scan geometry (however sparse) is preferred
    # over a parametric prefab or box fallback, so this only rejects results
    # too small to be a real surface at all, not merely "not much."
    if len(mesh.faces) < 4:
        return None
    comps = mesh.split(only_watertight=False)
    if len(comps) > 1:
        mesh = max(comps, key=lambda m: len(m.faces))
    return mesh


def _crop_dense_mesh(full_mesh: trimesh.Trimesh, world_center: np.ndarray,
                     yaw: float, size: list[float]) -> trimesh.Trimesh | None:
    """Cut this object's region directly out of the full-scene dense mesh
    (already in final world coordinates), de-rotating and re-centring to the
    same bottom-centre-at-origin local convention every object glb uses."""
    local_v = full_mesh.vertices - world_center
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    local_v = local_v @ R.T
    local_mesh = trimesh.Trimesh(
        vertices=local_v, faces=full_mesh.faces,
        vertex_colors=full_mesh.visual.vertex_colors, process=False,
    )
    # Tighter pad than the Poisson path's default (1.15): that padding exists
    # to give Poisson's characteristic bubble-hallucination past sparse data
    # room to be cropped without cutting into real geometry. A dense,
    # well-supported TSDF mesh doesn't bubble the same way, so a generous pad
    # here just makes the visible mesh noticeably bigger than the hover
    # bounding box (built from the exact detected `size`) instead of matching
    # it. 1.08, not tighter still (e.g. 1.03): real objects sit in sparsely-
    # triangulated regions of the dense mesh sometimes, and too little slack
    # means legitimately-present geometry gets rejected, falling back to a
    # prefab/box when real data was actually there to use.
    return _crop_to_local_box(local_mesh, size, pad=1.08)


def _scan_cutout_mesh(pts, cols, yaw, position, scale, size) -> trimesh.Trimesh | None:
    """Poisson-mesh the instance's own scan points (local frame: yaw-derotated,
    bottom-centre at origin). Returns None when there's too little data."""
    # Lowered from 800: this is the fallback of last resort before a prefab/
    # box, so it should only give up when Poisson genuinely has too little to
    # work with, not merely less than an arbitrary comfortable amount.
    if len(pts) < 300:
        return None
    try:
        from src.reconstruction.tsdf_fusion import _run_poisson_subprocess
        local = (pts * scale) - np.asarray(position)
        c, s = np.cos(-yaw), np.sin(-yaw)
        R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        local = local @ R.T
        verts, faces, vcols = _run_poisson_subprocess(
            local.astype(np.float64), cols.astype(np.uint8), 7
        )
        if len(verts) == 0:
            return None
        mesh = trimesh.Trimesh(
            vertices=verts, faces=faces,
            vertex_colors=(vcols[:, :3] * 255).clip(0, 255).astype(np.uint8) if len(vcols) else None,
            process=False,
        )
        return _crop_to_local_box(mesh, size)  # Poisson hallucinates a bubble past the real extent
    except Exception as exc:
        print(f"[SceneBuilder] Scan cut-out failed ({exc}); using box fallback.")
        return None


def _fuse_object_hires(predictions: dict, masks: list, obs_refs: list,
                       pl: dict, size_m: list[float], pos_m: list[float],
                       scale: float, offset: np.ndarray, T_total: np.ndarray,
                       conf_thres: float) -> trimesh.Trimesh | None:
    """Re-fuse one object in its own small, fine TSDF volume, restricted to
    its own SAM-masked pixels — instead of cropping the room-scale mesh.

    The room mesh is voxel-capped for memory (~3 cm voxels in a 6 m room), so
    an object cropped out of it is only a handful of voxels across — the
    "blocky" look. A grid fit to just the object's bounding box brings voxels
    down to ~1-2 cm at the same memory budget, and the include-mask keeps
    neighbouring geometry from bleeding in.

    Returns the mesh in the standard local convention (yaw-derotated,
    bottom-centre at origin, metres), or None to fall through to the coarse
    crop/Poisson/prefab chain.
    """
    from src.reconstruction.tsdf_fusion import TSDFConfig, fuse_tsdf_raw_mesh

    S, dH, dW = predictions["depth_conf"].shape
    include = np.zeros((S, dH, dW), dtype=bool)
    n_px = 0
    for fi, bi in obs_refs:
        if fi >= S or fi >= len(masks) or bi >= len(masks[fi]) or masks[fi][bi] is None:
            continue
        m = cv2.resize(masks[fi][bi].astype(np.uint8), (dW, dH),
                       interpolation=cv2.INTER_NEAREST)
        # Slight dilation: SAM edges are conservative, and TSDF needs a bit of
        # surround for a clean zero-crossing at the silhouette.
        m = cv2.dilate(m, np.ones((5, 5), np.uint8)).astype(bool)
        include[fi] |= m
        n_px += int(m.sum())
    if n_px < 500:
        return None

    # Object bounding box (leveled scene units, padded, floor to top) → raw
    # VGGT coordinates, where fusion runs.
    w, h, d = pl["size"]
    yaw = pl["yaw"]
    c, s = np.cos(-yaw), np.sin(-yaw)
    R_yaw = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    pad = 1.15
    corners_local = np.array([
        [sx * w / 2 * pad, y, sz * d / 2 * pad]
        for sx in (-1, 1) for sz in (-1, 1) for y in (-0.05 * h, h * 1.1)
    ])
    corners_leveled = corners_local @ R_yaw + np.asarray(pl["position"])
    corners_raw = _apply44(np.linalg.inv(T_total), corners_leveled)
    bounds = (corners_raw.min(axis=0), corners_raw.max(axis=0))

    try:
        mesh = fuse_tsdf_raw_mesh(
            predictions,
            TSDFConfig(conf_percentile=conf_thres, auto_resolution=96,
                       max_dim=112, min_cluster_faces=0),
            include_mask=include,
            bounds=bounds,
        )
    except ValueError:
        return None

    # Raw → final world → the shared local convention, then the same cleanup
    # crop every other scan path gets.
    mesh.apply_transform(T_total)
    mesh.apply_scale(scale)
    mesh.apply_translation(offset)
    local_v = (mesh.vertices - np.asarray(pos_m)) @ R_yaw.T
    local_mesh = trimesh.Trimesh(
        vertices=local_v, faces=mesh.faces,
        vertex_colors=mesh.visual.vertex_colors, process=False,
    )
    return _crop_to_local_box(local_mesh, size_m, pad=1.08)


def _save_object_photo(image_paths: list[str], inst: dict, scene_dir: str,
                       obj_id: str, max_px: int = 320) -> str | None:
    """Crop the object's detection box out of its best frame (the full-res
    source photo, not the 512px preprocessed copy) and save it next to the
    object's glb, so the editor's inspector can show what was actually
    detected. Returns the scene-relative path, or None when unavailable."""
    fi, box = inst.get("best_frame"), inst.get("best_box")
    if fi is None or box is None or fi >= len(image_paths):
        return None
    img = cv2.imread(image_paths[fi])
    if img is None:
        return None
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    pw, ph = 0.08 * (x2 - x1), 0.08 * (y2 - y1)      # a little context around the box
    x1, y1 = int(max(x1 - pw, 0)), int(max(y1 - ph, 0))
    x2, y2 = int(min(x2 + pw, W)), int(min(y2 + ph, H))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = img[y1:y2, x1:x2]
    s = max_px / max(crop.shape[:2])
    if s < 1.0:
        crop = cv2.resize(crop, (max(int(crop.shape[1] * s), 1), max(int(crop.shape[0] * s), 1)))
    rel = f"objects/{obj_id}.jpg"
    cv2.imwrite(os.path.join(scene_dir, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return rel


# ── Duplicate grouping (asset reuse) ─────────────────────────────────────────

def _group_duplicates(placed: list[dict], size_rel_tol: float = 0.15) -> list[list[int]]:
    """Partition placed instances into asset-reuse groups: same label with a
    matching metric footprint (all three extents agree within size_rel_tol).
    Returns a list of groups, each a list of indices into `placed`.

    Duplicates — a row of identical racks, matching office chairs — can then
    share one built asset: only the geometry is shared, while each member keeps
    its own position/yaw and a per-instance model_scale. The expensive step
    (scan re-fusion today, generative image-to-3D later) then runs once per
    group instead of once per instance.

    Size agreement is per-axis and *unsorted* on purpose (not on sorted
    extents): two objects whose width and depth are swapped sit 90° apart, so
    reusing one asset between them would render the shared mesh rotated wrong.
    A swap just leaves them in separate groups — each keeps its own mesh, a
    safe miss rather than a visible glitch.
    """
    from collections import defaultdict

    by_label: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(placed):
        by_label[p["label"].lower().strip()].append(i)

    groups: list[list[int]] = []
    for idxs in by_label.values():
        # Bigger, cleaner instances seed first — they make the better shared
        # asset, and greedy growth around a strong seed is more stable.
        idxs = sorted(idxs, key=lambda i: -float(np.prod(placed[i]["placement"]["size"])))
        used = [False] * len(idxs)
        for a in range(len(idxs)):
            if used[a]:
                continue
            sa = np.asarray(placed[idxs[a]]["placement"]["size"], float)
            group = [idxs[a]]
            used[a] = True
            for b in range(a + 1, len(idxs)):
                if used[b]:
                    continue
                sb = np.asarray(placed[idxs[b]]["placement"]["size"], float)
                if np.all(np.abs(sb - sa) / np.maximum(sa, 1e-6) <= size_rel_tol):
                    group.append(idxs[b])
                    used[b] = True
            groups.append(group)
    return groups


# ── Main entry point ──────────────────────────────────────────────────────────

def build_scene(
    predictions: dict,
    image_paths: list[str],
    dets_2d: list[dict],
    out_root: str,
    scene_name: str,
    conf_thres: float = 20.0,
    mesh_resolution: int = 256,
    rack_height_m: float = RACK_HEIGHT_M,
    manual_scale: float | None = None,
    abort_check: Callable[[], None] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict:
    """Build an editable scene directory. Returns the scene.json dict.

    abort_check, if given, is called between stages and per-frame/per-object
    inside the two slowest loops (SAM segmentation, mesh building) — it should
    raise to cancel a run in progress (see app.py's _abort_if_cancelled).
    progress_cb(fraction, description), if given, is called at the same
    checkpoints to report 0-1 progress through this function specifically.
    """
    from src.reconstruction.tsdf_fusion import TSDFConfig, fuse_tsdf_raw_mesh
    from visual_util import depth_edge

    depth = predictions["depth"]                      # (S, H, W, 1)
    depth_conf = predictions["depth_conf"]            # (S, H, W)
    images = predictions["images"]                    # (S, 3, H, W)
    extrinsic = predictions["extrinsic"]              # (S, 3, 4)
    intrinsic = predictions["intrinsic"]              # (S, 3, 3)
    world_pts = predictions["world_points_from_depth"]
    S, dH, dW = depth_conf.shape

    conf = depth_conf.copy()
    conf[depth_edge(depth[..., 0], rtol=0.03)] = 0.0
    valid = np.isfinite(conf) & (conf > 1e-5)
    conf_threshold = float(np.percentile(conf[valid], conf_thres)) if valid.any() and conf_thres > 0 else 0.0
    good_px = valid & (conf >= conf_threshold) & np.isfinite(world_pts).all(axis=-1)

    # ── 1. SAM masks + per-frame observations ────────────────────────────────
    print("[SceneBuilder] Segmenting detections with SAM…")
    masks = segment_detections(image_paths, dets_2d, abort_check=abort_check, progress_cb=progress_cb)

    T_align = _scene_transform(extrinsic)
    exclude_mask = np.zeros((S, dH, dW), dtype=bool)
    observations: list[dict] = []

    for fi, frame in enumerate(dets_2d):
        if fi >= S:
            break
        color_hwc = np.transpose(images[fi], (1, 2, 0))
        for bi, (label, score) in enumerate(zip(frame["labels"], frame["scores"])):
            if bi >= len(masks[fi]) or masks[fi][bi] is None:
                continue
            m_small = cv2.resize(masks[fi][bi].astype(np.uint8), (dW, dH),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            sel = m_small & good_px[fi]
            if sel.sum() < 30:
                continue
            pts = _apply44(T_align, world_pts[fi][sel])
            cols = (color_hwc[sel] * 255).clip(0, 255).astype(np.uint8)
            observations.append(_make_observation(str(label), float(score), fi, pts, cols,
                                                  box=np.asarray(frame["boxes"][bi]),
                                                  det_idx=bi))
            exclude_mask[fi] |= cv2.dilate(m_small.astype(np.uint8),
                                           np.ones((3, 3), np.uint8)).astype(bool)

    print(f"[SceneBuilder] {len(observations)} observations across {S} frames.")

    # ── 2. Level the floor, establish room frame ─────────────────────────────
    glob = world_pts[good_px]
    if len(glob) > 200_000:
        glob = glob[:: len(glob) // 200_000 + 1]
    glob = _apply44(T_align, glob)

    up_prior, up_trusted = _estimate_up(extrinsic, T_align)
    T_level = _level_transform(glob, up_prior=up_prior, trusted=up_trusted)
    T_total = T_level @ T_align
    glob = _apply44(T_level, glob)
    for o in observations:
        o["pts"] = _apply44(T_level, o["pts"])
        o["center"] = _apply44(T_level, o["center"][None])[0]

    floor_y = float(np.percentile(glob[:, 1], 3))
    ceil_y = float(np.percentile(glob[:, 1], 97))
    room_h = ceil_y - floor_y
    room_yaw, _ = _yaw_of_points(glob[:, [0, 2]])

    # ── 3. Cluster into instances, place them ────────────────────────────────
    instances = cluster_observations(observations)
    print(f"[SceneBuilder] {len(instances)} object instance(s) after clustering.")

    placed = []
    for inst in instances:
        placement = place_instance(inst, floor_y, room_yaw,
                                   floor_snap_tol=room_h * 0.12)
        placed.append({**inst, "placement": placement})

    # ── 4. Metric scale ───────────────────────────────────────────────────────
    if manual_scale:
        scale, calibrated = float(manual_scale), True
    else:
        scale, calibrated = rack_scale(placed, rack_height_m)
    print(f"[SceneBuilder] Scale: ×{scale:.3f} ({'rack-calibrated' if calibrated else 'uncalibrated'})")

    floor_y_m = floor_y * scale
    cx0 = float(np.median(glob[:, 0])) * scale
    cz0 = float(np.median(glob[:, 2])) * scale
    offset = np.array([-cx0, -floor_y_m, -cz0])       # floor → y=0, centre → origin

    # Every object is cut directly out of one dense, full-scene mesh when
    # possible — including recognized equipment types, which now only fall
    # back to a parametric prefab when there isn't enough real scan data to
    # build from. Real geometry from every frame's contribution, not just the
    # frames this one object happened to be masked in. Skipped entirely only
    # when there's nothing placed at all.
    needs_dense_mesh = len(placed) > 0
    full_mesh = None
    if needs_dense_mesh:
        if abort_check is not None:
            abort_check()
        if progress_cb is not None:
            progress_cb(0.5, "Fusing full-detail mesh for object cropping…")
        print("[SceneBuilder] Fusing full-detail mesh for object cropping…")
        full_mesh = fuse_tsdf_raw_mesh(
            predictions,
            TSDFConfig(conf_percentile=conf_thres, auto_resolution=mesh_resolution),
        )
        # Sharpen colors while still in VGGT coords: single-best-view sampling
        # replaces the TSDF's all-frames voxel average, which blurs/ghosts
        # wherever frames disagree slightly. Objects cropped from this mesh
        # inherit the sharper colors automatically.
        recolor_best_view(full_mesh, predictions, label="full mesh")
        full_mesh.apply_transform(T_total)
        full_mesh.apply_scale(scale)
        full_mesh.apply_translation(offset)

    # ── 5. Write scene directory ─────────────────────────────────────────────
    scene_dir = os.path.join(out_root, scene_name)
    if os.path.isdir(scene_dir):
        shutil.rmtree(scene_dir)
    os.makedirs(os.path.join(scene_dir, "objects"))

    objects_json = []
    n_placed = len(placed)
    for i, p in enumerate(placed):
        if abort_check is not None:
            abort_check()
        if progress_cb is not None:
            progress_cb(0.55 + 0.4 * i / max(n_placed, 1), f"Building object meshes: {i}/{n_placed}")
        pl = p["placement"]
        size_m = [s * scale for s in pl["size"]]
        pos_m = (np.array(pl["position"]) * scale + offset).tolist()
        obj_id = f"obj_{i:03d}"

        # First choice: per-object high-resolution re-fusion from this
        # object's own SAM masks (~1-2 cm voxels vs the room mesh's ~3 cm) —
        # the coarse-crop/Poisson/prefab chain is the fallback.
        mesh, source = None, None
        if p.get("obs_refs"):
            mesh = _fuse_object_hires(predictions, masks, p["obs_refs"], pl,
                                      size_m, pos_m, scale, offset, T_total,
                                      conf_thres)
            if mesh is not None:
                mesh, source = _ensure_material(mesh), "scan-hires"
        if mesh is None:
            mesh, source = build_object_mesh(
                p["label"], size_m, p["pts"], p["cols"], pl["yaw"],
                (np.array(pl["position"]) * scale).tolist(), scale,
                full_mesh=full_mesh, world_position=pos_m,
            )

        # Scan-derived meshes get the real photo projected on as a texture
        # when their best frame saw enough of them. All scan paths share the
        # same local convention (yaw-derotated, bottom-centre at origin), so
        # one inverse transform maps local verts back to VGGT camera space:
        # local → leveled scene units → undo T_total.
        if source in ("scan-hires", "scan-dense", "scan-poisson") and p.get("best_frame") is not None:
            fi = p["best_frame"]
            yaw = pl["yaw"]
            c_, s_ = np.cos(-yaw), np.sin(-yaw)
            R_yaw = np.array([[c_, 0, -s_], [0, 1, 0], [s_, 0, c_]])
            leveled = (np.asarray(mesh.vertices) @ R_yaw) / scale + np.asarray(pl["position"])
            verts_vggt = _apply44(np.linalg.inv(T_total), leveled)
            if apply_photo_texture(mesh, verts_vggt,
                                   np.transpose(images[fi], (1, 2, 0)),
                                   extrinsic[fi], intrinsic[fi], depth[fi, ..., 0]):
                source += "+photo"

        trimesh.Scene([mesh]).export(os.path.join(scene_dir, "objects", f"{obj_id}.glb"))
        photo_rel = _save_object_photo(image_paths, p, scene_dir, obj_id)

        objects_json.append({
            "id": obj_id,
            "label": p["label"],
            "score": round(p["score"], 3),
            "frames_seen": p["frames_seen"],
            "position": [round(v, 4) for v in pos_m],
            "yaw": round(pl["yaw"], 5),
            "size": [round(v, 4) for v in size_m],
            "source": source,
            "glb": f"objects/{obj_id}.glb",
            **({"photo": photo_rel} if photo_rel else {}),
        })

    # ── 6. Carved background + floor/ceiling/wall fill ───────────────────────
    if progress_cb is not None:
        progress_cb(0.95, "Fusing background mesh…")
    print("[SceneBuilder] Fusing carved background mesh…")
    bg = fuse_tsdf_raw_mesh(
        predictions,
        TSDFConfig(conf_percentile=conf_thres, auto_resolution=mesh_resolution),
        exclude_mask=exclude_mask,
    )
    recolor_best_view(bg, predictions, label="background")
    bg.apply_transform(T_total)
    bg.apply_scale(scale)
    bg.apply_translation(offset)

    lo = np.percentile(glob, 2, axis=0) * scale + offset
    hi = np.percentile(glob, 98, axis=0) * scale + offset

    # Named (not anonymous) so the editor can find and toggle the walls/ceiling
    # independently — they enclose the room and can block the camera's view
    # of the interior otherwise.
    room_scene = trimesh.Scene()
    room_scene.add_geometry(_ensure_material(bg), geom_name="background")

    floor_cols = _floor_color(glob, floor_y, images, good_px, world_pts, T_total)
    slab = _box([float(hi[0] - lo[0]), 0.02, float(hi[2] - lo[2])],
                [float((lo[0] + hi[0]) / 2), -0.011, float((lo[2] + hi[2]) / 2)],
                floor_cols)
    room_scene.add_geometry(_ensure_material(slab), geom_name="floor")

    # Ceiling: only synthesise when near-ceiling points actually spread across
    # the room (see _has_ceiling_support) — otherwise ceil_y is just "top of
    # the tallest observed object" and a slab there would be wrong.
    ceil_final_y = room_h * scale
    ceil_cols = floor_cols
    if _has_ceiling_support(glob, ceil_y, room_h):
        ceil_cols = _ceiling_color(glob, ceil_y, images, good_px, world_pts, T_total)
        ceiling = _box([float(hi[0] - lo[0]), 0.02, float(hi[2] - lo[2])],
                       [float((lo[0] + hi[0]) / 2), ceil_final_y + 0.011, float((lo[2] + hi[2]) / 2)],
                       ceil_cols)
        room_scene.add_geometry(_ensure_material(ceiling), geom_name="ceiling")

    # Walls: thin flat planes at the room's bounding rectangle, floor to
    # ceiling. Simplification: fills the whole rectangle regardless of real
    # doorways/openings — same simplification the floor slab already makes.
    # Named wall_0..wall_3 (not merged into one mesh) so the editor can toggle
    # them independently of the floor/ceiling/background.
    wall_cols = tuple(np.array([floor_cols, ceil_cols]).mean(axis=0).astype(int))
    wt = 0.02
    x0, _, z0 = lo; x1, _, z1 = hi
    walls = [
        _box([x1 - x0, ceil_final_y, wt], [(x0 + x1) / 2, ceil_final_y / 2, z0 - wt / 2], wall_cols),
        _box([x1 - x0, ceil_final_y, wt], [(x0 + x1) / 2, ceil_final_y / 2, z1 + wt / 2], wall_cols),
        _box([wt, ceil_final_y, z1 - z0], [x0 - wt / 2, ceil_final_y / 2, (z0 + z1) / 2], wall_cols),
        _box([wt, ceil_final_y, z1 - z0], [x1 + wt / 2, ceil_final_y / 2, (z0 + z1) / 2], wall_cols),
    ]
    for i, wall in enumerate(walls):
        room_scene.add_geometry(_ensure_material(wall), geom_name=f"wall_{i}")

    room_scene.export(os.path.join(scene_dir, "background.glb"))

    scene_json = {
        "name": scene_name,
        "metric": calibrated,
        "scale": round(scale, 5),
        "room": {
            "width": round(float(hi[0] - lo[0]), 2),
            "depth": round(float(hi[2] - lo[2]), 2),
            "height": round(room_h * scale, 2),
        },
        "room_yaw": round(room_yaw, 5),
        "objects": objects_json,
    }
    with open(os.path.join(scene_dir, "scene.json"), "w") as f:
        json.dump(scene_json, f, indent=2)

    print(f"[SceneBuilder] Scene written to {scene_dir}")
    return scene_json


def _floor_color(glob, floor_y, images, good_px, world_pts, T_total) -> tuple:
    """Median colour of near-floor points (fallback: neutral gray)."""
    try:
        cols = []
        S = images.shape[0]
        for fi in range(S):
            sel = good_px[fi]
            if not sel.any():
                continue
            pts = _apply44(T_total, world_pts[fi][sel])
            near = pts[:, 1] < floor_y + 0.1 * max(np.ptp(glob[:, 1]), 1e-6)
            if near.any():
                chw = np.transpose(images[fi], (1, 2, 0))
                cols.append((chw[sel][near] * 255).astype(np.uint8))
        if cols:
            return tuple(np.median(np.vstack(cols), axis=0).astype(int))
    except Exception:
        pass
    return (128, 126, 122)


def _ceiling_color(glob, ceil_y, images, good_px, world_pts, T_total) -> tuple:
    """Median colour of near-ceiling points (fallback: neutral light gray)."""
    try:
        cols = []
        S = images.shape[0]
        for fi in range(S):
            sel = good_px[fi]
            if not sel.any():
                continue
            pts = _apply44(T_total, world_pts[fi][sel])
            near = pts[:, 1] > ceil_y - 0.1 * max(np.ptp(glob[:, 1]), 1e-6)
            if near.any():
                chw = np.transpose(images[fi], (1, 2, 0))
                cols.append((chw[sel][near] * 255).astype(np.uint8))
        if cols:
            return tuple(np.median(np.vstack(cols), axis=0).astype(int))
    except Exception:
        pass
    return (235, 235, 230)


def _has_ceiling_support(glob: np.ndarray, ceil_y: float, room_h: float,
                         min_coverage: float = 0.35) -> bool:
    """True only when near-ceiling points spread across most of the room's XZ
    footprint. ceil_y is defined as a height percentile, so *some* points are
    always "near" it by construction — that alone doesn't mean a ceiling was
    actually seen. If the camera never looked up, ceil_y is really just "top
    of the tallest observed object" (e.g. a rack), and those points cluster
    over that one object's footprint rather than spreading across the room.
    """
    band = glob[glob[:, 1] > ceil_y - 0.08 * max(room_h, 1e-6)]
    if len(band) < 50:
        return False
    xz = band[:, [0, 2]]
    lo_xz = np.percentile(glob[:, [0, 2]], 2, axis=0)
    hi_xz = np.percentile(glob[:, [0, 2]], 98, axis=0)
    grid_n = 8
    cell = (hi_xz - lo_xz) / grid_n
    if not np.all(cell > 1e-9):
        return False
    idx = np.floor((xz - lo_xz) / cell).clip(0, grid_n - 1).astype(int)
    occupied = len(set(map(tuple, idx)))
    return (occupied / (grid_n * grid_n)) >= min_coverage

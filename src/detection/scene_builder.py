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
  6. The background is re-fused with object pixels carved out. Observed room
     surfaces are preserved; backing slabs fill only weakly supported boundaries.
     The aligned raw VGGT point cloud is exported as an editor reference layer.

Output layout (scenes/<name>/):
  scene.json        — object list with positions/yaw/size in metres
  background.glb    — scan-preserving room mesh with coverage-aware backing
  raw_scan.glb      — aligned/metric VGGT points used as the editor overlay
  objects/<id>.glb  — per-object geometry, origin at bottom-centre
"""
from __future__ import annotations

import io
import json
import os
import shutil
from typing import Callable

import cv2
import numpy as np
import trimesh

from src.detection.dedup import box_iou
from src.detection.photo_projection import apply_photo_texture, project_to_frame, recolor_best_view

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
    R = np.asarray(extrinsic)[:, :3, :3]              # (S,3,3) world→cam, OpenCV
    up_world = -np.median(R[:, 1, :], axis=0)         # world up ≈ -(median cam +Y)
    n = np.linalg.norm(up_world)
    if n < 1e-6:
        camera_up = np.array([0.0, 1.0, 0.0])
    else:
        camera_up = T_align[:3, :3] @ (up_world / n)  # into the aligned frame
        n = np.linalg.norm(camera_up)
        camera_up = camera_up / n if n > 1e-6 else np.array([0.0, 1.0, 0.0])

    path_up = _up_from_camera_path(extrinsic, T_align)
    if path_up is not None:
        # A compact walkthrough can accidentally sample a vertical plane: e.g.
        # moving left/right while pose drift changes the estimated camera
        # height. Its least-variance axis is then a WALL normal, not gravity,
        # even though the trajectory SVD otherwise looks well-conditioned.
        # Camera orientation is pitch-biased but still a reliable hemisphere
        # check; reject the path estimate when the two differ by more than 45°.
        agreement = float(np.clip(path_up @ camera_up, -1.0, 1.0))
        if agreement >= np.cos(np.radians(45.0)):
            return path_up, True
        print(
            f"[SceneBuilder] Rejecting camera-path up "
            f"({np.degrees(np.arccos(agreement)):.1f}° from camera up)."
        )

    return camera_up, False


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
                      det_idx: int | None = None,
                      reflector_inside_ratio: float = 0.0,
                      reflector_labels: list[str] | None = None) -> dict:
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
        "reflector_inside_ratio": float(reflector_inside_ratio),
        "reflector_labels": list(reflector_labels or []),
        "center": (lo + hi) / 2.0,
        "diag": float(np.linalg.norm(hi - lo)),
    }


def _instance_from_observations(group: list[dict]) -> dict:
    """Consolidate one physical object's observations and choose its label by
    cross-frame evidence rather than whichever phrase won a single frame."""
    from collections import defaultdict

    by_label: dict[str, list[dict]] = defaultdict(list)
    for obs in group:
        by_label[obs["label"]].append(obs)

    def label_evidence(item):
        _, obs = item
        return (
            len({o["frame"] for o in obs}),
            sum(float(o["score"]) for o in obs),
            max(float(o["score"]) for o in obs),
            sum(int(o.get("n_raw", len(o["pts"]))) for o in obs),
        )

    label, label_obs = max(by_label.items(), key=label_evidence)
    frames = {g["frame"] for g in group}
    # The representative photo must come from a detection that actually carried
    # the chosen label — not merely the largest mask in the group. After a
    # cross-label merge the biggest observation can belong to the *losing*
    # label, whose grounding box frames the object differently (or frames a
    # neighbour): that produced correct labels paired with the wrong thumbnail.
    # Geometry still pools every observation (more views → better mesh); only
    # the display frame/box/mask is label-scoped.
    best_obs = max(label_obs, key=lambda g: g.get("n_raw", len(g["pts"])))
    return {
        "label": label,
        "score": max(float(g["score"]) for g in label_obs),
        "frames_seen": len(frames),
        "best_frame": best_obs["frame"],
        "best_box": best_obs.get("box"),
        "best_det_idx": best_obs.get("det_idx"),
        "obs_refs": [(g["frame"], g["det_idx"]) for g in group
                     if g.get("det_idx") is not None],
        "pts": np.vstack([g["pts"] for g in group]),
        "cols": np.vstack([g["cols"] for g in group]),
        "_observations": list(group),
    }


def _instance_bounds(inst: dict) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(inst["pts"])
    return np.percentile(pts, 5, axis=0), np.percentile(pts, 95, axis=0)


def _instances_are_duplicate(a: dict, b: dict,
                             min_2d_iou: float = 0.60,
                             min_3d_containment: float = 0.45) -> bool:
    """True for instances supported by the same image region and 3-D volume.

    Requiring both spaces is important: chairs can touch a table in 3-D, while
    a monitor can overlap its desk in 2-D. A duplicated phrase grounding agrees
    in both spaces.
    """
    boxes_a: dict[int, list[np.ndarray]] = {}
    boxes_b: dict[int, list[np.ndarray]] = {}
    for obs in a.get("_observations", []):
        if obs.get("box") is not None:
            boxes_a.setdefault(obs["frame"], []).append(obs["box"])
    for obs in b.get("_observations", []):
        if obs.get("box") is not None:
            boxes_b.setdefault(obs["frame"], []).append(obs["box"])
    shared = boxes_a.keys() & boxes_b.keys()
    if not shared:
        return False
    max_iou = max(
        box_iou(ba, bb)
        for frame in shared for ba in boxes_a[frame] for bb in boxes_b[frame]
    )
    if max_iou < min_2d_iou:
        return False

    lo_a, hi_a = _instance_bounds(a)
    lo_b, hi_b = _instance_bounds(b)
    inter = np.maximum(np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b), 0.0)
    vol_a = float(np.prod(np.maximum(hi_a - lo_a, 1e-6)))
    vol_b = float(np.prod(np.maximum(hi_b - lo_b, 1e-6)))
    containment = float(np.prod(inter) / max(min(vol_a, vol_b), 1e-9))
    return containment >= min_3d_containment


def _instances_have_same_frame_separation(a: dict, b: dict) -> bool:
    """Whether the detector explicitly showed these as two objects together.

    Distinct same-frame boxes are hard cannot-link evidence. Near-identical
    boxes are repeated grounding and do not block a merge.
    """
    by_frame_a: dict[int, list[dict]] = {}
    by_frame_b: dict[int, list[dict]] = {}
    for obs in a.get("_observations", []):
        by_frame_a.setdefault(obs["frame"], []).append(obs)
    for obs in b.get("_observations", []):
        by_frame_b.setdefault(obs["frame"], []).append(obs)
    for frame in by_frame_a.keys() & by_frame_b.keys():
        for oa in by_frame_a[frame]:
            for ob in by_frame_b[frame]:
                # Missing boxes are not positive evidence of two objects. Only
                # veto a merge when the detector actually drew separate boxes.
                if _same_frame_distinct(oa, ob):
                    return True
    return False


def _point_cloud_overlap_support(
    a: dict, b: dict, tolerance_ratio: float = 0.075,
    max_points: int = 1200,
) -> tuple[float, float]:
    """Tolerant bidirectional overlap of two instance point clouds.

    Returns ``(containment, mutual)``. Containment is the stronger directional
    support and catches a partial track contained by a fuller one; mutual is
    the geometric mean and rejects a tiny nested patch. Distances are scaled by
    the smaller robust object diagonal, so this works before metric calibration.
    """
    pts_a = np.asarray(a.get("pts", []), np.float64)
    pts_b = np.asarray(b.get("pts", []), np.float64)
    if pts_a.ndim != 2 or pts_b.ndim != 2 or pts_a.shape[1:] != (3,) or pts_b.shape[1:] != (3,):
        return 0.0, 0.0
    pts_a = pts_a[np.isfinite(pts_a).all(axis=1)]
    pts_b = pts_b[np.isfinite(pts_b).all(axis=1)]
    if len(pts_a) < 10 or len(pts_b) < 10:
        return 0.0, 0.0
    if len(pts_a) > max_points:
        pts_a = pts_a[np.linspace(0, len(pts_a) - 1, max_points).astype(np.int64)]
    if len(pts_b) > max_points:
        pts_b = pts_b[np.linspace(0, len(pts_b) - 1, max_points).astype(np.int64)]
    lo_a, hi_a = np.percentile(pts_a, [5, 95], axis=0)
    lo_b, hi_b = np.percentile(pts_b, [5, 95], axis=0)
    scale = min(float(np.linalg.norm(hi_a - lo_a)),
                float(np.linalg.norm(hi_b - lo_b)))
    tolerance = max(tolerance_ratio * scale, 1e-6)

    from scipy.spatial import cKDTree
    dist_a = cKDTree(pts_b).query(pts_a, k=1, workers=1)[0]
    dist_b = cKDTree(pts_a).query(pts_b, k=1, workers=1)[0]
    support_a = float(np.mean(dist_a <= tolerance))
    support_b = float(np.mean(dist_b <= tolerance))
    return max(support_a, support_b), float(np.sqrt(support_a * support_b))


_REFLECTOR_WORDS = {
    "television", "tv", "monitor", "screen", "mirror", "window", "glass",
}
_PLANAR_PHYSICAL_WORDS = _REFLECTOR_WORDS | {"whiteboard", "door", "panel", "painting"}
_FLOOR_SUPPORT_WORDS = {
    "chair", "table", "desk", "cabinet", "rack", "sofa", "couch", "shelf",
}


def _label_has_word(label: str, vocabulary: set[str]) -> bool:
    value = str(label).lower().strip()
    tokens = set(value.replace("-", " ").split())
    return bool(tokens & vocabulary) or value in vocabulary


def _is_reflector_label(label: str) -> bool:
    return _label_has_word(label, _REFLECTOR_WORDS)


def _is_planar_physical_label(label: str) -> bool:
    return _label_has_word(label, _PLANAR_PHYSICAL_WORDS)


def _instance_reprojection_agreement(
    inst: dict, T_total: np.ndarray | None, extrinsic: np.ndarray | None,
    intrinsic: np.ndarray | None, depth: np.ndarray | None, masks: list | None,
    max_points: int = 1000,
) -> float | None:
    """Cross-project each view's points into the other observed masks."""
    if any(value is None for value in (T_total, extrinsic, intrinsic, depth, masks)):
        return None
    observations = list(inst.get("_observations", []))
    if len({obs.get("frame") for obs in observations}) < 2:
        return None
    inv_total = np.linalg.inv(np.asarray(T_total, float))
    pair_scores = []
    for source in observations:
        source_pts = np.asarray(source.get("pts", []), float)
        source_pts = source_pts[np.isfinite(source_pts).all(axis=1)] if source_pts.ndim == 2 and source_pts.shape[1:] == (3,) else np.empty((0, 3))
        if len(source_pts) > max_points:
            source_pts = source_pts[
                np.linspace(0, len(source_pts) - 1, max_points).astype(np.int64)
            ]
        if len(source_pts) < 10:
            continue
        raw_pts = _apply44(inv_total, source_pts)
        for target in observations:
            fi, di = target.get("frame"), target.get("det_idx")
            if fi == source.get("frame") or not isinstance(fi, (int, np.integer)):
                continue
            fi = int(fi)
            if (not isinstance(di, (int, np.integer)) or fi < 0 or
                    fi >= len(masks) or int(di) < 0 or int(di) >= len(masks[fi])):
                continue
            target_mask = masks[fi][int(di)]
            if target_mask is None:
                continue
            d_frame = np.asarray(depth[fi, ..., 0], float)
            H, W = d_frame.shape
            target_mask = cv2.resize(
                target_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            uv, z = project_to_frame(raw_pts, extrinsic[fi], intrinsic[fi])
            in_bounds = (
                np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 1e-6) &
                (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) &
                (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
            )
            if in_bounds.sum() < 10:
                continue
            ui = np.round(uv[in_bounds, 0]).astype(np.int32)
            vi = np.round(uv[in_bounds, 1]).astype(np.int32)
            expected_depth = d_frame[vi, ui]
            depth_ok = (
                np.isfinite(expected_depth) & (expected_depth > 1e-6) &
                (np.abs(z[in_bounds] - expected_depth) <=
                 0.12 * expected_depth + 0.08)
            )
            agreement = target_mask[vi, ui] & depth_ok
            pair_scores.append(float(np.mean(agreement)))
    return float(np.median(pair_scores)) if pair_scores else None


def _verify_instance_geometry(inst: dict, floor_y: float,
                              room_height: float, T_total: np.ndarray | None = None,
                              extrinsic: np.ndarray | None = None,
                              intrinsic: np.ndarray | None = None,
                              depth: np.ndarray | None = None,
                              masks: list | None = None) -> dict:
    """Conservative multi-view reflection verification for one instance.

    A hard rejection requires repeated containment inside a detected reflector
    plus geometry that conflicts with the candidate's physical category. Sparse
    or ambiguous evidence is retained and exposed through the audit fields.
    """
    observations = list(inst.get("_observations", []))
    frames = sorted({int(obs["frame"]) for obs in observations})
    positive_inside = [
        float(obs.get("reflector_inside_ratio", 0.0))
        for obs in observations
        if float(obs.get("reflector_inside_ratio", 0.0)) >= 0.50
    ]
    reflector_hits = len({
        int(obs["frame"]) for obs in observations
        if float(obs.get("reflector_inside_ratio", 0.0)) >= 0.70
    })
    inside_ratio = float(np.median(positive_inside)) if positive_inside else 0.0
    reflector_labels = sorted({
        label for obs in observations for label in obs.get("reflector_labels", [])
    })

    pair_mutual = []
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            if observations[i]["frame"] == observations[j]["frame"]:
                continue
            pair_mutual.append(_point_cloud_overlap_support(
                observations[i], observations[j]
            )[1])
    point_overlap = float(np.median(pair_mutual)) if pair_mutual else 0.0
    reprojection = _instance_reprojection_agreement(
        inst, T_total, extrinsic, intrinsic, depth, masks,
    )

    centers, extents = [], []
    for obs in observations:
        pts = np.asarray(obs.get("pts", []), float)
        pts = pts[np.isfinite(pts).all(axis=1)] if pts.ndim == 2 and pts.shape[1:] == (3,) else np.empty((0, 3))
        if len(pts) < 10:
            continue
        lo, hi = np.percentile(pts, [5, 95], axis=0)
        centers.append((lo + hi) / 2.0)
        extents.append(np.maximum(hi - lo, 1e-6))
    if centers:
        centers_arr, extents_arr = np.asarray(centers), np.asarray(extents)
        reference_extent = np.median(extents_arr, axis=0)
        reference_diag = max(float(np.linalg.norm(reference_extent)), 1e-6)
        center_stability = float(np.exp(-np.median(
            np.linalg.norm(centers_arr - np.median(centers_arr, axis=0), axis=1)
        ) / (0.20 * reference_diag)))
        extent_error = np.median(
            np.abs(extents_arr - reference_extent) / np.maximum(reference_extent, 1e-6)
        )
        extent_stability = float(np.exp(-extent_error / 0.35))
    else:
        center_stability = extent_stability = 0.0

    pts_all = np.asarray(inst.get("pts", []), float)
    pts_all = pts_all[np.isfinite(pts_all).all(axis=1)] if pts_all.ndim == 2 and pts_all.shape[1:] == (3,) else np.empty((0, 3))
    vertical_planar = False
    planarity = 0.0
    if len(pts_all) >= 30:
        centred = pts_all - np.median(pts_all, axis=0)
        cov = np.cov(centred.T)
        if np.isfinite(cov).all():
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            largest = max(float(eigenvalues[-1]), 1e-9)
            thickness_ratio = max(float(eigenvalues[0]), 0.0) / largest
            planarity = float(np.clip(1.0 - thickness_ratio / 0.08, 0.0, 1.0))
            normal = eigenvectors[:, 0]
            vertical_planar = planarity >= 0.75 and abs(float(normal[1])) <= 0.45

    label = str(inst.get("label", "object"))
    floor_gap = 0.0
    if len(pts_all) and _label_has_word(label, _FLOOR_SUPPORT_WORDS):
        floor_gap = max(0.0, float(np.percentile(pts_all[:, 1], 2) - floor_y))
    floor_gap_ratio = floor_gap / max(float(room_height), 1e-6)
    # Missing legs/support are common in monocular scans, so floor gap remains
    # diagnostic but is not sufficient for rejection. A vertical sheet is a
    # much stronger contradiction for an ordinarily volumetric category.
    category_conflict = not _is_planar_physical_label(label) and vertical_planar

    if reprojection is None:
        geometry_score = float(np.clip(
            0.50 * point_overlap + 0.25 * center_stability + 0.25 * extent_stability,
            0.0, 1.0,
        ))
    else:
        geometry_score = float(np.clip(
            0.30 * point_overlap + 0.30 * reprojection +
            0.20 * center_stability + 0.20 * extent_stability,
            0.0, 1.0,
        ))
    multi_view_conflict = bool(
        len(frames) >= 2 and point_overlap < 0.15 and
        ((reprojection is not None and reprojection < 0.25) or center_stability < 0.50)
    )
    reflection_conflict = category_conflict or multi_view_conflict
    reflection_score = (
        float(np.clip(
            0.55 * inside_ratio + 0.25 * float(reflection_conflict) +
            0.20 * (1.0 - point_overlap), 0.0, 1.0,
        )) if inside_ratio > 0 else 0.0
    )
    reflection_reject = bool(
        len(frames) >= 2 and reflector_hits >= 2 and inside_ratio >= 0.70 and
        reflection_conflict and reflection_score >= 0.65
    )
    # Floor gap alone is not evidence of a bad placement — monocular scans
    # routinely miss thin legs on well-observed furniture (see the 0.17-ish
    # ratios on this scene's well-tracked chairs). Only reject when the gap is
    # both large (nowhere near normal leg-occlusion noise) and the instance is
    # one of the thin-evidence fragments the rest of this module already
    # treats as unreliable.
    unsupported = bool(len(frames) <= 3 and floor_gap_ratio >= 0.30)
    reject = reflection_reject or unsupported
    if reflection_reject:
        status = "rejected_reflection"
    elif unsupported:
        status = "rejected_unsupported"
    elif inside_ratio >= 0.50 or len(frames) < 2:
        status = "uncertain"
    else:
        status = "accepted"
    return {
        "status": status,
        "geometry_score": round(geometry_score, 3),
        "reflection_score": round(reflection_score, 3),
        "point_overlap": round(point_overlap, 3),
        "reprojection_agreement": (
            round(float(reprojection), 3) if reprojection is not None else None
        ),
        "center_stability": round(center_stability, 3),
        "extent_stability": round(extent_stability, 3),
        "planarity": round(planarity, 3),
        "vertical_planar": bool(vertical_planar),
        "floor_gap": round(floor_gap, 4),
        "floor_gap_ratio": round(floor_gap_ratio, 3),
        "category_conflict": bool(category_conflict),
        "multi_view_conflict": bool(multi_view_conflict),
        "reflector_inside_ratio": round(inside_ratio, 3),
        "reflector_frames": int(reflector_hits),
        "reflector_labels": reflector_labels,
        "unsupported": unsupported,
        "frames": len(frames),
        "reject": reject,
    }


def _filter_reflection_instances(instances: list[dict], floor_y: float,
                                 room_height: float, **verification_context) -> tuple[list[dict], dict]:
    kept, rejected = [], []
    for inst in instances:
        audit = _verify_instance_geometry(
            inst, floor_y, room_height, **verification_context,
        )
        inst["geometry_verification"] = audit
        if audit["reject"]:
            rejected.append({
                "label": str(inst.get("label", "object")),
                "score": round(float(inst.get("score", 0.0)), 3),
                "reason": "unsupported" if audit["status"] == "rejected_unsupported" else "reflection",
                "verification": audit,
            })
        else:
            kept.append(inst)
    return kept, {
        "algorithm": "multiview-reflection-v1",
        "evaluated": len(instances),
        "accepted": len(kept),
        "rejected": len(rejected),
        "rejected_candidates": rejected,
    }


def _instances_are_split_track_duplicate(
    a: dict,
    b: dict,
    min_containment: float = 0.68,
    max_center_distance: float = 0.24,
) -> bool:
    """Detect two association tracks occupying the same physical object.

    This handles observations split by label changes or large centroid drift:
    there is no shared 2-D frame for the normal duplicate test, but the robust
    3-D boxes almost coincide. Per-axis size agreement and complete 3-D
    containment keep nested objects (monitor on desk) separate; explicit
    same-frame separation protects adjacent repeated furniture.
    """
    if _instances_have_same_frame_separation(a, b):
        return False
    lo_a, hi_a = _instance_bounds(a)
    lo_b, hi_b = _instance_bounds(b)
    extent_a = np.maximum(hi_a - lo_a, 1e-6)
    extent_b = np.maximum(hi_b - lo_b, 1e-6)
    ratio = extent_a / extent_b
    standard_size_match = not np.any((ratio < 0.50) | (ratio > 2.0))
    relaxed_size_match = not np.any((ratio < 0.38) | (ratio > 2.6))
    cloud_support = None
    if not standard_size_match:
        if not relaxed_size_match:
            return False
        cloud_support = _point_cloud_overlap_support(a, b)
        if cloud_support[0] < 0.68 or cloud_support[1] < 0.28:
            return False

    inter = np.maximum(np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b), 0.0)
    vol_a, vol_b = float(np.prod(extent_a)), float(np.prod(extent_b))
    containment = float(np.prod(inter) / max(min(vol_a, vol_b), 1e-9))
    # Strong direct cloud agreement permits a less complete bounding-volume
    # overlap, which is common when two camera arcs see opposite object sides.
    if containment < min_containment:
        if containment < 0.48:
            return False
        cloud_support = cloud_support or _point_cloud_overlap_support(a, b)
        if cloud_support[0] < 0.48 or cloud_support[1] < 0.20:
            return False

    center_a, center_b = (lo_a + hi_a) / 2.0, (lo_b + hi_b) / 2.0
    scale = max(float(np.linalg.norm(extent_a)), float(np.linalg.norm(extent_b)), 1e-6)
    return float(np.linalg.norm(center_a - center_b) / scale) <= max_center_distance


def _merge_cross_label_duplicates(instances: list[dict]) -> list[dict]:
    """Merge duplicate/split tracks and let repeated-frame evidence choose label."""
    ordered = sorted(
        instances,
        key=lambda inst: (inst["frames_seen"], inst["score"], len(inst["pts"])),
        reverse=True,
    )
    used = [False] * len(ordered)
    merged: list[dict] = []
    for i, base in enumerate(ordered):
        if used[i]:
            continue
        observations = list(base.get("_observations", []))
        components = [base]
        current = base
        used[i] = True
        for j in range(i + 1, len(ordered)):
            if used[j]:
                continue
            candidate = ordered[j]
            # Complete-link prevents a chain of individually close neighbours
            # from collapsing into one instance after the aggregate box grows.
            if not all(
                _instances_are_duplicate(component, candidate) or
                _instances_are_split_track_duplicate(component, candidate)
                for component in components
            ):
                continue
            observations.extend(ordered[j].get("_observations", []))
            components.append(candidate)
            current = _instance_from_observations(observations)
            used[j] = True
        merged.append(current)
    return merged


# Same-frame boxes above this IoU are one object detected twice (a repeated
# grounding); below it they are distinct objects the detector already separated.
_DUP_BOX_IOU = 0.5
# Only very low overlap is positive evidence of two different objects. The
# interval between this and _DUP_BOX_IOU is deliberately ambiguous: different
# phrases frequently produce a tight and a broad box for the same object, and
# their shared 3-D point support should be allowed to settle the decision.
_DISTINCT_BOX_MAX_IOU = 0.15


def _same_frame_distinct(a: dict, b: dict) -> bool:
    """Whether two detections are clearly separate objects in one image."""
    if a.get("box") is None or b.get("box") is None:
        return False
    # A reflected candidate and the physical surface containing it are not two
    # labels for one object. Their boxes are nested by definition, so ordinary
    # low-IoU separation cannot protect them from cross-label consolidation.
    a_reflector = _is_reflector_label(str(a.get("label", "")))
    b_reflector = _is_reflector_label(str(b.get("label", "")))
    if a_reflector != b_reflector:
        contained = b if a_reflector else a
        if float(contained.get("reflector_inside_ratio", 0.0)) >= 0.50:
            return True
    return box_iou(a["box"], b["box"]) <= _DISTINCT_BOX_MAX_IOU


def _same_frame_duplicate(a: dict, b: dict) -> bool:
    """Repeated phrase grounding, not merely overlapping nearby objects."""
    if a.get("box") is None or b.get("box") is None:
        return False
    if box_iou(a["box"], b["box"]) < _DUP_BOX_IOU:
        return False
    scale = max(float(a["diag"]), float(b["diag"]), 1e-6)
    return np.linalg.norm(np.asarray(a["center"]) - np.asarray(b["center"])) <= 0.35 * scale


def cluster_observations(observations: list[dict]) -> list[dict]:
    """Original merge-oriented clustering with one same-frame safeguard.

    Nearby observations with the same label use the original broad, scale-free
    radius. This intentionally favors one stable instance over fragmented
    tracks. The only added cannot-link rule is direct detector evidence: two
    clearly separate boxes in the same frame may never enter the same group.
    """
    from collections import defaultdict

    by_label: dict[str, list[dict]] = defaultdict(list)
    for o in observations:
        by_label[o["label"]].append(o)

    instances = []
    for _, label_obs in by_label.items():
        ordered = sorted(label_obs, key=lambda o: -o["score"])
        used = [False] * len(ordered)
        for i, seed in enumerate(ordered):
            if used[i]:
                continue
            group = [seed]
            used[i] = True
            # The original implementation consumed candidates in list order.
            # With several identical chairs that could attach a nearby chair
            # before the same chair's much closer observation in that frame.
            # Preserve the broad merge radius, but consume nearest observations
            # first so the same-frame cannot-link protects the correct identity.
            candidates = sorted(
                range(i + 1, len(ordered)),
                key=lambda j: float(np.linalg.norm(
                    seed["center"] - ordered[j]["center"]
                )),
            )
            for j in candidates:
                if used[j]:
                    continue
                candidate = ordered[j]
                threshold = 0.6 * max(seed["diag"], candidate["diag"], 1e-6)
                if np.linalg.norm(seed["center"] - candidate["center"]) >= threshold:
                    continue
                if any(
                    member["frame"] == candidate["frame"] and
                    _same_frame_distinct(member, candidate)
                    for member in group
                ):
                    continue
                group.append(candidate)
                used[j] = True

            frames_seen = {obs["frame"] for obs in group}
            if len(frames_seen) < 2 and max(obs["score"] for obs in group) < 0.45:
                continue
            instances.append(_instance_from_observations(group))

    # Same-label clustering intentionally mirrors the known-good first version.
    # This conservative pass handles any fragments left by label changes or
    # unusually large point-cloud drift.
    consolidated = _merge_cross_label_duplicates(instances)
    if len(consolidated) < len(instances):
        print(
            f"[SceneBuilder] Duplicate consolidation: {len(instances)} tracks → "
            f"{len(consolidated)} physical objects."
        )
    return consolidated


def _exclude_mask_for_instances(instances: list[dict], masks: list,
                                shape: tuple[int, int, int]) -> np.ndarray:
    """Rebuild background exclusions after semantic review/rejection."""
    S, dH, dW = shape
    result = np.zeros(shape, dtype=bool)
    for inst in instances:
        if _is_opening_label(str(inst.get("label", ""))):
            continue
        for fi, di in inst.get("obs_refs", []):
            if not (0 <= fi < min(S, len(masks)) and 0 <= di < len(masks[fi])):
                continue
            mask = masks[fi][di]
            if mask is None:
                continue
            small = cv2.resize(mask.astype(np.uint8), (dW, dH),
                               interpolation=cv2.INTER_NEAREST)
            result[fi] |= cv2.dilate(small, np.ones((3, 3), np.uint8)).astype(bool)
    return result


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


def _placement_aabb(placement: dict) -> tuple[np.ndarray, np.ndarray]:
    """World-axis bounds of the oriented box shown by the scene editor."""
    x, y, z = np.asarray(placement["position"], float)
    w, h, d = np.maximum(np.asarray(placement["size"], float), 1e-6)
    yaw = float(placement.get("yaw", 0.0))
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[-w / 2, -d / 2], [w / 2, -d / 2],
                      [w / 2, d / 2], [-w / 2, d / 2]])
    world_xz = local @ np.array([[c, s], [-s, c]]) + [x, z]
    lo = np.array([world_xz[:, 0].min(), y, world_xz[:, 1].min()])
    hi = np.array([world_xz[:, 0].max(), y + h, world_xz[:, 1].max()])
    return lo, hi


def _placed_are_duplicate(a: dict, b: dict,
                          min_containment: float = 0.72) -> bool:
    """Whether two final editor placements are duplicate representations.

    This is deliberately later than track association: floor snapping, robust
    extents and yaw are now applied, so it catches objects that will visibly
    overlap even when their partial raw point bounds differed. Same-frame
    separation remains a hard veto, and comparable dimensions prevent a small
    nested object from disappearing into its support furniture.
    """
    if _instances_have_same_frame_separation(a, b):
        return False
    pa, pb = a["placement"], b["placement"]
    sa = np.asarray(pa["size"], float)
    sb = np.asarray(pb["size"], float)
    # Width/depth may swap when two partial scans choose yaws 90 degrees apart.
    footprint_a, footprint_b = np.sort(sa[[0, 2]]), np.sort(sb[[0, 2]])
    footprint_ratio = footprint_a / np.maximum(footprint_b, 1e-6)
    height_ratio = sa[1] / max(sb[1], 1e-6)
    comparable = (
        np.all((footprint_ratio >= 0.50) & (footprint_ratio <= 2.0)) and
        0.50 <= height_ratio <= 2.0
    )
    relaxed_size = (
        np.all((footprint_ratio >= 0.38) & (footprint_ratio <= 2.6)) and
        0.38 <= height_ratio <= 2.6
    )
    cloud_support = None
    if not comparable:
        if not relaxed_size:
            return False
        cloud_support = _point_cloud_overlap_support(a, b)
        if cloud_support[0] < 0.68 or cloud_support[1] < 0.28:
            return False

    lo_a, hi_a = _placement_aabb(pa)
    lo_b, hi_b = _placement_aabb(pb)
    extent_a, extent_b = hi_a - lo_a, hi_b - lo_b
    inter = np.maximum(np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b), 0.0)
    vol_a, vol_b = float(np.prod(extent_a)), float(np.prod(extent_b))
    containment = float(np.prod(inter) / max(min(vol_a, vol_b), 1e-9))
    if containment < min_containment:
        if containment < 0.52:
            return False
        cloud_support = cloud_support or _point_cloud_overlap_support(a, b)
        if cloud_support[0] < 0.48 or cloud_support[1] < 0.20:
            return False
    center_a, center_b = (lo_a + hi_a) / 2, (lo_b + hi_b) / 2
    scale = max(float(np.linalg.norm(extent_a)), float(np.linalg.norm(extent_b)), 1e-6)
    return float(np.linalg.norm(center_a - center_b) / scale) <= 0.25


def _consolidate_overlapping_placements(
    instances: list[dict],
    floor_y: float,
    room_yaw: float,
    floor_snap_tol: float,
) -> list[dict]:
    """Merge complete-link groups that would strongly overlap in the editor."""
    provisional = [
        {**inst, "placement": place_instance(inst, floor_y, room_yaw, floor_snap_tol)}
        for inst in instances
    ]
    ordered = sorted(
        provisional,
        key=lambda p: (p["frames_seen"], p["score"], len(p["pts"])),
        reverse=True,
    )
    used = [False] * len(ordered)
    result = []
    for i, base in enumerate(ordered):
        if used[i]:
            continue
        group = [base]
        used[i] = True
        for j in range(i + 1, len(ordered)):
            if used[j] or not all(_placed_are_duplicate(member, ordered[j]) for member in group):
                continue
            group.append(ordered[j])
            used[j] = True
        observations = [
            obs for member in group for obs in member.get("_observations", [])
        ]
        result.append(_instance_from_observations(observations) if observations else base)
    return result


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
    tint = np.median(cols, axis=0).clip(0, 255).astype(int) if len(cols) else np.array([128, 128, 128])

    mesh, source = None, None
    if full_mesh is not None and world_position is not None:
        mesh = _crop_dense_mesh(full_mesh, np.asarray(world_position), yaw, size)
        source = "scan-dense"
    if mesh is None:
        mesh = _scan_cutout_mesh(pts, cols, yaw, position, scale, size)
        source = "scan-poisson"
    if mesh is None:
        mesh, source = _build_fallback_mesh(label, size, tint)

    return _ensure_material(mesh), source


def _build_fallback_mesh(label: str, size: list[float], tint) -> tuple[trimesh.Trimesh, str]:
    """Parametric/box fallback shared by initial reconstruction and the
    point-support quality gate."""
    w, h, d = size
    kind = _PREFAB_KINDS.get(label.lower().strip())
    if kind is not None:
        return _ensure_material(_PREFAB_BUILDERS[kind](w, h, d, tuple(tint))), "prefab"
    return _ensure_material(_box([w, h, d], [0, h / 2, 0], tuple(tint))), "box"


def _scan_mesh_quality(mesh: trimesh.Trimesh, observed_local: np.ndarray,
                       expected_size: list[float], frames_seen: int) -> dict:
    """Score whether a scan-derived mesh is supported by the original cloud.

    The score deliberately measures agreement, not visual smoothness: observed
    points should lie near the mesh, the mesh should span the detected object
    extents, and most faces should belong to one coherent component. This keeps
    a tiny surviving fragment from suppressing a better prefab/TRELLIS fallback.
    """
    pts = np.asarray(observed_local, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    verts = verts[np.isfinite(verts).all(axis=1)]
    faces = len(getattr(mesh, "faces", []))
    expected = np.maximum(np.asarray(expected_size, dtype=np.float64), 1e-6)

    empty = {
        "good": False, "score": 0.0, "points": int(len(pts)), "faces": int(faces),
        "support": 0.0, "bbox_agreement": 0.0, "component_ratio": 0.0,
    }
    if len(pts) < 3 or len(verts) < 3 or faces == 0:
        return empty

    mesh_extent = np.maximum(np.ptp(verts, axis=0), 1e-6)
    bbox_axis = np.minimum(mesh_extent, expected) / np.maximum(mesh_extent, expected)
    bbox_agreement = float(np.mean(bbox_axis))

    comps = mesh.split(only_watertight=False)
    component_ratio = (
        float(max((len(c.faces) for c in comps), default=0) / max(faces, 1))
    )

    # Point-to-sampled-surface distance avoids requiring trimesh's optional
    # rtree dependency. Fixed seed makes quality decisions reproducible.
    from scipy.spatial import cKDTree
    if len(pts) > 5000:
        pts = pts[np.linspace(0, len(pts) - 1, 5000).astype(np.int64)]
    sample_count = min(max(len(pts) * 2, 2500), 20_000)
    surface, _ = trimesh.sample.sample_surface(mesh, sample_count, seed=0)
    distances, _ = cKDTree(surface).query(pts, k=1)
    diag = float(np.linalg.norm(expected))
    tolerance = max(0.025, 0.06 * diag)
    support = float(np.mean(distances <= tolerance))

    density = min(len(pts) / 1000.0, 1.0)
    view_support = min(max(int(frames_seen), 0) / 2.0, 1.0)
    score = (0.45 * support + 0.25 * bbox_agreement +
             0.10 * component_ratio + 0.10 * density + 0.10 * view_support)
    good = (
        faces >= 100 and len(pts) >= 300 and support >= 0.25 and
        bbox_agreement >= 0.35 and component_ratio >= 0.65 and score >= 0.48
    )
    return {
        "good": bool(good),
        "score": round(float(score), 3),
        "points": int(len(pts)),
        "faces": int(faces),
        "support": round(support, 3),
        "bbox_agreement": round(bbox_agreement, 3),
        "component_ratio": round(component_ratio, 3),
    }


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
                       obj_id: str, max_px: int = 480) -> str | None:
    """Save the object's complete best frame with its detection highlighted.

    Keeping the full frame preserves phone-video portrait orientation and
    gives the editor useful scene context. A tight detection crop can be
    landscape even when its source video is portrait, which made inspector
    photos look incorrectly rotated. TRELLIS continues to use the separate
    masked ``input_crop`` produced by :func:`_save_object_input_crop`.
    """
    fi, box = inst.get("best_frame"), inst.get("best_box")
    if fi is None or box is None or fi >= len(image_paths):
        return None
    img = cv2.imread(image_paths[fi])
    if img is None:
        return None
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, y1 = int(max(x1, 0)), int(max(y1, 0))
    x2, y2 = int(min(x2, W)), int(min(y2, H))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    photo = img.copy()
    thickness = max(2, int(round(max(H, W) / 500)))
    cv2.rectangle(photo, (x1, y1), (x2, y2), (60, 220, 80), thickness)
    s = max_px / max(photo.shape[:2])
    if s < 1.0:
        photo = cv2.resize(
            photo,
            (max(int(photo.shape[1] * s), 1), max(int(photo.shape[0] * s), 1)),
            interpolation=cv2.INTER_AREA,
        )
    rel = f"objects/{obj_id}.jpg"
    cv2.imwrite(os.path.join(scene_dir, rel), photo, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return rel


def _select_object_support_observation(
    inst: dict,
    image_paths: list[str],
    masks: list,
    image_cache: dict[int, np.ndarray | None] | None = None,
) -> tuple[dict | None, dict]:
    """Choose the observation that best represents this final placed object.

    Raw mask size alone is unsafe after track consolidation: a broad grounding
    box can contain more points while actually framing a neighbour. Candidates
    are scored by point support inside the final editor bounds, 3-D centre
    agreement, detector confidence, mask availability, sharpness, and whether
    the box is clipped by the frame. Invalid candidates are skipped so another
    supporting frame can still provide the editor photo and TRELLIS crop.
    """
    cache = image_cache if image_cache is not None else {}
    observations = list(inst.get("_observations", []))
    if not observations:
        return None, {"reason": "no_observations"}
    # Cross-label consolidation deliberately pools geometry from compatible
    # detections, but a losing label can frame a neighbour or a larger parent
    # object. Keep the image/TRELLIS evidence label-scoped whenever possible;
    # only fall back to all observations for legacy instances whose reviewed
    # label no longer appears verbatim in their detector evidence.
    final_label = str(inst.get("label", "")).lower().strip()
    label_observations = [
        obs for obs in observations
        if str(obs.get("label", "")).lower().strip() == final_label
    ]
    label_scoped = bool(label_observations)
    if label_scoped:
        observations = label_observations
    placement = inst.get("placement")
    if placement is not None:
        target_lo, target_hi = _placement_aabb(placement)
        pad = 0.12 * np.maximum(target_hi - target_lo, 1e-6)
        target_lo, target_hi = target_lo - pad, target_hi + pad
        target_center = (target_lo + target_hi) / 2.0
        target_diag = max(float(np.linalg.norm(target_hi - target_lo)), 1e-6)
    else:
        target_lo, target_hi = _instance_bounds(inst)
        target_center = (target_lo + target_hi) / 2.0
        target_diag = max(float(np.linalg.norm(target_hi - target_lo)), 1e-6)

    ranked = []
    for obs in observations:
        fi, box = obs.get("frame"), obs.get("box")
        if not isinstance(fi, (int, np.integer)) or not (0 <= int(fi) < len(image_paths)):
            continue
        if box is None or np.asarray(box).size != 4 or not np.isfinite(box).all():
            continue
        fi = int(fi)
        if fi not in cache:
            cache[fi] = cv2.imread(image_paths[fi])
        image = cache[fi]
        if image is None:
            continue
        H, W = image.shape[:2]
        x1f, y1f, x2f, y2f = map(float, box)
        x1, y1 = int(max(x1f, 0)), int(max(y1f, 0))
        x2, y2 = int(min(x2f, W)), int(min(y2f, H))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue

        pts = np.asarray(obs.get("pts", []), float)
        finite = np.isfinite(pts).all(axis=1) if pts.ndim == 2 and pts.shape[1:] == (3,) else np.zeros(0, bool)
        pts = pts[finite] if len(finite) else np.empty((0, 3))
        support = float(np.mean(np.all((pts >= target_lo) & (pts <= target_hi), axis=1))) if len(pts) else 0.0
        obs_center = np.asarray(obs.get("center", target_center), float)
        center_score = max(0.0, 1.0 - float(np.linalg.norm(obs_center - target_center)) / (0.6 * target_diag))

        crop_gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())
        sharpness_score = min(np.log1p(max(sharpness, 0.0)) / 7.0, 1.0)
        det_score = float(np.clip(obs.get("score", 0.0), 0.0, 1.0))
        clipped = x1f <= 1 or y1f <= 1 or x2f >= W - 1 or y2f >= H - 1
        di = obs.get("det_idx")
        has_mask = (
            isinstance(di, (int, np.integer)) and fi < len(masks) and
            0 <= int(di) < len(masks[fi]) and masks[fi][int(di)] is not None
        )
        label_match = str(obs.get("label", "")).lower().strip() == str(inst.get("label", "")).lower().strip()
        quality = (
            0.42 * support + 0.23 * center_score + 0.12 * det_score +
            0.10 * sharpness_score + 0.08 * float(has_mask) +
            0.05 * float(label_match) - 0.08 * float(clipped)
        )
        ranked.append((quality, obs, {
            "frame": fi,
            "score": round(float(quality), 3),
            "point_support": round(support, 3),
            "center_agreement": round(center_score, 3),
            "sharpness": round(sharpness_score, 3),
            "has_mask": bool(has_mask),
            "clipped": bool(clipped),
            "label_scoped": label_scoped,
            "candidates": 0,
        }))

    if not ranked:
        return None, {"reason": "no_valid_image_observation", "candidates": 0}
    ranked.sort(key=lambda item: item[0], reverse=True)
    audit = ranked[0][2]
    audit["candidates"] = len(ranked)
    return ranked[0][1], audit


def _refresh_object_support_images(
    placed: list[dict], image_paths: list[str], masks: list,
) -> dict:
    """Refresh best-frame fields after all association/review/placement steps."""
    cache: dict[int, np.ndarray | None] = {}
    selected, missing = 0, 0
    for inst in placed:
        obs, audit = _select_object_support_observation(inst, image_paths, masks, cache)
        inst["photo_selection"] = audit
        if obs is None:
            inst["best_frame"] = None
            inst["best_box"] = None
            inst["best_det_idx"] = None
            missing += 1
            continue
        inst["best_frame"] = int(obs["frame"])
        inst["best_box"] = np.asarray(obs["box"], float)
        inst["best_det_idx"] = obs.get("det_idx")
        selected += 1
    return {"selected": selected, "missing": missing}


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


def _masked_crop_appearance_descriptor(crop: np.ndarray | None) -> dict | None:
    """Compact, dependency-free descriptor for conservative asset reuse.

    This is intentionally stricter than object tracking. Two physical objects
    may share a label and metric size while having different upholstery, shape,
    or controls. Only an RGBA crop with a useful SAM foreground is accepted;
    uncertain/maskless comparisons remain separate and get their own asset.
    """
    if crop is None or crop.ndim != 3 or crop.shape[2] != 4:
        return None
    alpha = np.asarray(crop[:, :, 3], np.uint8)
    mask = alpha >= 128
    if mask.sum() < 64 or mask.all():
        return None
    ys, xs = np.nonzero(mask)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgb = np.asarray(crop[y1:y2, x1:x2, :3], np.uint8)
    tight_mask = mask[y1:y2, x1:x2]
    if min(rgb.shape[:2]) < 2:
        return None

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], tight_mask.astype(np.uint8),
        [12, 4, 4], [0, 180, 0, 256, 0, 256],
    ).reshape(-1).astype(np.float32)
    hist_sum = float(hist.sum())
    if hist_sum <= 0:
        return None
    # Square-rooted probabilities make the dot product a bounded Hellinger
    # similarity and reduce sensitivity to illumination/exposure changes.
    hist = np.sqrt(hist / hist_sum)

    side, inner = 48, 44
    h, w = rgb.shape[:2]
    scale = inner / max(h, w)
    rh, rw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized_rgb = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(
        tight_mask.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    y0, x0 = (side - rh) // 2, (side - rw) // 2
    thumb = np.zeros((side, side, 3), np.uint8)
    silhouette = np.zeros((side, side), bool)
    thumb[y0:y0 + rh, x0:x0 + rw] = resized_rgb
    silhouette[y0:y0 + rh, x0:x0 + rw] = resized_mask
    thumb[~silhouette] = 0
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120) > 0
    edges &= cv2.dilate(silhouette.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return {
        "hist": hist,
        "thumb": thumb,
        "silhouette": silhouette,
        "edges": edges,
    }


def _appearance_similarity(a: dict | None, b: dict | None) -> float:
    """Return conservative [0, 1] similarity between two masked object crops."""
    if a is None or b is None:
        return 0.0
    color = float(np.clip(np.dot(a["hist"], b["hist"]), 0.0, 1.0))
    union = a["silhouette"] | b["silhouette"]
    if not union.any():
        return 0.0
    shape = float(np.sum(a["silhouette"] & b["silhouette"]) / np.sum(union))
    pixel_delta = np.abs(
        a["thumb"].astype(np.float32) - b["thumb"].astype(np.float32)
    ).mean(axis=2)
    texture = float(1.0 - np.mean(pixel_delta[union]) / 255.0)
    edge_union = a["edges"] | b["edges"]
    edge = (
        float(np.sum(a["edges"] & b["edges"]) / np.sum(edge_union))
        if edge_union.any() else 1.0
    )
    return float(np.clip(0.40 * color + 0.25 * texture + 0.20 * shape + 0.15 * edge,
                         0.0, 1.0))


def _group_reusable_assets(
    placed: list[dict], image_paths: list[str], masks: list,
    size_rel_tol: float = 0.10, min_appearance_similarity: float = 0.90,
) -> list[list[int]]:
    """Group only objects safe to render with the same generated/scan asset.

    Label and metric extent provide a cheap coarse filter. Every proposed group
    must then form a visual clique: each member's SAM crop must agree with every
    existing member. Missing or ambiguous masks fail closed to singleton groups.
    """
    coarse_groups = _group_duplicates(placed, size_rel_tol=size_rel_tol)
    descriptors: dict[int, dict | None] = {}
    for i, inst in enumerate(placed):
        descriptor = _masked_crop_appearance_descriptor(
            _object_crop_rgb(image_paths, masks, inst)
        )
        descriptors[i] = descriptor
        inst["_asset_appearance_descriptor"] = descriptor

    groups: list[list[int]] = []
    for coarse in coarse_groups:
        visual_groups: list[list[int]] = []
        for idx in coarse:
            matched = False
            for group in visual_groups:
                similarities = [
                    _appearance_similarity(descriptors[idx], descriptors[member])
                    for member in group
                ]
                if similarities and min(similarities) >= min_appearance_similarity:
                    group.append(idx)
                    matched = True
                    break
            if not matched:
                visual_groups.append([idx])
        groups.extend(visual_groups)
    return groups


# ── Generative object assets (TRELLIS) ───────────────────────────────────────

def _object_crop_rgb(image_paths: list[str], masks: list, inst: dict,
                     max_px: int = 1024) -> np.ndarray | None:
    """Cutout of an instance from its best (full-res) frame — the input handed
    to the image-to-3D generator. When the object's SAM mask for that frame is
    available it's returned as an RGBA subject cutout (background transparent);
    a clean, clutter-free subject is what most improves a single-image
    generator. Otherwise a plain RGB box crop. Padded for a little context and
    downscaled to max_px. Returns (H, W, 3|4) uint8, or None if unavailable."""
    fi, box = inst.get("best_frame"), inst.get("best_box")
    if fi is None or box is None or fi >= len(image_paths):
        return None
    img = cv2.imread(image_paths[fi])          # BGR, full-res
    if img is None:
        return None
    H, W = img.shape[:2]

    # This object's SAM mask in the best frame. Prefer the detection that
    # actually produced best_frame/best_box (best_det_idx); fall back to the
    # first obs in this frame. Scanning obs_refs alone is ambiguous after a
    # cross-label merge, where the best frame holds one detection per label.
    det_idx = inst.get("best_det_idx")
    if det_idx is None:
        det_idx = next((di for f, di in inst.get("obs_refs", []) if f == fi), None)
    mask = None
    if det_idx is not None and fi < len(masks) and det_idx < len(masks[fi]):
        mask = masks[fi][det_idx]
    if mask is not None and mask.shape[:2] != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    x1, y1, x2, y2 = [float(v) for v in box]
    pw, ph = 0.12 * (x2 - x1), 0.12 * (y2 - y1)      # a little context around the box
    x1, y1 = int(max(x1 - pw, 0)), int(max(y1 - ph, 0))
    x2, y2 = int(min(x2 + pw, W)), int(min(y2 + ph, H))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    rgb = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    if mask is not None:
        alpha = mask[y1:y2, x1:x2].astype(np.uint8) * 255
        crop = np.dstack([rgb, alpha])               # RGBA — background transparent
    else:
        crop = rgb

    s = max_px / max(crop.shape[:2])
    if s < 1.0:
        crop = cv2.resize(crop, (max(int(crop.shape[1] * s), 1), max(int(crop.shape[0] * s), 1)),
                          interpolation=cv2.INTER_AREA)
    return crop


def _save_object_input_crop(image_paths: list[str], masks: list, inst: dict,
                            scene_dir: str, obj_id: str) -> str | None:
    """Persist the RGBA subject cutout (the image-to-3D generator's input) next
    to the object glb, so the editor can regenerate this object with TRELLIS on
    demand — long after the source frames and SAM masks have been freed. Saved
    as PNG to keep the alpha. Returns the scene-relative path, or None."""
    crop = _object_crop_rgb(image_paths, masks, inst)
    if crop is None:
        return None
    rel = f"objects/{obj_id}_input.png"
    code = cv2.COLOR_RGBA2BGRA if crop.shape[2] == 4 else cv2.COLOR_RGB2BGR
    cv2.imwrite(os.path.join(scene_dir, rel), cv2.cvtColor(crop, code))
    return rel


def _glb_bytes_to_local_mesh(glb_bytes: bytes, size_m: list[float]) -> trimesh.Trimesh | None:
    """Load a generated GLB (canonical unit aabb) and place it in the standard
    object convention: scaled per-axis so its bbox matches the detected size,
    then bottom-centred at the origin (yaw is applied later by the editor). The
    generator's canonical 'front' is inherited by every instance that reuses the
    asset — the same yaw ambiguity every scan object already has."""
    try:
        loaded = trimesh.load(io.BytesIO(glb_bytes), file_type="glb", process=False)
        mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
        if mesh is None or len(getattr(mesh, "faces", [])) == 0:
            return None
        lo, hi = mesh.bounds
        mesh.apply_scale(np.asarray(size_m, float) / np.maximum(hi - lo, 1e-6))
        lo, hi = mesh.bounds
        center = (lo + hi) / 2.0
        mesh.apply_translation([-center[0], -lo[1], -center[2]])
        return mesh
    except Exception as exc:
        print(f"[SceneBuilder] Generated GLB load failed ({exc}); keeping fallback.")
        return None


# ── Room support: fitted boundaries + coverage-aware synthetic backing ────────

def _min_area_rect_yaw(pts_xz: np.ndarray) -> float:
    """Room orientation (radians) from the minimum-area bounding rectangle of
    the XZ footprint.

    Preferred over a PCA principal axis for the *room*: PCA follows where the
    bulk of points sits — i.e. the furniture layout — so a long row of racks
    can spin the estimate off the walls. The enclosing rectangle is instead
    pinned by the extreme points, which are the walls, so it locks onto the
    room. Returns the angle that rotates those walls onto the world axes, folded
    to [-45°, 45°) (a rectangle at θ and θ+90° describe the same alignment)."""
    pts = pts_xz - np.median(pts_xz, axis=0)
    if len(pts) > 20000:
        pts = pts[:: len(pts) // 20000 + 1]
    if len(pts) < 50:
        return 0.0

    def area_at(ang: float) -> float:
        c, s = np.cos(ang), np.sin(ang)
        u = pts[:, 0] * c + pts[:, 1] * s
        v = -pts[:, 0] * s + pts[:, 1] * c
        return float((u.max() - u.min()) * (v.max() - v.min()))

    best = min(np.linspace(0, np.pi / 2, 91), key=area_at)          # coarse 1° sweep
    best = min(np.linspace(best - np.radians(1), best + np.radians(1), 21), key=area_at)  # refine
    return float((best + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def _yaw_basis(room_yaw: float) -> np.ndarray:
    """Rotation matrix for the room's yaw frame (same convention every object
    uses). world→local: v @ R.T ; local→world: v @ R. Because the *same* R
    rotates the cloud in and the built geometry out, the shell lands exactly
    where the wall points are — the absolute yaw sign never matters."""
    c, s = np.cos(-room_yaw), np.sin(-room_yaw)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def _fit_room_rect(pts_local: np.ndarray, ceil_h: float,
                   min_support: int = 200) -> tuple[float, float, float, float]:
    """Fit the four wall planes in the yaw-aligned local frame (floor at y≈0).

    For each side, the plane is the robust median of the real wall-surface
    points on that side — the outer slab of a mid-height band, which excludes
    floor spread and low clutter. Where too few wall points were seen (an
    unobserved wall), it falls back to the 2/98 percentile extent of the whole
    cloud so the room still closes. Returns (xmin, xmax, zmin, zmax)."""
    x, y, z = pts_local[:, 0], pts_local[:, 1], pts_local[:, 2]
    band = (y > 0.15 * ceil_h) & (y < 0.85 * ceil_h)
    xb, zb = x[band], z[band]

    def side(vals_all: np.ndarray, vals_band: np.ndarray, upper: bool) -> float:
        if len(vals_band) >= min_support:
            thr = np.percentile(vals_band, 95 if upper else 5)
            outer = vals_band[vals_band > thr] if upper else vals_band[vals_band < thr]
            if len(outer) >= 20:
                return float(np.median(outer))          # plane fit to the wall surface
        return float(np.percentile(vals_all, 98 if upper else 2))   # unobserved → extent

    return (side(x, xb, False), side(x, xb, True),
            side(z, zb, False), side(z, zb, True))


def _oriented_box(extents, center_local, color, R: np.ndarray) -> trimesh.Trimesh:
    """A coloured box built at center_local in the yaw frame, rotated into world."""
    b = _box(extents, center_local, color)
    b.vertices = np.asarray(b.vertices) @ R
    return b


def _grid_coverage(points_2d: np.ndarray, lower: tuple[float, float],
                   upper: tuple[float, float], grid: tuple[int, int] = (10, 10)) -> float:
    """Fraction of surface grid cells touched by observed cloud points."""
    pts = np.asarray(points_2d, dtype=float)
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    span = hi - lo
    if len(pts) < 20 or np.any(span <= 1e-6):
        return 0.0
    inside = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts = pts[inside]
    if len(pts) < 20:
        return 0.0
    ij = np.floor((pts - lo) / span * np.asarray(grid)).astype(int)
    ij = np.clip(ij, [0, 0], np.asarray(grid) - 1)
    return float(len(np.unique(ij, axis=0)) / np.prod(grid))


def _room_surface_coverage(glob_m: np.ndarray, R: np.ndarray,
                           rect: tuple[float, float, float, float],
                           ceil_h: float) -> dict[str, float]:
    """Measure observed point support on the fitted floor/ceiling/wall planes.

    Coverage drives whether the structured scene needs a synthetic backing
    slab. Real TSDF surfaces are retained either way; slabs fill only weakly
    observed boundaries instead of replacing all scan geometry.
    """
    xmin, xmax, zmin, zmax = rect
    p = np.asarray(glob_m) @ R.T
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    width, depth = xmax - xmin, zmax - zmin
    wall_tol = max(0.04, 0.025 * max(min(width, depth), 1e-6))
    horizontal_tol = max(0.04, 0.04 * max(ceil_h, 1e-6))

    floor = np.abs(y) <= horizontal_tol
    ceiling = np.abs(y - ceil_h) <= horizontal_tol
    # Exclude the floor/ceiling bands from wall evidence; otherwise a dense
    # floor touching the room boundary can make an entirely unseen wall appear
    # supported along its bottom row.
    vertical = (y > 0.05 * ceil_h) & (y < 0.95 * ceil_h)
    x_lo = (np.abs(x - xmin) <= wall_tol) & vertical
    x_hi = (np.abs(x - xmax) <= wall_tol) & vertical
    z_lo = (np.abs(z - zmin) <= wall_tol) & vertical
    z_hi = (np.abs(z - zmax) <= wall_tol) & vertical
    return {
        "floor": _grid_coverage(p[floor][:, [0, 2]], (xmin, zmin), (xmax, zmax)),
        "ceiling": _grid_coverage(p[ceiling][:, [0, 2]], (xmin, zmin), (xmax, zmax)),
        "wall_0": _grid_coverage(p[x_lo][:, [2, 1]], (zmin, 0.0), (zmax, ceil_h)),
        "wall_1": _grid_coverage(p[x_hi][:, [2, 1]], (zmin, 0.0), (zmax, ceil_h)),
        "wall_2": _grid_coverage(p[z_lo][:, [0, 1]], (xmin, 0.0), (xmax, ceil_h)),
        "wall_3": _grid_coverage(p[z_hi][:, [0, 1]], (xmin, 0.0), (xmax, ceil_h)),
    }


_OPENING_WORDS = {"door", "doorway", "window", "opening", "entrance", "exit"}


def _is_opening_label(label: str) -> bool:
    tokens = set(label.lower().replace("-", " ").split())
    return bool(tokens & _OPENING_WORDS)


def _wall_spec(index: int, rect: tuple[float, float, float, float]) -> dict:
    xmin, xmax, zmin, zmax = rect
    if index == 0:
        return {"axis": 0, "plane": xmin, "sign": -1.0,
                "u_axis": 2, "u_min": zmin, "u_max": zmax}
    if index == 1:
        return {"axis": 0, "plane": xmax, "sign": 1.0,
                "u_axis": 2, "u_min": zmin, "u_max": zmax}
    if index == 2:
        return {"axis": 2, "plane": zmin, "sign": -1.0,
                "u_axis": 0, "u_min": xmin, "u_max": xmax}
    if index == 3:
        return {"axis": 2, "plane": zmax, "sign": 1.0,
                "u_axis": 0, "u_min": xmin, "u_max": xmax}
    raise ValueError(f"invalid wall index {index}")


def _wall_grid_shape(spec: dict, ceil_h: float,
                     target_cell_m: float = 0.20) -> tuple[int, int]:
    nu = int(np.clip(np.ceil((spec["u_max"] - spec["u_min"]) / target_cell_m), 1, 96))
    ny = int(np.clip(np.ceil(ceil_h / target_cell_m), 1, 64))
    return ny, nu


def _points_to_wall_cells(points_local: np.ndarray, spec: dict, ceil_h: float,
                          shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_local)
    ny, nu = shape
    u = pts[:, spec["u_axis"]]
    y = pts[:, 1]
    valid = ((u >= spec["u_min"]) & (u <= spec["u_max"]) &
             (y >= 0.0) & (y <= ceil_h) & np.isfinite(u) & np.isfinite(y))
    if not valid.any():
        return np.empty(0, int), np.empty(0, int)
    iu = np.floor((u[valid] - spec["u_min"]) /
                  max(spec["u_max"] - spec["u_min"], 1e-9) * nu).astype(int)
    iy = np.floor(y[valid] / max(ceil_h, 1e-9) * ny).astype(int)
    return np.clip(iy, 0, ny - 1), np.clip(iu, 0, nu - 1)


def _ray_wall_cells(camera_local: np.ndarray, endpoints_local: np.ndarray,
                    spec: dict, ceil_h: float, shape: tuple[int, int],
                    mode: str, beyond_tol: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Project rays onto a wall grid.

    mode="occluded" requires the wall intersection to lie behind the endpoint;
    mode="opening" accepts semantic door/window rays; mode="through" requires
    an observed endpoint beyond the wall, evidence of a real opening.
    """
    pts = np.asarray(endpoints_local)
    if len(pts) == 0:
        return np.empty(0, int), np.empty(0, int)
    axis = spec["axis"]
    direction = pts - np.asarray(camera_local)
    denom = direction[:, axis]
    valid = np.isfinite(direction).all(axis=1) & (np.abs(denom) > 1e-8)
    # Only cameras on the room side of this boundary provide meaningful rays.
    valid &= spec["sign"] * (camera_local[axis] - spec["plane"]) < beyond_tol
    t = np.full(len(pts), np.nan)
    t[valid] = (spec["plane"] - camera_local[axis]) / denom[valid]
    if mode == "occluded":
        valid &= t > 1.02
    elif mode == "through":
        valid &= (t > 0.0) & (t < 0.98)
        valid &= spec["sign"] * (pts[:, axis] - spec["plane"]) > beyond_tol
    elif mode == "opening":
        valid &= t > 0.0
    else:
        raise ValueError(f"invalid ray mode {mode}")
    if not valid.any():
        return np.empty(0, int), np.empty(0, int)
    intersections = np.asarray(camera_local) + t[valid, None] * direction[valid]
    return _points_to_wall_cells(intersections, spec, ceil_h, shape)


def _classify_wall_cells(observed: np.ndarray, occluded_views: np.ndarray,
                         semantic_opening: np.ndarray, through_views: np.ndarray,
                         min_through_views: int = 2,
                         complete_unknown: bool = False) -> np.ndarray:
    """Classify missing wall cells that are safe to complete.

    Object-occluded cells are always eligible. When a real wall plane has
    enough support, ``complete_unknown`` also fills unsupported cells connected
    to that observed plane. Semantic openings and repeated rays continuing
    beyond the wall remain hard exclusions in both modes.
    """
    from scipy.ndimage import binary_dilation, binary_propagation

    observed = binary_dilation(np.asarray(observed, bool), iterations=1)
    occluded = binary_dilation(np.asarray(occluded_views) > 0, iterations=1)
    opening = np.asarray(semantic_opening, bool) | (np.asarray(through_views) >= min_through_views)
    opening = binary_dilation(opening, iterations=1)
    eligible = occluded
    if complete_unknown and observed.any():
        # Flood only across the established wall surface. An opening spanning
        # the grid acts as a barrier instead of allowing invented geometry on
        # an unsupported region beyond it.
        connected_wall = binary_propagation(observed, mask=~opening)
        eligible = eligible | connected_wall
    return eligible & ~observed & ~opening


def _wall_plane_supported(observed: np.ndarray,
                          min_cells: int = 6) -> bool:
    """Whether real points establish a 2-D wall patch rather than an edge.

    Requiring support across both wall axes rejects floor-wall seams and single
    depth streaks, while still allowing a mostly dark or occluded wall to be
    completed from a small genuine patch.
    """
    cells = np.argwhere(np.asarray(observed, bool))
    if len(cells) < max(int(min_cells), 1):
        return False
    return len(np.unique(cells[:, 0])) >= 2 and len(np.unique(cells[:, 1])) >= 2


def _sample_mask_points_local(mask: np.ndarray, points_frame: np.ndarray,
                              T_total: np.ndarray, scale: float, offset: np.ndarray,
                              R: np.ndarray, max_points: int = 6000) -> np.ndarray:
    idx = np.flatnonzero(np.asarray(mask).ravel())
    if len(idx) > max_points:
        idx = idx[np.linspace(0, len(idx) - 1, max_points).astype(np.int64)]
    if not len(idx):
        return np.empty((0, 3), float)
    pts = points_frame.reshape(-1, 3)[idx]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if not len(pts):
        return np.empty((0, 3), float)
    pts_m = _apply44(T_total, pts) * scale + offset
    return pts_m @ R.T


def _wall_infill_evidence(
    glob_m: np.ndarray,
    world_pts: np.ndarray,
    good_px: np.ndarray,
    exclude_mask: np.ndarray,
    masks: list,
    dets_2d: list[dict],
    extrinsic: np.ndarray,
    T_total: np.ndarray,
    scale: float,
    offset: np.ndarray,
    R: np.ndarray,
    rect: tuple[float, float, float, float],
    ceil_h: float,
) -> list[dict]:
    """Accumulate per-view wall evidence from scan points and camera rays."""
    S, dH, dW = good_px.shape
    specs = [_wall_spec(i, rect) for i in range(4)]
    evidence = []
    glob_local = np.asarray(glob_m) @ R.T
    wall_tol = max(0.04, 0.025 * max(min(rect[1] - rect[0], rect[3] - rect[2]), 1e-6))
    for spec in specs:
        shape = _wall_grid_shape(spec, ceil_h)
        near = np.abs(glob_local[:, spec["axis"]] - spec["plane"]) <= wall_tol
        # Do not let a dense floor/ceiling seam falsely establish a wall plane.
        near &= ((glob_local[:, 1] > 0.05 * ceil_h) &
                 (glob_local[:, 1] < 0.95 * ceil_h))
        observed = np.zeros(shape, bool)
        iy, iu = _points_to_wall_cells(glob_local[near], spec, ceil_h, shape)
        observed[iy, iu] = True
        evidence.append({
            "spec": spec, "shape": shape, "observed": observed,
            "occluded_views": np.zeros(shape, np.uint16),
            "semantic_opening": np.zeros(shape, bool),
            "through_views": np.zeros(shape, np.uint16),
        })

    E = np.asarray(extrinsic, dtype=np.float64)
    cams_raw = -np.einsum("sij,sj->si", np.transpose(E[:, :3, :3], (0, 2, 1)), E[:, :3, 3])
    cams_local = (_apply44(T_total, cams_raw) * scale + offset) @ R.T

    for fi in range(S):
        opening_mask = np.zeros((dH, dW), bool)
        frame = dets_2d[fi] if fi < len(dets_2d) else {"labels": [], "boxes": []}
        for bi, label in enumerate(frame.get("labels", [])):
            if not _is_opening_label(str(label)):
                continue
            mask = masks[fi][bi] if fi < len(masks) and bi < len(masks[fi]) else None
            if mask is not None:
                opening_mask |= cv2.resize(mask.astype(np.uint8), (dW, dH),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
            elif bi < len(frame.get("boxes", [])):
                H0, W0 = frame.get("image_size", (dH, dW))
                x1, y1, x2, y2 = frame["boxes"][bi]
                xa, xb = np.clip(np.round([x1 / W0 * dW, x2 / W0 * dW]).astype(int), 0, dW)
                ya, yb = np.clip(np.round([y1 / H0 * dH, y2 / H0 * dH]).astype(int), 0, dH)
                opening_mask[ya:yb, xa:xb] = True

        object_mask = np.asarray(exclude_mask[fi], bool) & ~opening_mask
        through_mask = np.asarray(good_px[fi], bool) & ~np.asarray(exclude_mask[fi], bool)
        object_pts = _sample_mask_points_local(
            object_mask, world_pts[fi], T_total, scale, offset, R
        )
        opening_pts = _sample_mask_points_local(
            opening_mask, world_pts[fi], T_total, scale, offset, R
        )
        through_pts = _sample_mask_points_local(
            through_mask, world_pts[fi], T_total, scale, offset, R
        )

        for ev in evidence:
            for key, pts, mode in (
                ("occluded_views", object_pts, "occluded"),
                ("semantic_opening", opening_pts, "opening"),
                ("through_views", through_pts, "through"),
            ):
                iy, iu = _ray_wall_cells(
                    cams_local[fi], pts, ev["spec"], ceil_h, ev["shape"], mode
                )
                if not len(iy):
                    continue
                frame_cells = np.zeros(ev["shape"], bool)
                frame_cells[iy, iu] = True
                if key == "semantic_opening":
                    ev[key] |= frame_cells
                else:
                    ev[key] += frame_cells.astype(np.uint16)

    for ev in evidence:
        ev["plane_supported"] = _wall_plane_supported(ev["observed"])
        ev["fill"] = _classify_wall_cells(
            ev["observed"], ev["occluded_views"],
            ev["semantic_opening"], ev["through_views"],
            complete_unknown=ev["plane_supported"],
        )
    return evidence


def _mask_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Merge equal horizontal runs across rows; returns y0,y1,u0,u1 cells."""
    mask = np.asarray(mask, bool)
    active: dict[tuple[int, int], list[int]] = {}
    rectangles: list[tuple[int, int, int, int]] = []
    for y, row in enumerate(mask):
        padded = np.r_[False, row, False].astype(np.int8)
        edges = np.flatnonzero(np.diff(padded))
        runs = {(int(edges[k]), int(edges[k + 1])) for k in range(0, len(edges), 2)}
        for run in list(active):
            if run not in runs:
                y0, y1 = active.pop(run)
                rectangles.append((y0, y1, run[0], run[1]))
        for run in runs:
            if run in active:
                active[run][1] = y + 1
            else:
                active[run] = [y, y + 1]
    for run, (y0, y1) in active.items():
        rectangles.append((y0, y1, run[0], run[1]))
    return rectangles


def _wall_color_field(
    spec: dict,
    images: np.ndarray,
    world_pts: np.ndarray,
    sample_px: np.ndarray,
    T_total: np.ndarray,
    scale: float,
    offset: np.ndarray,
    R: np.ndarray,
    ceil_h: float,
    shape: tuple[int, int],
    fallback_color,
    wall_tol: float,
    min_points_per_view: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Locally varying wall colour field, mirroring _horizontal_color_field.

    Each frame contributes one mean RGB sample per supported grid cell, from
    points lying within `wall_tol` of this wall's plane — so a completed patch
    carries real captured wall texture sampled near it, not one flat colour
    for the whole wall. Missing cells use inverse-distance interpolation from
    up to four nearest observed neighbours, extending nearby real appearance.

    Returns (colors, observed_color_cells) where colors is (ny, nu, 3).
    """
    ny, nu = shape
    samples: list[list[np.ndarray]] = [[] for _ in range(ny * nu)]

    for fi in range(min(len(images), len(world_pts), len(sample_px))):
        sel = np.asarray(sample_px[fi], bool)
        if not sel.any():
            continue
        pts = _apply44(T_total, world_pts[fi][sel]) * scale + offset
        pts = pts @ R.T
        cols = np.transpose(images[fi], (1, 2, 0))[sel]
        if np.issubdtype(cols.dtype, np.floating):
            cols = cols * (255.0 if len(cols) and float(np.nanmax(cols)) <= 1.5 else 1.0)
        cols = np.asarray(cols, np.float64)
        u, y = pts[:, spec["u_axis"]], pts[:, 1]
        valid = (
            np.isfinite(pts).all(axis=1) & np.isfinite(cols).all(axis=1) &
            (np.abs(pts[:, spec["axis"]] - spec["plane"]) <= wall_tol) &
            (u >= spec["u_min"]) & (u <= spec["u_max"]) &
            (y >= 0.0) & (y <= ceil_h)
        )
        if not valid.any():
            continue
        uv, yv, colsv = u[valid], y[valid], cols[valid]
        iu = np.clip(np.floor((uv - spec["u_min"]) /
                              max(spec["u_max"] - spec["u_min"], 1e-9) * nu).astype(int), 0, nu - 1)
        iy = np.clip(np.floor(yv / max(ceil_h, 1e-9) * ny).astype(int), 0, ny - 1)
        cell = iy * nu + iu
        order = np.argsort(cell)
        cell, colsv = cell[order], colsv[order]
        ids, starts, counts = np.unique(cell, return_index=True, return_counts=True)
        sums = np.add.reduceat(colsv, starts, axis=0)
        for cid, total, count in zip(ids, sums, counts):
            if count >= max(int(min_points_per_view), 1):
                samples[int(cid)].append(total / count)

    fallback = np.asarray(fallback_color, np.float64)[:3]
    field = np.tile(fallback, (ny, nu, 1))
    observed = np.zeros((ny, nu), bool)
    for cid, values in enumerate(samples):
        if values:
            iy, iu = divmod(cid, nu)
            field[iy, iu] = np.median(np.asarray(values), axis=0)
            observed[iy, iu] = True

    if observed.any() and not observed.all():
        from scipy.spatial import cKDTree

        known = np.argwhere(observed)
        unknown = np.argwhere(~observed)
        k = min(4, len(known))
        distances, indices = cKDTree(known).query(unknown, k=k)
        distances = np.asarray(distances).reshape(len(unknown), k)
        indices = np.asarray(indices).reshape(len(unknown), k)
        weights = 1.0 / np.maximum(distances, 0.75) ** 2
        neighbour_colors = field[known[indices, 0], known[indices, 1]]
        interpolated = np.sum(neighbour_colors * weights[..., None], axis=1)
        interpolated /= np.sum(weights, axis=1, keepdims=True)
        field[unknown[:, 0], unknown[:, 1]] = interpolated

    return np.clip(field, 0, 255).astype(np.uint8), observed


def _wall_fill_mesh(index: int, fill: np.ndarray,
                    rect: tuple[float, float, float, float], ceil_h: float,
                    R: np.ndarray, color, color_field: np.ndarray | None = None,
                    thickness: float = 0.02,
                    gap: float = 0.01) -> trimesh.Trimesh | None:
    """Create backing patches for missing wall cells.

    With `color_field` each cell becomes its own thin coloured box carrying
    locally sampled real wall texture (see _wall_color_field), instead of one
    flat colour for every patch. A solid box — not a single-sided plane —
    keeps the inward-facing side correctly coloured regardless of winding, the
    same guarantee _oriented_box already gives every other wall/box mesh here.
    The legacy flat-colour path remains for callers without image evidence and
    merges contiguous cells into fewer, larger boxes.
    """
    spec = _wall_spec(index, rect)
    ny, nu = fill.shape
    center_axis = spec["plane"] + spec["sign"] * (gap + thickness / 2)

    if color_field is not None:
        colors = np.asarray(color_field)
        if colors.shape != (ny, nu, 3):
            raise ValueError(f"color_field shape {colors.shape} != {(ny, nu, 3)}")
        meshes = []
        for iy, iu in np.argwhere(fill):
            ya, yb = iy / ny * ceil_h, (iy + 1) / ny * ceil_h
            ua = spec["u_min"] + iu / nu * (spec["u_max"] - spec["u_min"])
            ub = spec["u_min"] + (iu + 1) / nu * (spec["u_max"] - spec["u_min"])
            if spec["axis"] == 0:
                extents = [thickness, yb - ya, ub - ua]
                center = [center_axis, (ya + yb) / 2, (ua + ub) / 2]
            else:
                extents = [ub - ua, yb - ya, thickness]
                center = [(ua + ub) / 2, (ya + yb) / 2, center_axis]
            meshes.append(_oriented_box(extents, center, colors[iy, iu].tolist(), R))
        return trimesh.util.concatenate(meshes) if meshes else None

    meshes = []
    for y0, y1, u0, u1 in _mask_rectangles(fill):
        ya, yb = y0 / ny * ceil_h, y1 / ny * ceil_h
        ua = spec["u_min"] + u0 / nu * (spec["u_max"] - spec["u_min"])
        ub = spec["u_min"] + u1 / nu * (spec["u_max"] - spec["u_min"])
        if spec["axis"] == 0:
            extents = [thickness, yb - ya, ub - ua]
            center = [center_axis, (ya + yb) / 2, (ua + ub) / 2]
        else:
            extents = [ub - ua, yb - ya, thickness]
            center = [(ua + ub) / 2, (ya + yb) / 2, center_axis]
        meshes.append(_oriented_box(extents, center, color, R))
    return trimesh.util.concatenate(meshes) if meshes else None


def _horizontal_infill_evidence(
    glob_m: np.ndarray,
    R: np.ndarray,
    rect: tuple[float, float, float, float],
    ceil_h: float,
    target_cell_m: float = 0.20,
    min_points_per_cell: int = 3,
) -> dict[str, dict]:
    """Find unobserved floor/ceiling cells inside the fitted room footprint.

    Horizontal surfaces do not have intentional door/window holes. Once a
    floor or ceiling plane has been established, any unsupported grid cell is
    therefore eligible for a backing patch. Multiple samples are required to
    mark a cell observed so isolated low-light depth noise cannot preserve a
    large hole. Patches sit behind the scan plane, so overlap at fragment edges
    does not require an exclusion halo.
    """
    xmin, xmax, zmin, zmax = rect
    nx = int(np.clip(np.ceil((xmax - xmin) / target_cell_m), 1, 96))
    nz = int(np.clip(np.ceil((zmax - zmin) / target_cell_m), 1, 96))
    shape = (nz, nx)
    pts = np.asarray(glob_m) @ R.T
    tol = max(0.04, 0.04 * max(ceil_h, 1e-6))
    result = {}
    for name, plane in (("floor", 0.0), ("ceiling", ceil_h)):
        near = np.abs(pts[:, 1] - plane) <= tol
        surface_pts = pts[near]
        inside = (
            (surface_pts[:, 0] >= xmin) & (surface_pts[:, 0] <= xmax) &
            (surface_pts[:, 2] >= zmin) & (surface_pts[:, 2] <= zmax) &
            np.isfinite(surface_pts).all(axis=1)
        )
        surface_pts = surface_pts[inside]
        support_count = np.zeros(shape, np.uint32)
        if len(surface_pts):
            ix = np.floor((surface_pts[:, 0] - xmin) /
                          max(xmax - xmin, 1e-9) * nx).astype(int)
            iz = np.floor((surface_pts[:, 2] - zmin) /
                          max(zmax - zmin, 1e-9) * nz).astype(int)
            np.add.at(
                support_count,
                (np.clip(iz, 0, nz - 1), np.clip(ix, 0, nx - 1)),
                1,
            )
        observed = support_count >= max(int(min_points_per_cell), 1)
        result[name] = {
            "shape": shape,
            "support_count": support_count,
            "observed": observed,
            "fill": ~observed,
        }
    return result


def _horizontal_color_field(
    surface: str,
    images: np.ndarray,
    world_pts: np.ndarray,
    sample_px: np.ndarray,
    T_total: np.ndarray,
    scale: float,
    offset: np.ndarray,
    R: np.ndarray,
    rect: tuple[float, float, float, float],
    ceil_h: float,
    shape: tuple[int, int],
    fallback_color,
    min_points_per_view: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a locally varying floor/ceiling colour field from video views.

    Each frame contributes one mean RGB sample per supported grid cell, so a
    close/dense frame cannot overwhelm the other viewpoints. The median of
    those per-view samples suppresses exposure changes and transient pixels.
    Missing cells use inverse-distance interpolation from their four nearest
    observed neighbours; this extends nearby appearance rather than painting
    every patch with one room-wide colour.

    Returns ``(colors, observed_color_cells)`` where colors is ``(nz,nx,3)``.
    ``sample_px`` should already contain the reconstruction confidence gate and
    exclude movable-object masks.
    """
    if surface not in {"floor", "ceiling"}:
        raise ValueError(f"invalid horizontal surface {surface}")
    nz, nx = shape
    xmin, xmax, zmin, zmax = rect
    plane = 0.0 if surface == "floor" else ceil_h
    tol = max(0.04, 0.04 * max(ceil_h, 1e-6))
    samples: list[list[np.ndarray]] = [[] for _ in range(nz * nx)]

    for fi in range(min(len(images), len(world_pts), len(sample_px))):
        sel = np.asarray(sample_px[fi], bool)
        if not sel.any():
            continue
        pts = _apply44(T_total, world_pts[fi][sel]) * scale + offset
        pts = pts @ R.T
        cols = np.transpose(images[fi], (1, 2, 0))[sel]
        if np.issubdtype(cols.dtype, np.floating):
            cols = cols * (255.0 if len(cols) and float(np.nanmax(cols)) <= 1.5 else 1.0)
        valid = (
            np.isfinite(pts).all(axis=1) & np.isfinite(cols).all(axis=1) &
            (np.abs(pts[:, 1] - plane) <= tol) &
            (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
            (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
        )
        if not valid.any():
            continue
        pts, cols = pts[valid], np.asarray(cols[valid], np.float64)
        ix = np.floor((pts[:, 0] - xmin) / max(xmax - xmin, 1e-9) * nx).astype(int)
        iz = np.floor((pts[:, 2] - zmin) / max(zmax - zmin, 1e-9) * nz).astype(int)
        cell = np.clip(iz, 0, nz - 1) * nx + np.clip(ix, 0, nx - 1)
        order = np.argsort(cell)
        cell, cols = cell[order], cols[order]
        ids, starts, counts = np.unique(cell, return_index=True, return_counts=True)
        sums = np.add.reduceat(cols, starts, axis=0)
        for cid, total, count in zip(ids, sums, counts):
            if count >= max(int(min_points_per_view), 1):
                samples[int(cid)].append(total / count)

    fallback = np.asarray(fallback_color, np.float64)[:3]
    field = np.tile(fallback, (nz, nx, 1))
    observed = np.zeros((nz, nx), bool)
    for cid, values in enumerate(samples):
        if values:
            iz, ix = divmod(cid, nx)
            field[iz, ix] = np.median(np.asarray(values), axis=0)
            observed[iz, ix] = True

    if observed.any() and not observed.all():
        from scipy.spatial import cKDTree

        known = np.argwhere(observed)
        unknown = np.argwhere(~observed)
        k = min(4, len(known))
        distances, indices = cKDTree(known).query(unknown, k=k)
        distances = np.asarray(distances).reshape(len(unknown), k)
        indices = np.asarray(indices).reshape(len(unknown), k)
        weights = 1.0 / np.maximum(distances, 0.75) ** 2
        neighbour_colors = field[known[indices, 0], known[indices, 1]]
        interpolated = np.sum(neighbour_colors * weights[..., None], axis=1)
        interpolated /= np.sum(weights, axis=1, keepdims=True)
        field[unknown[:, 0], unknown[:, 1]] = interpolated

    return np.clip(field, 0, 255).astype(np.uint8), observed


def _horizontal_fill_mesh(
    surface: str,
    fill: np.ndarray,
    rect: tuple[float, float, float, float],
    ceil_h: float,
    R: np.ndarray,
    color,
    color_field: np.ndarray | None = None,
    thickness: float = 0.02,
    gap: float = 0.01,
) -> trimesh.Trimesh | None:
    """Create backing patches for missing floor or ceiling cells.

    With ``color_field`` the visible plane is tessellated per cell and carries
    locally sampled vertex colours. The legacy flat-colour path remains for
    callers without image evidence and uses merged thin boxes.
    """
    if surface not in {"floor", "ceiling"}:
        raise ValueError(f"invalid horizontal surface {surface}")
    xmin, xmax, zmin, zmax = rect
    nz, nx = np.asarray(fill).shape
    plane = 0.0 if surface == "floor" else ceil_h
    sign = -1.0 if surface == "floor" else 1.0
    if color_field is not None:
        colors = np.asarray(color_field)
        if colors.shape != (nz, nx, 3):
            raise ValueError(f"color_field shape {colors.shape} != {(nz, nx, 3)}")
        vertices, faces, vertex_colors = [], [], []
        y = plane + sign * gap
        for iz, ix in np.argwhere(fill):
            xa = xmin + ix / nx * (xmax - xmin)
            xb = xmin + (ix + 1) / nx * (xmax - xmin)
            za = zmin + iz / nz * (zmax - zmin)
            zb = zmin + (iz + 1) / nz * (zmax - zmin)
            base = len(vertices)
            vertices.extend([[xa, y, za], [xb, y, za],
                             [xb, y, zb], [xa, y, zb]])
            if surface == "floor":                 # visible from above (+Y)
                faces.extend([[base, base + 2, base + 1],
                              [base, base + 3, base + 2]])
            else:                                  # visible from below (-Y)
                faces.extend([[base, base + 1, base + 2],
                              [base, base + 2, base + 3]])
            vertex_colors.extend([colors[iz, ix].tolist() + [255]] * 4)
        if not faces:
            return None
        mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices, float) @ R,
            faces=np.asarray(faces, np.int64),
            process=False,
        )
        mesh.visual.vertex_colors = np.asarray(vertex_colors, np.uint8)
        return mesh

    meshes = []
    for z0, z1, x0, x1 in _mask_rectangles(fill):
        xa = xmin + x0 / nx * (xmax - xmin)
        xb = xmin + x1 / nx * (xmax - xmin)
        za = zmin + z0 / nz * (zmax - zmin)
        zb = zmin + z1 / nz * (zmax - zmin)
        center_y = plane + sign * (gap + thickness / 2)
        meshes.append(_oriented_box(
            [xb - xa, thickness, zb - za],
            [(xa + xb) / 2, center_y, (za + zb) / 2],
            color, R,
        ))
    return trimesh.util.concatenate(meshes) if meshes else None


def _export_raw_scan(points_m: np.ndarray, colors: np.ndarray, path: str,
                     max_points: int = 300_000) -> int:
    """Export the editor-aligned VGGT cloud as a GLB POINTS primitive."""
    pts = np.asarray(points_m, dtype=np.float32)
    cols = np.asarray(colors)
    valid = np.isfinite(pts).all(axis=1)
    pts, cols = pts[valid], cols[valid]
    if len(pts) > max_points:
        take = np.linspace(0, len(pts) - 1, max_points).astype(np.int64)
        pts, cols = pts[take], cols[take]
    if np.issubdtype(cols.dtype, np.floating):
        cols = np.nan_to_num(cols, nan=0.0, posinf=255.0, neginf=0.0)
        color_scale = 255.0 if len(cols) and float(np.max(cols)) <= 1.5 else 1.0
        cols = (cols * color_scale).clip(0, 255).astype(np.uint8)
    else:
        cols = cols.clip(0, 255).astype(np.uint8)
    if len(pts) == 0:
        return 0
    scan = trimesh.Scene()
    scan.add_geometry(trimesh.PointCloud(vertices=pts, colors=cols), geom_name="raw_scan")
    scan.export(path)
    return int(len(pts))


def _object_fit_points(inst: dict, scale: float, offset: np.ndarray,
                       max_points: int = 384) -> list[list[float]]:
    """Return a compact scan sample in the editor object's local frame.

    These points are retained solely for fitting a later replacement GLB. They
    are voxel-deduplicated so repeatedly observed surfaces do not overwhelm a
    less frequently visible side, and clipped to the placement with a little
    tolerance to remove obvious background/support-surface leakage.
    """
    placement = inst["placement"]
    size = np.asarray(placement["size"], float) * scale
    position = np.asarray(placement["position"], float) * scale + offset
    points = np.asarray(inst.get("pts", []), float) * scale + offset
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        return []

    yaw = float(placement.get("yaw", 0.0))
    c, s = np.cos(-yaw), np.sin(-yaw)
    rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    local = (points - position) @ rotation.T
    local = local[np.isfinite(local).all(axis=1)]
    pad = size * np.array([0.15, 0.10, 0.15])
    keep = (
        (np.abs(local[:, 0]) <= size[0] / 2 + pad[0])
        & (local[:, 1] >= -pad[1])
        & (local[:, 1] <= size[1] + pad[1])
        & (np.abs(local[:, 2]) <= size[2] / 2 + pad[2])
    )
    local = local[keep]
    if len(local) == 0:
        return []

    voxel = max(float(np.linalg.norm(size)) / 70.0, 1e-5)
    keys = np.floor(local / voxel).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    local = local[np.sort(unique_idx)]
    if len(local) > max_points:
        take = np.linspace(0, len(local) - 1, max_points).astype(np.int64)
        local = local[take]
    return np.round(local, 4).tolist()


def _carve_room_shell(bg: trimesh.Trimesh, R: np.ndarray,
                      rect: tuple[float, float, float, float],
                      ceil_h: float, pad: float = 0.2) -> trimesh.Trimesh:
    """Drop background faces on the room boundary (walls/floor/ceiling) so the
    synthetic shell *replaces* them instead of overlapping. A face is boundary
    if its centroid (in the local frame) sits within `pad` of a fitted wall
    plane, below the floor band, or above the ceiling — while inside the room
    footprint. Interior structure (a central pillar, mid-height fixtures) is
    kept."""
    xmin, xmax, zmin, zmax = rect
    if len(bg.faces) == 0:
        return bg
    v_local = np.asarray(bg.vertices) @ R.T
    cen = v_local[bg.faces].mean(axis=1)
    xr, yr, zr = cen[:, 0], cen[:, 1], cen[:, 2]
    inside = (xr > xmin - pad) & (xr < xmax + pad) & (zr > zmin - pad) & (zr < zmax + pad)
    near_wall = ((np.abs(xr - xmin) < pad) | (np.abs(xr - xmax) < pad) |
                 (np.abs(zr - zmin) < pad) | (np.abs(zr - zmax) < pad))
    near_floor = yr < pad * 0.75
    near_ceil = yr > ceil_h - pad
    drop = inside & (near_wall | near_floor | near_ceil)
    bg = bg.copy()
    bg.update_faces(~drop)
    bg.remove_unreferenced_vertices()
    return bg


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
    reuse_duplicates: bool = False,
    generate_fn: Callable[[np.ndarray], bytes | None] | None = None,
    label_review_fn: Callable[[list[dict], list[str], list[dict]], tuple[list[dict], dict]] | None = None,
    abort_check: Callable[[], None] | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> dict:
    """Build an editable scene directory. Returns the scene.json dict.

    generate_fn, if given, is an image-to-3D generator: generate_fn(crop_rgb)
    returns GLB bytes (or None to decline). It's used only for objects that
    would otherwise fall back to a synthetic prefab/box — i.e. where there
    wasn't enough real scan geometry — turning those into a generated asset
    (source "trellis"). Real scan geometry is still preferred whenever present.
    Pairs naturally with reuse_duplicates: the generator then runs once per
    visually verified reusable-asset group (see modal_app.py's
    generate_object_glb container).

    reuse_duplicates: when True, instances must share a label, match closely in
    metric size, and pass a conservative masked-crop appearance comparison. One
    asset is built per verified group, and other members reference that GLB with
    per-instance model_scale/model_offset. Missing/uncertain SAM evidence fails
    closed to separate assets. Only geometry is shared — each object keeps its
    own position/yaw. Off by default: per-instance meshes remain the faithful
    default; reuse trades some build time for verified cross-object consistency.

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
        reflector_masks: list[tuple[str, np.ndarray]] = []
        for rbi, reflector_label in enumerate(frame["labels"]):
            if (not _is_reflector_label(str(reflector_label)) or
                    rbi >= len(masks[fi]) or masks[fi][rbi] is None):
                continue
            reflector_masks.append((
                str(reflector_label),
                cv2.resize(masks[fi][rbi].astype(np.uint8), (dW, dH),
                           interpolation=cv2.INTER_NEAREST).astype(bool),
            ))
        for bi, (label, score) in enumerate(zip(frame["labels"], frame["scores"])):
            if bi >= len(masks[fi]) or masks[fi][bi] is None:
                continue
            m_small = cv2.resize(masks[fi][bi].astype(np.uint8), (dW, dH),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
            # Doors and windows are architectural evidence, not movable scene
            # objects.  Keep their pixels in the background reconstruction and
            # let _wall_infill_evidence use their masks to veto wall patches.
            if _is_opening_label(str(label)):
                continue
            sel = m_small & good_px[fi]
            if sel.sum() < 30:
                continue
            inside_ratio, enclosing_labels = 0.0, []
            if not _is_reflector_label(str(label)):
                for reflector_label, reflector_mask in reflector_masks:
                    ratio = float(np.sum(m_small & reflector_mask) / max(np.sum(m_small), 1))
                    if ratio >= 0.50:
                        enclosing_labels.append(reflector_label)
                    inside_ratio = max(inside_ratio, ratio)
            pts = _apply44(T_align, world_pts[fi][sel])
            cols = (color_hwc[sel] * 255).clip(0, 255).astype(np.uint8)
            observations.append(_make_observation(str(label), float(score), fi, pts, cols,
                                                  box=np.asarray(frame["boxes"][bi]),
                                                  det_idx=bi,
                                                  reflector_inside_ratio=inside_ratio,
                                                  reflector_labels=enclosing_labels))
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

    # ── 2b. Align the room to the world grid ─────────────────────────────────
    # T_align leaves the scene at the first camera's arbitrary heading, so the
    # whole room sits skew to the editor's axis-aligned grid ("spun the wrong
    # way"). Rotate the entire scene about +Y so the room's walls run along the
    # world axes; the minimum-area rectangle locks onto the walls rather than
    # the furniture the way a PCA principal axis would.
    room_ang = _min_area_rect_yaw(glob[:, [0, 2]])
    ca, sa = np.cos(room_ang), np.sin(room_ang)
    T_yaw = np.eye(4)
    T_yaw[0, 0], T_yaw[0, 2] = ca, sa
    T_yaw[2, 0], T_yaw[2, 2] = -sa, ca
    T_total = T_yaw @ T_total
    glob = _apply44(T_yaw, glob)
    for o in observations:
        o["pts"] = _apply44(T_yaw, o["pts"])
        o["center"] = _apply44(T_yaw, o["center"][None])[0]

    floor_y = float(np.percentile(glob[:, 1], 3))
    ceil_y = float(np.percentile(glob[:, 1], 97))
    room_h = ceil_y - floor_y
    room_yaw = 0.0                       # scene is now grid-aligned

    # ── 3. Cluster into instances, place them ────────────────────────────────
    instances = cluster_observations(observations)
    clustered_count = len(instances)
    instances = _consolidate_overlapping_placements(
        instances, floor_y, room_yaw, floor_snap_tol=room_h * 0.12
    )
    if len(instances) < clustered_count:
        print(
            f"[SceneBuilder] Editor-overlap consolidation: {clustered_count} objects → "
            f"{len(instances)} physical objects."
        )
    print(f"[SceneBuilder] {len(instances)} object instance(s) after clustering.")
    association_summary = {
        "algorithm": "merge-oriented-same-label-v1",
        "observations": len(observations),
        "instances_before_editor_overlap_check": clustered_count,
        "instances_after_editor_overlap_check": len(instances),
        "instances_before_review": len(instances),
        "multi_view_instances": sum(inst.get("frames_seen", 0) >= 2 for inst in instances),
        "single_view_instances": sum(inst.get("frames_seen", 0) < 2 for inst in instances),
    }

    label_review_summary = {"enabled": False, "reviewed": 0, "relabeled": 0, "rejected": 0}
    if label_review_fn is not None and instances:
        if abort_check is not None:
            abort_check()
        if progress_cb is not None:
            progress_cb(0.42, "Reviewing object labels with Gemini…")
        try:
            instances, label_review_summary = label_review_fn(instances, image_paths, dets_2d)
            # A semantic review may identify an architectural opening that was
            # originally mislabeled as furniture. Keep it as wall evidence,
            # but never export it as a movable object.
            instances = [inst for inst in instances
                         if not _is_opening_label(str(inst.get("label", "")))]
            exclude_mask = _exclude_mask_for_instances(instances, masks, (S, dH, dW))
            print("[SceneBuilder] Gemini label review: "
                  f"{label_review_summary.get('relabeled', 0)} relabeled, "
                  f"{label_review_summary.get('rejected', 0)} rejected.")
        except Exception as exc:
            label_review_summary = {
                "enabled": True, "reviewed": 0, "relabeled": 0, "rejected": 0,
                "error": str(exc)[:300],
            }
            print(f"[SceneBuilder] Gemini label review failed; keeping detector labels ({exc}).")
    association_summary["instances_after_review"] = len(instances)

    # Reflection filtering is deliberately after optional semantic relabeling:
    # geometry profiles should use the final category, not a detector synonym.
    instances, reflection_summary = _filter_reflection_instances(
        instances, floor_y, room_h, T_total=T_total, extrinsic=extrinsic,
        intrinsic=intrinsic, depth=depth, masks=masks,
    )
    association_summary["reflection_verification"] = reflection_summary
    association_summary["instances_after_reflection_check"] = len(instances)
    if reflection_summary["rejected"]:
        print(
            f"[SceneBuilder] Geometry verification rejected "
            f"{reflection_summary['rejected']} candidate(s) "
            f"(reflection and/or unsupported floating placement)."
        )
    # Always rebuild this after all semantic and geometric rejection. Otherwise
    # a discarded reflected object would still carve a hole in the background.
    exclude_mask = _exclude_mask_for_instances(instances, masks, (S, dH, dW))

    placed = []
    for inst in instances:
        placement = place_instance(inst, floor_y, room_yaw,
                                   floor_snap_tol=room_h * 0.12)
        placed.append({**inst, "placement": placement})
    photo_selection_summary = _refresh_object_support_images(placed, image_paths, masks)
    association_summary["support_images"] = photo_selection_summary
    if photo_selection_summary["missing"]:
        print(
            f"[SceneBuilder] Support images: {photo_selection_summary['selected']} selected, "
            f"{photo_selection_summary['missing']} missing valid frame evidence."
        )

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

    # Preserve the original VGGT samples in exactly the same levelled, scaled
    # coordinate system as the editable meshes. Sample per frame so a long
    # video cannot let a few dense frames consume the entire overlay budget.
    raw_pts, raw_cols = [], []
    rgb_hwc = np.transpose(images, (0, 2, 3, 1))
    per_frame = max(300_000 // max(S, 1), 1)
    for fi in range(S):
        ij = np.flatnonzero(good_px[fi].ravel())
        if len(ij) > per_frame:
            ij = ij[np.linspace(0, len(ij) - 1, per_frame).astype(np.int64)]
        if len(ij):
            raw_pts.append(world_pts[fi].reshape(-1, 3)[ij])
            raw_cols.append(rgb_hwc[fi].reshape(-1, 3)[ij])
    raw_point_count = 0
    if raw_pts:
        aligned_raw = _apply44(T_total, np.vstack(raw_pts)) * scale + offset
        raw_point_count = _export_raw_scan(
            aligned_raw, np.vstack(raw_cols), os.path.join(scene_dir, "raw_scan.glb")
        )

    # ── 5b. Group visually equivalent objects for safe asset reuse ────────────
    # Off by default (every instance is its own group). When enabled, label and
    # size provide a coarse filter, then masked-crop appearance must agree. The
    # rest reference one verified mesh through model_scale/model_offset — the
    # same fit contract the editor's "Replace Model" upload applies.
    groups = (
        _group_reusable_assets(placed, image_paths, masks)
        if reuse_duplicates else [[i] for i in range(len(placed))]
    )
    rep_of: dict[int, int] = {}
    for g in groups:
        # Representative = the best-supported instance (most frames, then score,
        # then most points) — its scan geometry is the cleanest to share.
        rep = max(g, key=lambda i: (placed[i]["frames_seen"], placed[i]["score"],
                                    len(placed[i]["pts"])))
        for i in g:
            rep_of[i] = rep
    if reuse_duplicates and len(groups) < len(placed):
        print(f"[SceneBuilder] Asset reuse: {len(placed)} objects → {len(groups)} unique asset(s).")

    def _build_object_glb(i: int) -> tuple[str, np.ndarray, np.ndarray, str, dict | None]:
        """Build, texture and export instance i's mesh. Returns
        (glb_rel, bbox_lo, bbox_hi, source)."""
        p = placed[i]
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

        scan_quality = None
        if source.startswith("scan-"):
            yaw = pl["yaw"]
            c_, s_ = np.cos(-yaw), np.sin(-yaw)
            R_yaw = np.array([[c_, 0, -s_], [0, 1, 0], [s_, 0, c_]])
            observed_m = np.asarray(p["pts"]) * scale + offset
            observed_local = (observed_m - np.asarray(pos_m)) @ R_yaw.T
            scan_quality = _scan_mesh_quality(
                mesh, observed_local, size_m, p.get("frames_seen", 0)
            )
            if not scan_quality["good"]:
                print(
                    f"[SceneBuilder] {obj_id} {source} rejected by point support "
                    f"(score={scan_quality['score']}, support={scan_quality['support']}); "
                    "using fallback when one is available."
                )
                if generate_fn is not None or p["label"].lower().strip() in _PREFAB_KINDS:
                    tint = (
                        np.median(p["cols"], axis=0).clip(0, 255).astype(int)
                        if len(p["cols"]) else np.array([128, 128, 128])
                    )
                    mesh, source = _build_fallback_mesh(p["label"], size_m, tint)
                else:
                    # For an unknown category with no generator, a partial real
                    # surface remains more informative than an arbitrary box.
                    source = "scan-low-support"

        # Upgrade synthetic fallbacks (prefab/box — i.e. too little real scan
        # geometry) to a generated asset when a generator is wired in. The scan
        # tiers above are always preferred; generation only replaces what would
        # otherwise be a parametric stand-in. On OOM/failure generate_fn returns
        # None and the prefab/box is kept, so a scene never breaks on this.
        if source in ("prefab", "box") and generate_fn is not None:
            crop = _object_crop_rgb(image_paths, masks, p)
            if crop is not None:
                glb_bytes = generate_fn(crop)
                if glb_bytes:
                    gmesh = _glb_bytes_to_local_mesh(glb_bytes, size_m)
                    if gmesh is not None:
                        mesh, source = _ensure_material(gmesh), "trellis"

        # Scan-derived meshes get the real photo projected on as a texture
        # when their best frame saw enough of them. All scan paths share the
        # same local convention (yaw-derotated, bottom-centre at origin), so
        # one inverse transform maps local verts back to VGGT camera space:
        # local → leveled scene units → undo T_total.
        if source.startswith("scan-") and p.get("best_frame") is not None:
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
        lo_b, hi_b = mesh.bounds
        return (f"objects/{obj_id}.glb", np.asarray(lo_b, float),
                np.asarray(hi_b, float), source, scan_quality)

    # Pass 1 — build one glb per representative (the expensive work).
    built: dict[int, tuple[str, np.ndarray, np.ndarray, str, dict | None]] = {}
    reps = sorted(set(rep_of.values()))
    for ri, rep in enumerate(reps):
        if abort_check is not None:
            abort_check()
        if progress_cb is not None:
            progress_cb(0.55 + 0.4 * ri / max(len(reps), 1),
                        f"Building object meshes: {ri}/{len(reps)}")
        built[rep] = _build_object_glb(rep)

    # Pass 2 — one scene.json entry per instance. A member references its
    # representative's glb with a per-instance fit transform; a representative
    # (the only member when reuse is off) keeps the plain built glb.
    objects_json = []
    for i, p in enumerate(placed):
        pl = p["placement"]
        size_m = [s * scale for s in pl["size"]]
        pos_m = (np.array(pl["position"]) * scale + offset).tolist()
        obj_id = f"obj_{i:03d}"
        rep = rep_of[i]
        glb_rel, lo_b, hi_b, rep_source, scan_quality = built[rep]

        entry = {
            "id": obj_id,
            "label": p["label"],
            "score": round(p["score"], 3),
            "frames_seen": p["frames_seen"],
            "position": [round(v, 4) for v in pos_m],
            "yaw": round(pl["yaw"], 5),
            "size": [round(v, 4) for v in size_m],
            "glb": glb_rel,
        }
        fit_points = _object_fit_points(p, scale, offset)
        if fit_points:
            entry["fit_points"] = fit_points
        if rep == i:
            entry["source"] = rep_source
        else:
            # Fit the shared asset (built at the representative's size, with its
            # own bbox) to THIS instance's detected size — the same math the
            # editor applies to a Replace-Model upload: offset re-centres the
            # asset to bottom-centre, then scale stretches its bbox to size_m.
            bbox = np.maximum(hi_b - lo_b, 1e-6)
            center = (lo_b + hi_b) / 2.0
            entry["source"] = f"reuse:{rep_source}"
            entry["reuse_of"] = f"obj_{rep:03d}"
            entry["model_offset"] = [round(float(-center[0]), 5),
                                     round(float(-lo_b[1]), 5),
                                     round(float(-center[2]), 5)]
            entry["model_scale"] = [round(float(size_m[k] / bbox[k]), 5) for k in range(3)]
            similarity = _appearance_similarity(
                p.get("_asset_appearance_descriptor"),
                placed[rep].get("_asset_appearance_descriptor"),
            )
            entry["asset_reuse_similarity"] = round(similarity, 3)

        photo_rel = _save_object_photo(image_paths, p, scene_dir, obj_id)
        if photo_rel:
            entry["photo"] = photo_rel
        input_rel = _save_object_input_crop(image_paths, masks, p, scene_dir, obj_id)
        if input_rel:
            entry["input_crop"] = input_rel
        if scan_quality is not None:
            entry["scan_quality"] = scan_quality
        if p.get("original_label"):
            entry["original_label"] = p["original_label"]
        if p.get("label_review"):
            entry["label_review"] = p["label_review"]
        if p.get("photo_selection"):
            entry["photo_selection"] = p["photo_selection"]
        if p.get("geometry_verification"):
            entry["geometry_verification"] = p["geometry_verification"]
        objects_json.append(entry)

    # ── 6. Scan-preserving background + coverage-aware boundary fill ─────────
    if progress_cb is not None:
        progress_cb(0.95, "Fusing background mesh…")
    print("[SceneBuilder] Fusing scan-preserving background mesh…")
    bg = fuse_tsdf_raw_mesh(
        predictions,
        TSDFConfig(conf_percentile=conf_thres, auto_resolution=mesh_resolution),
        exclude_mask=exclude_mask,
    )
    recolor_best_view(bg, predictions, label="background")
    bg.apply_transform(T_total)
    bg.apply_scale(scale)
    bg.apply_translation(offset)

    # ── Fit the room rectangle to the real wall planes, in the yaw frame ──────
    glob_m = glob * scale + offset                       # metric, floor at y≈0
    R = _yaw_basis(room_yaw)
    ceil_h = float(room_h * scale)
    xmin, xmax, zmin, zmax = _fit_room_rect(glob_m @ R.T, ceil_h)

    surface_coverage = _room_surface_coverage(
        glob_m, R, (xmin, xmax, zmin, zmax), ceil_h
    )
    wall_evidence = _wall_infill_evidence(
        glob_m=glob_m,
        world_pts=world_pts,
        good_px=good_px,
        exclude_mask=exclude_mask,
        masks=masks,
        dets_2d=dets_2d,
        extrinsic=extrinsic,
        T_total=T_total,
        scale=scale,
        offset=offset,
        R=R,
        rect=(xmin, xmax, zmin, zmax),
        ceil_h=ceil_h,
    )
    horizontal_evidence = _horizontal_infill_evidence(
        glob_m, R, (xmin, xmax, zmin, zmax), ceil_h
    )

    # Named (not anonymous) so the editor can find and toggle the walls/ceiling
    # independently — they enclose the room and can block the camera's view of
    # the interior otherwise.
    room_scene = trimesh.Scene()
    if len(bg.faces) > 0:
        room_scene.add_geometry(_ensure_material(bg), geom_name="background")

    floor_cols = _floor_color(glob, floor_y, images, good_px, world_pts, T_total)
    surface_sample_px = good_px & ~exclude_mask
    floor_color_field, floor_color_observed = _horizontal_color_field(
        "floor", images, world_pts, surface_sample_px, T_total, scale, offset, R,
        (xmin, xmax, zmin, zmax), ceil_h,
        horizontal_evidence["floor"]["shape"], floor_cols,
    )
    floor_patch = _horizontal_fill_mesh(
        "floor", horizontal_evidence["floor"]["fill"],
        (xmin, xmax, zmin, zmax), ceil_h, R, floor_cols,
        color_field=floor_color_field,
    )
    if floor_patch is not None:
        room_scene.add_geometry(_ensure_material(floor_patch), geom_name="floor_fill")

    # Ceiling: only synthesise when near-ceiling points actually spread across
    # the room (see _has_ceiling_support) — otherwise ceil_y is just "top of
    # the tallest observed object" and a slab there would be wrong.
    ceil_cols = floor_cols
    has_ceiling = _has_ceiling_support(glob, ceil_y, room_h)
    ceiling_color_observed = np.zeros(
        horizontal_evidence["ceiling"]["shape"], bool
    )
    if has_ceiling:
        ceil_cols = _ceiling_color(glob, ceil_y, images, good_px, world_pts, T_total)
        ceiling_color_field, ceiling_color_observed = _horizontal_color_field(
            "ceiling", images, world_pts, surface_sample_px, T_total, scale, offset, R,
            (xmin, xmax, zmin, zmax), ceil_h,
            horizontal_evidence["ceiling"]["shape"], ceil_cols,
        )
        ceiling_patch = _horizontal_fill_mesh(
            "ceiling", horizontal_evidence["ceiling"]["fill"],
            (xmin, xmax, zmin, zmax), ceil_h, R, ceil_cols,
            color_field=ceiling_color_field,
        )
        if ceiling_patch is not None:
            room_scene.add_geometry(
                _ensure_material(ceiling_patch), geom_name="ceiling_fill"
            )

    # Walls: complete unsupported cells connected to an established real wall;
    # walls without sufficient plane support stay in occlusion-only mode.
    # Door/window masks and multi-view depth continuing beyond a wall protect
    # intentional openings from being closed.
    wall_cols = tuple(np.array([floor_cols, ceil_cols]).mean(axis=0).astype(int))
    wall_tol = max(0.04, 0.025 * max(min(xmax - xmin, zmax - zmin), 1e-6))
    wall_infill_summary = []
    for i, ev in enumerate(wall_evidence):
        wall_color_field, wall_color_observed = _wall_color_field(
            ev["spec"], images, world_pts, surface_sample_px, T_total, scale, offset, R,
            ceil_h, ev["shape"], wall_cols, wall_tol,
        )
        patch = _wall_fill_mesh(
            i, ev["fill"], (xmin, xmax, zmin, zmax), ceil_h, R, wall_cols,
            color_field=wall_color_field,
        )
        if patch is not None:
            room_scene.add_geometry(_ensure_material(patch), geom_name=f"wall_{i}_fill")
        wall_infill_summary.append({
            "wall": i,
            "grid": [int(ev["shape"][1]), int(ev["shape"][0])],
            "plane_supported": bool(ev["plane_supported"]),
            "completion_mode": (
                "connected-complete-shell" if ev["plane_supported"]
                else "occlusion-only"
            ),
            "observed_cells": int(np.count_nonzero(ev["observed"])),
            "occluded_cells": int(np.count_nonzero(ev["occluded_views"])),
            "opening_cells": int(np.count_nonzero(
                ev["semantic_opening"] | (ev["through_views"] >= 2)
            )),
            "filled_cells": int(np.count_nonzero(ev["fill"])),
            "locally_colored_cells": int(np.count_nonzero(wall_color_observed)),
            "appearance": "multi-view-local-interpolation-v1",
        })

    room_scene.export(os.path.join(scene_dir, "background.glb"))

    scene_json = {
        "name": scene_name,
        "metric": calibrated,
        "scale": round(scale, 5),
        "room": {
            "width": round(float(xmax - xmin), 2),
            "depth": round(float(zmax - zmin), 2),
            "height": round(ceil_h, 2),
        },
        "room_yaw": round(room_yaw, 5),
        "raw_scan": "raw_scan.glb" if raw_point_count else None,
        "raw_scan_points": raw_point_count,
        "surface_coverage": {k: round(v, 3) for k, v in surface_coverage.items()},
        "object_association": association_summary,
        "label_review": label_review_summary,
        "horizontal_infill": {
            name: {
                "grid": [int(ev["shape"][1]), int(ev["shape"][0])],
                "observed_cells": int(np.count_nonzero(ev["observed"])),
                "filled_cells": int(np.count_nonzero(ev["fill"])),
                "support_points": int(np.sum(ev["support_count"])),
                "locally_colored_cells": int(np.count_nonzero(
                    floor_color_observed if name == "floor" else ceiling_color_observed
                )),
                "appearance": "multi-view-local-interpolation-v1",
                "enabled": name == "floor" or has_ceiling,
            }
            for name, ev in horizontal_evidence.items()
        },
        "wall_infill": wall_infill_summary,
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

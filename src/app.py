"""
Room3D — Office & Data Center 3D Reconstruction Demo
Powered by VGGT-Omega (CVPR 2026 Oral), by Meta Research & VGG Oxford.

Prerequisites:
  1. Request model access at https://huggingface.co/facebook/VGGT-Omega
  2. huggingface-cli login
  3. bash setup.sh
  4. python app.py        # auto-downloads vggt_omega_1b_512.pt on first run
     OR manually:
     hf download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir checkpoints
"""

import os
import sys
import threading
import time
from typing import Callable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Multiple packages (PyTorch, PyMeshLab, scipy) each bundle libomp on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Route MPS-unsupported ops to CPU instead of segfaulting.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Disable MPS memory high-watermark abort.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
import json
import shutil
import tempfile
import traceback
import uuid
from urllib.parse import quote

try:
    import modal
    IS_MODAL = not modal.is_local()
except ImportError:
    IS_MODAL = False

import cv2
import numpy as np
from PIL import Image as PILImage
import torch
import gradio as gr

# ── VGGT Omega repo and src on path ───────────────────────────────────────────

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.dirname(SCRIPT_DIR)
VGGT_OMEGA_REPO = os.path.join(PROJECT_ROOT, "vggt_omega_repo")

if not os.path.exists(VGGT_OMEGA_REPO):
    raise RuntimeError("vggt_omega_repo/ not found. Please run:  bash setup.sh")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, VGGT_OMEGA_REPO)

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from visual_util import predictions_to_glb
from src.detection.dedup import suppress_overlapping_boxes
from src.reconstruction.tsdf_fusion import predictions_to_mesh_glb, predictions_to_mesh_poisson


# ── Device detection ──────────────────────────────────────────────────────────

def _detect_device():
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if cap >= 8 else torch.float16
        return "cuda", dtype
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


DEVICE, DTYPE = _detect_device()
print(f"[Room3D] device={DEVICE}  dtype={DTYPE}")

# MPS materialises the full attention matrix (no flash attention).
# 8 GB machines OOM at 256px; 16 GB machines handle it fine.
# Override via IMAGE_RESOLUTION / MAX_FRAMES env vars if needed.
_mps_ram_gb = 8
try:
    import subprocess
    out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
    _mps_ram_gb = int(out.strip()) // (1024 ** 3)
except Exception:
    pass

if DEVICE == "mps":
    if _mps_ram_gb >= 16:
        _DEFAULT_RESOLUTION = "256"
        MAX_FRAMES = 25
    else:
        _DEFAULT_RESOLUTION = "128"
        MAX_FRAMES = 10
else:
    _DEFAULT_RESOLUTION = "256"
    MAX_FRAMES = 60

IMAGE_RESOLUTION = int(os.environ.get("IMAGE_RESOLUTION", _DEFAULT_RESOLUTION))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", MAX_FRAMES))

_DEFAULT_CKPT = os.path.join(PROJECT_ROOT, "checkpoints", "vggt_omega_1b_512.pt")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", _DEFAULT_CKPT)

SCENES_DIR = os.path.join(PROJECT_ROOT, "scenes")
EDITOR_DIR = os.path.join(SCRIPT_DIR, "viewer", "editor")

# Injected by build_server() (Modal path) — see reconstruct()/run_pipeline_for_job().
RUN_JOB_FN = None
JOB_STATUS_DICT = None
COMMIT_SCENES_FN = None
RELOAD_SCENES_FN = None

# Injected by the GPU job (modal_app.run_reconstruction_job) when object
# generation is enabled: a callable crop_rgb -> GLB bytes backed by the separate
# TRELLIS Modal container. None everywhere else (local dev / CLI), which leaves
# the scan-only pipeline untouched.
OBJECT_GENERATOR_FN = None


def set_object_generator(fn):
    """Wire the image-to-3D generator used to upgrade prefab/box fallbacks.
    See modal_app.py's generate_object_glb and scene_builder.build_scene."""
    global OBJECT_GENERATOR_FN
    OBJECT_GENERATOR_FN = fn


# Async variant for the editor's on-demand "Regenerate with TRELLIS.2" endpoint,
# which runs in the web container and awaits the isolated TRELLIS Modal function
# via .remote.aio (so the ASGI event loop isn't blocked). None when generation
# isn't available (local dev, or TRELLIS not deployed) → the endpoint 503s.
OBJECT_GENERATOR_AIO = None


def set_object_generator_aio(fn):
    global OBJECT_GENERATOR_AIO
    OBJECT_GENERATOR_AIO = fn


# ── Depth unprojection (numpy, from vggt-omega demo_gradio.py) ────────────────

def unproject_depth_map_to_point_map(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    depth = depth_map[..., 0]                        # (S, H, W)
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [(x - cx) / fx * depth, (y - cy) / fy * depth, depth],
        axis=-1,
    )

    rotation    = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


# ── Video frame extraction ────────────────────────────────────────────────────

_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _display_rotate_code(video: "cv2.VideoCapture") -> int | None:
    """The cv2.rotate code that turns the stored (landscape) frame into the
    orientation the phone recorded it in — floor at the bottom.

    Phones store portrait clips as a landscape stream plus a rotation flag in
    the container. OpenCV does NOT apply that flag by default (its
    CAP_PROP_ORIENTATION_AUTO is off), so the raw decoded frame is sideways.
    VGGT then reconstructs a sideways room and the leveler mistakes a wall for
    the floor. Read the flag and rotate frames ourselves — more portable than
    relying on CAP_PROP_ORIENTATION_AUTO, which several OpenCV builds ignore."""
    try:
        deg = int(round(video.get(cv2.CAP_PROP_ORIENTATION_META))) % 360
    except Exception:
        return None
    return _ROTATE_CODES.get(deg)


def extract_video_frames(video_path: str, sample_fps: float = 1.0) -> list[str]:
    video = cv2.VideoCapture(video_path)
    rotate_code = _display_rotate_code(video)   # read before decoding starts
    src_fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_fps = max(float(sample_fps), 0.1)
    frame_interval = max(int(round((src_fps if src_fps > 0 else 1) / sample_fps)), 1)

    tmp_dir = tempfile.mkdtemp()
    paths: list[str] = []
    saved_idx = 0

    target_indices = list(range(0, total_frames, frame_interval)) if total_frames > 0 else []

    if not target_indices:
        # Fallback to sequential extraction if total_frames is empty/invalid
        frame_idx = 0
        while True:
            if _cancel_run.is_set():
                break
            ok, frame = video.read()
            if not ok:
                break
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            if frame_idx % frame_interval == 0:
                path = os.path.join(tmp_dir, f"{saved_idx:06}.png")
                cv2.imwrite(path, frame)
                paths.append(path)
                saved_idx += 1
            frame_idx += 1
        video.release()
        return paths

    # Single sequential decode pass. Each target still picks the sharpest of 5
    # consecutive candidate frames (Laplacian variance), but random
    # CAP_PROP_POS_FRAMES seeks are gone — each seek forces the decoder back to
    # the nearest keyframe, so the old ~5-seeks-per-kept-frame loop decoded
    # most of the video many times over. Same frames selected, one decode.
    from collections import deque

    next_t = 0                                    # cursor into target_indices
    active: deque[int] = deque()                  # windows [t, t+5) covering idx
    best: dict[int, tuple[float, np.ndarray]] = {}  # t → (variance, frame)

    def _flush(t: int) -> None:
        nonlocal saved_idx
        entry = best.pop(t, None)
        if entry is None:
            return
        path = os.path.join(tmp_dir, f"{saved_idx:06}.png")
        cv2.imwrite(path, entry[1])
        paths.append(path)
        saved_idx += 1

    for idx in range(total_frames):
        if _cancel_run.is_set():
            break
        ok, frame = video.read()
        if not ok:
            break
        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)
        while next_t < len(target_indices) and target_indices[next_t] <= idx:
            active.append(target_indices[next_t])
            next_t += 1
        while active and idx >= active[0] + 5:
            _flush(active.popleft())
        if active:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            for t in active:
                if variance > best.get(t, (-1.0, None))[0]:
                    best[t] = (variance, frame.copy())

    if not _cancel_run.is_set():
        while active:
            _flush(active.popleft())

    video.release()
    print(f"[Room3D] Extracted {len(paths)} blur-filtered sharp frames out of {total_frames} raw frames.")
    return paths



# ── Grounding DINO object detection ──────────────────────────────────────────

_gdino_model     = None
_gdino_processor = None

DEFAULT_DETECT_PROMPT = "chair . table . backpack . water bottle . laptop . whiteboard . television . remote control ."


def get_gdino():
    global _gdino_model, _gdino_processor
    if _gdino_model is None:
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError:
            raise RuntimeError("pip install transformers  # required for object detection")
        print("[Room3D] Loading Grounding DINO base…")
        repo = "IDEA-Research/grounding-dino-base"
        _gdino_processor = AutoProcessor.from_pretrained(repo)
        _gdino_model     = AutoModelForZeroShotObjectDetection.from_pretrained(repo).eval()
        # GDINO-base (~450MB) fits alongside the resident VGGT-Omega weights
        # (~4.3GB) with plenty of headroom on a 24GB L4 — run it on GPU where
        # available. CPU-only fallback for MPS/CPU hosts, since GDINO's Swin-B
        # backbone is slow enough per-frame on CPU that a long sequence of
        # frames can take tens of minutes there.
        _gdino_model     = _gdino_model.to(DEVICE if DEVICE == "cuda" else "cpu")
        print(f"[Room3D] Grounding DINO ready ({_gdino_model.device}).")
    return _gdino_model, _gdino_processor


def _prompt_categories(user_text: str) -> list[str]:
    """The user's requested labels, lowercased and de-duplicated (order kept)."""
    parts = [p.strip().lower() for p in user_text.replace(".", ",").split(",") if p.strip()]
    seen: dict[str, None] = {}
    for p in parts:
        seen.setdefault(p, None)
    return list(seen)


def _format_prompt(user_text: str) -> str:
    """Convert comma-separated labels to Grounding DINO's '. '-separated format."""
    return " . ".join(_prompt_categories(user_text)) + " ."


def _snap_label(span: str, categories: list[str]) -> str | None:
    """Map a Grounding DINO text span to the user category it best matches, or
    None to drop it.

    GDINO returns matched *sub-spans* of the prompt ("rack" for "server rack",
    "switch" for "network switch") and occasionally merges neighbours ("rack
    monitor"), so its raw labels aren't the categories you asked for. This snaps
    each span back to a requested category by word overlap — preferring whole
    containment — and drops spans that share no word with any category (e.g. a
    stray "person" when you never asked for one)."""
    s = span.strip().lower()
    if not s or not categories:
        return s or None
    if s in categories:
        return s
    s_toks = set(s.split())
    best, best_score = None, 0.0
    for cat in categories:
        c_toks = set(cat.split())
        shared = s_toks & c_toks
        if not shared:
            continue
        # Coverage in both directions, plus a bonus for substring containment.
        score = len(shared) / len(c_toks) + len(shared) / max(len(s_toks), 1)
        if s in cat or cat in s:
            score += 1.0
        if score > best_score:
            best, best_score = cat, score
    return best


def _text_labels(det: dict) -> list[str]:
    """Return string labels from a Grounding DINO result dict.
    transformers ≥4.51 uses 'text_labels'; older versions used 'labels'."""
    if "text_labels" in det:
        return det["text_labels"]
    raw = det["labels"]
    # Guard: if entries are already strings (old behaviour), return as-is
    if raw and isinstance(raw[0], str):
        return raw
    # Integer ids — shouldn't reach here, but avoid a silent crash
    return [str(r) for r in raw]


def detect_objects_2d(
    image_paths: list[str],
    prompt: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    progress_cb: Callable[[float, str], None] | None = None,
) -> list[dict]:
    """Run Grounding DINO on each image. Returns one dict per frame."""
    model, processor = get_gdino()
    formatted = _format_prompt(prompt)
    categories = _prompt_categories(prompt)
    results = []
    t0 = time.time()
    n = len(image_paths)

    for i, path in enumerate(image_paths):
        _abort_if_cancelled()
        if progress_cb is not None and (i % 10 == 0 or i == n - 1):
            progress_cb(i / max(n, 1), f"Detecting objects: {i}/{n} frames")
        img = PILImage.open(path).convert("RGB")
        W, H = img.size
        inputs = processor(images=img, text=formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning,
                                        module="transformers")
                det = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=[(H, W)],
                )[0]
        except TypeError:
            # newer transformers removed threshold params — filter manually
            det = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=[(H, W)],
            )[0]
            mask = det["scores"] > box_threshold
            det = {
                "boxes":  det["boxes"][mask],
                "labels": [_text_labels(det)[i] for i, m in enumerate(mask) if m],
                "scores": det["scores"][mask],
            }
        # Snap every Grounding DINO span to one of the requested categories and
        # drop anything that matches none, so the scene only ever contains the
        # labels you asked for.
        boxes = det["boxes"].cpu().numpy()
        scores = det["scores"].cpu().numpy()
        snapped = [_snap_label(lbl, categories) for lbl in _text_labels(det)]
        keep = [k for k, s in enumerate(snapped) if s is not None]
        boxes = boxes[keep]
        scores = scores[keep]
        labels = [snapped[k] for k in keep]
        keep = suppress_overlapping_boxes(boxes, labels, scores)
        results.append({
            "boxes":      boxes[keep],
            "labels":     [labels[k] for k in keep],
            "scores":     scores[keep],
            "image_size": (H, W),
        })
        if (i + 1) % 25 == 0 or (i + 1) == len(image_paths):
            print(f"[Room3D] Detection: {i + 1}/{len(image_paths)} frames "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

    return results


def project_detections_to_3d(
    detections_2d: list[dict],
    depth_np: np.ndarray,    # (S, H, W, 1)
    extrinsic_np: np.ndarray, # (S, 3, 4)
    intrinsic_np: np.ndarray, # (S, 3, 3)
) -> list[dict]:
    """Back-project each 2-D box into a 3-D AABB using VGGT depth maps."""
    S, dH, dW = depth_np.shape[:3]
    detections_3d = []

    for fi, frame_det in enumerate(detections_2d):
        if fi >= S:
            break
        orig_H, orig_W = frame_det["image_size"]
        sx, sy = dW / orig_W, dH / orig_H
        depth = depth_np[fi, :, :, 0]          # (H, W)
        K     = intrinsic_np[fi]               # (3, 3)
        E     = extrinsic_np[fi]               # (3, 4)  ← 3×4 from VGGT
        fx, fy   = K[0, 0], K[1, 1]
        ck_x, ck_y = K[0, 2], K[1, 2]
        R = E[:3, :3]
        t = E[:3, 3]

        for box, label, score in zip(frame_det["boxes"], frame_det["labels"], frame_det["scores"]):
            x1, y1, x2, y2 = box
            dx1 = int(np.clip(x1 * sx, 0, dW - 1))
            dy1 = int(np.clip(y1 * sy, 0, dH - 1))
            dx2 = int(np.clip(x2 * sx, 1, dW))
            dy2 = int(np.clip(y2 * sy, 1, dH))
            if dx2 <= dx1 or dy2 <= dy1:
                continue

            ys, xs = np.meshgrid(np.arange(dy1, dy2), np.arange(dx1, dx2), indexing="ij")
            d = depth[ys, xs]
            valid = d > 1e-4
            if valid.sum() < 4:
                continue
            d  = d[valid]; xs = xs[valid]; ys = ys[valid]

            # Outlier depth pruning: reject the outer 10% of depths to avoid boundary bleeding/noise
            if len(d) >= 10:
                d_lo, d_hi = np.percentile(d, [10, 90])
                inliers = (d >= d_lo) & (d <= d_hi)
                if inliers.any() and inliers.sum() >= 4:
                    d = d[inliers]; xs = xs[inliers]; ys = ys[inliers]

            # Camera-space unproject
            pts_cam = np.stack(
                [(xs - ck_x) / fx * d, (ys - ck_y) / fy * d, d], axis=-1
            )
            # World-space: R^T @ (p - t)
            pts_world = (pts_cam - t) @ R          # (N, 3)

            center = pts_world.mean(axis=0)
            size   = np.maximum(pts_world.max(axis=0) - pts_world.min(axis=0), 0.05)
            detections_3d.append(
                {"label": label, "score": float(score), "center": center, "size": size, "frame": fi}
            )

    return detections_3d


def cluster_detections(detections_3d: list[dict], distance_threshold: float = 0.5) -> list[dict]:
    """Merge nearby same-label detections across frames."""
    from collections import defaultdict
    by_label: dict[str, list] = defaultdict(list)
    for d in detections_3d:
        by_label[d["label"]].append(d)

    merged = []
    for label, dets in by_label.items():
        used = [False] * len(dets)
        for i, d in enumerate(dets):
            if used[i]:
                continue
            cluster = [d]; used[i] = True
            for j in range(i + 1, len(dets)):
                if not used[j] and np.linalg.norm(d["center"] - dets[j]["center"]) < distance_threshold:
                    cluster.append(dets[j]); used[j] = True
            merged.append({
                "label":  label,
                "score":  max(c["score"] for c in cluster),
                "center": np.mean([c["center"] for c in cluster], axis=0),
                "size":   np.max([c["size"] for c in cluster], axis=0),
                "count":  len(cluster),
            })
    return merged


def _scene_transform(extrinsic_np: np.ndarray) -> np.ndarray:
    """Recreate the 4×4 transform that visual_util.apply_scene_alignment uses."""
    opengl = np.eye(4); opengl[1, 1] = -1; opengl[2, 2] = -1
    E0 = np.eye(4); E0[:3, :4] = extrinsic_np[0]   # 3×4 → 4×4
    return np.linalg.inv(E0) @ opengl


def add_boxes_to_scene(scene, detections: list[dict], extrinsic_np: np.ndarray):
    """Add colour-coded wireframe 3-D boxes to a trimesh.Scene."""
    import trimesh
    from trimesh.path import Path3D
    from trimesh.path.entities import Line as LineEntity
    from matplotlib import colormaps

    T      = _scene_transform(extrinsic_np)
    cmap   = colormaps["tab10"]
    labels = sorted({d["label"] for d in detections})
    color_map = {
        lbl: (np.array(cmap(i % 10)[:3]) * 255).astype(np.uint8)
        for i, lbl in enumerate(labels)
    }

    EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

    for det in detections:
        c, s = det["center"], det["size"] / 2
        # 8 corners in world space
        corners = np.array([
            c + np.array(sx * s) for sx in [
                [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1],
            ]
        ])
        # Apply the same scene transform as the point cloud
        corners_h = np.hstack([corners, np.ones((8, 1))])
        corners_t = (T @ corners_h.T).T[:, :3]

        color = np.append(color_map[det["label"]], 255)
        entities = [LineEntity(points=[i, j], color=color) for i, j in EDGES]
        scene.add_geometry(Path3D(entities=entities, vertices=corners_t))

    return scene


# Dark navy background for the raw-mesh preview viewer.
VIEWER_BG = [0.04, 0.07, 0.20, 1.0]


# ── Lazy model loading ────────────────────────────────────────────────────────

_model: VGGTOmega | None = None


def get_model() -> VGGTOmega:
    global _model
    if _model is not None:
        return _model

    ckpt = CHECKPOINT_PATH
    if not os.path.exists(ckpt):
        print(f"[Room3D] Checkpoint not found at {ckpt}, downloading from HuggingFace…")
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(
            repo_id="facebook/VGGT-Omega",
            filename="vggt_omega_1b_512.pt",
            local_dir=os.path.join(SCRIPT_DIR, "checkpoints"),
        )

    # Sanity-check: a 1B-param checkpoint must be at least 1 GB
    ckpt_size = os.path.getsize(ckpt)
    if ckpt_size < 1 * 1024 ** 3:
        os.remove(ckpt)
        raise RuntimeError(
            f"Checkpoint at {ckpt} is only {ckpt_size // 1024**2} MB — likely a partial download. "
            "The file has been removed; re-run to trigger a fresh download."
        )

    print(f"[Room3D] Loading VGGT-Omega from {ckpt} ({ckpt_size / 1024**3:.1f} GB) …")
    enable_alignment = "text" in os.path.basename(ckpt)
    _model = VGGTOmega(enable_alignment=enable_alignment).eval()
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except (TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Timed out reading {ckpt}. "
            "If this file is in iCloud Drive, open Finder and double-click it to force a local download, "
            "then restart the app."
        ) from exc
    _model.load_state_dict(state)
    _model = _model.to(DEVICE)
    if DEVICE == "mps":
        torch.mps.synchronize()
    print("[Room3D] Model ready.", flush=True)
    return _model


# ── Inference ─────────────────────────────────────────────────────────────────

def run_vggt_omega(image_paths: list[str]) -> dict:
    model = get_model()
    images = load_and_preprocess_images(
        image_paths, image_resolution=IMAGE_RESOLUTION
    ).to(DEVICE)

    with torch.inference_mode():
        if DEVICE == "cuda":
            with torch.amp.autocast("cuda", dtype=DTYPE):
                predictions = model(images)
        elif DEVICE == "mps":
            # MPS segfaults on causal SDPA; the MATH kernel is safe.
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):
                predictions = model(images)
            torch.mps.synchronize()
        else:
            predictions = model(images)

    # Pose encoding → camera matrices (still on device as tensors)
    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"], predictions["images"].shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # All tensors → CPU numpy, squeeze batch dim (batch size is always 1)
    out: dict = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().float().cpu().numpy()
            if v.shape[0] == 1:
                v = v[0]
        out[k] = v

    # ── Bilateral Depth Smoothing ──
    # Smooth flat surfaces (like server rack panels/walls) while keeping silhouettes sharp
    smoothed_depths = []
    for s in range(out["depth"].shape[0]):
        d_frame = out["depth"][s, ..., 0]  # (H, W)
        d_min, d_max = float(d_frame.min()), float(d_frame.max())
        d_range = d_max - d_min
        if d_range > 1e-5:
            d_norm = (d_frame - d_min) / d_range
            # Bilateral filter expects float32, d=9, sigmaColor=0.03 (3% of scene range), sigmaSpace=9.0
            d_filtered = cv2.bilateralFilter(d_norm.astype(np.float32), d=9, sigmaColor=0.03, sigmaSpace=9.0)
            d_denorm = d_filtered * d_range + d_min
        else:
            d_denorm = d_frame
        smoothed_depths.append(d_denorm[..., None])
    out["depth"] = np.stack(smoothed_depths, axis=0)

    # Depth → world-space point map (numpy operation)
    out["world_points_from_depth"] = unproject_depth_map_to_point_map(
        out["depth"], out["extrinsic"], out["intrinsic"]
    )

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return out


# ── Gemini Vision label suggestion ───────────────────────────────────────────

_VISION_LABEL_PROMPT = (
    "You are analysing a photo of a room (office or data center). "
    "List every distinct object category you can see that would be useful for 3D space "
    "mapping and inventory. Be specific: use 'server rack' not 'equipment', "
    "'network switch' not 'device'. "
    "Return ONLY a comma-separated list, nothing else. "
    "Example: server rack, monitor, desk, chair, patch panel, cable tray"
)


def auto_detect_labels(image_files, video_file, sample_fps: float):
    """Send 1-2 frames to Gemini Vision and return suggested detection labels."""
    if video_file is not None:
        video_path = video_file if isinstance(video_file, str) else video_file["name"]
        paths = extract_video_frames(video_path, sample_fps)
    elif image_files:
        paths = [f.name if hasattr(f, "name") else str(f) for f in image_files]
    else:
        paths = []

    if not paths:
        return gr.update(), "*Upload images or a video first.*"

    indices = sorted({0, len(paths) // 2})
    samples = [paths[i] for i in indices if i < len(paths)]

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return gr.update(), "*Set `GEMINI_API_KEY` in `.env` to use auto-detection.*"

    try:
        from google import genai
    except ImportError:
        return gr.update(), "*`pip install google-genai` to use auto-detection.*"

    try:
        client = genai.Client(api_key=api_key)
        images = [PILImage.open(p).convert("RGB") for p in samples]
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[*images, _VISION_LABEL_PROMPT],
        )
        labels = response.text.strip().rstrip(".")
        n = len([l for l in labels.split(",") if l.strip()])
        return gr.update(value=labels), f"*Gemini suggested {n} categories from {len(samples)} frame(s).*"
    except Exception as exc:
        return gr.update(), f"*Gemini API error: {exc}*"


def _review_instance_labels(instances, image_paths, dets_2d):
    """Late import keeps Gemini optional for local scan-only workflows."""
    from src.detection.gemini_review import review_object_instances
    return review_object_instances(instances, image_paths, dets_2d)


# ── Run cancellation ──────────────────────────────────────────────────────────
# The Stop button sets this flag; the pipeline checks it between stages and
# between frames. A VGGT forward pass already in flight runs to completion
# (a few seconds); everything else aborts at the next checkpoint.
#
# On Modal, the pipeline runs in a spawned GPU job (see run_pipeline_for_job
# below), a separate container/process from the one handling the Stop click —
# so cancellation additionally goes through a shared modal.Dict keyed by
# job_id, not just the in-process Event.

_cancel_run = threading.Event()
_current_job_id: str | None = None
_job_status_handle = None


class RunCancelled(Exception):
    pass


def _job_status_dict():
    global _job_status_handle
    if JOB_STATUS_DICT is not None:
        return JOB_STATUS_DICT
    if not IS_MODAL:
        return None
    if _job_status_handle is None:
        try:
            import modal
            _job_status_handle = modal.Dict.from_name("room3d-job-status", create_if_missing=True)
        except Exception:
            return None
    return _job_status_handle


def _abort_if_cancelled():
    if _cancel_run.is_set():
        raise RunCancelled()
    if _current_job_id:
        d = _job_status_dict()
        if d is not None:
            entry = d.get(_current_job_id, {}) or {}
            if d.get(f"{_current_job_id}:cancel", False) or entry.get("cancel"):
                raise RunCancelled()


async def request_stop(job_id: str | None = None):
    _cancel_run.set()
    if job_id:
        if JOB_STATUS_DICT is not None:
            # Use a separate key for the authoritative flag. Progress and
            # completion updates share the main entry and cannot atomically
            # read/modify/write a modal.Dict value with this request.
            await JOB_STATUS_DICT.put.aio(f"{job_id}:cancel", True)
            entry = await JOB_STATUS_DICT.get.aio(job_id, {}) or {}
            entry["cancel"] = True
            await JOB_STATUS_DICT.put.aio(job_id, entry)
    return "⏹️ Stop requested — aborting at the next stage boundary…"


# ── Pipeline (no Gradio/Modal specifics — shared by local + spawned-job paths) ─

def _result(cancelled=False, error=None, message="", scene_name=None):
    return {"cancelled": cancelled, "error": error, "message": message, "scene_name": scene_name}


def _pipeline(
    image_paths_in: list[str] | None,
    video_path_in: str | None,
    sample_fps: float,
    conf_thres: float,
    show_cam: bool,
    mask_black_bg: bool,
    mask_white_bg: bool,
    output_type: str,
    mesh_resolution: int,
    detect_enabled: bool,
    detect_prompt: str,
    box_threshold: float,
    gemini_review_labels: bool,
    editable_scene: bool,
    glb_out_path: str,
    status_cb: Callable[[float, str], None],
) -> dict:
    """Runs frame extraction through scene export. Takes plain local paths
    (already resolved from Gradio file objects, already on this process's
    filesystem — a shared Volume mount when running as a spawned Modal job)
    and reports progress via status_cb instead of gr.Progress directly, so
    this same function works unchanged whether called in-process (local dev)
    or from run_pipeline_for_job() inside a spawned GPU container."""
    _cancel_run.clear()
    status_cb(0.0, "Extracting frames…")
    if video_path_in is not None:
        paths = extract_video_frames(video_path_in, sample_fps)
        if not paths:
            return _result(message="Could not extract frames from video.")
    else:
        paths = list(image_paths_in or [])

    if _cancel_run.is_set():
        return _result(cancelled=True, message="⏹️ Run cancelled — GPU work stopped.")
    if not paths:
        return _result(message="Upload images or a video to begin.")
    if len(paths) < 2:
        return _result(message="Need at least 2 frames for 3D reconstruction.")
    if len(paths) > MAX_FRAMES:
        # Evenly resample across the whole clip so long videos don't silently
        # lose coverage of everything after the frame budget runs out.
        idx = np.linspace(0, len(paths) - 1, MAX_FRAMES).round().astype(int)
        paths = [paths[i] for i in idx]

    try:
        t0 = time.time()
        _abort_if_cancelled()
        status_cb(0.1, f"Running VGGT-Omega on {len(paths)} frames…")
        preds = run_vggt_omega(paths)
        _abort_if_cancelled()
        elapsed = time.time() - t0

        status_cb(0.5, f"Building {output_type.lower()}…")

        if output_type == "Mesh (TSDF)":
            scene = predictions_to_mesh_glb(
                preds,
                conf_thres=conf_thres,
                show_cam=show_cam,
                mesh_resolution=mesh_resolution,
            )
            mode = "mesh"
        elif output_type == "Mesh (Poisson)":
            scene = predictions_to_mesh_poisson(
                preds,
                conf_thres=conf_thres,
                show_cam=show_cam,
                mesh_resolution=mesh_resolution,
            )
            mode = "mesh"
        else:
            scene = predictions_to_glb(
                preds,
                conf_thres=conf_thres,
                mask_black_bg=mask_black_bg,
                mask_white_bg=mask_white_bg,
                show_cam=show_cam,
                mask_sky=False,
            )
            mode = "point cloud"

        # Object detection
        dets_3d: list[dict] = []
        dets_2d: list[dict] = []
        _abort_if_cancelled()
        if detect_enabled and detect_prompt.strip():
            status_cb(0.55, "Detecting objects…")
            dets_2d = detect_objects_2d(
                paths, detect_prompt, box_threshold,
                progress_cb=lambda frac, desc: status_cb(0.55 + 0.15 * frac, desc),
            )
            dets_3d = cluster_detections(
                project_detections_to_3d(
                    dets_2d, preds["depth"], preds["extrinsic"], preds["intrinsic"]
                )
            )
            if dets_3d:
                add_boxes_to_scene(scene, dets_3d, preds["extrinsic"])

        # Editable scene (segmented, movable objects + web editor)
        editor_note = ""
        scene_name = None
        _abort_if_cancelled()
        if editable_scene and dets_2d:
            try:
                from src.detection import scene_builder
                if video_path_in is not None:
                    scene_name = os.path.splitext(os.path.basename(video_path_in))[0]
                else:
                    scene_name = "photos"
                status_cb(0.7, "Building editable scene…")
                scene_json = scene_builder.build_scene(
                    preds, paths, dets_2d,
                    out_root=SCENES_DIR,
                    scene_name=scene_name,
                    conf_thres=conf_thres,
                    mesh_resolution=mesh_resolution,
                    # When a generator is wired in (Modal + TRELLIS enabled),
                    # also dedupe so it runs once per identical-object group.
                    generate_fn=OBJECT_GENERATOR_FN,
                    reuse_duplicates=OBJECT_GENERATOR_FN is not None,
                    label_review_fn=_review_instance_labels if gemini_review_labels else None,
                    abort_check=_abort_if_cancelled,
                    progress_cb=lambda frac, desc: status_cb(0.7 + 0.3 * frac, desc),
                )
                n_obj = len(scene_json["objects"])
                metric_note = (
                    "metric scale calibrated from rack height"
                    if scene_json["metric"]
                    else "relative scale — no rack found to calibrate against"
                )
                editor_note = (
                    f"\n\n### Editable Scene\n"
                    f"**{n_obj} movable object(s)** · {metric_note} · room "
                    f"{scene_json['room']['width']} × {scene_json['room']['depth']} "
                    f"× {scene_json['room']['height']}\n\n"
                    f"Editor is embedded below — "
                    f"**[open full screen](/editor/?scene={quote(scene_name)})**"
                )
            except RunCancelled:
                raise
            except Exception:
                scene_name = None
                editor_note = (
                    f"\n\n⚠️ Editable scene build failed:\n```\n{traceback.format_exc()}\n```"
                )
        elif editable_scene:
            editor_note = "\n\n⚠️ Editable scene needs object detection enabled with a prompt."

        scene.export(glb_out_path)

        # ── Status message ──
        src = "frames" if video_path_in is not None else "views"
        det_note = f", {len(dets_3d)} objects detected" if detect_enabled else ""
        msg = f"Done in {elapsed:.1f}s — {len(paths)} {src} @ {IMAGE_RESOLUTION}px on {DEVICE} ({mode}{det_note})."
        msg += editor_note

        if dets_3d:
            msg += "\n\n### Detected Objects\n"
            for i, det in enumerate(sorted(dets_3d, key=lambda d: -d["score"]), 1):
                c, s = det["center"], det["size"]
                count = det.get("count", 1)
                msg += (
                    f"**{i}. {det['label'].upper()}** ({det['score']:.0%})\n"
                    f"   • Location: ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) m\n"
                    f"   • Size: {s[0]:.2f} × {s[1]:.2f} × {s[2]:.2f} m\n"
                )
                if count > 1:
                    msg += f"   • Seen in {count} frames\n"
                msg += "\n"

        return _result(message=msg, scene_name=scene_name)

    except RunCancelled:
        return _result(cancelled=True, message="⏹️ Run cancelled — GPU work stopped.")
    except Exception:
        return _result(error=traceback.format_exc(), message=f"Error:\n{traceback.format_exc()}")


def run_pipeline_for_job(job_id: str, params: dict) -> dict:
    """Entry point for the spawned Modal GPU job (see modal_app.py's
    run_reconstruction_job). Runs fully decoupled from the web container's
    request lifecycle: if the web request that started this job gets
    cancelled (browser reconnect, idle SSE timeout, etc.), this job — a
    separate Modal input in a separate container — keeps running to
    completion unaffected. That decoupling is the actual fix for the
    recurring prod issue where a cancelled web request force-killed the
    whole GPU container mid-run (Modal docs: cancelling one input under
    @modal.concurrent on a synchronous function kills the entire container
    after a 30s grace period — confirmed in prod logs as 'killing task'
    during scene fusion, well before the pipeline reached scene.json)."""
    global _current_job_id
    _current_job_id = job_id
    _cancel_run.clear()
    status_dict = _job_status_dict()

    def status_cb(frac, desc):
        if status_dict is not None:
            # Merge, don't overwrite — request_stop() writes a "cancel" flag
            # into this same dict entry from the web container; a blind
            # put({"frac":..., "desc":...}) here would silently erase it.
            entry = status_dict.get(job_id, {}) or {}
            entry["state"] = "running"
            entry["frac"] = frac
            entry["desc"] = desc
            status_dict.put(job_id, entry)
        print(f"[Room3D] job {job_id}: {desc} ({frac:.0%})", flush=True)

    try:
        return _pipeline(
            params.get("image_paths"), params.get("video_path"),
            params["sample_fps"], params["conf_thres"], params["show_cam"],
            params["mask_black_bg"], params["mask_white_bg"], params["output_type"],
            params["mesh_resolution"], params["detect_enabled"], params["detect_prompt"],
            params["box_threshold"], params.get("gemini_review_labels", False),
            params["editable_scene"], params["glb_out_path"],
            status_cb,
        )
    finally:
        _current_job_id = None


def _build_editor_update(scene_name: str | None):
    if not scene_name:
        return gr.update(visible=False)
    # Unique per build: rebuilding a scene with the same name would otherwise
    # produce an identical iframe src, so Gradio skips the DOM update and the
    # editor keeps showing the PREVIOUS build's scene (stale labels/layout, and
    # no raw_scan). The token forces a fresh iframe → fresh scene.json fetch.
    # The editor reads `scene` and ignores `v`, threading it onto its own asset
    # fetches so the browser can't serve a cached scene.json/background/raw_scan.
    v = uuid.uuid4().hex[:8]
    return gr.update(
        value=(
            f'<iframe src="/editor/?scene={quote(scene_name)}&v={v}" '
            f'style="width:100%;height:640px;border:1px solid #30363d;'
            f'border-radius:8px;background:#0d1117;" '
            f'title="Scene Editor"></iframe>'
        ),
        visible=True,
    )


# ── Gradio callback ───────────────────────────────────────────────────────────
# Modal submissions return as soon as the GPU job has spawned. A gr.Timer then
# makes independent, short status requests; there is no reconstruction-long
# Gradio SSE request for a browser reconnect to cancel. The latest job_id is a
# gr.BrowserState, so refreshes and web-container restarts can resume polling.

async def reconstruct(
    image_files,
    video_file,
    sample_fps: float,
    conf_thres: float,
    show_cam: bool,
    mask_black_bg: bool,
    mask_white_bg: bool,
    output_type: str,
    mesh_resolution: int,
    detect_enabled: bool,
    detect_prompt: str,
    box_threshold: float,
    gemini_review_labels: bool,
    editable_scene: bool,
    progress: gr.Progress = gr.Progress(),
):
    import asyncio
    import inspect

    no_editor = gr.update(visible=False)
    _cancel_run.clear()

    video_path_in = None
    image_paths_in: list[str] = []
    if video_file is not None:
        video_path_in = video_file if isinstance(video_file, str) else video_file["name"]
    elif image_files:
        image_paths_in = [f.name if hasattr(f, "name") else str(f) for f in image_files]

    if not video_path_in and not image_paths_in:
        return None, "Upload images or a video to begin.", no_editor, "", gr.Timer(active=False)

    job_id = uuid.uuid4().hex[:12]

    if RUN_JOB_FN is None:
        # Local dev (no Modal spawn available) — run in a worker thread so the
        # event loop stays free; behavior otherwise matches the pre-split code.
        glb_path = tempfile.mktemp(suffix=".glb")
        result = await asyncio.to_thread(
            _pipeline,
            image_paths_in, video_path_in, sample_fps, conf_thres, show_cam,
            mask_black_bg, mask_white_bg, output_type, mesh_resolution,
            detect_enabled, detect_prompt, box_threshold, gemini_review_labels, editable_scene,
            glb_path, lambda f, d: progress(f, desc=d),
        )
        out_glb = (
            glb_path if not result["cancelled"] and not result["error"]
            and os.path.exists(glb_path) else None
        )
        return (
            out_glb, result["message"], _build_editor_update(result["scene_name"]),
            job_id, gr.Timer(active=False),
        )

    # Modal: persist uploads to the shared Volume, spawn a decoupled GPU job,
    # and poll for progress/result. See run_pipeline_for_job's docstring for
    # why this — not tuning around the symptom again — is the actual fix.
    def _stage_uploads():
        job_dir = os.path.join(SCENES_DIR, "_jobs", job_id)
        uploads_dir = os.path.join(job_dir, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        video_dst = None
        image_dsts = None
        if video_path_in:
            video_dst = os.path.join(uploads_dir, os.path.basename(video_path_in))
            shutil.copy(video_path_in, video_dst)
        else:
            image_dsts = []
            for p in image_paths_in:
                dst = os.path.join(uploads_dir, os.path.basename(p))
                shutil.copy(p, dst)
                image_dsts.append(dst)
        return job_dir, video_dst, image_dsts

    job_dir, job_video_path, job_image_paths = await asyncio.to_thread(_stage_uploads)
    glb_out_path = os.path.join(job_dir, "output.glb")
    if COMMIT_SCENES_FN:
        res = COMMIT_SCENES_FN()
        if inspect.isawaitable(res):
            await res

    params = dict(
        image_paths=job_image_paths, video_path=job_video_path, sample_fps=sample_fps,
        conf_thres=conf_thres, show_cam=show_cam, mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg, output_type=output_type, mesh_resolution=mesh_resolution,
        detect_enabled=detect_enabled, detect_prompt=detect_prompt, box_threshold=box_threshold,
        gemini_review_labels=gemini_review_labels,
        editable_scene=editable_scene, glb_out_path=glb_out_path,
    )
    if JOB_STATUS_DICT is not None:
        # Create the durable record before spawning. Writing a whole entry
        # after spawn races a sufficiently fast worker's progress/completion
        # update and can accidentally change "complete" back to "queued".
        await JOB_STATUS_DICT.put.aio(job_id, {
            "state": "queued",
            "frac": 0.0,
            "desc": "Waiting for GPU worker…",
            "glb_out_path": glb_out_path,
        })
    await RUN_JOB_FN.spawn.aio(job_id, params)
    return (
        None,
        f"🚀 Job `{job_id}` started on the GPU worker. You can safely refresh this page.",
        no_editor,
        job_id,
        gr.Timer(active=True),
    )


async def poll_job(job_id: str | None):
    """One short, reconnect-safe status check for the browser timer."""
    import inspect

    no_editor = gr.update(visible=False)
    if not job_id or JOB_STATUS_DICT is None:
        return gr.update(), gr.update(), gr.update(), gr.Timer(active=False)

    entry = await JOB_STATUS_DICT.get.aio(job_id, {}) or {}
    if not entry:
        return (
            gr.update(),
            f"⚠️ Job `{job_id}` was not found (it may have expired).",
            no_editor,
            gr.Timer(active=False),
        )

    state = entry.get("state", "queued")
    if state == "failed":
        return (
            gr.update(),
            f"Error in job `{job_id}`: {entry.get('error', 'unknown worker failure')}",
            no_editor,
            gr.Timer(active=False),
        )

    if state != "complete":
        frac = entry.get("frac", 0.0)
        desc = entry.get("desc", "Waiting for GPU worker…")
        return (
            gr.update(),
            f"⏳ Job `{job_id}` — {desc} ({frac:.0%})",
            gr.update(),
            gr.Timer(active=True),
        )

    result = entry.get("result") or _result(error="Missing job result", message="Missing job result")
    if RELOAD_SCENES_FN:
        refreshed = RELOAD_SCENES_FN()
        if inspect.isawaitable(refreshed):
            await refreshed
    glb_out_path = entry.get("glb_out_path")
    out_glb = (
        glb_out_path if glb_out_path and not result["cancelled"] and not result["error"]
        and os.path.exists(glb_out_path) else None
    )
    return (
        out_glb,
        result["message"],
        _build_editor_update(result["scene_name"]),
        gr.Timer(active=False),
    )


# ── UI ────────────────────────────────────────────────────────────────────────

DESCRIPTION = """
# Room3D
### Office & Data Center 3D Reconstruction
Powered by [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) — CVPR 2026 Oral (Meta Research & VGG Oxford)

Upload **5–30 photos** taken from different angles around your office or data center.
VGGT-Omega reconstructs a dense 3D point cloud in seconds — no calibration, no markers needed.

**Tips for best results**
- Walk around the room / between server racks, taking overlapping shots
- Vary height and angle — include overhead shots for equipment tops
- Consistent, even lighting; avoid flash glare on screens and metal surfaces
- Minimize motion blur; keep the camera steady
"""

with gr.Blocks(
    title="Room3D — Office & Data Center 3D Reconstruction",
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row():

        with gr.Column(scale=1, min_width=320):

            with gr.Tabs():
                with gr.Tab("Images"):
                    image_upload = gr.Files(
                        label="Upload Images  (JPG / PNG)",
                        file_types=["image"],
                        file_count="multiple",
                    )
                with gr.Tab("Video"):
                    video_upload = gr.Video(
                        label="Upload Video  (MP4 / MOV)",
                    )
                    sample_fps = gr.Slider(
                        minimum=0.1, maximum=10.0, value=1.0, step=0.1,
                        label="Sample Rate (frames per second)",
                        info="1 fps = 1 frame per second of video",
                    )

            gr.Markdown("#### Output")

            output_type = gr.Radio(
                choices=["Point Cloud", "Mesh (TSDF)", "Mesh (Poisson)"],
                value="Point Cloud",
                label="Output Type",
                info="TSDF = watertight (slower); Poisson = smooth surfaces (faster)",
            )

            mesh_resolution = gr.Slider(
                minimum=64, maximum=512, value=256, step=32,
                label="Mesh Resolution",
                info="Voxels along the longest axis (Mesh only)",
                visible=False,
            )

            gr.Markdown("#### Visualisation Settings")

            conf_thres = gr.Slider(
                minimum=2.0, maximum=50.0, value=20.0, step=1.0,
                label="Confidence Threshold",
                info="Higher = fewer, more reliable points",
            )

            with gr.Row():
                show_cam      = gr.Checkbox(value=True,  label="Show Cameras")
                mask_black_bg = gr.Checkbox(value=True,  label="Mask Black BG")
                mask_white_bg = gr.Checkbox(value=False, label="Mask White BG")

            gr.Markdown("#### Object Detection  *(Grounding DINO)*")

            detect_enabled = gr.Checkbox(value=False, label="Detect objects")

            with gr.Column(visible=False) as detect_options:
                auto_label_btn = gr.Button(
                    "Auto-detect labels with Gemini Vision",
                    variant="secondary",
                    size="sm",
                )
                auto_label_status = gr.Markdown("")
                detect_prompt = gr.Textbox(
                    value=DEFAULT_DETECT_PROMPT,
                    label="What to detect",
                    info="Comma or '. '-separated labels. Edit freely after auto-detection.",
                    lines=2,
                )
                box_threshold = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                    label="Detection Confidence",
                    info="Lower = more detections, higher = fewer but more confident",
                )
                gemini_review_labels = gr.Checkbox(
                    value=False,
                    label="Review object labels with Gemini",
                    info="After 3D clustering, send up to 24 representative object crops "
                         "to Gemini to conservatively relabel or reject mistakes.",
                )
                editable_scene = gr.Checkbox(
                    value=False,
                    label="Build editable scene",
                    info="Segment objects (SAM) into separate movable nodes with real "
                         "scanned detail, auto-calibrate scale from racks, fill in "
                         "unseen ceiling/walls/floor, open in web editor",
                )

            detect_enabled.change(
                fn=lambda v: gr.update(visible=v),
                inputs=detect_enabled,
                outputs=detect_options,
            )

            auto_label_btn.click(
                fn=auto_detect_labels,
                inputs=[image_upload, video_upload, sample_fps],
                outputs=[detect_prompt, auto_label_status],
            )

            with gr.Row():
                run_btn = gr.Button(
                    "Reconstruct 3D Space", variant="primary", size="lg", scale=3
                )
                stop_btn = gr.Button(
                    "⏹ Stop", variant="stop", size="lg", scale=1
                )

            output_type.change(
                fn=lambda v: gr.update(visible=(v in ["Mesh (TSDF)", "Mesh (Poisson)"])),
                inputs=output_type,
                outputs=mesh_resolution,
            )

        with gr.Column(scale=2):
            viewer = gr.Model3D(
                label="3D Reconstruction",
                clear_color=VIEWER_BG,
            )
            status = gr.Markdown(label="Status")
            editor_panel = gr.HTML(visible=False, padding=True)

    # Persist only the opaque latest job ID. This lets the timer reconnect to
    # Modal's shared status store after a tab refresh or web-container restart.
    job_id_state = gr.BrowserState(
        "", storage_key="room3d-latest-job-v1", secret="room3d-job-id"
    )
    job_poll_timer = gr.Timer(value=2.0, active=True)

    run_btn.click(
        fn=reconstruct,
        inputs=[image_upload, video_upload, sample_fps, conf_thres,
                show_cam, mask_black_bg, mask_white_bg, output_type, mesh_resolution,
                detect_enabled, detect_prompt, box_threshold,
                gemini_review_labels, editable_scene],
        outputs=[viewer, status, editor_panel, job_id_state, job_poll_timer],
    )

    job_poll_timer.tick(
        fn=poll_job,
        inputs=[job_id_state],
        outputs=[viewer, status, editor_panel, job_poll_timer],
        show_progress="hidden",
    )

    stop_btn.click(fn=request_stop, inputs=[job_id_state], outputs=[status])


# ── Serve Gradio + scene editor + scene files on one port ────────────────────

def build_server(run_job_fn=None, job_status_dict=None, commit_scenes_fn=None,
                 reload_scenes_fn=None, object_generator_fn=None):
    from fastapi import FastAPI, Request, UploadFile, File, Form
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles

    global RUN_JOB_FN, JOB_STATUS_DICT, COMMIT_SCENES_FN, RELOAD_SCENES_FN
    RUN_JOB_FN = run_job_fn
    JOB_STATUS_DICT = job_status_dict
    COMMIT_SCENES_FN = commit_scenes_fn
    RELOAD_SCENES_FN = reload_scenes_fn
    # Async generator for the editor's on-demand regenerate endpoint (Modal path).
    set_object_generator_aio(object_generator_fn)

    os.makedirs(SCENES_DIR, exist_ok=True)
    api = FastAPI()

    @api.post("/api/scenes/{name}/layout")
    async def save_layout(name: str, request: Request):
        if "/" in name or "\\" in name or ".." in name:
            return JSONResponse({"error": "bad scene name"}, status_code=400)
        path = os.path.join(SCENES_DIR, name, "scene.json")
        if not os.path.exists(path):
            return JSONResponse({"error": "scene not found"}, status_code=404)

        payload = await request.json()
        updates = {o["id"]: o for o in payload.get("objects", []) if "id" in o}
        for upd in updates.values():
            if "label" not in upd:
                continue
            label = upd["label"]
            if not isinstance(label, str):
                return JSONResponse({"error": "label must be text"}, status_code=400)
            label = label.strip()
            if not label or len(label) > 100 or any(ord(ch) < 32 for ch in label):
                return JSONResponse(
                    {"error": "label must be 1-100 printable characters"},
                    status_code=400,
                )
            upd["label"] = label
        with open(path) as f:
            scene_json = json.load(f)
        n = 0
        for obj in scene_json["objects"]:
            upd = updates.get(obj["id"])
            if upd:
                obj["position"] = [round(float(v), 4) for v in upd["position"]]
                obj["yaw"] = round(float(upd["yaw"]), 5)
                if "label" in upd:
                    obj["label"] = upd["label"]
                n += 1
        with open(path, "w") as f:
            json.dump(scene_json, f, indent=2)
        return {"ok": True, "updated": n}

    @api.post("/api/scenes/{name}/objects/{obj_id}/model")
    async def replace_object_model(
        name: str, obj_id: str,
        file: UploadFile = File(...), scale: str = Form(...), offset: str = Form("[0,0,0]"),
    ):
        """Replace a detected object's mesh with an uploaded GLB. `scale` is a
        JSON [sx, sy, sz] computed client-side to fit the upload's own
        bounding box to the object's already-detected size, and `offset` is a
        JSON [ox, oy, oz] re-centring the upload's own (arbitrary) origin to
        the bottom-centre-at-origin convention every other object glb uses.
        Both are stored as metadata, not baked into the mesh, so they're
        applied the same way position/yaw already are, and stay reversible."""
        if "/" in name or "\\" in name or ".." in name:
            return JSONResponse({"error": "bad scene name"}, status_code=400)
        if "/" in obj_id or "\\" in obj_id or ".." in obj_id:
            return JSONResponse({"error": "bad object id"}, status_code=400)
        scene_dir = os.path.join(SCENES_DIR, name)
        path = os.path.join(scene_dir, "scene.json")
        if not os.path.exists(path):
            return JSONResponse({"error": "scene not found"}, status_code=404)

        with open(path) as f:
            scene_json = json.load(f)
        obj = next((o for o in scene_json["objects"] if o["id"] == obj_id), None)
        if obj is None:
            return JSONResponse({"error": "object not found"}, status_code=404)

        try:
            model_scale = [float(v) for v in json.loads(scale)]
            if len(model_scale) != 3 or any(v <= 0 for v in model_scale):
                raise ValueError
        except (ValueError, TypeError):
            return JSONResponse({"error": "scale must be [sx, sy, sz], all > 0"}, status_code=400)

        try:
            model_offset = [float(v) for v in json.loads(offset)]
            if len(model_offset) != 3:
                raise ValueError
        except (ValueError, TypeError):
            return JSONResponse({"error": "offset must be [ox, oy, oz]"}, status_code=400)

        content = await file.read()
        if content[:4] != b"glTF":
            return JSONResponse({"error": "file must be a binary .glb"}, status_code=400)

        glb_path = os.path.join(scene_dir, "objects", f"{obj_id}.glb")
        with open(glb_path, "wb") as f:
            f.write(content)

        obj["model_scale"] = [round(v, 5) for v in model_scale]
        obj["model_offset"] = [round(v, 5) for v in model_offset]
        obj["source"] = "custom-upload"
        with open(path, "w") as f:
            json.dump(scene_json, f, indent=2)
        return {"ok": True}

    @api.delete("/api/scenes/{name}/objects/{obj_id}")
    async def delete_object(name: str, obj_id: str):
        if "/" in name or "\\" in name or ".." in name:
            return JSONResponse({"error": "bad scene name"}, status_code=400)
        if "/" in obj_id or "\\" in obj_id or ".." in obj_id:
            return JSONResponse({"error": "bad object id"}, status_code=400)
        scene_dir = os.path.join(SCENES_DIR, name)
        path = os.path.join(scene_dir, "scene.json")
        if not os.path.exists(path):
            return JSONResponse({"error": "scene not found"}, status_code=404)

        with open(path) as f:
            scene_json = json.load(f)
        before = len(scene_json["objects"])
        scene_json["objects"] = [o for o in scene_json["objects"] if o["id"] != obj_id]
        if len(scene_json["objects"]) == before:
            return JSONResponse({"error": "object not found"}, status_code=404)
        with open(path, "w") as f:
            json.dump(scene_json, f, indent=2)

        glb_path = os.path.join(scene_dir, "objects", f"{obj_id}.glb")
        if os.path.exists(glb_path):
            os.remove(glb_path)
        return {"ok": True}

    @api.post("/api/scenes/{name}/objects/{obj_id}/regenerate")
    async def regenerate_object(name: str, obj_id: str):
        """Regenerate one object's mesh with TRELLIS.2 from its saved input crop.
        Long-running (tens of seconds on the GPU): awaits the isolated TRELLIS
        Modal function, so the client shows a spinner while this is in flight."""
        if "/" in name or "\\" in name or ".." in name:
            return JSONResponse({"error": "bad scene name"}, status_code=400)
        if "/" in obj_id or "\\" in obj_id or ".." in obj_id:
            return JSONResponse({"error": "bad object id"}, status_code=400)
        if OBJECT_GENERATOR_AIO is None:
            return JSONResponse(
                {"error": "object generation is not enabled on this deployment"},
                status_code=503)
        scene_dir = os.path.join(SCENES_DIR, name)
        path = os.path.join(scene_dir, "scene.json")
        if not os.path.exists(path):
            return JSONResponse({"error": "scene not found"}, status_code=404)

        with open(path) as f:
            scene_json = json.load(f)
        obj = next((o for o in scene_json["objects"] if o["id"] == obj_id), None)
        if obj is None:
            return JSONResponse({"error": "object not found"}, status_code=404)

        # Input image: the RGBA subject cutout if we saved one, else the photo.
        crop = None
        for rel in (obj.get("input_crop"), obj.get("photo")):
            if rel and os.path.exists(os.path.join(scene_dir, rel)):
                img = cv2.imread(os.path.join(scene_dir, rel), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                crop = (cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA) if img.ndim == 3 and img.shape[2] == 4
                        else cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                break
        if crop is None:
            return JSONResponse({"error": "no source image saved for this object"}, status_code=400)

        try:
            glb_bytes = await OBJECT_GENERATOR_AIO(crop)
        except Exception as exc:
            return JSONResponse({"error": f"generator call failed: {exc}"}, status_code=502)
        if not glb_bytes:
            return JSONResponse(
                {"error": "generation failed (out of memory or empty result)"}, status_code=502)

        from src.detection.scene_builder import _glb_bytes_to_local_mesh, _ensure_material
        import trimesh
        mesh = _glb_bytes_to_local_mesh(glb_bytes, obj["size"])
        if mesh is None:
            return JSONResponse({"error": "generated mesh could not be processed"}, status_code=502)
        glb_path = os.path.join(scene_dir, "objects", f"{obj_id}.glb")
        trimesh.Scene([_ensure_material(mesh)]).export(glb_path)

        # The mesh is now built at the object's metric size, bottom-centred, so
        # reset the fit transform and mark the source.
        obj["model_scale"] = [1, 1, 1]
        obj["model_offset"] = [0, 0, 0]
        obj["source"] = "trellis"
        obj.pop("reuse_of", None)
        with open(path, "w") as f:
            json.dump(scene_json, f, indent=2)
        return {"ok": True, "source": "trellis"}

    api.mount("/editor", StaticFiles(directory=EDITOR_DIR, html=True), name="editor")
    api.mount("/scenes", StaticFiles(directory=SCENES_DIR), name="scenes")
    return gr.mount_gradio_app(api, demo, path="/", theme=gr.themes.Soft())


if __name__ == "__main__":
    import uvicorn
    print("[Room3D] http://localhost:7860  ·  editor at /editor/?scene=<name>")
    uvicorn.run(build_server(), host="127.0.0.1", port=7860)

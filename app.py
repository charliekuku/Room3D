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
import time

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
import tempfile
import traceback

import cv2
import numpy as np
from PIL import Image as PILImage
import torch
import gradio as gr

# ── VGGT Omega repo on path ───────────────────────────────────────────────────

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
VGGT_OMEGA_REPO = os.path.join(SCRIPT_DIR, "vggt_omega_repo")

if not os.path.exists(VGGT_OMEGA_REPO):
    raise RuntimeError("vggt_omega_repo/ not found. Please run:  bash setup.sh")

sys.path.insert(0, VGGT_OMEGA_REPO)

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from visual_util import predictions_to_glb
from tsdf_fusion import predictions_to_mesh_glb, predictions_to_mesh_poisson


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
    import subprocess, re
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

_DEFAULT_CKPT = os.path.join(SCRIPT_DIR, "checkpoints", "vggt_omega_1b_512.pt")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", _DEFAULT_CKPT)


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

def extract_video_frames(video_path: str, sample_fps: float = 1.0) -> list[str]:
    video = cv2.VideoCapture(video_path)
    src_fps = video.get(cv2.CAP_PROP_FPS)
    sample_fps = max(float(sample_fps), 0.1)
    frame_interval = max(int(round((src_fps if src_fps > 0 else 1) / sample_fps)), 1)

    tmp_dir = tempfile.mkdtemp()
    paths: list[str] = []
    frame_idx = saved_idx = 0

    while True:
        ok, frame = video.read()
        if not ok:
            break
        if frame_idx % frame_interval == 0:
            path = os.path.join(tmp_dir, f"{saved_idx:06}.png")
            cv2.imwrite(path, frame)
            paths.append(path)
            saved_idx += 1
        frame_idx += 1

    video.release()
    return paths


# ── Grounding DINO object detection ──────────────────────────────────────────

_gdino_model     = None
_gdino_processor = None

DEFAULT_DETECT_PROMPT = "server rack . monitor . desk . chair . person . computer . printer . UPS . network switch ."


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
        # Run GDINO on CPU to avoid competing with VGGT for GPU memory
        _gdino_model     = _gdino_model.to("cpu")
        print("[Room3D] Grounding DINO ready.")
    return _gdino_model, _gdino_processor


def _format_prompt(user_text: str) -> str:
    """Convert comma-separated labels to Grounding DINO's '. '-separated format."""
    parts = [p.strip().lower() for p in user_text.replace(".", ",").split(",") if p.strip()]
    return " . ".join(parts) + " ."


def detect_objects_2d(
    image_paths: list[str],
    prompt: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> list[dict]:
    """Run Grounding DINO on each image. Returns one dict per frame."""
    model, processor = get_gdino()
    formatted = _format_prompt(prompt)
    results = []

    for path in image_paths:
        img = PILImage.open(path).convert("RGB")
        W, H = img.size
        inputs = processor(images=img, text=formatted, return_tensors="pt")  # CPU
        with torch.no_grad():
            outputs = model(**inputs)
        try:
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
                "labels": [det["labels"][i] for i, m in enumerate(mask) if m],
                "scores": det["scores"][mask],
            }
        results.append({
            "boxes":      det["boxes"].cpu().numpy(),
            "labels":     det["labels"],
            "scores":     det["scores"].cpu().numpy(),
            "image_size": (H, W),
        })

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


# ── Blueprint styling ─────────────────────────────────────────────────────────

BLUEPRINT_BG = [0.04, 0.07, 0.20, 1.0]   # viewer background in blueprint mode

_BP_LOW  = np.array([ 20,  80, 180, 255], dtype=np.uint8)   # cold blue  (low points)
_BP_HIGH = np.array([200, 235, 255, 255], dtype=np.uint8)   # near-white (high points)
_BP_EDGE = np.array([100, 190, 255, 255], dtype=np.uint8)   # wireframe edge colour
_BP_GRID = np.array([ 45, 105, 195, 160], dtype=np.uint8)   # floor grid colour
_BP_MESH = np.array([ 10,  25,  70, 220], dtype=np.uint8)   # solid mesh fill


def _height_gradient(y: np.ndarray) -> np.ndarray:
    """Map Y values to (N,4) uint8 blueprint height gradient (dark blue → near-white)."""
    y_min, y_max = float(y.min()), float(y.max())
    t = np.clip((y - y_min) / max(y_max - y_min, 1e-6), 0.0, 1.0)[:, None]
    return (_BP_LOW.astype(float) * (1 - t) + _BP_HIGH.astype(float) * t).astype(np.uint8)


def _add_floor_grid(scene, cell_size: float = 0.5, cells: int = 24) -> None:
    from trimesh.path.entities import Line as LineEntity
    from trimesh.path.path import Path3D

    y_vals = [
        geom.vertices[:, 1].min()
        for geom in scene.geometry.values()
        if hasattr(geom, "vertices") and len(geom.vertices) > 0
    ]
    y_floor = min(y_vals) if y_vals else 0.0
    half = cells * cell_size / 2
    verts, entities, idx = [], [], 0
    for i in range(cells + 1):
        t = -half + i * cell_size
        for p0, p1 in [
            ([t, y_floor, -half], [t, y_floor,  half]),
            ([-half, y_floor,  t], [ half, y_floor,  t]),
        ]:
            verts += [p0, p1]
            entities.append(LineEntity(points=[idx, idx + 1], color=_BP_GRID))
            idx += 2
    scene.add_geometry(
        Path3D(entities=entities, vertices=np.array(verts, dtype=float)),
        geom_name="blueprint_floor_grid",
    )


def _wireframe_from_mesh(mesh) -> "Path3D":
    from trimesh.path.entities import Line as LineEntity
    from trimesh.path.path import Path3D
    import trimesh.graph as tg

    try:
        angles = tg.face_adjacency_angles(mesh)
        edges = mesh.face_adjacency_edges[angles > np.radians(12)]
    except Exception:
        edges = np.empty((0, 2), int)
    if len(edges) == 0:
        edges = mesh.edges_unique
    if len(edges) > 40_000:
        edges = edges[:: len(edges) // 40_000 + 1]
    flat_v = mesh.vertices[edges.reshape(-1)]
    n = len(edges)
    ents = [LineEntity(points=[i * 2, i * 2 + 1], color=_BP_EDGE) for i in range(n)]
    return Path3D(entities=ents, vertices=flat_v)


def detect_room_structure(scene) -> dict:
    """Estimate floor/ceiling/wall planes from the scene's geometry vertices."""
    all_pts = [
        geom.vertices
        for geom in scene.geometry.values()
        if hasattr(geom, "vertices") and len(geom.vertices) > 0
    ]
    if not all_pts:
        return {}
    pts = np.vstack(all_pts)
    for ax in range(3):
        lo, hi = np.percentile(pts[:, ax], 2), np.percentile(pts[:, ax], 98)
        pts = pts[(pts[:, ax] >= lo) & (pts[:, ax] <= hi)]
    if len(pts) < 10:
        return {}
    y, x, z = pts[:, 1], pts[:, 0], pts[:, 2]
    floor_y   = float(np.percentile(y, 4))
    ceiling_y = float(np.percentile(y, 96))
    return {
        "floor_y":   floor_y,
        "ceiling_y": ceiling_y,
        "height":    ceiling_y - floor_y,
        "width":     float(np.percentile(x, 97) - np.percentile(x, 3)),
        "depth":     float(np.percentile(z, 97) - np.percentile(z, 3)),
        "x_min":     float(np.percentile(x, 3)),
        "x_max":     float(np.percentile(x, 97)),
        "z_min":     float(np.percentile(z, 3)),
        "z_max":     float(np.percentile(z, 97)),
    }


def add_structural_markers(scene, room: dict) -> None:
    """Add colour-coded floor / ceiling / wall outlines to the scene."""
    from trimesh.path.entities import Line as LineEntity
    from trimesh.path.path import Path3D

    f, c = room["floor_y"], room["ceiling_y"]
    x0, x1 = room["x_min"], room["x_max"]
    z0, z1 = room["z_min"], room["z_max"]

    def _rect(name, pts4, color):
        ents = [LineEntity(points=[i, (i + 1) % 4], color=color) for i in range(4)]
        scene.add_geometry(
            Path3D(entities=ents, vertices=np.array(pts4, dtype=float)),
            geom_name=name,
        )

    def _seg(name, p0, p1, color):
        ent = [LineEntity(points=[0, 1], color=color)]
        scene.add_geometry(
            Path3D(entities=ent, vertices=np.array([p0, p1], dtype=float)),
            geom_name=name,
        )

    _rect("bp_floor",   [[x0,f,z0],[x1,f,z0],[x1,f,z1],[x0,f,z1]],
          np.array([ 80, 140, 220, 220], dtype=np.uint8))
    _rect("bp_ceiling", [[x0,c,z0],[x1,c,z0],[x1,c,z1],[x0,c,z1]],
          np.array([200, 235, 255, 220], dtype=np.uint8))
    for i, (xi, zi) in enumerate([[x0,z0],[x1,z0],[x1,z1],[x0,z1]]):
        _seg(f"bp_wall_{i}", [xi, f, zi], [xi, c, zi],
             np.array([100, 190, 255, 200], dtype=np.uint8))


def apply_blueprint_style(scene) -> None:
    """Restyle point clouds and meshes to architectural blueprint look."""
    import trimesh as _tm

    for name, geom in list(scene.geometry.items()):
        if isinstance(geom, _tm.PointCloud):
            geom.colors = _height_gradient(geom.vertices[:, 1])
        elif isinstance(geom, _tm.Trimesh):
            nv = len(geom.vertices)
            geom.visual.vertex_colors = np.tile(_BP_MESH, (nv, 1))
            scene.add_geometry(_wireframe_from_mesh(geom), geom_name=name + "_bp_wire")

    _add_floor_grid(scene)


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
            with torch.cuda.amp.autocast(dtype=DTYPE):
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


# ── Gradio callback ───────────────────────────────────────────────────────────

def reconstruct(
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
    blueprint_mode: bool,
):
    paths: list[str] = []

    if video_file is not None:
        video_path = video_file if isinstance(video_file, str) else video_file["name"]
        paths = extract_video_frames(video_path, sample_fps)
        if not paths:
            return None, "Could not extract frames from video."
    elif image_files:
        paths = [f.name if hasattr(f, "name") else str(f) for f in image_files]

    if not paths:
        return None, "Upload images or a video to begin."
    if len(paths) < 2:
        return None, "Need at least 2 frames for 3D reconstruction."
    if len(paths) > MAX_FRAMES:
        paths = paths[:MAX_FRAMES]  # silently cap rather than error

    try:
        t0 = time.time()
        preds = run_vggt_omega(paths)
        elapsed = time.time() - t0

        glb_path = tempfile.mktemp(suffix=".glb")

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
        if detect_enabled and detect_prompt.strip():
            dets_2d = detect_objects_2d(paths, detect_prompt, box_threshold)
            dets_3d = cluster_detections(
                project_detections_to_3d(
                    dets_2d, preds["depth"], preds["extrinsic"], preds["intrinsic"]
                )
            )
            if dets_3d:
                add_boxes_to_scene(scene, dets_3d, preds["extrinsic"])

        # Blueprint styling + structural labelling
        room_info: dict = {}
        if blueprint_mode:
            apply_blueprint_style(scene)
            room_info = detect_room_structure(scene)
            if room_info:
                add_structural_markers(scene, room_info)

        scene.export(glb_path)

        # ── Status message ──
        src = "frames" if video_file is not None else "views"
        det_note = f", {len(dets_3d)} objects detected" if detect_enabled else ""
        bp_note  = " · Blueprint mode" if blueprint_mode else ""
        msg = f"Done in {elapsed:.1f}s — {len(paths)} {src} @ {IMAGE_RESOLUTION}px on {DEVICE} ({mode}{det_note}{bp_note})."

        if blueprint_mode and room_info:
            msg += (
                f"\n\n### Room Analysis\n"
                f"**Dimensions:** {room_info['width']:.1f} m wide "
                f"× {room_info['depth']:.1f} m deep "
                f"× {room_info['height']:.1f} m tall\n\n"
                f"**Structural elements** *(colour-coded in the 3D view)*\n"
                f"- 🟦 Floor — Y ≈ {room_info['floor_y']:.2f} m\n"
                f"- ⬜ Ceiling — Y ≈ {room_info['ceiling_y']:.2f} m\n"
                f"- 🔷 Walls — {room_info['width']:.1f} m × {room_info['depth']:.1f} m footprint\n"
            )

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

        return glb_path, msg

    except Exception:
        return None, f"Error:\n{traceback.format_exc()}"


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
    theme=gr.themes.Soft(),
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

            blueprint_mode = gr.Checkbox(
                value=False,
                label="Blueprint Mode",
                info="Wireframe geometry + height-gradient colours + floor/wall/ceiling labels",
            )

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

            run_btn = gr.Button(
                "Reconstruct 3D Space", variant="primary", size="lg"
            )

            output_type.change(
                fn=lambda v: gr.update(visible=(v in ["Mesh (TSDF)", "Mesh (Poisson)"])),
                inputs=output_type,
                outputs=mesh_resolution,
            )

        with gr.Column(scale=2):
            viewer = gr.Model3D(
                label="3D Reconstruction",
                clear_color=BLUEPRINT_BG,
            )
            status = gr.Markdown(label="Status")

    run_btn.click(
        fn=reconstruct,
        inputs=[image_upload, video_upload, sample_fps, conf_thres,
                show_cam, mask_black_bg, mask_white_bg, output_type, mesh_resolution,
                detect_enabled, detect_prompt, box_threshold, blueprint_mode],
        outputs=[viewer, status],
    )


if __name__ == "__main__":
    demo.launch(server_port=7860)

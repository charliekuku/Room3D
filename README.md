# Room3D

**3D reconstruction of office and data center spaces from photos or video.**

Powered by [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) (CVPR 2026 Oral — Meta Research & VGG Oxford). Upload 5–30 overlapping photos and get a dense 3D point cloud or mesh in seconds — no calibration, no markers.

## Features

- **Point cloud & mesh output** — point cloud, TSDF (watertight), or Poisson surface reconstruction
- **Object detection** — Grounding DINO back-projects detected objects (server racks, desks, monitors…) into 3D bounding boxes
- **Editable hybrid scenes** — SAM segments detections into movable objects while the aligned raw scan remains available as an accuracy overlay; edit labels, compare Raw/Structured/Both, move and rotate objects, then save the layout
- **Serverless GPU deploy** — `modal_app.py` runs the whole app on an L4 GPU on [Modal](https://modal.com), scaling to zero when idle
- **Metric auto-calibration** — detected server racks (42U ≈ 2.0 m) anchor the scene to real-world metres
- **Gemini Vision auto-labelling** — one click to detect what's in your scene and populate the detection prompt
- **Video input** — sample frames from a walkthrough video instead of uploading individual photos
- **Device-aware** — runs on CUDA, Apple MPS, or CPU; resolution and frame cap auto-adjusted to available memory

## Prerequisites

- Python 3.10+
- PyTorch 2.3+ (install separately before `setup.sh` — see note below)
- Hugging Face account with access to [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega)

## Setup

**1. Install PyTorch** (skip if already installed):

```bash
# CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon
pip install torch torchvision

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**2. Clone and install dependencies:**

```bash
git clone https://github.com/charliekuku/Room3D.git
cd Room3D
bash setup.sh
```

**3. Authenticate with Hugging Face and request model access:**

```bash
huggingface-cli login
# Then request access at https://huggingface.co/facebook/VGGT-Omega
```

**4. Run:**

```bash
python src/app.py
# Opens at http://localhost:7860
# The model (~4 GB) downloads automatically on first run.
```

Or download the checkpoint manually:

```bash
hf download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir checkpoints
```

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Enables Gemini Vision auto-label button |
| `CHECKPOINT_PATH` | `checkpoints/vggt_omega_1b_512.pt` | Path to a custom checkpoint |
| `IMAGE_RESOLUTION` | 256 (128 on low-RAM MPS) | Inference resolution in pixels |

## Tips for best results

- Walk around the room / between server racks, taking overlapping shots from multiple angles
- Include overhead shots to capture equipment tops
- Even, consistent lighting — avoid flash glare on screens and metal surfaces
- Minimise motion blur; keep the camera steady
- 10–20 photos is usually enough for a mid-size room; more frames = slower but denser

## Project structure

```
src/
  app.py                        # Gradio UI + inference pipeline + FastAPI server (editor & scenes)
  build_scene_cli.py            # Headless scene builder (video/photos → scenes/<name>/)
  detection/scene_builder.py    # SAM segmentation → movable objects, rack calibration, GLB export
  reconstruction/tsdf_fusion.py # TSDF and Poisson mesh reconstruction
  viewer/editor/                # three.js scene editor (served at /editor/?scene=<name>)
modal_app.py        # Serverless GPU deployment on Modal (CUDA L4)
scenes/             # Built editable scenes (gitignored)
tests/              # Phase tests (pytest tests/)
setup.sh            # Clones vggt_omega_repo and installs deps
requirements.txt    # App-level deps (heavy deps come from vggt_omega_repo)
checkpoints/        # Model weights (gitignored — downloaded on first run)
vggt_omega_repo/    # Upstream VGGT-Omega repo (gitignored — cloned by setup.sh)
```

## Deploy to Modal (CUDA)

One-time setup (secrets + weight caching) is documented at the top of [modal_app.py](modal_app.py). Then:

```bash
modal serve modal_app.py    # dev: temporary URL, hot-reloads src/ on save
modal deploy modal_app.py   # prod: persistent URL, scales to zero when idle
```

## Editable scene workflow

1. Run `python src/app.py`, upload a video or photos.
2. Enable **Detect objects** (optionally auto-label with Gemini), tick **Build editable scene**, and reconstruct.
3. The editor appears embedded below the status panel (or use the full-screen link): every
   detected object is a separate, labeled, movable node — parametric prefabs for
   racks/desks/monitors/chairs, or for everything else, a piece cut directly out of the
   dense scanned mesh (falling back to an isolated reconstruction, then a plain box).
   Observed room surfaces are preserved and synthetic backing fills only weakly scanned
   boundaries. Use **View → Raw Scan / Structured / Both** to inspect alignment; hover an
   object for its bounding box, point-support quality, label, and detection confidence.
4. Edit labels or move (G) / rotate (R) objects, then **Save layout** — changes persist to
   `scenes/<name>/scene.json`.

Or headless: `python src/build_scene_cli.py data/office.mp4 --fps 1`

How positions stay accurate: each object is placed from SAM-masked depth pixels aggregated
across *all* frames it appears in (median centre, percentile extents), single-frame
low-confidence detections are discarded, bases snap to the RANSAC-fitted floor plane, and
yaw snaps to the room's 90° grid when close.

## Generate one GLB from your own photo with TRELLIS.2

After deploying `modal_app.py`, an authenticated Modal user can upload a local
JPG, WebP, or PNG directly to the isolated TRELLIS.2 L4 function:

```bash
modal run modal_app.py::trellis_upload \
  --image-path data/chair.jpg \
  --output-path outputs/chair.glb
```

Ordinary photos are automatically segmented with the ungated MIT-licensed
`ZhengPeng7/BiRefNet` model. A PNG that already contains useful transparency
keeps its supplied alpha instead. For best results, use one clearly visible
object, avoid severe occlusion, include the complete silhouette, and leave a
little space around it. The first call after scale-to-zero loads the background
remover and TRELLIS weights; later calls reuse the warm container.

## License

Room3D code is MIT licensed. The bundled models have their own licenses:
- **VGGT-Omega** — see [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) for terms
- **Grounding DINO** — Apache 2.0
- **Gemini API** — Google API Terms of Service

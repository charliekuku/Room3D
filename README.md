# Room3D

**3D reconstruction of office and data center spaces from photos or video.**

Powered by [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) (CVPR 2026 Oral — Meta Research & VGG Oxford). Upload 5–30 overlapping photos and get a dense 3D point cloud or mesh in seconds — no calibration, no markers.

![Room3D demo UI](https://github.com/user-attachments/assets/placeholder)

## Features

- **Point cloud & mesh output** — point cloud, TSDF (watertight), or Poisson surface reconstruction
- **Blueprint mode** — architectural wireframe view with floor/ceiling/wall detection and height-gradient colouring
- **Object detection** — Grounding DINO back-projects detected objects (server racks, desks, monitors…) into 3D bounding boxes
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
git clone https://github.com/your-username/Room3D.git
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
python app.py
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
app.py              # Gradio UI + inference pipeline
tsdf_fusion.py      # TSDF and Poisson mesh reconstruction
setup.sh            # Clones vggt_omega_repo and installs deps
requirements.txt    # App-level deps (heavy deps come from vggt_omega_repo)
checkpoints/        # Model weights (gitignored — downloaded on first run)
vggt_omega_repo/    # Upstream VGGT-Omega repo (gitignored — cloned by setup.sh)
```

## License

Room3D code is MIT licensed. The bundled models have their own licenses:
- **VGGT-Omega** — see [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) for terms
- **Grounding DINO** — Apache 2.0
- **Gemini API** — Google API Terms of Service

#!/usr/bin/env bash
# Room3D setup — clones VGGT-Omega and installs all dependencies
set -eu
set -o pipefail

echo "=== Room3D Setup (VGGT-Omega) ==="

# Check Python version
python3 -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required (got ' + sys.version + ')'" || exit 1

# 1. Clone VGGT-Omega into vggt_omega_repo/
if [ -d "vggt_omega_repo" ]; then
    echo "[1/3] vggt_omega_repo/ already exists — skipping clone."
else
    echo "[1/3] Cloning VGGT-Omega from GitHub…"
    git clone https://github.com/facebookresearch/vggt-omega.git vggt_omega_repo
fi

# 2. Install VGGT-Omega as an editable package (required by its pyproject.toml)
echo "[2/3] Installing Python dependencies…"

# PyTorch: install a version that matches your system BEFORE running this script.
# For CUDA 12.x:   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For Apple MPS:   pip install torch torchvision
# For CPU only:    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#
# Uncomment and run the appropriate line above if you haven't installed PyTorch yet.

pip install -r vggt_omega_repo/requirements.txt
pip install -e vggt_omega_repo/          # editable install for the vggt_omega package

# requirements_demo.txt minus onnxruntime: it only backs visual_util's sky-
# segmentation feature, which app.py never enables (mask_sky is hardcoded
# False — indoor scans don't need it) and is a sizeable, otherwise-unused dep.
grep -v '^onnxruntime' vggt_omega_repo/requirements_demo.txt > vggt_omega_repo/requirements_demo_filtered.txt
pip install -r vggt_omega_repo/requirements_demo_filtered.txt

# 3. Install our app-level requirements
echo "[3/3] Installing app requirements…"
pip install -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo ""
echo "Before first run, download the model weights:"
echo "  1. Request access at https://huggingface.co/facebook/VGGT-Omega"
echo "  2. Once approved: huggingface-cli login"
echo "  3. The app will auto-download on first launch, OR manually:"
echo "       huggingface-cli download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir checkpoints"
echo ""
echo "Start the demo:"
echo "  python src/app.py"
echo ""
echo "Or point to a different checkpoint:"
echo "  CHECKPOINT_PATH=path/to/model.pt python src/app.py"

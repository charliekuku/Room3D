"""
trellis_gen.py — Image-to-3D object generation with Microsoft TRELLIS.2.

Runs *inside the dedicated TRELLIS Modal container* (its own CUDA image with
trellis2 + the compiled extensions), invoked as a separate Modal function from
the reconstruction job — never in the VGGT pipeline container. TRELLIS.2-4B
peaks near the L4's full 24 GB, so it must not share a GPU process with VGGT
(see modal_app.py's generate_object_glb).

Contract used by scene_builder: generate_glb_from_image(crop_rgb) -> GLB bytes,
or None on any failure (including CUDA OOM). None means "fall back" — the caller
keeps the prefab/box mesh it would otherwise have produced, so a failed or
out-of-memory generation never breaks a scene.

DEPLOY-TO-VERIFY: TRELLIS.2's exact call surface (pipeline.run return shape,
o_voxel.postprocess.to_glb kwargs) can only be confirmed against the compiled
package on the GPU container. This mirrors the documented API in the
microsoft/TRELLIS.2 README; adjust here if the installed version differs.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

# Reuse one loaded pipeline across calls in a warm container (the 4B weights are
# ~10-12 GB — reloading per object would dominate runtime).
_pipeline = None


class _AlphaOnlyBackgroundRemover:
    """Stand-in for TRELLIS.2's eagerly-created local BiRefNet.

    Room3D supplies the foreground mask as the input image's alpha channel, so
    TRELLIS.2's ``preprocess_image`` never needs to call a background-removal
    model.  Upstream nevertheless constructs BiRefNet while loading the
    pipeline, which downloads the separately gated briaai/RMBG-2.0 weights.
    Replacing that constructor with this device-compatible stand-in avoids the
    unused download.  Calling it is an error: maskless inputs are rejected
    before the pipeline is loaded instead of silently using an uncut photo.
    """

    def __init__(self, *args, **kwargs):
        pass

    def to(self, device):
        return self

    def __call__(self, image):
        raise RuntimeError(
            "Room3D TRELLIS generation requires an RGBA input with a foreground mask"
        )


def _has_foreground_alpha(arr: np.ndarray) -> bool:
    """Whether an image carries a non-empty, non-opaque foreground mask."""
    if arr.ndim != 3 or arr.shape[2] != 4:
        return False
    alpha = arr[:, :, 3]
    return bool(np.any(alpha > 0) and np.any(alpha < 255))


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        # expandable_segments is TRELLIS.2's own recommended allocator setting —
        # it's what keeps 512³ inference under 24 GB on an L4/4090-class card.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

        # The upstream repository is cloned, not installed as a wheel. Modal's
        # default /root entry makes the OUTER /root/trellis2 directory look like
        # a namespace package called `trellis2`; its actual package (and
        # pipelines/) is one level below that. Put the repository root first so
        # Python resolves /root/trellis2/trellis2/__init__.py instead.
        repo_root = os.environ.get("TRELLIS2_ROOT", "/root/trellis2")
        if not os.path.isfile(os.path.join(repo_root, "trellis2", "__init__.py")):
            raise RuntimeError(f"TRELLIS.2 package not found under {repo_root}")
        if repo_root in sys.path:
            sys.path.remove(repo_root)
        sys.path.insert(0, repo_root)

        # Clear an outer-directory namespace package if another import created
        # it before this loader ran; otherwise Python keeps the bad resolution
        # cached even after sys.path is corrected.
        loaded = sys.modules.get("trellis2")
        if loaded is not None and getattr(loaded, "__file__", None) is None:
            del sys.modules["trellis2"]
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        from trellis2.pipelines import rembg as rembg_module
        import transformers

        if not transformers.__version__.startswith("4.57."):
            raise RuntimeError(
                "TRELLIS.2 requires transformers 4.57.x; found "
                f"{transformers.__version__}. Rebuild the Modal image."
            )

        # Trellis2ImageTo3DPipeline.from_pretrained looks up the configured
        # remover as getattr(rembg, "BiRefNet") and instantiates it eagerly.
        # Room3D already has a SAM mask, so substitute a no-download stand-in.
        rembg_module.BiRefNet = _AlphaOnlyBackgroundRemover

        print(f"[TRELLIS] Loading TRELLIS.2-4B (transformers {transformers.__version__}; "
              "using Room3D SAM alpha; local RMBG disabled)…", flush=True)
        _pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        _pipeline.cuda()
        print("[TRELLIS] Ready.", flush=True)
    return _pipeline


def generate_glb_from_image(
    crop_rgb: np.ndarray,
    decimation_target: int = 200_000,
    texture_size: int = 1024,
) -> bytes | None:
    """Generate a textured GLB from an RGBA crop whose alpha is the SAM mask.

    Returns GLB bytes, or None on CUDA OOM / any failure so the caller can fall
    back to its prefab/box mesh. An RGBA crop carries a clean subject cutout, so
    TRELLIS uses that alpha as the foreground mask instead of re-running its own
    background removal on a box that may contain neighbouring clutter. Inputs
    without a useful alpha mask are declined because local RMBG is intentionally
    disabled; the caller retains its existing prefab/box fallback.
    """
    import torch
    from PIL import Image
    import o_voxel

    try:
        arr = np.ascontiguousarray(crop_rgb)
        if not _has_foreground_alpha(arr):
            print("[TRELLIS] No usable SAM alpha mask; keeping fallback mesh.", flush=True)
            return None
        image = Image.fromarray(arr, mode="RGBA")
        pipeline = _get_pipeline()
        with torch.no_grad():
            mesh = pipeline.run(image)[0]

        # PBR-ready GLB. extension_webp=False → PNG textures: three.js's
        # GLTFLoader (the editor) renders those without the EXT_texture_webp
        # dependency, so keep textures as PNG for portability.
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tf:
            tmp_path = tf.name
        try:
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=texture_size,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=False,
            )
            glb.export(tmp_path, extension_webp=False)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except torch.cuda.OutOfMemoryError as exc:
        print(f"[TRELLIS] CUDA OOM — falling back to prefab ({exc}).", flush=True)
        torch.cuda.empty_cache()
        return None
    except Exception as exc:
        print(f"[TRELLIS] Generation failed ({type(exc).__name__}: {exc}).", flush=True)
        return None

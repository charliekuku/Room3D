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
import time

import numpy as np

# Reuse one loaded pipeline across calls in a warm container (the 4B weights are
# ~10-12 GB — reloading per object would dominate runtime).
_pipeline = None


class _AutoBackgroundRemover:
    """TRELLIS-compatible, ungated BiRefNet foreground extractor.

    Some TRELLIS.2 revisions configure a gated BRIA repository in pipeline.json.
    Room3D deliberately ignores that configured model name and uses the official
    MIT-licensed ``ZhengPeng7/BiRefNet`` checkpoint instead. TRELLIS's low-VRAM
    preprocessing moves this model to CUDA only while producing alpha, then
    returns it to CPU before the much larger 3D models run.
    """

    MODEL_NAME = "ZhengPeng7/BiRefNet"

    def __init__(self, *args, **kwargs):
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        print(f"[TRELLIS] Loading background remover {self.MODEL_NAME}…", flush=True)
        self.model = AutoModelForImageSegmentation.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True,
        ).eval()
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.device = torch.device("cpu")

    def to(self, device):
        import torch

        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")

    def __call__(self, image):
        import torch
        from PIL import Image
        from torchvision import transforms

        original = image.convert("RGB")
        tensor = self.transform(original).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            prediction = self.model(tensor)[-1].sigmoid()[0].squeeze().float().cpu()
        mask = transforms.ToPILImage()(prediction).resize(
            original.size, resample=Image.Resampling.BILINEAR,
        )
        result = original.convert("RGBA")
        result.putalpha(mask)
        return result


def _has_foreground_alpha(arr: np.ndarray) -> bool:
    """Whether an image carries a non-empty, non-opaque foreground mask."""
    if arr.ndim != 3 or arr.shape[2] != 4:
        return False
    alpha = arr[:, :, 3]
    return bool(np.any(alpha > 0) and np.any(alpha < 255))


def _prepare_input_image(arr: np.ndarray):
    """Convert an RGB/RGBA array into the PIL mode TRELLIS should preprocess.

    A useful existing alpha mask is preserved. RGB and fully opaque RGBA inputs
    are passed without alpha, which asks TRELLIS to run BiRefNet automatically.
    Fully transparent inputs are invalid because they contain no foreground.
    """
    from PIL import Image

    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size and np.nanmax(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0).clip(0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4) or min(arr.shape[:2]) < 2:
        raise ValueError("input image must be an H×W RGB or RGBA array")
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        if not np.any(alpha > 0):
            raise ValueError("input image is fully transparent")
        if _has_foreground_alpha(arr):
            return Image.fromarray(arr), "provided-alpha"
        arr = arr[:, :, :3]
    return Image.fromarray(arr), "automatic-birefnet"


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

        # Force an ungated remover even when an older cached TRELLIS checkout's
        # pipeline.json names the gated BRIA model. The wrapper intentionally
        # ignores constructor model-name arguments.
        rembg_module.BiRefNet = _AutoBackgroundRemover

        print(f"[TRELLIS] Loading TRELLIS.2-4B (transformers {transformers.__version__}; "
              "SAM alpha or automatic BiRefNet background removal)…", flush=True)
        _pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        _pipeline.cuda()
        print("[TRELLIS] Ready.", flush=True)
    return _pipeline


def generate_glb_from_image(
    crop_rgb: np.ndarray,
    decimation_target: int = 200_000,
    texture_size: int = 1024,
) -> bytes | None:
    """Generate a textured GLB from an RGB photo or masked RGBA crop.

    Returns GLB bytes, or None on CUDA OOM / any failure so the caller can fall
    back to its prefab/box mesh. A useful RGBA alpha mask is used directly;
    ordinary RGB/JPG and fully opaque PNG inputs are automatically segmented by
    the ungated BiRefNet remover before TRELLIS inference.
    """
    import torch
    import o_voxel

    started = time.perf_counter()
    try:
        image, masking = _prepare_input_image(crop_rgb)
        print(f"[TRELLIS] Input {image.size[0]}×{image.size[1]}; mask={masking}.", flush=True)
        pipeline_started = time.perf_counter()
        pipeline = _get_pipeline()
        pipeline_ready = time.perf_counter()
        with torch.no_grad():
            mesh = pipeline.run(image)[0]
        inference_done = time.perf_counter()

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
                data = f.read()
            finished = time.perf_counter()
            print(
                "[TRELLIS] Completed "
                f"(pipeline-ready={pipeline_ready - pipeline_started:.1f}s, "
                f"inference={inference_done - pipeline_ready:.1f}s, "
                f"postprocess/export={finished - inference_done:.1f}s, "
                f"total={finished - started:.1f}s).",
                flush=True,
            )
            return data
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

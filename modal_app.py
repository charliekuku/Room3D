"""
Room3D on Modal — serverless GPU deployment.

One-time setup:
  1. pip install modal && modal setup
  2. Get the gated VGGT checkpoint into the Volume (upload your local copy):
       modal volume create room3d-checkpoints
       modal volume put room3d-checkpoints checkpoints/vggt_omega_1b_512.pt vggt_omega_1b_512.pt
  3. Secret for the Gemini auto-label button (or remove GEMINI_SECRET below):
       modal secret create gemini GEMINI_API_KEY=xxx
  4. TRELLIS.2 requires Meta's gated DINOv3 encoder. Request/accept access at
       https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
     using the account that owns your read token, then create the Modal secret:
       modal secret create huggingface HF_TOKEN=hf_xxx
     Room3D supplies SAM alpha masks and disables TRELLIS.2's otherwise eager
     local RMBG-2.0 loader, so briaai/RMBG-2.0 access is not required.

GDINO + SAM weights are ungated and download automatically on the first run
(get_model / the detection path fetch them into the hf-cache Volume).

Run:
  modal serve modal_app.py     # dev: live-reloads src/ on save, temporary URL
  modal deploy modal_app.py    # prod: persistent URL, scales to zero when idle
"""

import modal

APP_ROOT = "/root/room3d"
HF_CACHE = "/cache/huggingface"
CHECKPOINT = f"{APP_ROOT}/checkpoints/vggt_omega_1b_512.pt"

# GPU for inference. L4 (24 GB) runs VGGT-Omega 1B at full 512px.
# Swap to "T4" for cheaper smoke tests (16 GB, fp16 — keep frame counts low).
GPU = "L4"

app = modal.App("room3d")

# Weights persist across containers so they download once, not per cold start.
checkpoints_vol = modal.Volume.from_name("room3d-checkpoints", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("room3d-hf-cache", create_if_missing=True)
scenes_vol = modal.Volume.from_name("room3d-scenes", create_if_missing=True)

# Progress/cancellation channel between `web` (polls it) and the spawned GPU
# job (writes to it) — see run_reconstruction_job / src/app.py's
# run_pipeline_for_job for why these run as decoupled Modal inputs now.
job_status = modal.Dict.from_name("room3d-job-status", create_if_missing=True)

VOLUMES = {
    f"{APP_ROOT}/checkpoints": checkpoints_vol,
    HF_CACHE: hf_cache_vol,
    f"{APP_ROOT}/scenes": scenes_vol,
}

# gemini: only needed for the auto-label button in the UI.
GEMINI_SECRET = [modal.Secret.from_name("gemini", required_keys=["GEMINI_API_KEY"])]
# TRELLIS.2 loads Meta's gated DINOv3 image encoder at runtime. The token's
# Hugging Face account must first be granted access to
# facebook/dinov3-vitl16-pretrain-lvd1689m.
HF_SECRET = [modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])]

# Mirrors setup.sh: clone VGGT-Omega, install its requirements + ours.
image = (
    modal.Image.debian_slim(python_version="3.11")
    # libopengl0: PyMeshLab's filter plugins (libfilter_meshing.so — Poisson
    # reconstruction + normal estimation) link against libOpenGL.so.0. Without
    # it the plugins silently fail to load and MeshSet "loses" methods like
    # compute_normal_for_point_clouds (the AttributeError seen in prod was
    # this, not a wrong pymeshlab version).
    .apt_install("git", "libgl1", "libglib2.0-0", "libegl1", "libopengl0")
    .run_commands(
        f"git clone https://github.com/facebookresearch/vggt-omega.git {APP_ROOT}/vggt_omega_repo",
        f"pip install -r {APP_ROOT}/vggt_omega_repo/requirements.txt",
        f"pip install -e {APP_ROOT}/vggt_omega_repo",
        # requirements_demo.txt minus onnxruntime: it only backs visual_util's
        # sky-segmentation feature, which app.py never enables (mask_sky is
        # hardcoded False — indoor scans don't need it) and is a sizeable dep.
        f"grep -v '^onnxruntime' {APP_ROOT}/vggt_omega_repo/requirements_demo.txt "
        f"> {APP_ROOT}/vggt_omega_repo/requirements_demo_filtered.txt",
        f"pip install -r {APP_ROOT}/vggt_omega_repo/requirements_demo_filtered.txt",
    )
    .pip_install(  # requirements.txt (app-level)
        # Overrides requirements_demo.txt's gradio==5.50.0 + gradio-client==1.14.0 —
        # testing whether Gradio 6's queue/connection handling helps the mid-run
        # cancellations seen on Modal (see requirements.txt comment for detail).
        "gradio==6.20.0",
        "gradio-client==2.5.0",
        "scikit-image>=0.19",
        "huggingface_hub>=0.20",
        "python-dotenv>=1.0",
        "transformers==5.13.0",  # exact-pinned — see requirements.txt comment
        "pymeshlab==2025.7.post1",  # exact-pinned — see requirements.txt comment
        "google-genai>=1.0",
        "fastapi[standard]>=0.100",  # bundles uvicorn[standard] via the extra
        "trimesh",       # GLB export, scene graphs (scene_builder.py, tsdf_fusion.py)
        "matplotlib",    # colormaps (tsdf_fusion.py, app.py)
        "scipy",         # required by vggt_omega_repo/visual_util.py (module-level import)
        "requests",      # required by vggt_omega_repo/visual_util.py (module-level import)
    )
    .env({
        "HF_HOME": HF_CACHE,
        "CHECKPOINT_PATH": CHECKPOINT,
        "IMAGE_RESOLUTION": "512",  # full checkpoint resolution — no MPS caps here
        # Real CUDA OOM at 146 frames on the L4 (needed ~24.4 GB, had 23.66 GB) —
        # the published peak-memory-vs-frames table this was based on
        # (~0.074 GB/frame) clearly assumed a lower resolution than our 512px;
        # actual cost here refits to ~0.126 GB/frame. 100 leaves real margin
        # under the ~103-119 GB-budget range that refit implies. Only one real
        # OOM data point so far — if you push this back up, step in small
        # increments (+20) and watch `modal serve` logs, don't jump straight
        # back to 180.
        "MAX_FRAMES": "100",
        # "1" makes the reconstruction job upgrade prefab/box fallback objects
        # into generated TRELLIS.2 assets (see generate_object_glb below). Off by
        # default — flip to "1" and redeploy once the trellis_image builds clean.
        "ROOM3D_GENERATE_OBJECTS": "1",
    })
    # Added at runtime, not baked in: `modal serve` hot-reloads on local edits.
    .add_local_dir("src", remote_path=f"{APP_ROOT}/src")
)


# ── TRELLIS.2 image-to-3D generation (isolated GPU container) ─────────────────
# Its own image + GPU container on purpose: TRELLIS.2-4B peaks near 24 GB at
# 512³, so it must never share a GPU process with VGGT. Called via .remote()
# from run_reconstruction_job, one crop at a time.
#
# DEPLOY-TO-VERIFY: this image compiles six CUDA extensions from TRELLIS.2's
# setup.sh (flash-attn, nvdiffrast, nvdiffrec, CuMesh, o-voxel, FlexGEMM). The
# build can only be validated with `modal deploy` on a GPU builder — expect to
# iterate on the exact setup.sh invocation / arch flags below.
TRELLIS_ROOT = "/root/trellis2"
trellis_hf_vol = modal.Volume.from_name("room3d-trellis-cache", create_if_missing=True)

trellis_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    # libjpeg-dev + zlib1g-dev: setup.sh installs pillow-simd from source and
    # tries to apt-get these via `sudo`, which doesn't exist in Modal's build
    # container (the "sudo: command not found" → pillow-simd zlib error). Provide
    # them here so pillow-simd compiles.
    .apt_install("git", "build-essential", "ninja-build", "libgl1",
                 "libglib2.0-0", "libegl1", "libjpeg-dev", "zlib1g-dev")
    .env({
        "HF_HOME": HF_CACHE,
        # Where trellis_gen.py finds the cloned repo (it imports trellis2 from here).
        "TRELLIS2_ROOT": TRELLIS_ROOT,
        # Ada (L4 / RTX 4090) = 8.9. Add "9.0" if you move this function to H100.
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "ATTN_BACKEND": "flash-attn",  # L4 is Ada → flash-attn supported (no xformers fallback)
        "SPCONV_ALGO": "native",
        "MAX_JOBS": "4",  # cap flash-attn's parallel nvcc jobs so the builder doesn't OOM
    })
    .pip_install("torch==2.6.0", "torchvision",
                 index_url="https://download.pytorch.org/whl/cu124")
    # setup.sh assumes a conda env that already has these. Its local extensions
    # build with --no-build-isolation against THIS interpreter, so without `wheel`
    # each one dies with "invalid command 'bdist_wheel'".
    .pip_install("wheel", "setuptools", "packaging", "ninja", "psutil", "einops")
    # flash-attn's sdist imports torch at build time, but setup.sh installs it
    # WITH build isolation (fresh env, no torch → "No module named 'torch'").
    # Pre-build it here with isolation off — torch is present — so setup.sh's
    # `pip install flash-attn==2.7.3` finds it already satisfied and skips.
    .pip_install("flash-attn==2.7.3", extra_options="--no-build-isolation")
    # clang: Modal's add_python interpreter was built with clang, so distutils
    # invokes `clang` to compile pillow-simd's C extension (the CUDA extensions
    # use g++/nvcc and are fine). Placed AFTER flash-attn on purpose — adding it
    # to the first apt layer would invalidate the cached ~30-45 min flash-attn
    # build and force a full recompile.
    .apt_install("clang")
    .run_commands(
        f"git clone -b main --recursive https://github.com/microsoft/TRELLIS.2.git {TRELLIS_ROOT}",
    )
    # setup.sh probes for a GPU ("No supported GPU found") and won't compile the
    # CUDA extensions without one — but Modal builds on CPU by default. Attach a
    # GPU to just this build step so the probe passes and nvcc targets Ada (8.9).
    # No --new-env: install into the container interpreter, not a conda env Modal
    # wouldn't activate. flash-attn is already satisfied (above), so --flash-attn
    # here is a no-op skip; the remaining extensions now find `wheel`.
    .run_commands(
        f"cd {TRELLIS_ROOT} && bash setup.sh --basic --flash-attn --nvdiffrast "
        f"--nvdiffrec --cumesh --o-voxel --flexgemm",
        gpu="L4",
    )
    # setup.sh installs an unconstrained `transformers`, which now resolves to
    # 5.x. TRELLIS.2's DINOv3FeatureExtractor accesses the 4.57 model layout
    # (`DINOv3ViTModel.layer`); 5.x removed that attribute and fails only when
    # the first object is generated. Match Microsoft's working Space exactly.
    # This image is isolated from Room3D's reconstruction image, which keeps
    # transformers 5.13 for Grounding DINO + SAM2.
    .pip_install("transformers==4.57.3")
    .run_commands(
        f"PYTHONPATH={TRELLIS_ROOT} python -c \"from trellis2.pipelines import "
        f"Trellis2ImageTo3DPipeline; import transformers; "
        f"assert transformers.__version__ == '4.57.3'; "
        f"print('TRELLIS.2 pipeline import OK, transformers', transformers.__version__)\"",
        # Importing FlexGEMM initializes Triton's active CUDA driver, so even
        # this no-inference validation must run on a GPU builder.
        gpu="L4",
    )
    .add_local_dir("src", remote_path=f"{APP_ROOT}/src")
)


@app.function(
    image=trellis_image,
    gpu="L4",  # 24 GB — the whole card to itself. Bump to "L40S" if you see OOM/latency.
    volumes={HF_CACHE: trellis_hf_vol, f"{APP_ROOT}/scenes": scenes_vol},
    secrets=HF_SECRET,
    timeout=1800,
    scaledown_window=120,
    max_containers=1,  # keep the ~10-12 GB 4B weights warm across a scene's objects
)
def generate_object_glb(crop_rgb) -> bytes | None:
    """Generate a textured GLB from one object crop (numpy RGB). Returns GLB
    bytes, or None on OOM/failure so the caller keeps its prefab/box fallback."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "trellis_gen", f"{APP_ROOT}/src/detection/trellis_gen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = mod.generate_glb_from_image(crop_rgb)
    trellis_hf_vol.commit()  # persist the 4B weights after first-run download
    return data


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    # GDINO + SAM now run on CUDA (moved off CPU earlier), so this no longer
    # needs to hold their weights alongside the VGGT checkpoint in CPU RAM —
    # just torch.load of the 4.3 GB checkpoint (~2x transient) plus buffers.
    # Was 24576: Modal reported scheduling delays waiting for an L4 worker
    # with that much free memory ("Relaxing requirements may lead to faster
    # scheduling"). Watch logs for CPU OOM and raise this back up if hit.
    memory=16384,
    volumes=VOLUMES,
    # Grounding DINO downloads from Hugging Face in this worker. Supplying the
    # token avoids unauthenticated Hub requests and their lower rate limits.
    secrets=GEMINI_SECRET + HF_SECRET,
    timeout=3600,
    scaledown_window=120,  # idle seconds before scale-to-zero stops billing
    # No @modal.concurrent here on purpose: this function does the actual
    # VGGT-Omega/detection/scene-building work and is only ever invoked via
    # .spawn() from `web`, never directly by an HTTP request. One input per
    # container is the safe default Modal itself recommends for exactly this
    # kind of synchronous, long-running work — see the note on `web` below.
    max_containers=1,  # keep one warm container so the 4.3GB checkpoint stays loaded
)
def run_reconstruction_job(job_id: str, params: dict) -> dict:
    """Runs the reconstruction pipeline as its own Modal input, decoupled
    from any web request's lifecycle. `web` spawns this and short browser-timer
    requests poll `job_status` for progress/result instead of running or
    awaiting the pipeline in one request handler.

    This replaces the previous design, where reconstruct() ran synchronously
    inside `web`'s own (concurrent, asgi) request handler. That combination is
    explicitly called out in Modal's docs: "When using input concurrency with
    a synchronous Function, a single input cancellation will terminate the
    entire container." Prod logs confirmed this exactly — a cancellation
    signal would arrive at an arbitrary point (mid-detection in one run, right
    after SAM loaded in another), the synchronous pipeline code had no way to
    notice or yield to it, and Modal force-killed the whole container after a
    30s grace period ("killing task"), always before scene.json was ever
    written. Running the pipeline here means even if the web request that
    triggered it gets cancelled, this input is unaffected and runs to
    completion in its own container.
    """
    import os
    import sys

    sys.path.insert(0, f"{APP_ROOT}/src")
    import app as app_module
    from app import run_pipeline_for_job

    # Wire the TRELLIS generator into the pipeline when enabled. .remote runs it
    # in the isolated generate_object_glb container (its own L4), so VGGT never
    # shares a GPU process with TRELLIS.2. Off unless ROOM3D_GENERATE_OBJECTS=1.
    if os.environ.get("ROOM3D_GENERATE_OBJECTS") == "1":
        app_module.set_object_generator(generate_object_glb.remote)

    try:
        result = run_pipeline_for_job(job_id, params)
        # Make the output visible to the web container before advertising the
        # job as complete. Otherwise a fast status poll can observe "complete"
        # while output.glb / scene.json is still only in this container's view
        # of the Volume.
        scenes_vol.commit()
        entry = job_status.get(job_id, {}) or {}
        entry.update(
            state="complete",
            result=result,
            glb_out_path=params.get("glb_out_path"),
        )
        job_status.put(job_id, entry)
        return result
    except Exception as exc:
        # No browser request holds the FunctionCall object open, so persist a
        # terminal failure here as well. A refreshed page can then show the
        # failure instead of polling an apparently-running job forever.
        entry = job_status.get(job_id, {}) or {}
        entry.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        job_status.put(job_id, entry)
        raise


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    volumes=VOLUMES,
    secrets=GEMINI_SECRET,
    timeout=600,
    scaledown_window=120,  # idle seconds before scale-to-zero stops billing
    max_containers=1,  # Gradio queue state is per-process — keep one container
)
# max_inputs=20: `web` now only serves Gradio UI, static assets, small API
# endpoints, and spawns/polls GPU jobs — no more long synchronous blocking
# calls in a request handler. Concurrency is what keeps asset loads/polls from
# queueing behind each other (see git history for the stalling incident this
# fixed originally). The trade-off Modal warns about for this combination —
# a cancelled input killing the whole container — no longer bites the GPU
# work at all now that it runs in run_reconstruction_job instead; the worst
# a stray cancellation can do here is interrupt a cheap poll/asset request.
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, f"{APP_ROOT}/src")
    from app import build_server  # src/app.py — device auto-detects CUDA

    # Async on purpose — these run inside the ASGI event loop, and a blocking
    # Modal call there both stalls the loop and (worse) can't be cancelled,
    # which turns a dropped client connection into a whole-container kill
    # after Modal's 30s cancellation grace period.
    async def commit_scenes():
        await scenes_vol.commit.aio()

    async def reload_scenes():
        await scenes_vol.reload.aio()

    server = build_server(
        run_job_fn=run_reconstruction_job,
        job_status_dict=job_status,
        commit_scenes_fn=commit_scenes,
        reload_scenes_fn=reload_scenes,
        # Editor's on-demand "Regenerate with TRELLIS.2" awaits this in the ASGI
        # loop. Always wired (independent of ROOM3D_GENERATE_OBJECTS, which only
        # gates auto-generation during a build) so per-object regen works even
        # when the build pipeline runs scan-only.
        object_generator_fn=generate_object_glb.remote.aio,
    )

    # The editor's save endpoint writes scenes/<name>/scene.json; commit the
    # Volume after each write so layouts survive container recycling.
    @server.middleware("http")
    async def commit_scenes_mw(request, call_next):
        response = await call_next(request)
        if request.method == "POST" and request.url.path.startswith("/api/scenes/"):
            await scenes_vol.commit.aio()
        return response

    return server

# 🎯 Room3D — Phased Project Architecture

Welcome to the organized **Room3D** workspace! The codebase has been physically restructured and segmented into modular directories aligning directly with your multi-phase project roadmap. All module paths, file endpoints, and unit tests have been updated and validated.

---

## 📁 restructred Codebase Layout

```bash
Room3D/
├── src/                                  # Modular source directory
│   ├── app.py                            # Main Gradio & FastAPI Web Server
│   ├── build_scene_cli.py                # Headless command-line runner
│   │
│   ├── reconstruction/                   # PHASE 1 — Reconstruction Core
│   │   └── tsdf_fusion.py                # Dense TSDF voxel fusion to watertight meshes
│   │
│   ├── detection/                        # PHASE 2 — Object Detection & Positioning
│   │   └── scene_builder.py              # SAM mask segmenter & multi-frame OBB clusterer
│   │
│   └── viewer/                           # PHASE 3 & 4 — Interactive Web Viewer & BMS
│       └── editor/
│           └── index.html                # Three.js Custom Visual Layout Editor
│
├── modal_app.py                          # Serverless CUDA deployment (Modal, L4 GPU)
├── tests/                                # Automated Testing Framework
│   └── test_phase{0..4}_*.py             # Per-phase unit tests (pytest tests/)
│
├── data/                                 # Test videos and photosets (Phase 0, gitignored)
├── checkpoints/                          # Model weights (vggt_omega_1b_512.pt)
└── scenes/                               # Auto-generated scene layout outputs
```

---

## 🚀 Phase-by-Phase Implementation Guide & Run Commands

### 📸 Phase 0 — Capture Protocol & Test Data
* **Purpose:** Standardize image/video inputs and gather baseline datasets.
* **Working Assets:** 
  * `data/hospital.mp4`, `data/office.mp4` (video walkthroughs).
  * `data/samples/cyrusone/` (data center b-roll frame sequence).
  * `data/samples/rgbd_dataset_freiburg1_desk/` (TUM benchmark desk sequence).
* **Capture Best Practices:**
  * **Angle:** Capture server racks and walls obliquely ($45^\circ$) rather than head-on to give depth models strong parallax.
  * **Motion:** Run the camera in a slow, continuous loop at chest height; avoid quick vertical jerks.
  * **Lighting:** Keep illumination even; minimize blinking LED exposure glare on metal surfaces.

---

### 🧱 Phase 1 — Reconstruction Core
* **Purpose:** Process images/video into absolute metric depth coordinates and watertight room shell meshes.
* **Modules:**
  * `src/reconstruction/tsdf_fusion.py`: Fuses feed-forward predictions into high-quality TSDF meshes.
* **Execution Commands:**
  * **Reconstruct Point Cloud or TSDF/Poisson Mesh (via Web UI):**
    ```bash
    python src/app.py
    ```
    Open `http://localhost:7860` in your browser.

> The earlier Metric3D + Blender MCP experiments (`video_to_3d.py`, synthetic room
> generators, `hospital.blend`) were superseded by the VGGT pipeline and now live
> untracked in `blender/`.

---

### 🔍 Phase 2 — Object Detection, Labeling & Positioning
* **Purpose:** Run Grounding DINO to spot equipment, Segment Anything (SAM) to isolate pixels, and cluster 3D detections across frames to place parametric bounding boxes.
* **Modules:**
  * `src/detection/scene_builder.py`: Contains RANSAC floor-leveling, median rack-height metric calibration, 3D coordinate snapping, and Oriented Bounding Box (OBB) solvers.
* **Execution Commands (Headless CLI scene builder):**
  ```bash
  python src/build_scene_cli.py data/office.mp4 --prompt "server rack, desk, monitor, chair"
  ```
  This creates `scenes/office/scene.json`, `background.glb`, and discrete asset components in `scenes/office/objects/*.glb`.

* **Automated Unit Tests:**
  Verify the core geometric algorithms, principal-axis yaw calculations, floor-snapping correctness, and metric rack scaling without loading heavy PyTorch models:
  ```bash
  pytest tests/
  ```

---

### 🕹️ Phase 3 — Interactive Scene Editor
* **Purpose:** Drag, rotate, and interact with objects on a floor-snapped grid in a web browser, saving layouts back to your local files.
* **Modules:**
  * `src/viewer/editor/index.html`: WebGL application powered by Three.js, OrbitControls, and TransformControls. Served dynamically on `/editor/` by FastAPI.
* **Execution & Testing:**
  1. Spin up the server: `python src/app.py`
  2. Open any processed scene in the web browser:
     `http://localhost:7860/editor/?scene=office`
  3. Drag racks/chairs around and hit **Save layout** to write coordinates back to `scenes/office/scene.json`.

---

### 📊 Phase 4 — BMS Story & Validation
* **Purpose:** Quantify reconstruction error and demonstrate asset-management (BMS) integration.
* **Modules:**
  * `src/app.py`: Contains the REST API (`POST /api/scenes/{name}/layout`) that serves static scene configurations and links layout nodes to physical server rack attributes.
* **Deliverables:**
  * Check the terminal output after running `pytest tests/` to assert mathematical accuracy down to centimeters.
  * Scene coordinates in `scene.json` can be directly queried to feed rack IDs into temperature sensor charts, energy allocation systems, or external datacenter inventory tables.

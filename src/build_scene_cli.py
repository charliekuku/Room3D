#!/usr/bin/env python3
"""
build_scene_cli.py — Headless editable-scene builder.

  python build_scene_cli.py data/office.mp4 --fps 1
  python build_scene_cli.py img1.png img2.png ... --prompt "server rack, desk"

Writes scenes/<name>/ (scene.json + background.glb + objects/*.glb),
viewable at http://localhost:7860/editor/?scene=<name> when app.py is running.
"""
import argparse
import json
from pathlib import Path

_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def main():
    ap = argparse.ArgumentParser(description="Video/photos → editable 3D scene")
    ap.add_argument("input", nargs="+", help="One video file, or multiple images")
    ap.add_argument("--fps", type=float, default=1.0, help="Video sample rate")
    ap.add_argument("--prompt", default=None, help="Detection labels (comma-separated)")
    ap.add_argument("--name", default=None, help="Scene name (default: input stem)")
    ap.add_argument("--conf", type=float, default=20.0, help="Depth confidence percentile")
    ap.add_argument("--mesh-resolution", type=int, default=192)
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--reuse-duplicates", action="store_true",
                    help="Share one built asset across same-label, same-size "
                         "objects (e.g. rows of identical racks/chairs)")
    args = ap.parse_args()

    import os, sys
    import numpy as np
    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    sys.path.insert(0, PROJECT_ROOT)

    from src import app            # sets up vggt repo path, device, frame caps
    from src.detection import scene_builder

    first = args.input[0]
    if first.lower().endswith(_VIDEO_EXTS):
        paths = app.extract_video_frames(first, args.fps)
        name = args.name or Path(first).stem
    else:
        paths = list(args.input)
        name = args.name or "photos"

    if len(paths) > app.MAX_FRAMES:
        # Evenly resample across the whole clip — same as app.py's reconstruct() —
        # so long videos don't silently lose coverage after the frame budget runs out.
        idx = np.linspace(0, len(paths) - 1, app.MAX_FRAMES).round().astype(int)
        paths = [paths[i] for i in idx]
    if len(paths) < 2:
        raise SystemExit("Need at least 2 frames.")
    print(f"[CLI] {len(paths)} frames @ {app.IMAGE_RESOLUTION}px on {app.DEVICE}")

    preds = app.run_vggt_omega(paths)
    print("[CLI] Reconstruction done; detecting objects…")

    prompt = args.prompt or app.DEFAULT_DETECT_PROMPT
    dets_2d = app.detect_objects_2d(paths, prompt, args.box_threshold)
    n_det = sum(len(f["labels"]) for f in dets_2d)
    print(f"[CLI] {n_det} detections across frames.")

    scene_json = scene_builder.build_scene(
        preds, paths, dets_2d,
        out_root=app.SCENES_DIR,
        scene_name=name,
        conf_thres=args.conf,
        mesh_resolution=args.mesh_resolution,
        reuse_duplicates=args.reuse_duplicates,
    )
    print(json.dumps(
        {k: scene_json[k] for k in ("name", "metric", "scale", "room")}
        | {"objects": [(o["label"], o["source"], o["frames_seen"]) for o in scene_json["objects"]]},
        indent=2,
    ))
    print(f"[CLI] Open: http://localhost:7860/editor/?scene={name}")


if __name__ == "__main__":
    main()

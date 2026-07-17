"""Conservative Gemini review of clustered object labels.

Gemini is advisory: it may keep, relabel, or reject an instance, but never
changes geometry or placement. Decisions are schema-constrained and then
validated locally before being applied.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from PIL import Image


_REVIEW_PROMPT = """Review these object detections from one indoor room scan.
Each image is preceded by its numeric ID and Grounding DINO label. For every
ID, choose exactly one action:
- keep: the label is a reasonable concise object category;
- relabel: the pictured object is clearly a different category;
- reject: this is clearly background, a fragment, or not one distinct object.

Be conservative. Partial visibility, blur, or uncertainty means keep. Use a
short singular category such as 'office chair', 'desk', 'monitor', 'door', or
'window'; do not guess brands or model numbers. Confidence is your confidence
in changing/rejecting the existing result, not detection confidence.
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "action": {"type": "string", "enum": ["keep", "relabel", "reject"]},
                    "label": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["id", "action", "label", "confidence", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}


def _instance_crop(inst: dict, image_paths: list[str], max_px: int = 640) -> Image.Image | None:
    fi, box = inst.get("best_frame"), inst.get("best_box")
    if fi is None or box is None or not (0 <= int(fi) < len(image_paths)):
        return None
    try:
        image = Image.open(image_paths[int(fi)]).convert("RGB")
    except Exception:
        return None
    width, height = image.size
    x1, y1, x2, y2 = [float(v) for v in box]
    px, py = 0.12 * (x2 - x1), 0.12 * (y2 - y1)
    crop_box = (
        max(int(x1 - px), 0), max(int(y1 - py), 0),
        min(int(x2 + px), width), min(int(y2 + py), height),
    )
    if crop_box[2] - crop_box[0] < 8 or crop_box[3] - crop_box[1] < 8:
        return None
    crop = image.crop(crop_box)
    if max(crop.size) > max_px:
        crop.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    return crop


def _clean_label(value: Any) -> str | None:
    label = re.sub(r"\s+", " ", str(value or "").strip().lower())
    label = label.strip(" .,:;-/")
    if not label or len(label) > 60 or not re.search(r"[a-z0-9]", label):
        return None
    return label


def apply_review_decisions(
    instances: list[dict],
    dets_2d: list[dict],
    decisions: list[dict],
    relabel_threshold: float = 0.70,
    reject_threshold: float = 0.85,
) -> tuple[list[dict], dict]:
    """Apply only well-formed, high-confidence decisions to selected instances."""
    by_id = {}
    for raw in decisions:
        try:
            idx = int(raw["id"])
            confidence = float(raw["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(instances) or idx in by_id:
            continue
        action = str(raw.get("action", "keep")).lower()
        if action not in {"keep", "relabel", "reject"}:
            continue
        by_id[idx] = {**raw, "action": action, "confidence": confidence}

    kept = []
    summary = {"reviewed": len(by_id), "relabeled": 0, "rejected": 0, "changes": []}
    for idx, inst in enumerate(instances):
        decision = by_id.get(idx)
        if decision is None:
            kept.append(inst)
            continue
        action, confidence = decision["action"], decision["confidence"]
        old_label = str(inst.get("label", "object"))
        if action == "reject" and confidence >= reject_threshold:
            summary["rejected"] += 1
            summary["changes"].append({
                "id": idx, "from": old_label, "to": None,
                "confidence": round(confidence, 3), "reason": str(decision.get("reason", ""))[:200],
            })
            continue
        new_label = _clean_label(decision.get("label"))
        if action == "relabel" and confidence >= relabel_threshold and new_label and new_label != old_label.lower():
            inst["original_label"] = old_label
            inst["label"] = new_label
            inst["label_review"] = {
                "provider": "gemini", "confidence": round(confidence, 3),
                "reason": str(decision.get("reason", ""))[:200],
            }
            for obs in inst.get("_observations", []):
                obs["label"] = new_label
                fi, di = obs.get("frame"), obs.get("det_idx")
                if (isinstance(fi, int) and isinstance(di, int) and
                        0 <= fi < len(dets_2d) and 0 <= di < len(dets_2d[fi].get("labels", []))):
                    dets_2d[fi]["labels"][di] = new_label
            summary["relabeled"] += 1
            summary["changes"].append({
                "id": idx, "from": old_label, "to": new_label,
                "confidence": round(confidence, 3), "reason": str(decision.get("reason", ""))[:200],
            })
        kept.append(inst)
    return kept, summary


def review_object_instances(
    instances: list[dict],
    image_paths: list[str],
    dets_2d: list[dict],
    api_key: str | None = None,
    model: str = "gemini-2.5-flash",
    max_objects: int = 24,
    batch_size: int = 12,
) -> tuple[list[dict], dict]:
    """Review representative object crops in small multimodal batches."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key or not instances:
        return instances, {"enabled": bool(api_key), "reviewed": 0, "relabeled": 0, "rejected": 0}

    from google import genai

    ranked = sorted(
        range(len(instances)),
        key=lambda i: (instances[i].get("frames_seen", 0), instances[i].get("score", 0)),
        reverse=True,
    )[:max_objects]
    available = [(idx, _instance_crop(instances[idx], image_paths)) for idx in ranked]
    available = [(idx, crop) for idx, crop in available if crop is not None]
    client = genai.Client(api_key=api_key)
    decisions = []
    for start in range(0, len(available), batch_size):
        contents: list[Any] = [_REVIEW_PROMPT]
        for idx, crop in available[start:start + batch_size]:
            inst = instances[idx]
            contents.extend([
                f"ID {idx}; current label: {inst['label']}; "
                f"detector confidence: {float(inst.get('score', 0)):.2f}; "
                f"seen in {int(inst.get('frames_seen', 1))} frame(s)",
                crop,
            ])
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": _RESPONSE_SCHEMA,
                "temperature": 0.1,
            },
        )
        payload = json.loads(response.text)
        decisions.extend(payload.get("decisions", []))

    reviewed, summary = apply_review_decisions(instances, dets_2d, decisions)
    summary.update({"enabled": True, "model": model, "submitted": len(available)})
    return reviewed, summary

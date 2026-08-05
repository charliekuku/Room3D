"""Canonical object labels shared by Gemini suggestion and review paths."""
from __future__ import annotations

import re
from typing import Any


# Keep this intentionally small. Every extra Grounding DINO phrase increases
# the chance that one physical region is grounded under multiple similar names.
# Installed data-centre components remain because they are important inventory
# even when smaller than room furniture.
CANONICAL_MAJOR_LABELS = (
    "server rack",
    "cabinet",
    "table",
    "chair",
    "monitor",
    "whiteboard",
    "shelf",
    "printer",
    "sofa",
    "door",
    "window",
    "network switch",
    "patch panel",
    "cable tray",
)


_ALIASES = {
    "rack": "server rack",
    "server cabinet": "server rack",
    "rack cabinet": "server rack",
    "equipment rack": "server rack",
    "storage cabinet": "cabinet",
    "cupboard": "cabinet",
    "desk": "table",
    "work desk": "table",
    "office desk": "table",
    "workstation": "table",
    "workstation desk": "table",
    "office chair": "chair",
    "desk chair": "chair",
    "computer chair": "chair",
    "seat": "chair",
    "screen": "monitor",
    "display": "monitor",
    "computer monitor": "monitor",
    "computer screen": "monitor",
    "television": "monitor",
    "tv": "monitor",
    "bookcase": "shelf",
    "shelving": "shelf",
    "glass door": "door",
    "doorway": "door",
    "office window": "window",
    "switch": "network switch",
    "server switch": "network switch",
}


def clean_label(value: Any) -> str | None:
    label = re.sub(r"\s+", " ", str(value or "").strip().lower())
    label = label.strip(" .,:;-/")
    if not label or len(label) > 60 or not re.search(r"[a-z0-9]", label):
        return None
    return label


def canonicalize_major_label(value: Any) -> str | None:
    """Map a known synonym to one Grounding DINO category.

    Unknown labels return ``None`` for auto-suggestion, whose vocabulary is
    deliberately closed. Callers reviewing an existing object may keep their
    cleaned unknown label instead.
    """
    label = clean_label(value)
    if label is None:
        return None
    if label in CANONICAL_MAJOR_LABELS:
        return label
    return _ALIASES.get(label)


def normalize_major_suggestions(
    suggestions: list[dict],
    sampled_frames: int,
    confidence_threshold: float = 0.65,
    single_view_confidence: float = 0.85,
    max_labels: int = 10,
) -> list[str]:
    """Filter, rank and de-duplicate Gemini's category suggestions."""
    best: dict[str, tuple[float, int]] = {}
    for item in suggestions:
        if not isinstance(item, dict) or item.get("major") is not True:
            continue
        label = canonicalize_major_label(item.get("label"))
        if label is None:
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
            frames_seen = int(item.get("frames_seen", 0))
        except (TypeError, ValueError):
            continue
        confidence = min(max(confidence, 0.0), 1.0)
        frames_seen = min(max(frames_seen, 0), max(sampled_frames, 1))
        is_opening = label in {"door", "window"}
        supported = (
            sampled_frames <= 1 or frames_seen >= 2 or
            confidence >= single_view_confidence or is_opening
        )
        if confidence < confidence_threshold or not supported:
            continue
        if label not in best or (confidence, frames_seen) > best[label]:
            best[label] = (confidence, frames_seen)

    ranked = sorted(
        best,
        key=lambda label: (best[label][0], best[label][1]),
        reverse=True,
    )
    return ranked[:max(int(max_labels), 1)]

from src.detection.label_taxonomy import (
    canonicalize_major_label,
    normalize_major_suggestions,
)


def test_major_label_aliases_collapse_similar_categories():
    assert canonicalize_major_label("office chair") == "chair"
    assert canonicalize_major_label("desk") == "table"
    assert canonicalize_major_label("computer screen") == "monitor"
    assert canonicalize_major_label("server cabinet") == "server rack"
    assert canonicalize_major_label("water bottle") is None


def test_major_suggestions_filter_clutter_support_and_duplicates():
    suggestions = [
        {"label": "chair", "confidence": 0.91, "frames_seen": 4, "major": True},
        {"label": "office chair", "confidence": 0.88, "frames_seen": 3, "major": True},
        {"label": "monitor", "confidence": 0.72, "frames_seen": 1, "major": True},
        {"label": "door", "confidence": 0.70, "frames_seen": 1, "major": True},
        {"label": "table", "confidence": 0.90, "frames_seen": 2, "major": False},
        {"label": "water bottle", "confidence": 0.99, "frames_seen": 5, "major": True},
    ]

    labels = normalize_major_suggestions(suggestions, sampled_frames=5)

    assert labels == ["chair", "door"]


def test_single_image_allows_one_view_major_category():
    labels = normalize_major_suggestions([
        {"label": "table", "confidence": 0.70, "frames_seen": 1, "major": True},
    ], sampled_frames=1)
    assert labels == ["table"]

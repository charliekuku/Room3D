"""Pure tests for conservative Gemini label-decision application."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detection.gemini_review import apply_review_decisions


def _instance(label, frame, det_idx):
    return {
        "label": label, "_observations": [{"label": label, "frame": frame, "det_idx": det_idx}],
    }


def test_review_relabels_and_propagates_to_frame_detection():
    instances = [_instance("seat", 0, 0)]
    dets = [{"labels": ["seat"]}]
    kept, summary = apply_review_decisions(instances, dets, [{
        "id": 0, "action": "relabel", "label": "Office Chair",
        "confidence": 0.92, "reason": "Visible wheeled office chair",
    }])
    assert kept[0]["label"] == "office chair"
    assert kept[0]["original_label"] == "seat"
    assert dets[0]["labels"][0] == "office chair"
    assert summary["relabeled"] == 1


def test_review_rejects_only_at_high_confidence():
    instances = [_instance("monitor", 0, 0), _instance("desk", 0, 1)]
    dets = [{"labels": ["monitor", "desk"]}]
    kept, summary = apply_review_decisions(instances, dets, [
        {"id": 0, "action": "reject", "label": "monitor", "confidence": 0.7, "reason": "uncertain"},
        {"id": 1, "action": "reject", "label": "desk", "confidence": 0.95, "reason": "background"},
    ])
    assert [item["label"] for item in kept] == ["monitor"]
    assert summary["rejected"] == 1


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)}/{len(tests)} passed")

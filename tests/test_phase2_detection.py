"""Unit tests for scene_builder's placement / clustering / calibration math.

Run:  python tests/test_scene_builder.py
(no VGGT model or SAM download needed — synthetic point sets only)
"""
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vggt_omega_repo"))

from src.detection import scene_builder as sb
from src.detection.dedup import suppress_overlapping_boxes


def _box_points(center, size, yaw=0.0, n=3000, seed=0):
    """Random points filling a box centred at `center` with extents `size`,
    rotated by `yaw` about +Y in the exporter's convention (x toward z)."""
    rng = np.random.default_rng(seed)
    pts = (rng.random((n, 3)) - 0.5) * np.asarray(size)
    x, z = pts[:, 0], pts[:, 2]
    pts = np.stack([np.cos(yaw) * x - np.sin(yaw) * z,
                    pts[:, 1],
                    np.sin(yaw) * x + np.cos(yaw) * z], axis=1)
    return pts + np.asarray(center)


def test_yaw_recovery():
    # Elongated footprint at 30° should be recovered (mod 180°)
    pts = _box_points([0, 1, 0], [2.0, 1.0, 0.5], yaw=np.radians(30))
    yaw, ratio = sb._yaw_of_points(pts[:, [0, 2]])
    assert ratio > 1.5, f"expected elongated footprint, ratio={ratio}"
    err = abs((yaw - np.radians(30) + np.pi / 2) % np.pi - np.pi / 2)
    assert err < np.radians(4), f"yaw error {np.degrees(err):.1f}°"


def test_snap_yaw():
    assert abs(sb._snap_yaw(np.radians(4), 0.0) - 0.0) < 1e-9
    assert abs(sb._snap_yaw(np.radians(87), 0.0) - np.pi / 2) < 1e-9
    # 30° off the grid → left alone
    assert abs(sb._snap_yaw(np.radians(30), 0.0) - np.radians(30)) < 1e-9
    # snapping is relative to the room's own orientation
    assert abs(sb._snap_yaw(np.radians(25), np.radians(20)) - np.radians(20)) < 1e-9


def test_cluster_merges_same_object_and_keeps_separate_ones():
    cols = np.zeros((3000, 3), dtype=np.uint8)
    obs = []
    # Rack A seen from 3 frames (slightly shifted centres)
    for f in range(3):
        pts = _box_points([0 + 0.03 * f, 1.0, 0], [0.6, 2.0, 1.0], seed=f)
        obs.append(sb._make_observation("server rack", 0.6, f, pts, cols))
    # Rack B 3 m away, 2 frames
    for f in range(2):
        pts = _box_points([3.0, 1.0, 0], [0.6, 2.0, 1.0], seed=10 + f)
        obs.append(sb._make_observation("server rack", 0.55, f, pts, cols))
    # Single-frame low-score ghost → should be dropped
    obs.append(sb._make_observation(
        "server rack", 0.3, 0, _box_points([8, 1, 8], [0.5, 0.5, 0.5]), cols))

    instances = sb.cluster_observations(obs)
    assert len(instances) == 2, f"expected 2 instances, got {len(instances)}"
    assert sorted(i["frames_seen"] for i in instances) == [2, 3]


def test_cross_label_duplicate_uses_multi_frame_label_evidence():
    cols = np.zeros((3000, 3), dtype=np.uint8)
    obs = []
    # Correct label appears consistently; a higher-scoring wrong phrase appears
    # once on the same box and same 3-D points.
    for frame in range(3):
        obs.append(sb._make_observation(
            "table", 0.58, frame,
            _box_points([0, 0.4, 0], [1.6, 0.8, 0.8], seed=frame), cols,
            box=np.array([20, 20, 180, 140]), det_idx=0,
        ))
    obs.append(sb._make_observation(
        "cabinet", 0.82, 1,
        _box_points([0.01, 0.4, 0.01], [1.6, 0.8, 0.8], seed=20), cols,
        box=np.array([22, 21, 179, 141]), det_idx=1,
    ))
    # A real monitor overlaps the table spatially but not with a near-identical
    # image box, so it must remain its own instance.
    obs.append(sb._make_observation(
        "monitor", 0.75, 1,
        _box_points([0, 0.85, 0], [0.5, 0.35, 0.15], seed=30), cols,
        box=np.array([75, 30, 125, 80]), det_idx=2,
    ))

    instances = sb.cluster_observations(obs)

    assert sorted(i["label"] for i in instances) == ["monitor", "table"]
    table = next(i for i in instances if i["label"] == "table")
    assert table["frames_seen"] == 3
    assert len(table["obs_refs"]) == 4


def test_cluster_keeps_same_frame_neighbours_separate():
    """Two adjacent chairs, close enough for the proximity threshold to merge
    them, but drawn as distinct boxes in every shared frame. The detector
    already separated them, so they must stay two instances — this is the
    chair-around-a-table over-merge that dropped one chair from the scene."""
    cols = np.zeros((3000, 3), dtype=np.uint8)
    obs = []
    for frame in range(3):
        # Chair A and chair B ~0.45 m apart (< 0.6·diag ≈ 0.6 m ⇒ proximity
        # alone would merge them), non-overlapping image boxes.
        obs.append(sb._make_observation(
            "chair", 0.66, frame,
            _box_points([0.0, 0.4, 0.0], [0.5, 0.85, 0.5], seed=frame), cols,
            box=np.array([10, 50, 90, 200]), det_idx=0,
        ))
        obs.append(sb._make_observation(
            "chair", 0.64, frame,
            _box_points([0.45, 0.4, 0.0], [0.5, 0.85, 0.5], seed=10 + frame), cols,
            box=np.array([95, 50, 175, 200]), det_idx=1,
        ))

    instances = sb.cluster_observations(obs)
    assert len(instances) == 2, f"adjacent chairs collapsed: got {len(instances)}"
    assert all(i["frames_seen"] == 3 for i in instances)


def test_cluster_still_merges_duplicate_box_same_frame():
    """A repeated grounding of one object in a single frame (high box-IoU) is a
    duplicate, not a second object — the same-frame guard must let it merge so
    genuine duplicates don't split into phantom instances."""
    cols = np.zeros((3000, 3), dtype=np.uint8)
    obs = []
    for frame in range(3):
        obs.append(sb._make_observation(
            "chair", 0.66, frame,
            _box_points([0.0, 0.4, 0.0], [0.5, 0.85, 0.5], seed=frame), cols,
            box=np.array([10, 50, 90, 200]), det_idx=0,
        ))
    # Duplicate detection of the same chair in frame 0 (near-identical box).
    obs.append(sb._make_observation(
        "chair", 0.60, 0,
        _box_points([0.01, 0.4, 0.01], [0.5, 0.85, 0.5], seed=99), cols,
        box=np.array([12, 52, 92, 202]), det_idx=1,
    ))

    instances = sb.cluster_observations(obs)
    assert len(instances) == 1, f"duplicate box split object: got {len(instances)}"


def test_cluster_merges_ambiguous_overlapping_boxes_using_3d_support():
    """Different phrases can draw moderately different boxes around one object.
    Such overlap is not positive evidence of two objects and must not become a
    permanent cannot-link before the point-cloud cleanup can assess it."""
    cols = np.zeros((3000, 3), dtype=np.uint8)
    obs = [
        sb._make_observation(
            "chair", 0.70, 0,
            _box_points([0.0, 0.4, 0.0], [0.5, 0.85, 0.5], seed=0), cols,
            box=np.array([20, 40, 120, 200]), det_idx=0,
        ),
        # IoU is deliberately between the distinct and duplicate thresholds.
        sb._make_observation(
            "chair", 0.65, 0,
            _box_points([0.02, 0.4, 0.01], [0.5, 0.85, 0.5], seed=1), cols,
            box=np.array([55, 45, 145, 195]), det_idx=1,
        ),
        sb._make_observation(
            "chair", 0.68, 1,
            _box_points([0.01, 0.4, 0.0], [0.5, 0.85, 0.5], seed=2), cols,
            box=np.array([25, 42, 125, 202]), det_idx=0,
        ),
    ]

    assert sb._same_frame_distinct(obs[0], obs[1]) is False
    instances = sb.cluster_observations(obs)
    assert len(instances) == 1
    assert len(instances[0]["_observations"]) == 3


def test_cluster_associates_same_object_across_different_labels():
    cols = np.zeros((3000, 3), dtype=np.uint8)
    obs = [
        sb._make_observation(
            label, score, frame,
            _box_points([0.02 * frame, 0.4, 0], [0.5, 0.85, 0.5], seed=frame), cols,
            box=np.array([10, 50, 90, 200]), det_idx=0,
        )
        for frame, (label, score) in enumerate([
            ("chair", 0.62), ("office seat", 0.70), ("chair", 0.64),
        ])
    ]

    instances = sb.cluster_observations(obs)
    assert len(instances) == 1
    assert instances[0]["frames_seen"] == 3
    assert instances[0]["label"] == "chair", "multi-frame label evidence should win"


def test_duplicate_cleanup_merges_disjoint_tracks_with_same_3d_volume():
    """Ambiguity can split one physical object into early/late tracks which
    never share a frame. Strong 3-D coincidence should reunite them."""
    cols = np.zeros((3000, 3), dtype=np.uint8)
    early = [
        sb._make_observation(
            "chair", 0.65, frame,
            _box_points([0.01 * frame, 0.4, 0], [0.5, 0.85, 0.5], seed=frame), cols,
            box=np.array([10, 50, 90, 200]), det_idx=0,
        )
        for frame in (0, 1)
    ]
    late = [
        sb._make_observation(
            "office seat", 0.62, frame,
            _box_points([0.01, 0.4, 0.01], [0.5, 0.85, 0.5], seed=20 + frame), cols,
            box=np.array([12, 52, 92, 202]), det_idx=0,
        )
        for frame in (2, 3)
    ]

    merged = sb._merge_cross_label_duplicates([
        sb._instance_from_observations(early),
        sb._instance_from_observations(late),
    ])

    assert len(merged) == 1
    assert merged[0]["frames_seen"] == 4
    assert merged[0]["label"] == "chair"


def test_duplicate_cleanup_respects_same_frame_distinct_boxes():
    """Even coincident noisy depth must not merge objects seen separately."""
    cols = np.zeros((3000, 3), dtype=np.uint8)
    a = sb._make_observation(
        "chair", 0.68, 0, _box_points([0, 0.4, 0], [0.5, 0.85, 0.5]), cols,
        box=np.array([10, 50, 80, 200]), det_idx=0,
    )
    b = sb._make_observation(
        "chair", 0.66, 0, _box_points([0.02, 0.4, 0], [0.5, 0.85, 0.5], seed=1), cols,
        box=np.array([100, 50, 170, 200]), det_idx=1,
    )

    merged = sb._merge_cross_label_duplicates([
        sb._instance_from_observations([a]),
        sb._instance_from_observations([b]),
    ])
    assert len(merged) == 2


def test_duplicate_cleanup_does_not_merge_nested_different_scale_objects():
    cols = np.zeros((3000, 3), dtype=np.uint8)
    table = sb._make_observation(
        "table", 0.7, 0, _box_points([0, 0.5, 0], [1.6, 0.8, 0.8]), cols,
        box=np.array([10, 30, 180, 180]), det_idx=0,
    )
    monitor = sb._make_observation(
        "monitor", 0.7, 1, _box_points([0, 0.7, 0], [0.5, 0.4, 0.15], seed=1), cols,
        box=np.array([60, 40, 120, 100]), det_idx=0,
    )
    merged = sb._merge_cross_label_duplicates([
        sb._instance_from_observations([table]),
        sb._instance_from_observations([monitor]),
    ])
    assert len(merged) == 2


def test_point_cloud_overlap_support_distinguishes_duplicate_and_neighbor():
    base = _box_points([0, 0.4, 0], [0.5, 0.85, 0.5], n=1200, seed=7)
    duplicate = base + np.array([0.025, 0.0, 0.018])
    neighbor = base + np.array([0.65, 0.0, 0.0])
    duplicate_support = sb._point_cloud_overlap_support({"pts": base}, {"pts": duplicate})
    neighbor_support = sb._point_cloud_overlap_support({"pts": base}, {"pts": neighbor})

    assert duplicate_support[0] > 0.9 and duplicate_support[1] > 0.9
    assert neighbor_support[0] < 0.1 and neighbor_support[1] < 0.1


def test_editor_overlap_cleanup_merges_coincident_disjoint_tracks():
    cols = np.zeros((3000, 3), dtype=np.uint8)
    instances = []
    for frame, label, center, size in [
        (0, "chair", [0.00, 0.42, 0.00], [0.55, 0.86, 0.55]),
        (2, "office seat", [0.03, 0.43, 0.02], [0.52, 0.84, 0.50]),
    ]:
        obs = sb._make_observation(
            label, 0.65, frame, _box_points(center, size, seed=frame), cols,
            box=np.array([10, 50, 90, 200]), det_idx=0,
        )
        instances.append(sb._instance_from_observations([obs]))

    result = sb._consolidate_overlapping_placements(
        instances, floor_y=0.0, room_yaw=0.0, floor_snap_tol=0.1,
    )
    assert len(result) == 1


def test_editor_overlap_cleanup_preserves_adjacent_same_frame_objects():
    cols = np.zeros((3000, 3), dtype=np.uint8)
    observations = [
        sb._make_observation(
            "chair", 0.65, 0,
            _box_points([0.00, 0.42, 0.00], [0.55, 0.86, 0.55], seed=0), cols,
            box=np.array([10, 50, 80, 200]), det_idx=0,
        ),
        # Deliberately noisy/coincident 3-D placement; distinct boxes prove two.
        sb._make_observation(
            "chair", 0.64, 0,
            _box_points([0.02, 0.42, 0.01], [0.55, 0.86, 0.55], seed=1), cols,
            box=np.array([100, 50, 170, 200]), det_idx=1,
        ),
    ]
    result = sb._consolidate_overlapping_placements(
        [sb._instance_from_observations([o]) for o in observations],
        floor_y=0.0, room_yaw=0.0, floor_snap_tol=0.1,
    )
    assert len(result) == 2


def test_editor_overlap_cleanup_reduces_54_split_tracks_to_12_objects():
    """Regression at the scale reported by the Modal reconstruction log."""
    instances = []
    frame = 0
    # Six objects have five disjoint track fragments and six have four:
    # 6*5 + 6*4 = 54 tracks representing 12 physical objects.
    for object_id in range(12):
        fragments = 5 if object_id < 6 else 4
        center_x = 1.5 * (object_id % 4)
        center_z = 1.5 * (object_id // 4)
        for fragment in range(fragments):
            pts = _box_points(
                [center_x + 0.008 * fragment, 0.43,
                 center_z + 0.006 * fragment],
                [0.52 + 0.005 * fragment, 0.86, 0.50],
                n=500, seed=1000 + frame,
            )
            obs = sb._make_observation(
                "chair", 0.62, frame, pts,
                np.zeros((len(pts), 3), dtype=np.uint8),
                box=np.array([10, 50, 90, 200]), det_idx=0,
            )
            instances.append(sb._instance_from_observations([obs]))
            frame += 1

    result = sb._consolidate_overlapping_placements(
        instances, floor_y=0.0, room_yaw=0.0, floor_snap_tol=0.1,
    )
    assert len(instances) == 54
    assert len(result) == 12
    assert all(inst["frames_seen"] in (4, 5) for inst in result)


def test_support_image_selection_prefers_final_placement_over_largest_mask(tmp_path):
    wrong = np.full((120, 160, 3), 90, np.uint8)
    correct = np.indices((120, 160)).sum(axis=0) % 2 * 255
    correct = np.repeat(correct[:, :, None].astype(np.uint8), 3, axis=2)
    paths = [str(tmp_path / "wrong.jpg"), str(tmp_path / "correct.jpg")]
    assert sb.cv2.imwrite(paths[0], wrong)
    assert sb.cv2.imwrite(paths[1], correct)

    wrong_pts = _box_points([3.0, 0.4, 0], [0.7, 0.9, 0.7], n=3000, seed=1)
    correct_pts = _box_points([0.0, 0.4, 0], [0.5, 0.85, 0.5], n=500, seed=2)
    observations = [
        sb._make_observation(
            "chair", 0.8, 0, wrong_pts, np.zeros((len(wrong_pts), 3), np.uint8),
            box=np.array([20, 20, 140, 110]), det_idx=0,
        ),
        sb._make_observation(
            "chair", 0.65, 1, correct_pts, np.zeros((len(correct_pts), 3), np.uint8),
            box=np.array([40, 20, 120, 110]), det_idx=0,
        ),
    ]
    inst = sb._instance_from_observations(observations)
    assert inst["best_frame"] == 0, "old raw-point rule should prefer the wrong view"
    inst["placement"] = {"position": [0.0, 0.0, 0.0], "size": [0.5, 0.85, 0.5], "yaw": 0.0}
    masks = [[np.ones((120, 160), bool)], [np.ones((120, 160), bool)]]

    selected, audit = sb._select_object_support_observation(inst, paths, masks)
    assert selected is observations[1]
    assert audit["frame"] == 1
    assert audit["candidates"] == 2


def test_support_image_selection_is_scoped_to_final_label(tmp_path):
    paths = [str(tmp_path / "chair.png"), str(tmp_path / "cabinet.png")]
    assert sb.cv2.imwrite(paths[0], np.full((100, 120, 3), 80, np.uint8))
    assert sb.cv2.imwrite(paths[1], np.full((100, 120, 3), 180, np.uint8))
    target_pts = _box_points([0, 0.4, 0], [0.5, 0.8, 0.5], n=500)
    observations = [
        sb._make_observation(
            "chair", 0.62, 0, target_pts, np.zeros((500, 3), np.uint8),
            box=np.array([20, 10, 90, 90]), det_idx=0,
        ),
        # This losing cross-label observation would otherwise win on detector
        # confidence and sharpness despite not carrying the final label.
        sb._make_observation(
            "cabinet", 0.99, 1, target_pts, np.zeros((500, 3), np.uint8),
            box=np.array([10, 5, 110, 95]), det_idx=0,
        ),
    ]
    inst = sb._instance_from_observations([
        observations[0], observations[0], observations[1],
    ])
    inst["placement"] = {
        "position": [0.0, 0.0, 0.0], "size": [0.5, 0.8, 0.5], "yaw": 0.0,
    }
    masks = [[np.ones((100, 120), bool)], [np.ones((100, 120), bool)]]

    selected, audit = sb._select_object_support_observation(inst, paths, masks)

    assert inst["label"] == "chair"
    assert selected["label"] == "chair"
    assert audit["frame"] == 0
    assert audit["label_scoped"] is True
    assert audit["candidates"] == 2


def test_support_image_refresh_falls_back_from_invalid_best_frame(tmp_path):
    image_path = str(tmp_path / "valid.jpg")
    assert sb.cv2.imwrite(image_path, np.full((80, 100, 3), 140, np.uint8))
    pts = _box_points([0, 0.4, 0], [0.5, 0.8, 0.5], n=500)
    obs = sb._make_observation(
        "chair", 0.65, 0, pts, np.zeros((len(pts), 3), np.uint8),
        box=np.array([10, 10, 90, 70]), det_idx=0,
    )
    inst = sb._instance_from_observations([obs])
    inst.update(
        placement={"position": [0.0, 0.0, 0.0], "size": [0.5, 0.8, 0.5], "yaw": 0.0},
        best_frame=99,
        best_box=None,
    )
    summary = sb._refresh_object_support_images(
        [inst], [image_path], [[np.ones((80, 100), bool)]],
    )
    assert summary == {"selected": 1, "missing": 0}
    assert inst["best_frame"] == 0
    assert np.array_equal(inst["best_box"], np.array([10, 10, 90, 70]))


def test_cluster_merge_bias_reunites_nearby_disjoint_fragments():
    """Without same-frame evidence, prefer one stable object to duplicate
    fragments. This restores the behavior of the original scene builder."""
    cols = np.zeros((3000, 3), dtype=np.uint8)
    obs = []
    for frame, x, score in [(0, 0.00, 0.80), (1, 0.02, 0.75),
                            (2, 0.45, 0.68), (3, 0.46, 0.66)]:
        obs.append(sb._make_observation(
            "object", score, frame,
            _box_points([x, 0.4, 0], [0.5, 0.85, 0.5], seed=frame), cols,
            box=np.array([20, 40, 100, 190]), det_idx=0,
        ))

    instances = sb.cluster_observations(obs)
    assert len(instances) == 1
    assert instances[0]["frames_seen"] == 4


def test_cluster_broad_radius_suppresses_center_jitter_duplicates():
    cols = np.zeros((3000, 3), dtype=np.uint8)
    obs = []
    # Large centroid movement from partial masks should not create a new asset
    # when no frame contains evidence of two separate objects.
    for frame, x in enumerate([0.00, 0.18, 0.42, 0.43]):
        obs.append(sb._make_observation(
            "item", 0.72 - frame * 0.01, frame,
            _box_points([x, 0.4, 0], [0.5, 0.85, 0.5], seed=20 + frame), cols,
            box=np.array([15, 45, 95, 195]), det_idx=0,
        ))

    instances = sb.cluster_observations(obs)
    assert len(instances) == 1


def test_cluster_preserves_six_repeated_objects_under_partial_visibility():
    """An anchor frame with six separate boxes protects all six identities even
    when later frames only see subsets of the repeated objects."""
    cols = np.zeros((3000, 3), dtype=np.uint8)
    visible = {
        0: [0, 1, 2, 3, 4, 5],
        1: [0, 1, 2, 3, 5],
        2: [2, 3, 4, 5],
    }
    obs = []
    for frame, object_ids in visible.items():
        for slot, object_id in enumerate(object_ids):
            x = 0.45 * object_id + 0.01 * frame
            obs.append(sb._make_observation(
                "chair", 0.66, frame,
                _box_points([x, 0.4, 0], [0.5, 0.85, 0.5],
                            seed=100 + 10 * frame + object_id), cols,
                box=np.array([10 + 90 * slot, 50, 85 + 90 * slot, 200]),
                det_idx=slot,
            ))

    instances = sb.cluster_observations(obs)
    assert len(instances) == 6, f"expected six physical instances, got {len(instances)}"
    assert all(inst["frames_seen"] >= 1 for inst in instances)


def test_frame_nms_keeps_cross_label_candidates_for_3d_voting():
    boxes = np.array([
        [10, 10, 100, 100],
        [11, 10, 101, 100],
        [10, 10, 100, 100],
    ])
    labels = ["table", "table", "cabinet"]
    scores = np.array([0.8, 0.7, 0.75])

    assert suppress_overlapping_boxes(boxes, labels, scores) == [0, 2]


def test_group_duplicates():
    # Two near-identical chairs, one distinctly bigger chair, two matching racks.
    placed = [
        {"label": "chair", "placement": {"size": [0.50, 0.90, 0.50]}},
        {"label": "chair", "placement": {"size": [0.52, 0.88, 0.49]}},   # dup of [0]
        {"label": "chair", "placement": {"size": [0.90, 1.40, 0.90]}},   # different chair
        {"label": "server rack", "placement": {"size": [0.60, 2.00, 1.00]}},
        {"label": "server rack", "placement": {"size": [0.60, 2.00, 1.02]}},  # dup of [3]
    ]
    groups = [sorted(g) for g in sb._group_duplicates(placed, size_rel_tol=0.15)]
    assert sorted(groups) == [[0, 1], [2], [3, 4]], groups


def test_group_duplicates_respects_orientation_swap():
    # Same label + same extents but width/depth swapped → 90° apart. Reusing one
    # asset would render it rotated wrong, so they must stay separate groups.
    placed = [
        {"label": "server rack", "placement": {"size": [0.60, 2.00, 1.00]}},
        {"label": "server rack", "placement": {"size": [1.00, 2.00, 0.60]}},
    ]
    assert len(sb._group_duplicates(placed)) == 2


def test_group_duplicates_singletons_when_disabled_semantics():
    # Distinct labels never merge, even at identical sizes.
    placed = [
        {"label": "desk", "placement": {"size": [1.2, 0.75, 0.6]}},
        {"label": "monitor", "placement": {"size": [1.2, 0.75, 0.6]}},
    ]
    assert len(sb._group_duplicates(placed)) == 2


def test_reusable_asset_groups_require_masked_crop_appearance_match(tmp_path):
    paths, masks, placed = [], [], []
    colors = [
        (20, 40, 220),   # red-ish object A (BGR)
        (20, 40, 220),   # matching object B
        (220, 40, 20),   # blue-ish object C
        (20, 40, 220),   # A-like pixels, but no valid SAM mask
    ]
    for frame, color in enumerate(colors):
        image = np.zeros((100, 100, 3), np.uint8)
        image[20:80, 25:75] = color
        path = str(tmp_path / f"appearance_{frame}.png")
        assert sb.cv2.imwrite(path, image)
        paths.append(path)
        mask = np.zeros((100, 100), bool)
        if frame != 3:
            mask[20:80, 25:75] = True
        masks.append([mask if frame != 3 else None])
        placed.append({
            "label": "chair",
            "best_frame": frame,
            "best_box": np.array([20, 15, 80, 85]),
            "best_det_idx": 0,
            "obs_refs": [(frame, 0)],
            "placement": {"size": [0.5, 0.9, 0.5]},
        })

    groups = [sorted(group) for group in sb._group_reusable_assets(
        placed, paths, masks, min_appearance_similarity=0.90,
    )]

    assert sorted(groups) == [[0, 1], [2], [3]]
    assert sb._appearance_similarity(
        placed[0]["_asset_appearance_descriptor"],
        placed[1]["_asset_appearance_descriptor"],
    ) > 0.99


def _vertical_reflection_points(frame: int, n: int = 1200) -> np.ndarray:
    rng = np.random.default_rng(700 + frame)
    return np.column_stack([
        rng.uniform(-0.7, 0.7, n),
        rng.uniform(0.8, 1.8, n),
        np.full(n, 2.0) + rng.normal(0, 0.003, n),
    ])


def test_geometry_verification_rejects_volumetric_label_on_reflector_plane():
    observations = []
    for frame in range(3):
        pts = _vertical_reflection_points(frame)
        observations.append(sb._make_observation(
            "table", 0.72, frame, pts, np.zeros((len(pts), 3), np.uint8),
            box=np.array([25, 20, 140, 110]), det_idx=0,
            reflector_inside_ratio=0.92, reflector_labels=["television"],
        ))
    inst = sb._instance_from_observations(observations)

    audit = sb._verify_instance_geometry(inst, floor_y=0.0, room_height=3.0)

    assert audit["status"] == "rejected_reflection"
    assert audit["reject"] is True
    assert audit["vertical_planar"] is True
    assert audit["category_conflict"] is True
    assert audit["reflector_frames"] == 3


def test_geometry_verification_keeps_real_table_and_planar_tv():
    table_obs = []
    for frame in range(3):
        pts = _box_points([0, 0.4, 0], [1.4, 0.8, 0.8], n=1200, seed=800 + frame)
        table_obs.append(sb._make_observation(
            "table", 0.70, frame, pts, np.zeros((len(pts), 3), np.uint8),
        ))
    table = sb._instance_from_observations(table_obs)
    table_audit = sb._verify_instance_geometry(table, floor_y=0.0, room_height=3.0)
    assert table_audit["reject"] is False

    tv_obs = []
    for frame in range(3):
        pts = _vertical_reflection_points(frame)
        tv_obs.append(sb._make_observation(
            "television", 0.68, frame, pts, np.zeros((len(pts), 3), np.uint8),
        ))
    tv = sb._instance_from_observations(tv_obs)
    tv_audit = sb._verify_instance_geometry(tv, floor_y=0.0, room_height=3.0)
    assert tv_audit["vertical_planar"] is True
    assert tv_audit["category_conflict"] is False
    assert tv_audit["reject"] is False


def test_geometry_verification_keeps_single_frame_reflection_as_uncertain():
    pts = _vertical_reflection_points(0)
    obs = sb._make_observation(
        "table", 0.75, 0, pts, np.zeros((len(pts), 3), np.uint8),
        reflector_inside_ratio=0.95, reflector_labels=["mirror"],
    )
    inst = sb._instance_from_observations([obs])

    audit = sb._verify_instance_geometry(inst, floor_y=0.0, room_height=3.0)

    assert audit["status"] == "uncertain"
    assert audit["reject"] is False


def test_geometry_verification_rejects_floating_object_with_thin_evidence():
    # Table-sized box centred well above the floor (~1.05 gap in a 3.0-tall
    # room => floor_gap_ratio ~0.35), tracked in only 2 frames. Mirrors the
    # real obj_014 case: 2 frames, floor_gap_ratio 0.48, no detected reflector.
    observations = []
    for frame in range(2):
        pts = _box_points([0, 1.2, 0], [0.6, 0.3, 0.6], n=1200, seed=900 + frame)
        observations.append(sb._make_observation(
            "table", 0.42, frame, pts, np.zeros((len(pts), 3), np.uint8),
        ))
    inst = sb._instance_from_observations(observations)

    audit = sb._verify_instance_geometry(inst, floor_y=0.0, room_height=3.0)

    assert audit["floor_gap_ratio"] >= 0.30
    assert audit["unsupported"] is True
    assert audit["status"] == "rejected_unsupported"
    assert audit["reject"] is True


def test_geometry_verification_keeps_floating_object_with_strong_evidence():
    # Same elevated placement as above, but well-observed (>3 frames) — e.g. a
    # wall-mounted shelf. Thin evidence is required to reject on floor gap
    # alone, so this must stay accepted despite the identical gap.
    observations = []
    for frame in range(5):
        pts = _box_points([0, 1.2, 0], [0.6, 0.3, 0.6], n=1200, seed=910 + frame)
        observations.append(sb._make_observation(
            "shelf", 0.80, frame, pts, np.zeros((len(pts), 3), np.uint8),
        ))
    inst = sb._instance_from_observations(observations)

    audit = sb._verify_instance_geometry(inst, floor_y=0.0, room_height=3.0)

    assert audit["floor_gap_ratio"] >= 0.30
    assert audit["unsupported"] is False
    assert audit["reject"] is False


def test_reflector_and_nested_reflection_are_never_cross_label_merged():
    plane = _vertical_reflection_points(0)
    tv = sb._make_observation(
        "television", 0.68, 0, plane, np.zeros((len(plane), 3), np.uint8),
        box=np.array([10, 10, 190, 150]), det_idx=0,
    )
    reflected_table = sb._make_observation(
        "table", 0.74, 0, plane, np.zeros((len(plane), 3), np.uint8),
        box=np.array([35, 30, 165, 135]), det_idx=1,
        reflector_inside_ratio=0.95, reflector_labels=["television"],
    )

    assert sb._same_frame_distinct(tv, reflected_table)
    instances = sb.cluster_observations([tv, reflected_table])
    assert sorted(inst["label"] for inst in instances) == ["table", "television"]


def test_instance_reprojection_agreement_uses_masks_and_depth():
    rng = np.random.default_rng(901)
    pts = np.column_stack([
        rng.uniform(-0.2, 0.2, 800),
        rng.uniform(-0.2, 0.2, 800),
        np.ones(800),
    ])
    observations = [
        sb._make_observation(
            "chair", 0.7, frame, pts.copy(), np.zeros((len(pts), 3), np.uint8),
            det_idx=0,
        )
        for frame in range(2)
    ]
    inst = sb._instance_from_observations(observations)
    f = 50.0
    intrinsic = np.repeat(
        np.array([[[f, 0, 32], [0, f, 32], [0, 0, 1.0]]]), 2, axis=0,
    )
    extrinsic = np.repeat(
        np.hstack([np.eye(3), np.zeros((3, 1))])[None], 2, axis=0,
    )
    depth = np.ones((2, 64, 64, 1), np.float32)
    mask = np.zeros((64, 64), bool)
    mask[20:45, 20:45] = True

    agreement = sb._instance_reprojection_agreement(
        inst, np.eye(4), extrinsic, intrinsic, depth, [[mask], [mask]],
    )

    assert agreement is not None
    assert agreement > 0.95


def test_scan_mesh_quality_uses_original_point_support():
    good = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    good.apply_translation([0, 0.5, 0])
    observed, _ = trimesh.sample.sample_surface(good, 3000, seed=2)

    quality = sb._scan_mesh_quality(good, observed, [1, 1, 1], frames_seen=3)
    assert quality["good"]
    assert quality["support"] > 0.95

    fragment = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
    fragment.apply_translation([0, 0.05, 0])
    quality = sb._scan_mesh_quality(fragment, observed, [1, 1, 1], frames_seen=3)
    assert not quality["good"]
    assert quality["bbox_agreement"] < 0.2


def test_raw_scan_glb_round_trip():
    import tempfile
    from pathlib import Path

    pts = np.random.default_rng(0).normal(size=(500, 3)).astype(np.float32)
    cols = np.tile(np.array([30, 120, 240], np.uint8), (len(pts), 1))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "raw_scan.glb"
        assert sb._export_raw_scan(pts, cols, str(path)) == len(pts)
        loaded = trimesh.load(path, process=False)
        geometries = list(loaded.geometry.values())
        assert len(geometries) == 1
        assert isinstance(geometries[0], trimesh.PointCloud)
        assert len(geometries[0].vertices) == len(pts)


def test_object_fit_points_are_compact_and_object_local():
    rng = np.random.default_rng(4)
    local = np.column_stack([
        rng.uniform(-0.5, 0.5, 900),
        rng.uniform(0.0, 2.0, 900),
        rng.uniform(-0.25, 0.25, 900),
    ])
    yaw = np.radians(30)
    c, s = np.cos(yaw), np.sin(yaw)
    world = local @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]]).T + [3, 0, -2]
    inst = {
        "pts": world,
        "placement": {
            "position": [3, 0, -2],
            "yaw": yaw,
            "size": [1, 2, 0.5],
        },
    }

    sampled = np.asarray(sb._object_fit_points(inst, 1.0, np.zeros(3)))

    assert 24 <= len(sampled) <= 384
    assert np.max(np.abs(sampled[:, 0])) <= 0.55
    assert sampled[:, 1].min() >= -0.01
    assert sampled[:, 1].max() <= 2.01
    assert np.max(np.abs(sampled[:, 2])) <= 0.30


def test_room_surface_coverage_distinguishes_observed_and_missing_planes():
    rng = np.random.default_rng(4)
    floor = np.column_stack([
        rng.uniform(-2, 2, 4000), np.zeros(4000), rng.uniform(-3, 3, 4000),
    ])
    wall_xmin = np.column_stack([
        np.full(2500, -2.0), rng.uniform(0, 3, 2500), rng.uniform(-3, 3, 2500),
    ])
    coverage = sb._room_surface_coverage(
        np.vstack([floor, wall_xmin]), np.eye(3), (-2, 2, -3, 3), 3.0,
    )

    assert coverage["floor"] > 0.9
    assert coverage["wall_0"] > 0.7
    assert coverage["wall_1"] == 0.0


def test_wall_cell_classifier_fills_occlusion_but_preserves_opening():
    observed = np.zeros((8, 10), bool)
    occluded = np.zeros_like(observed, dtype=np.uint16)
    semantic = np.zeros_like(observed)
    through = np.zeros_like(observed, dtype=np.uint16)
    occluded[4, 5] = 1

    fill = sb._classify_wall_cells(observed, occluded, semantic, through)
    assert fill[4, 5]

    semantic[4, 5] = True
    fill = sb._classify_wall_cells(observed, occluded, semantic, through)
    assert not fill[4, 5], "door/window evidence must override occlusion"

    semantic[:] = False
    through[4, 5] = 2
    fill = sb._classify_wall_cells(observed, occluded, semantic, through)
    assert not fill[4, 5], "multi-view depth beyond the wall indicates an opening"


def test_wall_cell_classifier_completes_unknown_supported_wall():
    observed = np.zeros((8, 10), bool)
    observed[2:4, 2:5] = True
    occluded = np.zeros_like(observed, dtype=np.uint16)
    semantic = np.zeros_like(observed)
    through = np.zeros_like(observed, dtype=np.uint16)

    assert sb._wall_plane_supported(observed)
    conservative = sb._classify_wall_cells(
        observed, occluded, semantic, through, complete_unknown=False,
    )
    complete = sb._classify_wall_cells(
        observed, occluded, semantic, through, complete_unknown=True,
    )

    assert not conservative.any()
    assert complete[6, 7], "low-light/unsupported wall region should be filled"
    assert not complete[2, 3], "observed scan geometry must remain untouched"


def test_complete_wall_still_preserves_opening_and_rejects_edge_only_support():
    observed = np.zeros((8, 10), bool)
    observed[2:4, 1:4] = True
    semantic = np.zeros_like(observed)
    semantic[:, 5] = True
    zeros = np.zeros_like(observed, dtype=np.uint16)

    fill = sb._classify_wall_cells(
        observed, zeros, semantic, zeros, complete_unknown=True,
    )
    assert not fill[:, 5].any(), "door/window cells must remain empty"
    assert fill[6, 2]

    edge_only = np.zeros_like(observed)
    edge_only[0, :8] = True
    assert not sb._wall_plane_supported(edge_only)


def test_wall_ray_classification_distinguishes_object_and_opening():
    rect = (-2.0, 2.0, -3.0, 3.0)
    spec = sb._wall_spec(3, rect)  # z=+3 wall
    shape = sb._wall_grid_shape(spec, ceil_h=3.0)
    camera = np.array([0.0, 1.5, 0.0])

    # Object is in front of the wall, so its ray reaches the wall after t=1.
    iy, iu = sb._ray_wall_cells(
        camera, np.array([[0.0, 1.5, 1.0]]), spec, 3.0, shape, "occluded"
    )
    assert len(iy) == 1

    # A reconstructed point beyond the wall crosses it before reaching depth.
    ty, tu = sb._ray_wall_cells(
        camera, np.array([[0.0, 1.5, 4.0]]), spec, 3.0, shape, "through"
    )
    assert len(ty) == 1
    assert (iy[0], iu[0]) == (ty[0], tu[0])


def test_wall_fill_mesh_patches_only_selected_cells():
    fill = np.zeros((4, 6), bool)
    fill[1:3, 2:4] = True
    mesh = sb._wall_fill_mesh(
        0, fill, (-2.0, 2.0, -3.0, 3.0), 3.0, np.eye(3), (128, 128, 128)
    )
    assert mesh is not None
    assert len(mesh.faces) == 12, "contiguous cells should merge into one box"
    assert mesh.bounds[1, 0] < -2.0, "patch sits behind the fitted wall plane"


def test_wall_color_field_interpolates_neighbouring_surface_appearance():
    # Mirrors test_horizontal_color_field_interpolates_neighbouring_surface_
    # appearance, with the wall's plane axis (x, fixed at 0) and u-axis (z)
    # standing in for the floor's y (fixed) and x.
    images = np.zeros((1, 3, 2, 2), np.float32)
    images[0, 0, :, 0] = 1.0       # two red samples in the left cell
    images[0, 2, :, 1] = 1.0       # two blue samples in the right cell
    world_pts = np.array([[[[0.0, 0.3, 0.2], [0.0, 0.3, 2.8]],
                            [[0.0, 0.7, 0.2], [0.0, 0.7, 2.8]]]])
    spec = sb._wall_spec(0, (0.0, 2.0, 0.0, 3.0))
    colors, observed = sb._wall_color_field(
        spec, images, world_pts, np.ones((1, 2, 2), bool),
        np.eye(4), 1.0, np.zeros(3), np.eye(3),
        3.0, (1, 3), (128, 128, 128), wall_tol=0.1,
    )

    assert observed.tolist() == [[True, False, True]]
    assert colors[0, 0, 0] > 240 and colors[0, 0, 2] < 10
    assert colors[0, 2, 2] > 240 and colors[0, 2, 0] < 10
    assert colors[0, 1, 0] > 80 and colors[0, 1, 2] > 80


def test_wall_fill_mesh_uses_spatially_varying_cell_colors():
    fill = np.array([[True, True]])
    colors = np.array([[[240, 20, 10], [10, 20, 240]]], np.uint8)
    mesh = sb._wall_fill_mesh(
        0, fill, (0.0, 2.0, 0.0, 1.0), 3.0,
        np.eye(3), (128, 128, 128), color_field=colors,
    )

    assert mesh is not None
    assert len(mesh.faces) == 24, "two per-cell boxes, unmerged, 12 faces each"
    assert mesh.bounds[1, 0] < 0.0, "patch sits behind the fitted wall plane"
    unique = np.unique(np.asarray(mesh.visual.vertex_colors)[:, :3], axis=0)
    assert {tuple(c) for c in unique} == {(240, 20, 10), (10, 20, 240)}


def test_opening_label_vocabulary():
    assert sb._is_opening_label("glass door")
    assert sb._is_opening_label("office window")
    assert not sb._is_opening_label("cabinet")


def test_instance_exclusion_mask_omits_rejected_and_opening_objects():
    masks = [[np.ones((8, 8), bool), np.ones((8, 8), bool)]]
    furniture = {"label": "chair", "obs_refs": [(0, 0)]}
    opening = {"label": "door", "obs_refs": [(0, 1)]}

    kept = sb._exclude_mask_for_instances([furniture, opening], masks, (1, 4, 4))
    opening_only = sb._exclude_mask_for_instances([opening], masks, (1, 4, 4))
    rejected = sb._exclude_mask_for_instances([], masks, (1, 4, 4))

    assert kept.all(), "movable object pixels should remain excluded from background TSDF"
    assert not opening_only.any(), "doors/windows should remain in the architectural background"
    assert not rejected.any(), "rejected detections must not leave background holes"


def test_horizontal_infill_fills_only_missing_floor_and_ceiling_cells():
    # Multiple samples support the centre of each plane. Distant unseen cells
    # remain eligible for infill.
    points = np.array([
        [0.0, 0.0, 0.0], [0.01, 0.0, 0.01], [0.02, 0.0, 0.02],
        [0.0, 3.0, 0.0], [0.01, 3.0, 0.01], [0.02, 3.0, 0.02],
    ])
    evidence = sb._horizontal_infill_evidence(
        points, np.eye(3), (-1.0, 1.0, -1.0, 1.0), 3.0,
    )

    for surface in ("floor", "ceiling"):
        ev = evidence[surface]
        assert ev["observed"].sum() == 1
        assert not ev["fill"][5, 5]
        assert ev["fill"][0, 0]


def test_horizontal_infill_ignores_isolated_depth_noise():
    evidence = sb._horizontal_infill_evidence(
        np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]),
        np.eye(3), (-1.0, 1.0, -1.0, 1.0), 3.0,
    )

    assert evidence["floor"]["support_count"][5, 5] == 1
    assert evidence["floor"]["fill"][5, 5]
    assert evidence["ceiling"]["fill"][5, 5]


def test_horizontal_fill_mesh_sits_behind_the_surface_plane():
    fill = np.zeros((4, 6), bool)
    fill[1:3, 2:4] = True
    floor = sb._horizontal_fill_mesh(
        "floor", fill, (-2.0, 2.0, -3.0, 3.0), 3.0,
        np.eye(3), (128, 128, 128),
    )
    ceiling = sb._horizontal_fill_mesh(
        "ceiling", fill, (-2.0, 2.0, -3.0, 3.0), 3.0,
        np.eye(3), (128, 128, 128),
    )

    assert floor is not None and ceiling is not None
    assert len(floor.faces) == 12 and len(ceiling.faces) == 12
    assert floor.bounds[1, 1] < 0.0
    assert ceiling.bounds[0, 1] > 3.0


def test_horizontal_color_field_interpolates_neighbouring_surface_appearance():
    images = np.zeros((1, 3, 2, 2), np.float32)
    images[0, 0, :, 0] = 1.0       # two red samples in the left cell
    images[0, 2, :, 1] = 1.0       # two blue samples in the right cell
    world_pts = np.array([[[[0.2, 0.0, 0.3], [2.8, 0.0, 0.3]],
                            [[0.2, 0.0, 0.7], [2.8, 0.0, 0.7]]]])
    colors, observed = sb._horizontal_color_field(
        "floor", images, world_pts, np.ones((1, 2, 2), bool),
        np.eye(4), 1.0, np.zeros(3), np.eye(3),
        (0.0, 3.0, 0.0, 1.0), 3.0, (1, 3), (128, 128, 128),
    )

    assert observed.tolist() == [[True, False, True]]
    assert colors[0, 0, 0] > 240 and colors[0, 0, 2] < 10
    assert colors[0, 2, 2] > 240 and colors[0, 2, 0] < 10
    assert colors[0, 1, 0] > 80 and colors[0, 1, 2] > 80


def test_horizontal_fill_mesh_uses_spatially_varying_cell_colors():
    fill = np.array([[True, True]])
    colors = np.array([[[240, 20, 10], [10, 20, 240]]], np.uint8)
    mesh = sb._horizontal_fill_mesh(
        "floor", fill, (0.0, 2.0, 0.0, 1.0), 3.0,
        np.eye(3), (128, 128, 128), color_field=colors,
    )

    assert mesh is not None and len(mesh.faces) == 4
    assert mesh.bounds[1, 1] < 0.0
    unique = np.unique(np.asarray(mesh.visual.vertex_colors)[:, :3], axis=0)
    assert {tuple(c) for c in unique} == {(240, 20, 10), (10, 20, 240)}


def test_glb_bytes_to_local_mesh_fits_and_grounds():
    # A generated GLB comes in at an arbitrary size/offset; the normalizer must
    # scale its bbox to the detected size and bottom-centre it at the origin.
    import trimesh
    box = trimesh.creation.box(extents=[3.0, 4.0, 5.0])
    box.apply_translation([10.0, -7.0, 2.0])          # arbitrary origin
    glb_bytes = trimesh.Scene([box]).export(file_type="glb")

    size_m = [1.0, 2.0, 0.5]
    mesh = sb._glb_bytes_to_local_mesh(glb_bytes, size_m)
    assert mesh is not None
    lo, hi = mesh.bounds
    assert np.allclose(hi - lo, size_m, atol=1e-4), f"extents {hi - lo} != {size_m}"
    assert abs(lo[1]) < 1e-4, "bottom should sit on y=0"
    assert abs((lo[0] + hi[0]) / 2) < 1e-4 and abs((lo[2] + hi[2]) / 2) < 1e-4, "not XZ-centred"


def test_glb_bytes_to_local_mesh_rejects_garbage():
    assert sb._glb_bytes_to_local_mesh(b"not a glb", [1, 1, 1]) is None


def test_object_crop_rgba_from_mask():
    import tempfile
    import cv2

    img = np.zeros((100, 100, 3), np.uint8)
    img[:] = (30, 60, 90)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        path = tf.name
    cv2.imwrite(path, img)

    mask = np.zeros((100, 100), bool)
    mask[30:70, 30:70] = True                       # object occupies the centre
    inst = {"best_frame": 0, "best_box": [25, 25, 75, 75], "obs_refs": [(0, 0)]}

    crop = sb._object_crop_rgb([path], [[mask]], inst)
    assert crop is not None and crop.shape[2] == 4, "expected an RGBA cutout"
    assert crop[..., 3].max() == 255 and crop[..., 3].min() == 0, "alpha should carry fg+bg"

    # No usable mask → plain RGB box crop (3 channels).
    rgb = sb._object_crop_rgb([path], [[None]], inst)
    assert rgb is not None and rgb.shape[2] == 3


def test_min_area_rect_yaw_aligns_room_not_furniture():
    rng = np.random.default_rng(0)
    # A 4×2 m room outline (points along the 4 walls) plus a dense row of
    # "furniture" running diagonally — a PCA axis would chase the furniture; the
    # min-area rectangle must lock onto the walls.
    xs = np.concatenate([rng.uniform(-2, 2, 2000), rng.uniform(-2, 2, 2000),
                         np.full(2000, -2.0), np.full(2000, 2.0)])
    zs = np.concatenate([np.full(2000, -1.0), np.full(2000, 1.0),
                         rng.uniform(-1, 1, 2000), rng.uniform(-1, 1, 2000)])
    room = np.stack([xs, zs], axis=1)
    t = np.linspace(-1.5, 1.5, 3000)
    furniture = np.stack([t, 0.5 * t], axis=1)       # diagonal clutter inside
    pts = np.vstack([room, furniture])

    th = np.radians(25)                              # rotate the whole thing 25°
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    rot = pts @ Rm.T

    ang = sb._min_area_rect_yaw(rot)
    ca, sa = np.cos(ang), np.sin(ang)
    Txz = np.array([[ca, sa], [-sa, ca]])            # the builder's T_yaw XZ block
    aligned = rot @ Txz.T
    assert abs(sb._min_area_rect_yaw(aligned)) < np.radians(1.5), "room not grid-aligned"
    ext = np.sort([np.ptp(aligned[:, 0]), np.ptp(aligned[:, 1])])
    assert abs(ext[0] - 2) < 0.1 and abs(ext[1] - 4) < 0.1, f"wrong room extents {ext}"


def test_yaw_basis_is_orthonormal_roundtrip():
    R = sb._yaw_basis(np.radians(37))
    v = np.random.default_rng(0).random((50, 3))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.allclose(v @ R.T @ R, v, atol=1e-9)      # world→local→world identity


def test_fit_room_rect_finds_wall_planes_not_clutter():
    rng = np.random.default_rng(0)
    ceil_h = 3.0

    def wall_x(val, n=3000):                            # plane at x=val
        p = np.empty((n, 3))
        p[:, 0] = val + rng.normal(0, 0.01, n)
        p[:, 1] = rng.uniform(0.2, 2.8, n)
        p[:, 2] = rng.uniform(-3, 3, n)
        return p

    def wall_z(val, n=3000):                            # plane at z=val
        p = np.empty((n, 3))
        p[:, 0] = rng.uniform(-2, 2, n)
        p[:, 1] = rng.uniform(0.2, 2.8, n)
        p[:, 2] = val + rng.normal(0, 0.01, n)
        return p

    rack = rng.uniform([-0.3, 0, -0.3], [0.3, 2.0, 0.3], (2000, 3))   # interior clutter
    pts = np.vstack([wall_x(-2), wall_x(2), wall_z(-3), wall_z(3), rack])
    xmin, xmax, zmin, zmax = sb._fit_room_rect(pts, ceil_h)
    assert abs(xmin + 2) < 0.05 and abs(xmax - 2) < 0.05, (xmin, xmax)
    assert abs(zmin + 3) < 0.05 and abs(zmax - 3) < 0.05, (zmin, zmax)


def test_carve_room_shell_drops_boundary_keeps_interior():
    import trimesh
    R = sb._yaw_basis(0.0)                              # yaw 0 → identity
    rect = (-2.0, 2.0, -3.0, 3.0)
    wall = sb._box([0.02, 2.0, 4.0], [2.0, 1.0, 0.0], (128, 128, 128))   # boundary wall at x=2
    interior = sb._box([0.4, 0.4, 0.4], [0.0, 1.5, 0.0], (128, 128, 128))  # mid-room fixture
    combined = trimesh.util.concatenate([wall, interior])

    carved = sb._carve_room_shell(combined, R, rect, ceil_h=3.0)
    assert len(carved.faces) < len(combined.faces), "boundary faces should be dropped"
    cen = carved.vertices[carved.faces].mean(axis=1)
    assert cen[:, 0].max() < 1.0, "the x=2 wall should be gone, interior kept"


def test_placement_accuracy():
    # Rack: bottom at y=0.05 (near floor at 0), centred at (1.5, ·, -2)
    pts = _box_points([1.5, 1.05, -2.0], [0.6, 2.0, 1.0], yaw=0.0, n=6000)
    inst = {"pts": pts}
    p = sb.place_instance(inst, floor_y=0.0, room_yaw=0.0, floor_snap_tol=0.3)
    assert abs(p["position"][0] - 1.5) < 0.05
    assert abs(p["position"][2] + 2.0) < 0.05
    assert p["position"][1] == 0.0, "bottom should snap to floor"
    # major axis (depth=1.0) along Z → local X extent should be ~1.0 after yaw fold
    dims = sorted(p["size"])
    assert abs(dims[-1] - 2.0) < 0.15         # height
    assert abs(p["size"][1] - 2.0) < 0.15     # size[1] is height


def test_rack_scale():
    placed = [
        {"label": "server rack", "placement": {"size": [0.3, 1.0, 0.5]}},
        {"label": "server rack", "placement": {"size": [0.3, 1.1, 0.5]}},
        {"label": "desk", "placement": {"size": [1.0, 0.4, 0.6]}},
    ]
    scale, calibrated = sb.rack_scale(placed)
    assert calibrated
    # median rack height 1.05 → scale = 2.0 / 1.05
    assert abs(scale - 2.0 / 1.05) < 1e-6

    scale, calibrated = sb.rack_scale([placed[2]])
    assert not calibrated and scale == 1.0


def test_level_transform():
    # Floor tilted 10° about X, plus wall points so the fit has to pick the band
    rng = np.random.default_rng(1)
    floor = np.stack([rng.uniform(-5, 5, 4000), np.zeros(4000), rng.uniform(-5, 5, 4000)], axis=1)
    wall = np.stack([rng.uniform(-5, 5, 2000), rng.uniform(0, 3, 2000), np.full(2000, 5.0)], axis=1)
    tilt = np.radians(10)
    R = np.array([[1, 0, 0],
                  [0, np.cos(tilt), -np.sin(tilt)],
                  [0, np.sin(tilt), np.cos(tilt)]])
    pts = np.vstack([floor, wall]) @ R.T
    T = sb._level_transform(pts)
    levelled = pts @ T[:3, :3].T
    floor_levelled = levelled[:4000]
    spread = np.percentile(floor_levelled[:, 1], 98) - np.percentile(floor_levelled[:, 1], 2)
    assert spread < 0.1, f"floor not level after correction, Y spread {spread:.3f}"


def test_level_transform_steep_tilt_with_prior():
    # A capture tilted 50° in the aligned frame (e.g. shot down an aisle):
    # the real floor is no longer the lowest-Y band and a wall dominates it,
    # so the legacy +Y assumption rotates the WALL flat and leaves a wall as
    # the ground. With a gravity prior the true floor levels regardless of tilt.
    rng = np.random.default_rng(2)
    floor = np.stack([rng.uniform(-5, 5, 4000), np.zeros(4000), rng.uniform(-5, 5, 4000)], axis=1)
    wall = np.stack([rng.uniform(-5, 5, 3000), rng.uniform(0, 3, 3000), np.full(3000, 5.0)], axis=1)
    tilt = np.radians(50)
    R = np.array([[1, 0, 0],
                  [0, np.cos(tilt), -np.sin(tilt)],
                  [0, np.sin(tilt), np.cos(tilt)]])
    pts = np.vstack([floor, wall]) @ R.T
    up_prior = R @ np.array([0.0, 1.0, 0.0])          # gravity in the aligned frame

    # Legacy assumption (no prior) fails: floor stays far from level.
    legacy = pts[:4000] @ sb._level_transform(pts)[:3, :3].T
    legacy_spread = np.percentile(legacy[:, 1], 98) - np.percentile(legacy[:, 1], 2)
    assert legacy_spread > 0.5, "expected the no-prior path to mishandle a 50° tilt"

    # With a trusted gravity prior the true floor levels (the 50° correction
    # exceeds the untrusted 30° cap, so trusted=True is required here).
    T = sb._level_transform(pts, up_prior=up_prior, trusted=True)
    floor_levelled = pts[:4000] @ T[:3, :3].T
    spread = np.percentile(floor_levelled[:, 1], 98) - np.percentile(floor_levelled[:, 1], 2)
    assert spread < 0.15, f"floor not level after prior-guided correction, Y spread {spread:.3f}"


def _orbit_extrinsics(n=8, radius=3.0, height=1.6, target=np.array([0, 0, 1.2])):
    """Cameras orbiting the room at constant height (real up = world +Z), each
    pitched slightly down toward a target — OpenCV convention (x right, y down,
    z forward)."""
    def look_at(pos, tgt, world_up=np.array([0, 0, 1.0])):
        f = tgt - pos; f /= np.linalg.norm(f)          # +z_cam (forward)
        r = np.cross(f, world_up); r /= np.linalg.norm(r)
        d = np.cross(f, r)                              # +y_cam (down)
        Rwc = np.stack([r, d, f], axis=0)              # world→cam
        return np.hstack([Rwc, (-Rwc @ pos)[:, None]])
    return np.array([
        look_at(np.array([radius * np.cos(a), radius * np.sin(a), height]), tgt=target)
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False)
    ])


def test_estimate_up_recovers_gravity():
    # A constant-height orbit gives the camera-path estimator a clean 2-D
    # horizontal patch, so it recovers the same up T_align maps world-up onto —
    # regardless of frame 0 — and reports it as trusted.
    exts = _orbit_extrinsics()
    T_align = sb._scene_transform(exts)
    up, trusted = sb._estimate_up(exts, T_align)
    gt = T_align[:3, :3] @ np.array([0, 0, 1.0]); gt /= np.linalg.norm(gt)
    assert float(up @ gt) > 0.98, f"estimated up off from gravity: dot={up @ gt:.3f}"
    assert trusted, "constant-height orbit should yield a trusted camera-path estimate"


def test_estimate_up_falls_back_untrusted_on_degenerate_path():
    # A straight-line walk: camera centres lie on a 1-D line, so height is not a
    # clear least-variance minimum (a horizontal axis is just as flat). The path
    # estimator must bail and _estimate_up fall back to an untrusted median.
    def look_at(pos, tgt, world_up=np.array([0, 0, 1.0])):
        f = tgt - pos; f /= np.linalg.norm(f)
        r = np.cross(f, world_up); r /= np.linalg.norm(r)
        d = np.cross(f, r)
        Rwc = np.stack([r, d, f], axis=0)
        return np.hstack([Rwc, (-Rwc @ pos)[:, None]])

    exts = np.array([
        look_at(np.array([x, 0.0, 1.6]), np.array([x, 5.0, 1.2]))   # walk along +X
        for x in np.linspace(-3, 3, 8)
    ])
    T_align = sb._scene_transform(exts)
    _, trusted = sb._estimate_up(exts, T_align)
    assert not trusted, "straight-line walk should not yield a trusted path estimate"


def test_estimate_up_rejects_vertical_camera_path_plane():
    """A compact walkthrough plus height drift can form a vertical trajectory
    plane. Its normal is horizontal, so it must not be accepted as gravity even
    when the trajectory SVD itself is well-conditioned."""
    def look_at(pos, tgt, world_up=np.array([0, 0, 1.0])):
        f = tgt - pos; f /= np.linalg.norm(f)
        r = np.cross(f, world_up); r /= np.linalg.norm(r)
        d = np.cross(f, r)
        Rwc = np.stack([r, d, f], axis=0)
        return np.hstack([Rwc, (-Rwc @ pos)[:, None]])

    # Positions cover X/Z but have no Y spread: trajectory PCA incorrectly
    # proposes ±Y as "up", while the upright cameras consistently report +Z.
    positions = [
        np.array([x, 0.0, z])
        for x in np.linspace(-2, 2, 4)
        for z in np.linspace(0.8, 2.4, 3)
    ]
    exts = np.array([look_at(p, p + np.array([0, 5, 0])) for p in positions])
    T_align = sb._scene_transform(exts)

    path_up = sb._up_from_camera_path(exts, T_align)
    assert path_up is not None, "fixture must exercise the path sanity check"
    up, trusted = sb._estimate_up(exts, T_align)
    expected = T_align[:3, :3] @ np.array([0, 0, 1.0])
    expected /= np.linalg.norm(expected)

    assert not trusted
    assert float(up @ expected) > 0.98


def test_floor_snap_is_label_aware():
    # A tabletop slab floating at 0.62 m — legs too thin to survive depth
    # filtering, the common real-scan case. Tables must still ground.
    rng = np.random.default_rng(0)
    pts = rng.random((3000, 3)) * [0.8, 0.15, 0.8] + [0, 0.62, 0]
    tol = 0.3

    table = sb.place_instance({"pts": pts, "label": "table"},
                              floor_y=0.0, room_yaw=0.0, floor_snap_tol=tol)
    assert table["position"][1] == 0.0, "floor-standing kind should snap"
    assert abs(table["size"][1] - 0.77) < 0.05, "snapped height spans floor→top"

    monitor = sb.place_instance({"pts": pts, "label": "monitor"},
                                floor_y=0.0, room_yaw=0.0, floor_snap_tol=tol)
    assert monitor["position"][1] > 0.5, "wall/stand-mounted kinds must not snap"

    unlabeled = sb.place_instance({"pts": pts},
                                  floor_y=0.0, room_yaw=0.0, floor_snap_tol=tol)
    assert unlabeled["position"][1] > 0.5, "unknown labels keep strict tolerance"


def test_region_restricted_tsdf_fusion():
    """include_mask + bounds fuse only the masked pixels into a small
    high-res grid — the basis of per-object re-fusion."""
    from src.reconstruction.tsdf_fusion import TSDFConfig, fuse_tsdf_raw_mesh

    H = W = 64
    f = 50.0
    K = np.array([[f, 0, 32], [0, f, 32], [0, 0, 1]])
    E = np.hstack([np.eye(3), np.zeros((3, 1))])
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    preds = {
        "depth": np.full((1, H, W, 1), 1.0, np.float32),
        "depth_conf": np.full((1, H, W), 10.0, np.float32),
        "images": np.full((1, 3, H, W), 0.5, np.float32),
        "extrinsic": E[None], "intrinsic": K[None],
        "world_points_from_depth":
            np.stack([(u - 32) / f, (v - 32) / f, np.ones_like(u, float)], -1)[None],
    }
    inc = np.zeros((1, H, W), bool)
    inc[0, 20:44, 20:44] = True
    bounds = (np.array([-0.4, -0.4, 0.7]), np.array([0.4, 0.4, 1.3]))
    cfg = TSDFConfig(conf_percentile=0, auto_resolution=96, max_dim=112,
                     filter_depth_edges=False)

    mesh = fuse_tsdf_raw_mesh(preds, cfg, include_mask=inc, bounds=bounds)
    vs = np.asarray(mesh.vertices)
    assert len(vs) > 50
    assert abs(np.median(vs[:, 2]) - 1.0) < 0.05, "plane should sit at z=1"
    # mask spans pixels 20..43 → xy within ±0.24 at z=1, plus truncation slack
    assert np.abs(vs[:, :2]).max() < 0.32, "geometry leaked outside the mask"

    # same bounds without the mask fill wider — the mask is what restricts
    full = fuse_tsdf_raw_mesh(preds, cfg, bounds=bounds)
    assert np.abs(np.asarray(full.vertices)[:, :2]).max() > 0.35


def test_prefabs_have_expected_bounds():
    for kind, dims in [("rack", (0.6, 2.0, 1.0)), ("desk", (1.4, 0.75, 0.7)),
                       ("monitor", (0.6, 0.4, 0.2)), ("chair", (0.5, 0.9, 0.5)),
                       ("cabinet", (0.8, 1.8, 0.5))]:
        mesh = sb._PREFAB_BUILDERS[kind](*dims, (120, 120, 120))
        lo, hi = mesh.bounds
        assert lo[1] > -1e-6, f"{kind} dips below floor: {lo[1]}"
        assert abs(hi[1] - dims[1]) < dims[1] * 0.1, f"{kind} height {hi[1]} vs {dims[1]}"
        for ax in (0, 2):
            assert hi[ax] - lo[ax] <= max(dims[0], dims[2]) * 1.15, f"{kind} axis {ax} too big"

    # rack with major axis on X gets rotated: extents must still match input
    mesh = sb._PREFAB_BUILDERS["rack"](1.0, 2.0, 0.6, (120, 120, 120))
    lo, hi = mesh.bounds
    assert abs((hi[0] - lo[0]) - 1.0) < 0.12
    assert abs((hi[2] - lo[2]) - 0.6) < 0.12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {fn.__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)

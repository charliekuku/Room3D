import pytest
import json
import shutil
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
import sys
import os
import cv2
import numpy as np
import trimesh

# Add src to python path to import app
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.app import build_server, _enhance_dark_planar_image
from PIL import Image as PILImage


def test_dark_planar_detection_enhancement_lifts_shadows():
    gradient = np.tile(np.arange(10, 70, dtype=np.uint8), (80, 2))
    rgb = np.repeat(gradient[:, :, None], 3, axis=2)
    enhanced = np.asarray(_enhance_dark_planar_image(PILImage.fromarray(rgb)))

    assert enhanced.shape == rgb.shape
    assert enhanced.mean() > rgb.mean() + 10

@pytest.fixture
def mock_scene_dir():
    """Create a temporary scene directory with a dummy scene.json."""
    temp_dir = tempfile.mkdtemp()
    scene_name = "test_viewer_scene"
    scene_path = Path(temp_dir) / scene_name
    scene_path.mkdir(parents=True)
    
    scene_data = {
        "name": scene_name,
        "metric": True,
        "scale": 1.0,
        "room": {"width": 10.0, "depth": 10.0, "height": 3.0},
        "room_yaw": 0.0,
        "objects": [
            {
                "id": "obj_000",
                "label": "server rack",
                "position": [0.0, 0.0, 0.0],
                "yaw": 0.0,
                "size": [0.6, 2.0, 1.0]
            }
        ]
    }
    
    with open(scene_path / "scene.json", "w") as f:
        json.dump(scene_data, f)
        
    # Monkey-patch SCENES_DIR in app module to point to our temp dir
    import src.app
    original_scenes_dir = src.app.SCENES_DIR
    src.app.SCENES_DIR = str(temp_dir)
    
    yield scene_name, scene_path
    
    # Teardown
    src.app.SCENES_DIR = original_scenes_dir
    shutil.rmtree(temp_dir)

def test_viewer_api_save_layout(mock_scene_dir):
    """Phase 3 Gate: Verify the viewer API successfully saves 3D object layout changes to disk."""
    scene_name, scene_path = mock_scene_dir
    
    calls = {"commit": 0}

    async def commit():
        calls["commit"] += 1

    # Initialize TestClient
    app = build_server(commit_scenes_fn=commit)
    client = TestClient(app)
    
    # Simulate a drag-and-drop event from the Three.js viewer
    update_payload = {
        "objects": [
            {
                "id": "obj_000",
                "label": "network cabinet",
                "position": [1.5, 0.0, -2.5],
                "yaw": 1.5708
            }
        ]
    }
    
    response = client.post(f"/api/scenes/{scene_name}/layout", json=update_payload)
    
    assert response.status_code == 200
    assert response.json()["updated"] == 1
    assert response.json()["ok"] is True
    assert calls["commit"] == 1
    
    # Verify the file was actually written to disk correctly
    with open(scene_path / "scene.json", "r") as f:
        updated_scene = json.load(f)
        
    updated_obj = updated_scene["objects"][0]
    assert updated_obj["position"] == [1.5, 0.0, -2.5], "Position was not updated on disk"
    assert updated_obj["yaw"] == 1.5708, "Yaw was not updated on disk"
    assert updated_obj["label"] == "network cabinet", "Label was not updated on disk"


def test_add_uploaded_object_and_persist_manual_scale(mock_scene_dir):
    scene_name, scene_path = mock_scene_dir
    mesh = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
    mesh.apply_translation([10.0, -2.0, 5.0])
    calls = {"commit": 0}

    async def commit():
        calls["commit"] += 1

    client = TestClient(build_server(commit_scenes_fn=commit))
    response = client.post(
        f"/api/scenes/{scene_name}/objects",
        files={"file": ("meeting-table.glb", mesh.export(file_type="glb"),
                        "model/gltf-binary")},
        data={"label": "Meeting table"},
    )

    assert response.status_code == 200, response.text
    obj = response.json()["object"]
    assert obj["id"].startswith("custom_")
    assert obj["label"] == "Meeting table"
    assert obj["source"] == "custom-upload"
    assert obj["size"] == [2.0, 3.0, 4.0]
    assert obj["model_offset"] == [-10.0, 3.5, -5.0]
    assert obj["editor_scale"] == [1.0, 1.0, 1.0]
    assert (scene_path / obj["glb"]).exists()
    assert calls["commit"] == 1

    response = client.post(
        f"/api/scenes/{scene_name}/layout",
        json={"objects": [{
            "id": obj["id"], "label": "Large meeting table",
            "position": [1.0, 0.0, -2.0], "yaw": 0.75,
            "editor_scale": [0.5, 2.0, 1.5],
        }]},
    )
    assert response.status_code == 200, response.text
    assert calls["commit"] == 2
    with open(scene_path / "scene.json") as f:
        saved = json.load(f)
    added = next(o for o in saved["objects"] if o["id"] == obj["id"])
    assert added["label"] == "Large meeting table"
    assert added["position"] == [1.0, 0.0, -2.0]
    assert added["yaw"] == 0.75
    assert added["editor_scale"] == [0.5, 2.0, 1.5]


def test_add_object_rejects_invalid_glb_and_layout_scale(mock_scene_dir):
    scene_name, _ = mock_scene_dir
    client = TestClient(build_server())

    response = client.post(
        f"/api/scenes/{scene_name}/objects",
        files={"file": ("bad.glb", b"glTFnot-a-real-file", "model/gltf-binary")},
        data={"label": "Bad model"},
    )
    assert response.status_code == 400

    response = client.post(
        f"/api/scenes/{scene_name}/layout",
        json={"objects": [{
            "id": "obj_000", "label": "server rack",
            "position": [0, 0, 0], "yaw": 0,
            "editor_scale": [-1, 1, 1],
        }]},
    )
    assert response.status_code == 400


def test_viewer_mutations_commit_and_uploaded_model_detaches_shared_asset(mock_scene_dir):
    scene_name, scene_path = mock_scene_dir
    objects_dir = scene_path / "objects"
    objects_dir.mkdir()
    shared_rel = "objects/obj_001.glb"
    (scene_path / shared_rel).write_bytes(trimesh.creation.box().export(file_type="glb"))

    with open(scene_path / "scene.json") as f:
        scene = json.load(f)
    scene["objects"] = [
        {**scene["objects"][0], "glb": shared_rel, "reuse_of": "obj_001"},
        {**scene["objects"][0], "id": "obj_001", "glb": shared_rel},
    ]
    with open(scene_path / "scene.json", "w") as f:
        json.dump(scene, f)

    calls = {"commit": 0}

    async def commit():
        calls["commit"] += 1

    client = TestClient(build_server(commit_scenes_fn=commit))
    replacement = trimesh.creation.icosphere(subdivisions=1).export(file_type="glb")
    response = client.post(
        f"/api/scenes/{scene_name}/objects/obj_000/model",
        files={"file": ("replacement.glb", replacement, "model/gltf-binary")},
        data={"scale": "[1, 2, 3]", "offset": "[0, 0.5, 0]"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["glb"] == "objects/obj_000_custom.glb"
    assert calls["commit"] == 1
    assert (objects_dir / "obj_000_custom.glb").exists()
    with open(scene_path / "scene.json") as f:
        updated = json.load(f)
    selected, shared = updated["objects"]
    assert selected["glb"] == "objects/obj_000_custom.glb"
    assert selected["source"] == "custom-upload"
    assert "reuse_of" not in selected
    assert shared["glb"] == shared_rel

    response = client.delete(f"/api/scenes/{scene_name}/objects/obj_001")
    assert response.status_code == 200
    assert calls["commit"] == 2
    assert not (scene_path / shared_rel).exists()


def test_deleting_reused_object_preserves_shared_asset(mock_scene_dir):
    scene_name, scene_path = mock_scene_dir
    objects_dir = scene_path / "objects"
    objects_dir.mkdir()
    shared_rel = "objects/obj_001.glb"
    (scene_path / shared_rel).write_bytes(trimesh.creation.box().export(file_type="glb"))
    with open(scene_path / "scene.json") as f:
        scene = json.load(f)
    scene["objects"] = [
        {**scene["objects"][0], "glb": shared_rel, "reuse_of": "obj_001"},
        {**scene["objects"][0], "id": "obj_001", "glb": shared_rel},
    ]
    with open(scene_path / "scene.json", "w") as f:
        json.dump(scene, f)

    client = TestClient(build_server())
    response = client.delete(f"/api/scenes/{scene_name}/objects/obj_000")

    assert response.status_code == 200
    assert (scene_path / shared_rel).exists()


def test_viewer_api_rejects_empty_label(mock_scene_dir):
    scene_name, _ = mock_scene_dir
    app = build_server()
    client = TestClient(app)

    response = client.post(
        f"/api/scenes/{scene_name}/layout",
        json={"objects": [{
            "id": "obj_000", "label": "   ",
            "position": [0, 0, 0], "yaw": 0,
        }]},
    )

    assert response.status_code == 400

def test_viewer_api_invalid_scene():
    """Verify security controls against path traversal or missing scenes."""
    app = build_server()
    client = TestClient(app)
    
    # Try directory traversal (FastAPI automatically 404s slashes in path variables)
    response = client.post("/api/scenes/../../etc/layout", json={"objects": []})
    assert response.status_code == 404
    
    # Try non-existent scene
    response = client.post("/api/scenes/ghost_scene_999/layout", json={"objects": []})
    assert response.status_code == 404


def test_regenerate_detaches_reused_asset_and_commits(mock_scene_dir):
    scene_name, scene_path = mock_scene_dir
    objects_dir = scene_path / "objects"
    objects_dir.mkdir()
    shared_rel = "objects/obj_001.glb"
    (scene_path / shared_rel).write_bytes(
        trimesh.creation.box().export(file_type="glb")
    )

    rgba = np.zeros((32, 32, 4), np.uint8)
    rgba[6:26, 6:26, :3] = 160
    rgba[6:26, 6:26, 3] = 255
    assert cv2.imwrite(str(objects_dir / "obj_000_input.png"),
                       cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

    with open(scene_path / "scene.json") as f:
        scene = json.load(f)
    scene["objects"] = [
        {
            **scene["objects"][0],
            "glb": shared_rel,
            "reuse_of": "obj_001",
            "input_crop": "objects/obj_000_input.png",
        },
        {
            **scene["objects"][0],
            "id": "obj_001",
            "glb": shared_rel,
        },
    ]
    with open(scene_path / "scene.json", "w") as f:
        json.dump(scene, f)

    calls = {"generate": 0, "commit": 0}

    async def generate(_crop):
        calls["generate"] += 1
        return trimesh.creation.icosphere(subdivisions=1).export(file_type="glb")

    async def commit():
        calls["commit"] += 1

    client = TestClient(build_server(
        object_generator_fn=generate,
        commit_scenes_fn=commit,
    ))
    response = client.post(
        f"/api/scenes/{scene_name}/objects/obj_000/regenerate"
    )

    assert response.status_code == 200, response.text
    assert response.json()["glb"] == "objects/obj_000_trellis.glb"
    assert calls == {"generate": 1, "commit": 1}
    assert (objects_dir / "obj_000_trellis.glb").exists()
    with open(scene_path / "scene.json") as f:
        updated = json.load(f)
    selected, shared = updated["objects"]
    assert selected["glb"] == "objects/obj_000_trellis.glb"
    assert selected["source"] == "trellis"
    assert "reuse_of" not in selected
    assert shared["glb"] == shared_rel


def test_write_auth_no_op_when_unset(mock_scene_dir, monkeypatch):
    """No ROOM3D_API_TOKEN configured -> writes succeed with no header at all,
    matching every other test in this file (local dev needs no setup)."""
    monkeypatch.delenv("ROOM3D_API_TOKEN", raising=False)
    scene_name, _ = mock_scene_dir
    client = TestClient(build_server())

    response = client.post(f"/api/scenes/{scene_name}/layout", json={"objects": []})

    assert response.status_code == 200


def test_write_auth_rejects_missing_or_wrong_token(mock_scene_dir, monkeypatch):
    monkeypatch.setenv("ROOM3D_API_TOKEN", "correct-token")
    scene_name, _ = mock_scene_dir
    client = TestClient(build_server())

    no_header = client.post(f"/api/scenes/{scene_name}/layout", json={"objects": []})
    wrong_header = client.post(
        f"/api/scenes/{scene_name}/layout", json={"objects": []},
        headers={"X-Room3D-Token": "wrong-token"},
    )

    assert no_header.status_code == 401
    assert wrong_header.status_code == 401


def test_write_auth_accepts_correct_token(mock_scene_dir, monkeypatch):
    monkeypatch.setenv("ROOM3D_API_TOKEN", "correct-token")
    scene_name, _ = mock_scene_dir
    client = TestClient(build_server())

    response = client.post(
        f"/api/scenes/{scene_name}/layout", json={"objects": []},
        headers={"X-Room3D-Token": "correct-token"},
    )

    assert response.status_code == 200


def test_write_auth_covers_delete_and_regenerate(mock_scene_dir, monkeypatch):
    """The two endpoints without a json body/Request param originally — make
    sure adding auth to them didn't just silently no-op."""
    monkeypatch.setenv("ROOM3D_API_TOKEN", "correct-token")
    scene_name, _ = mock_scene_dir
    client = TestClient(build_server())

    delete_resp = client.delete(f"/api/scenes/{scene_name}/objects/obj_000")
    regen_resp = client.post(f"/api/scenes/{scene_name}/objects/obj_000/regenerate")

    assert delete_resp.status_code == 401
    assert regen_resp.status_code == 401

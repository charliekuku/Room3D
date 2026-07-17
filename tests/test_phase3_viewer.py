import pytest
import json
import shutil
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
import sys
import os

# Add src to python path to import app
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.app import build_server

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
    
    # Initialize TestClient
    app = build_server()
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
    
    # Verify the file was actually written to disk correctly
    with open(scene_path / "scene.json", "r") as f:
        updated_scene = json.load(f)
        
    updated_obj = updated_scene["objects"][0]
    assert updated_obj["position"] == [1.5, 0.0, -2.5], "Position was not updated on disk"
    assert updated_obj["yaw"] == 1.5708, "Yaw was not updated on disk"
    assert updated_obj["label"] == "network cabinet", "Label was not updated on disk"


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

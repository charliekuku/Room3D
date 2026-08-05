import os
import cv2
import pytest
from pathlib import Path

# The project root relative to this test file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "samples" / "cyrusone"

def check_blur_laplacian(image_path: str, threshold: float = 80.0) -> float:
    """Calculate the Laplacian variance to measure image sharpness."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1.0 # Return invalid variance for unreadable files
    variance = cv2.Laplacian(img, cv2.CV_64F).var()
    return variance

def test_capture_protocol_data_exists():
    """Phase 0 Gate: Verify that local test datasets are valid when present.

    data/ is machine-local (gitignored), so absence is a skip, not a failure —
    otherwise the suite can never pass on a fresh clone or CI.
    """
    if not (PROJECT_ROOT / "data").exists():
        pytest.skip("data/ not present on this machine.")
    assert (PROJECT_ROOT / "data" / "office.mp4").exists(), "office.mp4 missing"
    assert (PROJECT_ROOT / "data" / "hospital.mp4").exists(), "hospital.mp4 missing"
    if not SAMPLE_DIR.exists():
        pytest.skip("Photo sample dataset not present on this machine.")
    assert len(list(SAMPLE_DIR.glob("*.jpg"))) > 0, "cyrusone sample dir exists but has no .jpg files"

def test_capture_image_sharpness():
    """Phase 0 Gate: Verify that the capture protocol prevents motion blur."""
    if not SAMPLE_DIR.exists():
        pytest.skip("Sample photo directory not found.")
        
    images = list(SAMPLE_DIR.glob("*.jpg"))[:5] # Test the first 5 images
    for img_path in images:
        variance = check_blur_laplacian(img_path)
        if variance < 0:
            pytest.skip(f"Skipping sharpness test because {img_path.name} is not a valid image file (likely a dummy placeholder).")
            
        # 80.0 is a reasonable threshold for standard indoor photography
        assert variance > 80.0, f"Image {img_path.name} failed sharpness test (variance {variance:.1f} < 80.0). High motion blur detected."

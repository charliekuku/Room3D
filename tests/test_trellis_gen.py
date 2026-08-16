import os
import sys

import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detection.trellis_gen import (
    _has_foreground_alpha,
    _prepare_input_image,
)


def test_foreground_alpha_requires_rgba_with_foreground_and_background():
    assert not _has_foreground_alpha(np.zeros((8, 8, 3), dtype=np.uint8))
    assert not _has_foreground_alpha(np.zeros((8, 8, 4), dtype=np.uint8))
    assert not _has_foreground_alpha(np.full((8, 8, 4), 255, dtype=np.uint8))

    cutout = np.zeros((8, 8, 4), dtype=np.uint8)
    cutout[2:6, 2:6, :3] = 127
    cutout[2:6, 2:6, 3] = 255
    assert _has_foreground_alpha(cutout)


def test_prepare_input_preserves_useful_alpha_and_auto_masks_rgb():
    rgb = np.full((8, 10, 3), 127, dtype=np.uint8)
    image, masking = _prepare_input_image(rgb)
    assert image.mode == "RGB"
    assert image.size == (10, 8)
    assert masking == "automatic-birefnet"

    opaque = np.dstack([rgb, np.full((8, 10), 255, dtype=np.uint8)])
    image, masking = _prepare_input_image(opaque)
    assert image.mode == "RGB"
    assert masking == "automatic-birefnet"

    cutout = opaque.copy()
    cutout[:2, :, 3] = 0
    image, masking = _prepare_input_image(cutout)
    assert image.mode == "RGBA"
    assert masking == "provided-alpha"

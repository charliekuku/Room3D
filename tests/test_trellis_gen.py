import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detection.trellis_gen import (
    _AlphaOnlyBackgroundRemover,
    _has_foreground_alpha,
)


def test_foreground_alpha_requires_rgba_with_foreground_and_background():
    assert not _has_foreground_alpha(np.zeros((8, 8, 3), dtype=np.uint8))
    assert not _has_foreground_alpha(np.zeros((8, 8, 4), dtype=np.uint8))
    assert not _has_foreground_alpha(np.full((8, 8, 4), 255, dtype=np.uint8))

    cutout = np.zeros((8, 8, 4), dtype=np.uint8)
    cutout[2:6, 2:6, :3] = 127
    cutout[2:6, 2:6, 3] = 255
    assert _has_foreground_alpha(cutout)


def test_alpha_only_remover_is_device_compatible_but_never_callable():
    remover = _AlphaOnlyBackgroundRemover(model_name="unused")
    assert remover.to("cuda") is remover
    with pytest.raises(RuntimeError, match="requires an RGBA input"):
        remover(None)

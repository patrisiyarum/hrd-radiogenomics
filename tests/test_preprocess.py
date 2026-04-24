"""Smoke tests for the preprocess contract that drug-cell-viz also uses."""

from __future__ import annotations

from radiogenomics import HU_WINDOW, TARGET_SHAPE


def test_target_shape_is_96_cubed():
    assert TARGET_SHAPE == (96, 96, 96)


def test_hu_window_targets_soft_tissue():
    lo, hi = HU_WINDOW
    assert lo < 0 and hi > 0
    # 0 HU is the water reference and must sit inside the window so normal
    # tissue doesn't clip to the floor.
    assert lo < 0 < hi


def test_version_is_semver():
    from radiogenomics import __version__

    assert __version__.count(".") == 2

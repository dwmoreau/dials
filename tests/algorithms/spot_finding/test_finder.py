from __future__ import annotations

import pytest

from dials.algorithms.spot_finding.factory import (
    _mask_depends_on_wavelength,
    phil_scope,
)
from dials.algorithms.spot_finding.finder import ExtractPixelsFromImage, ExtractSpots
from dials.algorithms.spot_finding.threshold import DispersionThresholdStrategy
from dials.array_family import flex
from dials.util.phil import parse


def filter_params(overrides=""):
    return phil_scope.fetch(parse(overrides)).extract().spotfinder.filter


def test_mask_is_wavelength_independent_by_default():
    assert not _mask_depends_on_wavelength(filter_params())


@pytest.mark.parametrize(
    "override",
    [
        "spotfinder.filter.d_min=2.5",
        "spotfinder.filter.d_max=50",
        "spotfinder.filter.resolution_range=3,4",
        "spotfinder.filter.ice_rings.filter=True",
    ],
)
def test_every_resolution_mask_makes_the_mask_wavelength_dependent(override):
    assert _mask_depends_on_wavelength(filter_params(override))


class FakeImageset:
    def __init__(self, n_panels):
        self.n_panels = n_panels

    def get_detector(self):
        return [None] * self.n_panels


def test_update_imageset_replaces_the_imageset_and_the_mask():
    first, second = FakeImageset(2), FakeImageset(2)
    mask = (flex.bool(flex.grid(4, 4), True),) * 2
    extractor = ExtractPixelsFromImage(
        imageset=first,
        threshold_function=None,
        mask=mask,
        region_of_interest=None,
        max_strong_pixel_fraction=1.0,
        compute_mean_background=False,
    )
    assert extractor.imageset is first

    other = (flex.bool(flex.grid(4, 4), False),) * 2
    extractor.update_imageset(second, other)
    assert extractor.imageset is second
    assert extractor.mask is other


def test_update_imageset_rejects_a_mask_of_the_wrong_length():
    extractor = ExtractPixelsFromImage(
        imageset=FakeImageset(2),
        threshold_function=None,
        mask=(flex.bool(flex.grid(4, 4), True),) * 2,
        region_of_interest=None,
        max_strong_pixel_fraction=1.0,
        compute_mean_background=False,
    )
    with pytest.raises(AssertionError):
        extractor.update_imageset(FakeImageset(2), (flex.bool(flex.grid(4, 4), True),))


def dispersion_strategy(**kwargs):
    kwargs.setdefault("kernel_size", (3, 3))
    kwargs.setdefault("n_sigma_b", 6)
    kwargs.setdefault("n_sigma_s", 3)
    kwargs.setdefault("min_count", 2)
    kwargs.setdefault("global_threshold", 0)
    return DispersionThresholdStrategy(**kwargs)


def flat_image(shape):
    image = flex.double(flex.grid(*shape), 1.0)
    return image, flex.bool(flex.grid(*shape), True)


def test_images_of_one_shape_share_a_thresholder_through_the_cache():
    strategy = dispersion_strategy()
    image, mask = flat_image((20, 20))
    cache = {}

    strategy(image, mask, algorithm_cache=cache)
    assert list(cache) == [(20, 20)]
    thresholder = cache[(20, 20)]

    strategy(image, mask, algorithm_cache=cache)
    assert cache[(20, 20)] is thresholder


def test_the_thresholder_is_not_retained_on_the_strategy():
    strategy = dispersion_strategy()
    image, mask = flat_image((20, 20))
    strategy(image, mask)
    assert not hasattr(strategy, "algorithm")


def test_a_reused_strategy_follows_the_image_shape():
    strategy = dispersion_strategy(gain=2.0)
    small, small_mask = flat_image((8, 8))
    large, large_mask = flat_image((16, 16))

    strategy(small, small_mask)
    assert strategy._gain_map.all() == (8, 8)

    strategy(large, large_mask)
    assert strategy._gain_map.all() == (16, 16)


def test_extract_spots_does_not_hold_a_mask():
    """A retained mask is a byte per detector pixel for the life of the process, and
    ExtractSpots outlives an imageset only when the mask does not."""
    assert not hasattr(ExtractSpots(), "mask")

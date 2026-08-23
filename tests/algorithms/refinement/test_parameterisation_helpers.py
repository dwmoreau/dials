"""Tests for the C++ helpers of the hierarchical detector parameterisation"""

from __future__ import annotations

import math

import pytest

from dxtbx.model import Detector
from scitbx import matrix
from scitbx.array_family import flex

from dials_refinement_helpers_ext import (
    panel_group_centroid,
    panel_offsets_and_directions,
)


def _add_panel(node, name, fast, slow, origin):
    panel = node.add_panel()
    panel.set_name(name)
    panel.set_image_size((100, 200))
    panel.set_pixel_size((0.1, 0.05))
    panel.set_local_frame(fast, slow, origin)
    return panel


def _tilted_detector(n_groups=2, n_panels=3):
    """A hierarchy whose groups and panels are all differently oriented"""
    detector = Detector()
    root = detector.hierarchy()
    root.set_local_frame((1, 0, 0), (0, 1, 0), (0.0, 0.0, -120.0))
    for i in range(n_groups):
        group = root.add_group()
        group.set_name(f"group{i}")
        angle = math.radians(7.0 * (i + 1))
        group.set_local_frame(
            (math.cos(angle), math.sin(angle), 0.0),
            (-math.sin(angle), math.cos(angle), 0.0),
            (12.0 * i, -3.0 * i, 4.0 * i),
        )
        for j in range(n_panels):
            beta = math.radians(2.0 * (j + 1))
            _add_panel(
                group,
                f"panel{i}{j}",
                (math.cos(beta), 0.0, math.sin(beta)),
                (0.0, 1.0, 0.0),
                (11.0 * j, 6.0 * j, 1.5 * j),
            )
    return detector


def _panel_centre(panel):
    size = panel.get_image_size_mm()
    return (
        matrix.col(panel.get_origin())
        + 0.5 * matrix.col(panel.get_fast_axis()) * size[0]
        + 0.5 * matrix.col(panel.get_slow_axis()) * size[1]
    )


def test_centroid_of_a_single_panel_is_its_centre():
    detector = _tilted_detector(n_groups=1, n_panels=1)

    centroid = panel_group_centroid(detector, flex.size_t([0]))

    assert centroid == pytest.approx(_panel_centre(detector[0]).elems)


def test_centroid_is_the_mean_of_the_panel_centres():
    detector = _tilted_detector()
    ids = list(range(len(detector)))

    centroid = panel_group_centroid(detector, flex.size_t(ids))

    expected = matrix.col((0.0, 0.0, 0.0))
    for i in ids:
        expected += _panel_centre(detector[i])
    expected /= len(ids)
    assert centroid == pytest.approx(expected.elems)


def test_centroid_covers_only_the_panels_asked_for():
    detector = _tilted_detector()

    centroid = panel_group_centroid(detector, flex.size_t([1, 4]))

    expected = (_panel_centre(detector[1]) + _panel_centre(detector[4])) / 2.0
    assert centroid == pytest.approx(expected.elems)


def test_offsets_and_directions_reconstruct_each_panel_frame():
    detector = _tilted_detector(n_groups=1, n_panels=4)
    group = detector.hierarchy()[0]
    ids = list(range(len(detector)))
    d1 = matrix.col(group.get_fast_axis())
    d2 = matrix.col(group.get_slow_axis())
    dn = matrix.col(group.get_normal())
    dorg = matrix.col(group.get_origin())

    offsets, dir1s, dir2s = panel_offsets_and_directions(
        detector, flex.size_t(ids), dorg, d1, d2, dn
    )

    assert len(offsets) == len(dir1s) == len(dir2s) == len(ids)
    for i, panel in ((i, detector[i]) for i in ids):
        j = ids.index(i)
        origin = dorg + offsets[j][0] * d1 + offsets[j][1] * d2 + offsets[j][2] * dn
        fast = dir1s[j][0] * d1 + dir1s[j][1] * d2 + dir1s[j][2] * dn
        slow = dir2s[j][0] * d1 + dir2s[j][1] * d2 + dir2s[j][2] * dn
        assert origin.elems == pytest.approx(panel.get_origin())
        assert fast.elems == pytest.approx(panel.get_fast_axis())
        assert slow.elems == pytest.approx(panel.get_slow_axis())


def test_offsets_and_directions_follow_the_order_of_the_panel_ids():
    detector = _tilted_detector()
    basis = (matrix.col((1, 0, 0)), matrix.col((0, 1, 0)), matrix.col((0, 0, 1)))
    dorg = matrix.col((1.0, 2.0, 3.0))

    forward = panel_offsets_and_directions(
        detector, flex.size_t([0, 3, 5]), dorg, *basis
    )
    reversed_ = panel_offsets_and_directions(
        detector, flex.size_t([5, 3, 0]), dorg, *basis
    )

    for a, b in zip(forward, reversed_):
        assert list(a) == list(b)[::-1]

"""Tests for the refinement helper functions"""

from __future__ import annotations

from dxtbx.model import Detector

from dials.algorithms.refinement.refinement_helpers import (
    get_panel_groups_at_depth,
    get_panel_ids_at_root,
)


def _add_panel(node, name, origin):
    panel = node.add_panel()
    panel.set_name(name)
    panel.set_image_size((10, 10))
    panel.set_pixel_size((0.1, 0.1))
    panel.set_local_frame((1, 0, 0), (0, 1, 0), origin)
    return panel


def _hierarchical_detector(n_groups, n_panels):
    """A detector of n_groups groups, each holding n_panels distinct panels"""
    detector = Detector()
    root = detector.hierarchy()
    for i in range(n_groups):
        group = root.add_group()
        group.set_name(f"group{i}")
        for j in range(n_panels):
            _add_panel(group, f"panel{i}{j}", (10.0 * j, 10.0 * i, -100.0))
    return detector


def test_single_panel():
    detector = Detector()
    _add_panel(detector.hierarchy(), "panel", (0.0, 0.0, -100.0))

    assert get_panel_ids_at_root(detector.hierarchy()) == [0]


def test_ids_are_the_panels_own_positions():
    detector = _hierarchical_detector(4, 8)
    panels = list(detector)

    ids = get_panel_ids_at_root(detector.hierarchy())

    assert sorted(ids) == list(range(len(detector)))
    leaves = [panel for group in detector.hierarchy().children() for panel in group]
    for i, leaf in zip(ids, leaves):
        assert panels[i].get_name() == leaf.get_name()


def test_ids_are_distinct_for_panels_that_compare_equal():
    detector = Detector()
    root = detector.hierarchy()
    for _ in range(3):
        _add_panel(root.add_group(), "panel", (0.0, 0.0, -100.0))
    assert detector[0] == detector[1] == detector[2]

    assert get_panel_ids_at_root(root) == [0, 1, 2]


def test_group_ordering_is_preserved():
    detector = _hierarchical_detector(3, 2)
    groups = get_panel_groups_at_depth(detector.hierarchy(), 1)

    ids_by_group = [get_panel_ids_at_root(group) for group in groups]

    assert ids_by_group == [[0, 1], [2, 3], [4, 5]]
    assert get_panel_ids_at_root(detector.hierarchy()) == [0, 1, 2, 3, 4, 5]

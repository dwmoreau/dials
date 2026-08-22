from __future__ import annotations

import math

import pytest

from cctbx import sgtbx, uctbx
from scitbx import matrix

from dials.algorithms.spot_prediction import IndexGenerator, StillsIndexGenerator

# cells chosen to exercise the metric tensor: cubic is diagonal, the triclinic and
# monoclinic ones are not, and the orthorhombic one carries systematic absences
CELLS = [
    ("P 1", (100, 100, 100, 90, 90, 90)),
    ("P 1", (30, 35, 40, 85, 95, 105)),
    ("P 1 21 1", (40, 55, 45, 90, 112, 90)),
    ("P 21 21 21", (35, 60, 75, 90, 90, 90)),
    ("P 6/m m m", (45, 45, 90, 90, 90, 120)),
]
WAVELENGTH = 1.2
D_MIN = 3.0


def _models(space_group_symbol, cell_parameters):
    """A UB matrix in a general orientation, and a beam that is not down an axis."""
    unit_cell = uctbx.unit_cell(cell_parameters)
    space_group_type = sgtbx.space_group_info(space_group_symbol).group().type()

    # a general orientation, so the walk is never aligned with the beam
    rotation = matrix.sqr((0.36, -0.48, 0.8, 0.8, 0.6, 0.0, -0.48, 0.64, 0.6))
    b_matrix = matrix.sqr(unit_cell.fractionalization_matrix()).transpose()
    ub = rotation * b_matrix

    s0 = matrix.col((0.05, -0.1, 1.0)).normalize() / WAVELENGTH
    return unit_cell, space_group_type, ub, s0


def _ewald_offsets(ub, s0, indices):
    offsets = []
    for h in indices:
        q = ub * matrix.col(h)
        offsets.append(q.dot(q) + 2.0 * q.dot(s0))
    return offsets


def _cutoff(s0):
    return 2.0 * 2.0 * s0.length() * 0.0015 / D_MIN


@pytest.mark.parametrize("space_group_symbol,cell_parameters", CELLS)
def test_emits_every_index_within_the_cutoff(space_group_symbol, cell_parameters):
    unit_cell, space_group_type, ub, s0 = _models(space_group_symbol, cell_parameters)
    eps_cut = _cutoff(s0)

    box = list(IndexGenerator(unit_cell, space_group_type, D_MIN).to_array())
    offsets = _ewald_offsets(ub, s0, box)
    wanted = [h for h, eps in zip(box, offsets) if abs(eps) <= eps_cut]
    assert len(wanted) > 0
    assert len(wanted) < len(box) // 50

    emitted = list(
        StillsIndexGenerator(
            unit_cell, space_group_type, D_MIN, ub, s0, eps_cut
        ).to_array()
    )

    assert len(set(emitted)) == len(emitted)
    assert set(wanted) <= set(emitted)
    # emitting a little beyond the cutoff is allowed, emitting the whole box is not
    assert len(emitted) < len(box) // 20
    # and what it does emit must stay in bounding-box order
    order = {h: i for i, h in enumerate(box)}
    assert all(h in order for h in emitted)
    assert [order[h] for h in emitted] == sorted(order[h] for h in emitted)


@pytest.mark.parametrize("space_group_symbol,cell_parameters", CELLS)
def test_without_a_cutoff_it_is_the_bounding_box_walk(
    space_group_symbol, cell_parameters
):
    unit_cell, space_group_type, ub, s0 = _models(space_group_symbol, cell_parameters)

    box = list(IndexGenerator(unit_cell, space_group_type, D_MIN).to_array())
    emitted = list(
        StillsIndexGenerator(
            unit_cell, space_group_type, D_MIN, ub, s0, float("inf")
        ).to_array()
    )
    assert emitted == box


def test_a_degenerate_ub_falls_back_to_the_bounding_box_walk():
    unit_cell, space_group_type, ub, s0 = _models(*CELLS[0])
    flattened = matrix.sqr((ub[0], ub[1], ub[2], ub[3], ub[4], ub[5], 0, 0, 0))

    box = list(IndexGenerator(unit_cell, space_group_type, D_MIN).to_array())
    emitted = list(
        StillsIndexGenerator(
            unit_cell, space_group_type, D_MIN, flattened, s0, _cutoff(s0)
        ).to_array()
    )
    assert emitted == box


def test_next_terminates_with_a_zero_index():
    unit_cell, space_group_type, ub, s0 = _models(*CELLS[1])
    generator = StillsIndexGenerator(
        unit_cell, space_group_type, D_MIN, ub, s0, _cutoff(s0)
    )
    count = 0
    while True:
        h = generator.next()
        if h == (0, 0, 0):
            break
        count += 1
    assert count > 0
    assert generator.next() == (0, 0, 0)


def test_systematic_absences_are_still_applied():
    """P 21 21 21 extinguishes the odd axial reflections, in both walks."""
    unit_cell, space_group_type, ub, s0 = _models(
        "P 21 21 21", (35, 60, 75, 90, 90, 90)
    )
    emitted = list(
        StillsIndexGenerator(
            unit_cell, space_group_type, D_MIN, ub, s0, float("inf")
        ).to_array()
    )
    axial = [h for h in emitted if (h[1], h[2]) == (0, 0)]
    assert axial
    assert all(h[0] % 2 == 0 for h in axial)


def test_the_shell_narrows_as_the_cutoff_does():
    unit_cell, space_group_type, ub, s0 = _models(*CELLS[1])
    counts = [
        len(
            StillsIndexGenerator(
                unit_cell, space_group_type, D_MIN, ub, s0, _cutoff(s0) * scale
            ).to_array()
        )
        for scale in (1.0, 10.0, 100.0)
    ]
    assert counts[0] < counts[1] < counts[2]
    assert math.isclose(counts[1] / counts[0], 10.0, rel_tol=0.5)

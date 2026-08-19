from __future__ import annotations

import itertools
import logging
import math
from itertools import combinations as iter_combinations

import numpy as np

from cctbx import sgtbx
from cctbx.sgtbx.bravais_types import bravais_lattice
from cctbx.uctbx.reduction_base import iteration_limit_exceeded
from dxtbx.model import Crystal

from dials.algorithms.indexing import DialsIndexError
from dials.algorithms.indexing.compare_orientation_matrices import (
    difference_rotation_matrix_axis_angle,
)
from dials.algorithms.indexing.symmetry import find_matching_symmetry

logger = logging.getLogger(__name__)

# Widen the volume window by this relative amount so a candidate sitting on the
# bound cannot be lost to rounding.  The window spans a factor of two, so the cost
# in selectivity is nil.
_VOLUME_WINDOW_SLACK = 1e-6
_ANGLE_EPS = 1e-9


def candidate_orientation_matrices(basis_vectors, max_combinations=None):
    # select unique combinations of input vectors to test
    # the order of combinations is such that combinations comprising vectors
    # nearer the beginning of the input list will appear before combinations
    # comprising vectors towards the end of the list
    n = len(basis_vectors)
    if n < 3:
        return []
    # hardcoded limit on number of vectors, fixes issue #72
    # https://github.com/dials/dials/issues/72
    n = min(n, 100)
    basis_vectors = basis_vectors[:n]

    # Build sorted (i<j<k) index array. itertools.combinations already enforces
    # i<j<k, replacing the flex filter. Sort by squared index-norm to match the
    # original ordering (smallest-indexed vectors — best candidates — first).
    idxs = np.array(list(iter_combinations(range(n), 3)), dtype=np.int32)
    norms_sq = (idxs**2).sum(axis=1)
    idxs = idxs[np.argsort(norms_sq, kind="stable")]

    if max_combinations is not None and max_combinations < len(idxs):
        idxs = idxs[:max_combinations]

    half_pi = 0.5 * math.pi
    min_angle = 20 / 180 * math.pi  # 20 degrees, arbitrary cutoff

    # Convert basis vectors to a (n, 3) numpy array.
    bv = np.array([v.elems for v in basis_vectors])

    # Extract all (a, b, c) vector triplets.
    a_arr = bv[idxs[:, 0]]  # (N, 3)
    b_arr = bv[idxs[:, 1]]  # (N, 3)
    c_arr = bv[idxs[:, 2]]  # (N, 3)

    # Precompute squared norms for all vectors in each triplet.
    a_ns = np.einsum("ij,ij->i", a_arr, a_arr)
    b_ns = np.einsum("ij,ij->i", b_arr, b_arr)
    c_ns = np.einsum("ij,ij->i", c_arr, c_arr)

    # Filter 1: angle(a, b) not too close to 0° or 180°.
    ab_dot = np.einsum("ij,ij->i", a_arr, b_arr)
    angle_ab = np.arccos(np.clip(ab_dot / np.sqrt(a_ns * b_ns), -1.0, 1.0))
    m1 = (angle_ab >= min_angle) & ((np.pi - angle_ab) >= min_angle)
    a_arr, b_arr, c_arr = a_arr[m1], b_arr[m1], c_arr[m1]
    b_ns, c_ns, angle_ab = b_ns[m1], c_ns[m1], angle_ab[m1]

    # Flip b (and implicitly a×b) where angle_ab < half_pi so all angles obtuse.
    flip_b = (angle_ab < half_pi)[:, None]
    b_arr = np.where(flip_b, -b_arr, b_arr)

    # Compute a × b after the b flip.
    a_cross_b = np.cross(a_arr, b_arr)  # (M1, 3)
    acb_ns = np.einsum("ij,ij->i", a_cross_b, a_cross_b)

    # Filter 2: angle(a×b, c) not too close to 90° (would give degenerate cell).
    acb_c_dot = np.einsum("ij,ij->i", a_cross_b, c_arr)
    angle_acb_c = np.arccos(np.clip(acb_c_dot / np.sqrt(acb_ns * c_ns), -1.0, 1.0))
    m2 = np.abs(half_pi - angle_acb_c) >= min_angle
    a_arr, b_arr, c_arr = a_arr[m2], b_arr[m2], c_arr[m2]
    a_cross_b, b_ns, c_ns = a_cross_b[m2], b_ns[m2], c_ns[m2]

    # Flip c where angle(b, c) < half_pi so all angles obtuse.
    # alpha is in radians — consistent with half_pi (fixes a deg=True bug in the
    # original code where alpha was computed in degrees but compared to half_pi).
    bc_dot = np.einsum("ij,ij->i", b_arr, c_arr)
    alpha = np.arccos(np.clip(bc_dot / np.sqrt(b_ns * c_ns), -1.0, 1.0))
    flip_c = (alpha < half_pi)[:, None]
    c_arr = np.where(flip_c, -c_arr, c_arr)

    # Ensure right-handed basis: invert all vectors if a×b · c < 0.
    acb_dot_c = np.einsum("ij,ij->i", a_cross_b, c_arr)
    flip_all = (acb_dot_c < 0)[:, None]
    a_arr = np.where(flip_all, -a_arr, a_arr)
    b_arr = np.where(flip_all, -b_arr, b_arr)
    c_arr = np.where(flip_all, -c_arr, c_arr)

    # Crystal creation and Niggli reduction are C++ and cannot be batched.
    # The loop runs only over the fraction of combinations that passed the filters.
    # Pass the real-space vectors as plain Python lists (one bulk .tolist() per
    # array) straight to the C++ Crystal constructor, and reuse a single P 1 space
    # group, rather than round-tripping each row through scitbx.matrix.col and
    # re-parsing the "P 1" symbol on every iteration.
    sg_p1 = sgtbx.space_group()
    for a_row, b_row, c_row in zip(a_arr.tolist(), b_arr.tolist(), c_arr.tolist()):
        model = Crystal(a_row, b_row, c_row, space_group=sg_p1)
        uc = model.get_unit_cell()
        try:
            cb_op_to_niggli = uc.change_of_basis_op_to_niggli_cell()
        except iteration_limit_exceeded as e:
            raise DialsIndexError(e)
        model = model.change_basis(cb_op_to_niggli)

        uc = model.get_unit_cell()
        params = uc.parameters()
        if uc.volume() > (params[0] * params[1] * params[2] / 100):
            # unit cell volume cutoff from labelit 2004 paper
            yield model


def _volume_window(
    target_unit_cell, relative_length_tolerance, absolute_angle_tolerance
):
    """Bound the volume ratio of any cell that :func:`filter_known_symmetry` accepts.

    ``uctbx.unit_cell.is_similar_to`` compares the six cell parameters elementwise, so
    an accepted cell has each length within a factor ``[1 - rel, 1 / (1 - rel)]`` of the
    target's and each angle within ``absolute_angle_tolerance`` of it.  Writing
    ``V = a b c f(alpha, beta, gamma)``, those two constraints bound ``V`` independently.

    Returns the ``(lo, hi)`` bounds on ``V / target_unit_cell.volume()``, or ``None`` if
    the tolerances do not bound it.
    """
    if relative_length_tolerance >= 1:
        return None
    angles = target_unit_cell.parameters()[3:]
    f_target = _cell_volume_factor(*(math.cos(math.radians(a)) for a in angles))
    if f_target <= 0:
        return None
    f_lo, f_hi = _volume_factor_extrema(angles, absolute_angle_tolerance)
    length_lo = (1 - relative_length_tolerance) ** 3
    return (
        length_lo * f_lo / f_target * (1 - _VOLUME_WINDOW_SLACK),
        f_hi / (length_lo * f_target) * (1 + _VOLUME_WINDOW_SLACK),
    )


def _cell_volume_factor(cos_alpha, cos_beta, cos_gamma):
    square = (
        1
        - cos_alpha**2
        - cos_beta**2
        - cos_gamma**2
        + 2 * cos_alpha * cos_beta * cos_gamma
    )
    return math.sqrt(square) if square > 0 else 0


def _volume_factor_extrema(angles, tolerance):
    """Exact extrema of ``f`` over the box of angles within ``tolerance`` of ``angles``.

    ``d(f**2)/d(cos alpha) = -2 cos alpha + 2 cos beta cos gamma``, so a critical point
    interior in any coordinate requires that coordinate's cosine to be zero.  The
    extrema therefore lie among the points whose cosines are interval endpoints or zero.
    """
    axes = []
    for angle in angles:
        lo = math.cos(math.radians(max(angle - tolerance, _ANGLE_EPS)))
        hi = math.cos(math.radians(min(angle + tolerance, 180 - _ANGLE_EPS)))
        cosines = {lo, hi}
        if min(lo, hi) <= 0 <= max(lo, hi):
            cosines.add(0.0)
        axes.append(cosines)
    factors = [
        f
        for f in (_cell_volume_factor(*point) for point in itertools.product(*axes))
        if f > 0
    ]
    return min(factors), max(factors)


def filter_known_symmetry(
    crystal_models,
    target_symmetry,
    relative_length_tolerance=0.1,
    absolute_angle_tolerance=5,
    max_delta=5,
):
    """Filter crystal models for known symmetry.

    Args:
        crystal_models (list): A list of :class:`dxtbx.model.Crystal` objects.
        target_symmetry (cctbx.crystal.symmetry): The target symmetry for filtering.
        relative_length_tolerance (float): Relative tolerance for unit cell lengths in
            unit cell comparison (default value is 0.1).
        absolute_angle_tolerance (float): Angular tolerance (in degrees) in unit cell
            comparison (default value is 5).
        max_delta (float): Maximum allowed Le Page delta used in searching for basis
            vector combinations that are consistent with the given symmetry (default
            value is 5).
    """

    n_matched = 0

    cb_op_ref_to_primitive = target_symmetry.change_of_basis_op_to_primitive_setting()

    volume_window = None
    if target_symmetry.unit_cell() is not None:
        target_symmetry_primitive = target_symmetry.change_basis(cb_op_ref_to_primitive)
        target_unit_cell = (
            target_symmetry.as_reference_setting().best_cell().unit_cell()
        )
        ratio_window = _volume_window(
            target_unit_cell, relative_length_tolerance, absolute_angle_tolerance
        )
        if ratio_window is not None:
            # find_matching_symmetry only returns subgroups of the target's Bravais
            # type, so the accepted cell is the candidate's primitive volume scaled by
            # the target's centring multiplicity; comparing against the target's
            # primitive volume cancels that factor.
            primitive_volume = target_symmetry_primitive.unit_cell().volume()
            volume_window = tuple(r * primitive_volume for r in ratio_window)
    else:
        target_symmetry_primitive = target_symmetry.customized_copy(
            space_group_info=target_symmetry.space_group_info().change_basis(
                cb_op_ref_to_primitive
            )
        )
        target_unit_cell = None
    target_bravais_str = str(
        bravais_lattice(
            group=target_symmetry_primitive.space_group_info()
            .reference_setting()
            .group()
        )
    )

    for model in crystal_models:
        uc = model.get_unit_cell()
        if volume_window is not None and not (
            volume_window[0] <= uc.volume() <= volume_window[1]
        ):
            logger.debug(
                "Rejecting crystal model inconsistent with input symmetry:\n"
                f"  Unit cell: {str(uc)}"
            )
            continue
        best_subgroup = find_matching_symmetry(
            uc, None, max_delta=max_delta, target_bravais_str=target_bravais_str
        )
        if best_subgroup is not None:
            if target_symmetry.unit_cell() is not None and not (
                best_subgroup["best_subsym"]
                .unit_cell()
                .is_similar_to(
                    target_unit_cell,
                    relative_length_tolerance=relative_length_tolerance,
                    absolute_angle_tolerance=absolute_angle_tolerance,
                )
            ):
                logger.debug(
                    "Rejecting crystal model inconsistent with input symmetry:\n"
                    f"  Unit cell: {str(model.get_unit_cell())}"
                )
                continue

            n_matched += 1
            yield model
    if not n_matched:
        logger.warning(
            "No crystal models remaining after comparing with known symmetry"
        )


def filter_similar_orientations(
    crystal_models, other_crystal_models, minimum_angular_separation=5
):
    for cryst in crystal_models:
        orientation_too_similar = False
        for cryst_a in other_crystal_models:
            R_ab, axis, angle, cb_op_ab = difference_rotation_matrix_axis_angle(
                cryst_a, cryst
            )
            if abs(angle) < minimum_angular_separation:  # degrees
                orientation_too_similar = True
                break
        if orientation_too_similar:
            logger.debug("skipping crystal: too similar to other crystals")
            continue
        yield cryst

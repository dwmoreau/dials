from __future__ import annotations

from math import radians, sqrt

import pytest

from scitbx import matrix


class Model:
    def __init__(
        self,
        test_nave_model=False,
        half_mosaicity_deg=500,
        domain_size_ang=0.2,
        cell_edge=100,
    ):
        # Set up experimental models with regular geometry
        from dxtbx.model import BeamFactory, DetectorFactory, GoniometerFactory

        # Beam along the Z axis
        self.beam = BeamFactory.make_beam(unit_s0=matrix.col((0, 0, 1)), wavelength=1.0)

        # Goniometer (used only for index generation) along X axis
        self.goniometer = GoniometerFactory.known_axis(matrix.col((1, 0, 0)))

        # Detector fast, slow along X, -Y; beam in the centre, 200 mm distance
        dir1 = matrix.col((1, 0, 0))
        dir2 = matrix.col((0, -1, 0))
        centre = matrix.col((0, 0, 200))
        npx_fast = npx_slow = 1000
        pix_size = 0.2
        origin = centre - (
            0.5 * npx_fast * pix_size * dir1 + 0.5 * npx_slow * pix_size * dir2
        )
        self.detector = DetectorFactory.make_detector(
            "PAD",
            dir1,
            dir2,
            origin,
            (pix_size, pix_size),
            (npx_fast, npx_slow),
            (0, 1.0e6),
        )

        # Cubic crystal
        a = matrix.col((cell_edge, 0, 0))
        b = matrix.col((0, cell_edge, 0))
        c = matrix.col((0, 0, cell_edge))

        if test_nave_model:
            from dxtbx.model import MosaicCrystalSauter2014

            self.crystal = MosaicCrystalSauter2014(a, b, c, space_group_symbol="P 1")
            self.crystal.set_half_mosaicity_deg(half_mosaicity_deg)
            self.crystal.set_domain_size_ang(domain_size_ang)
        else:
            from dxtbx.model import Crystal

            self.crystal = Crystal(a, b, c, space_group_symbol="P 1")

        # Collect these models in an Experiment (ignoring the goniometer)
        from dxtbx.model.experiment_list import Experiment

        self.experiment = Experiment(
            beam=self.beam,
            detector=self.detector,
            goniometer=None,
            scan=None,
            crystal=self.crystal,
            imageset=None,
        )

        # Generate some reflections
        self.reflections = self.generate_reflections()

    def generate_reflections(self):
        """Use reeke_model to generate indices of reflections near to the Ewald
        sphere that might be observed on a still image. Build a reflection_table
        of these."""
        from cctbx.sgtbx import space_group_info

        space_group_type = space_group_info("P 1").group().type()

        # create a ReekeIndexGenerator
        UB = self.crystal.get_A()
        axis = self.goniometer.get_rotation_axis()
        s0 = self.beam.get_s0()
        # use the same UB at the beginning and end - the margin parameter ensures
        # we still have indices close to the Ewald sphere generated
        from dials.algorithms.spot_prediction import ReekeIndexGenerator

        r = ReekeIndexGenerator(UB, UB, space_group_type, axis, s0, dmin=1.5, margin=1)

        # generate indices
        hkl = r.to_array()
        nref = len(hkl)

        # create a reflection table
        from dials.array_family import flex

        table = flex.reflection_table()
        table["flags"] = flex.size_t(nref, 0)
        table["id"] = flex.int(nref, 0)
        table["panel"] = flex.size_t(nref, 0)
        table["miller_index"] = flex.miller_index(hkl)
        table["entering"] = flex.bool(nref, True)
        table["s1"] = flex.vec3_double(nref)
        table["xyzcal.mm"] = flex.vec3_double(nref)
        table["xyzcal.px"] = flex.vec3_double(nref)

        return table


@pytest.mark.parametrize("nave_model", [True, False], ids=["nave", "native"])
def test(nave_model):
    model = Model(test_nave_model=nave_model)

    # cache objects from the model
    UB = matrix.sqr(model.crystal.get_A())
    s0 = matrix.col(model.beam.get_s0())
    es_radius = s0.length()

    # create the predictor and predict for reflection table
    from dials.algorithms.spot_prediction import StillsReflectionPredictor

    predictor = StillsReflectionPredictor(model.experiment)
    predictor.for_reflection_table(model.reflections, UB)

    # for every reflection, reconstruct relp rotated to the Ewald sphere (vector
    # r) and unrotated relp (vector q), calculate the angle between them and
    # compare with delpsical.rad
    from libtbx.test_utils import approx_equal

    for ref in model.reflections.rows():
        r = matrix.col(ref["s1"]) - s0
        q = UB * matrix.col(ref["miller_index"])
        tst_radius = (s0 + q).length()
        sgn = -1 if tst_radius > es_radius else 1
        delpsi = sgn * r.accute_angle(q)
        assert approx_equal(delpsi, ref["delpsical.rad"])


def test_spherical_relps():
    model = Model()

    # cache objects from the model
    UB = matrix.sqr(model.crystal.get_A())
    s0 = matrix.col(model.beam.get_s0())
    es_radius = s0.length()

    # create the predictor and predict for reflection table
    from dials.algorithms.spot_prediction import StillsReflectionPredictor

    predictor = StillsReflectionPredictor(model.experiment, spherical_relp=True)
    predictor.for_reflection_table(model.reflections, UB)

    # for every reflection, reconstruct relp centre q, calculate s1 according
    # to the formula in stills_prediction_nave3.pdf and compare
    from libtbx.test_utils import approx_equal

    for ref in model.reflections.rows():
        q = UB * matrix.col(ref["miller_index"])
        radicand = q.length_sq() + 2.0 * q.dot(s0) + s0.length_sq()
        assert radicand > 0.0
        denom = sqrt(radicand)
        s1 = es_radius * (q + s0) / denom
        assert approx_equal(s1, ref["s1"])


def _stills_predictors(experiment, dmin):
    """One predictor of each stills class, all built on the same experiment."""
    from dials.algorithms.spot_prediction import (
        NaveStillsReflectionPredictor,
        SphericalRelpStillsReflectionPredictor,
        StillsDeltaPsiReflectionPredictor,
    )

    crystal = experiment.crystal
    common = (
        experiment.beam,
        experiment.detector,
        crystal.get_A(),
        crystal.get_unit_cell(),
        crystal.get_space_group().type(),
        dmin,
    )
    return {
        "nave": NaveStillsReflectionPredictor(
            *common,
            crystal.get_half_mosaicity_deg(),
            crystal.get_domain_size_ang(),
        ),
        "delta_psi": StillsDeltaPsiReflectionPredictor(*common),
        "spherical_relp": SphericalRelpStillsReflectionPredictor(*common),
    }


def _acceptance_threshold(kind, experiment, indices):
    """The delta psi bound each index is tested against."""
    from math import radians

    from dials.array_family import flex

    if kind == "nave":
        crystal = experiment.crystal
        d = crystal.get_unit_cell().d(indices)
        return (
            d / crystal.get_domain_size_ang()
            + radians(crystal.get_half_mosaicity_deg()) / 2.0
        )
    return flex.double(len(indices), 0.0015)


def _predict_then_select(predictor, kind, experiment, indices):
    """Predict every index, then keep the ones that meet the delta psi bound.

    __call__ runs the same per-index prediction as for_ub with no delta psi test and
    appends in input order, so this reproduces for_ub's table independently of how
    for_ub decides which indices to skip.
    """
    from dials.array_family import flex

    table = predictor(indices)
    threshold = _acceptance_threshold(kind, experiment, table["miller_index"])
    return table.select(flex.abs(table["delpsical.rad"]) < threshold)


def _assert_tables_identical(reference, predicted):
    assert len(predicted) == len(reference)
    for column in (
        "miller_index",
        "panel",
        "entering",
        "s1",
        "xyzcal.px",
        "xyzcal.mm",
        "flags",
        "delpsical.rad",
    ):
        assert list(predicted[column]) == list(reference[column]), column


def _generated_indices(experiment, dmin):
    from dials.algorithms.spot_prediction import IndexGenerator

    return IndexGenerator(
        experiment.crystal.get_unit_cell(),
        experiment.crystal.get_space_group().type(),
        dmin,
    ).to_array()


@pytest.fixture
def realistic_model():
    """A still whose mosaic parameters accept a small fraction of the indices."""
    return Model(
        test_nave_model=True,
        half_mosaicity_deg=0.05,
        domain_size_ang=3000,
        cell_edge=100,
    )


@pytest.mark.parametrize("kind", ["nave", "delta_psi", "spherical_relp"])
def test_for_ub_matches_predicting_every_index(realistic_model, kind):
    dmin = 4.0
    experiment = realistic_model.experiment
    indices = _generated_indices(experiment, dmin)
    predictor = _stills_predictors(experiment, dmin)[kind]

    reference = _predict_then_select(predictor, kind, experiment, indices)
    assert len(reference) > 0

    _assert_tables_identical(reference, predictor.for_ub(experiment.crystal.get_A()))


def _ewald_offsets(experiment, indices, ub=None):
    """q.q + 2 q.s0 for every index, the quantity the prefilter tests."""
    from dials.array_family import flex

    ub = matrix.sqr(experiment.crystal.get_A()) if ub is None else ub
    s0 = matrix.col(experiment.beam.get_s0())
    q = flex.vec3_double([tuple(ub * matrix.col(h)) for h in indices])
    return q.dot(q) + 2.0 * q.dot(flex.vec3_double(len(q), s0.elems))


@pytest.mark.parametrize("kind", ["nave", "delta_psi", "spherical_relp"])
def test_for_ub_cutoff_is_necessary_but_not_sufficient(realistic_model, kind):
    """Every prediction meets the Ewald offset cutoff, and most indices meeting it
    are still rejected by the delta psi test that follows."""
    from dials.array_family import flex

    dmin = 4.0
    experiment = realistic_model.experiment
    indices = _generated_indices(experiment, dmin)
    predictor = _stills_predictors(experiment, dmin)[kind]

    delta_psi_at_dmin = (
        dmin / experiment.crystal.get_domain_size_ang()
        + radians(experiment.crystal.get_half_mosaicity_deg()) / 2.0
        if kind == "nave"
        else 0.0015
    )
    s0_length = matrix.col(experiment.beam.get_s0()).length()
    eps_cut = 2.0 * 2.0 * s0_length * delta_psi_at_dmin / dmin

    predicted = predictor.for_ub(experiment.crystal.get_A())
    assert len(predicted) > 0
    assert (
        flex.max(flex.abs(_ewald_offsets(experiment, predicted["miller_index"])))
        <= eps_cut
    )

    passes_cut = flex.abs(_ewald_offsets(experiment, indices)) <= eps_cut
    assert passes_cut.count(True) > len(predicted)
    assert passes_cut.count(True) < len(indices) // 10


def test_for_ub_does_not_filter_a_ub_that_disagrees_with_the_unit_cell(realistic_model):
    """The cutoff assumes |ub h| is 1 / d(h); where it is not, nothing is skipped."""
    from dials.algorithms.spot_prediction import NaveStillsReflectionPredictor

    dmin = 4.0
    experiment = realistic_model.experiment
    crystal = experiment.crystal
    indices = _generated_indices(experiment, dmin)
    scaled = matrix.sqr(crystal.get_A()) * 1.05

    mismatched = NaveStillsReflectionPredictor(
        experiment.beam,
        experiment.detector,
        scaled,
        crystal.get_unit_cell(),
        crystal.get_space_group().type(),
        dmin,
        crystal.get_half_mosaicity_deg(),
        crystal.get_domain_size_ang(),
    )

    reference = _predict_then_select(mismatched, "nave", experiment, indices)
    assert len(reference) > 0
    _assert_tables_identical(reference, mismatched.for_ub(scaled))


def test_for_ub_below_the_ewald_limit_still_raises():
    """Indices past |q| = 2|s0| have no ray, and asking for them is an error."""
    model = Model(
        test_nave_model=True,
        half_mosaicity_deg=0.05,
        domain_size_ang=3000,
        cell_edge=10,
    )
    experiment = model.experiment
    dmin = 0.4  # below half the 1.0 A wavelength
    predictor = _stills_predictors(experiment, dmin)["nave"]

    with pytest.raises(RuntimeError):
        predictor.for_ub(experiment.crystal.get_A())

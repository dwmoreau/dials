from __future__ import annotations

import math
from random import randint, seed

import numpy as np

from dxtbx.model.experiment_list import ExperimentList
from scitbx import matrix

from dials.algorithms.profile_model.gaussian_rs import (
    CoordinateSystem,
    MaskCalculator3D,
)
from dials.algorithms.shoebox import MaskCode
from dials.array_family import flex


def test(dials_data):
    experiment = ExperimentList.from_file(
        dials_data("centroid_test_data") / "experiments.json"
    )

    beam = experiment[0].beam
    detector = experiment[0].detector
    goniometer = experiment[0].goniometer
    scan = experiment[0].scan
    delta_b = experiment[0].profile.delta_b()
    delta_m = experiment[0].profile.delta_m()

    assert len(detector) == 1

    # Get the function object to mask the foreground
    mask_foreground = MaskCalculator3D(
        beam, detector, goniometer, scan, delta_b, delta_m
    )

    s0 = beam.get_s0()
    m2 = goniometer.get_rotation_axis()
    s0_length = matrix.col(beam.get_s0()).length()
    width, height = detector[0].get_image_size()
    zrange = scan.get_array_range()
    phi0, dphi = scan.get_oscillation(deg=False)

    # Generate some reflections
    reflections = generate_reflections(detector, beam, scan, experiment, 10)

    # Mask the foreground in each
    mask_foreground(
        reflections["shoebox"],
        reflections["s1"],
        reflections["xyzcal.px"].parts()[2],
        reflections["panel"],
    )

    # Loop through all the reflections and check the mask values
    shoebox = reflections["shoebox"]
    beam_vector = reflections["s1"]
    rotation_angle = reflections["xyzcal.mm"].parts()[2]
    for l in range(len(reflections)):
        mask = shoebox[l].mask
        x0, x1, y0, y1, z0, z1 = shoebox[l].bbox
        s1 = beam_vector[l]
        phi = rotation_angle[l]
        cs = CoordinateSystem(m2, s0, s1, phi)

        def rs_coord(i, j, k):
            s1d = detector[0].get_pixel_lab_coord((i, j))
            s1d = matrix.col(s1d).normalize() * s0_length
            e1, e2 = cs.from_beam_vector(s1d)
            e3 = cs.from_rotation_angle_fast(phi0 + (k - zrange[0]) * dphi)
            return e1, e2, e3

        new_mask = flex.int(mask.accessor(), 0)
        for k in range(z1 - z0):
            for j in range(y1 - y0):
                for i in range(x1 - x0):
                    # value1 = mask[k, j, i]
                    e11, e12, e13 = rs_coord(x0 + i, y0 + j, z0 + k)
                    e21, e22, e23 = rs_coord(x0 + i + 1, y0 + j, z0 + k)
                    e31, e32, e33 = rs_coord(x0 + i, y0 + j + 1, z0 + k)
                    e41, e42, e43 = rs_coord(x0 + i, y0 + j, z0 + k + 1)
                    e51, e52, e53 = rs_coord(x0 + i + 1, y0 + j + 1, z0 + k)
                    e61, e62, e63 = rs_coord(x0 + i + 1, y0 + j, z0 + k + 1)
                    e71, e72, e73 = rs_coord(x0 + i, y0 + j + 1, z0 + k + 1)
                    e81, e82, e83 = rs_coord(x0 + i + 1, y0 + j + 1, z0 + k + 1)
                    de1 = (e11 / delta_b) ** 2 + (
                        e12 / delta_b
                    ) ** 2  # +(e13/delta_m)**2
                    de2 = (e21 / delta_b) ** 2 + (
                        e22 / delta_b
                    ) ** 2  # +(e23/delta_m)**2
                    de3 = (e31 / delta_b) ** 2 + (
                        e32 / delta_b
                    ) ** 2  # +(e33/delta_m)**2
                    de4 = (e41 / delta_b) ** 2 + (
                        e42 / delta_b
                    ) ** 2  # +(e43/delta_m)**2
                    de5 = (e51 / delta_b) ** 2 + (
                        e52 / delta_b
                    ) ** 2  # +(e53/delta_m)**2
                    de6 = (e61 / delta_b) ** 2 + (
                        e62 / delta_b
                    ) ** 2  # +(e63/delta_m)**2
                    de7 = (e71 / delta_b) ** 2 + (
                        e72 / delta_b
                    ) ** 2  # +(e73/delta_m)**2
                    de8 = (e81 / delta_b) ** 2 + (
                        e82 / delta_b
                    ) ** 2  # +(e83/delta_m)**2
                    de = math.sqrt(min([de1, de2, de3, de4, de5, de6, de7, de8]))
                    if (
                        x0 + i < 0
                        or y0 + j < 0
                        or x0 + i >= width
                        or y0 + j >= height
                        or z0 + k < zrange[0]
                        or z0 + k >= zrange[1]
                    ):
                        value2 = MaskCode.Valid
                    else:
                        if de <= 1.0:
                            value2 = MaskCode.Valid | MaskCode.Foreground
                        else:
                            value2 = MaskCode.Valid | MaskCode.Background
                    new_mask[k, j, i] = value2

        if not all(m1 == m2 for m1, m2 in zip(mask, new_mask)):
            np.set_printoptions(threshold=10000)
            diff = (mask == new_mask).as_numpy_array()
            print(diff.astype(int))
            # print mask.as_numpy_array()
            # print new_mask.as_numpy_array()
            # print (new_mask.as_numpy_array()[:,:,:] %2) * (new_mask.as_numpy_array() == 5)
            assert False


def generate_reflections(detector, beam, scan, experiment, num):
    seed(0)
    assert len(detector) == 1
    beam_vector = flex.vec3_double(num)
    xyzcal_px = flex.vec3_double(num)
    xyzcal_mm = flex.vec3_double(num)
    panel = flex.size_t(num)
    s0_length = matrix.col(beam.get_s0()).length()
    for i in range(num):
        x = randint(0, 2000)
        y = randint(0, 2000)
        z = randint(0, 8)
        s1 = detector[0].get_pixel_lab_coord((x, y))
        s1 = matrix.col(s1).normalize() * s0_length
        phi = scan.get_angle_from_array_index(z, deg=False)
        beam_vector[i] = s1
        xyzcal_px[i] = (x, y, z)
        (x, y) = detector[0].pixel_to_millimeter((x, y))
        xyzcal_mm[i] = (x, y, phi)
        panel[i] = 0

    rlist = flex.reflection_table()
    rlist["id"] = flex.int(len(beam_vector), 0)
    rlist["s1"] = beam_vector
    rlist["panel"] = panel
    rlist["xyzcal.px"] = xyzcal_px
    rlist["xyzcal.mm"] = xyzcal_mm
    rlist["bbox"] = rlist.compute_bbox(experiment)
    index = []
    image_size = experiment[0].detector[0].get_image_size()
    array_range = experiment[0].scan.get_array_range()
    bbox = rlist["bbox"]
    for i in range(len(rlist)):
        x0, x1, y0, y1, z0, z1 = bbox[i]
        if (
            x0 < 0
            or x1 > image_size[0]
            or y0 < 0
            or y1 > image_size[1]
            or z0 < array_range[0]
            or z1 > array_range[1]
        ):
            index.append(i)
    rlist.del_selected(flex.size_t(index))
    rlist["shoebox"] = flex.shoebox(rlist["panel"], rlist["bbox"])
    rlist["shoebox"].allocate_data_with_value(MaskCode.Valid)
    return rlist


def _stills_detector_and_beam():
    from dxtbx.model import ParallaxCorrectedPxMmStrategy
    from dxtbx.model.beam import BeamFactory
    from dxtbx.model.detector import DetectorFactory

    beam = BeamFactory.simple(wavelength=1)
    detector = DetectorFactory.simple(
        sensor=DetectorFactory.sensor("PAD"),
        distance=150,
        beam_centre=[50, 50],
        fast_direction="+x",
        slow_direction="+y",
        pixel_size=[0.1, 0.1],
        image_size=[1000, 1000],
    )
    for panel in detector:
        panel.set_px_mm_strategy(ParallaxCorrectedPxMmStrategy(0.5, 0.32))
    return detector, beam


def _stills_shoeboxes(detector, beam):
    """Shoeboxes of deliberately mixed sizes, including the 1x1 degenerate case."""
    from dials.algorithms.shoebox import MaskCode

    s0_length = matrix.col(beam.get_s0()).length()
    sizes = [(1, 1), (1, 7), (7, 1), (2, 2), (5, 9), (12, 3), (9, 9), (20, 15)]
    bbox = flex.int6()
    s1 = flex.vec3_double()
    for index, (width, height) in enumerate(sizes):
        x0 = 200 + 37 * index
        y0 = 150 + 53 * index
        bbox.append((x0, x0 + width, y0, y0 + height, 0, 1))
        centre = detector[0].get_pixel_lab_coord((x0 + 0.5 * width, y0 + 0.5 * height))
        s1.append(tuple(matrix.col(centre).normalize() * s0_length))

    panel = flex.size_t(len(bbox), 0)
    shoeboxes = flex.shoebox(panel, bbox, allocate=True)
    shoeboxes.allocate_data_with_value(MaskCode.Valid)
    return shoeboxes, s1, panel


def _reference_dxy(detector, beam, bbox, s1, delta_b):
    """The corner loop written out, converting one coordinate at a time."""
    from dials.algorithms.profile_model.gaussian_rs import CoordinateSystem2d

    s0 = beam.get_s0()
    s0_length = matrix.col(s0).length()
    cs = CoordinateSystem2d(s0, s1)
    x0, x1, y0, y1, _, _ = bbox
    xsize, ysize = x1 - x0, y1 - y0

    dxy = {}
    for j in range(ysize + 1):
        for i in range(xsize + 1):
            lab = detector[0].get_pixel_lab_coord((x0 + i, y0 + j))
            s_dash = matrix.col(lab).normalize() * s0_length
            e1, e2 = cs.from_beam_vector(tuple(s_dash))
            dxy[j, i] = (e1 * e1 + e2 * e2) / (delta_b * delta_b)
    return dxy


def _reference_mask(detector, beam, shoebox, s1, delta_b):
    """The stills mask, from the reference corner values."""
    from dials.algorithms.shoebox import MaskCode

    dxy = _reference_dxy(detector, beam, shoebox.bbox, s1, delta_b)
    x0, x1, y0, y1, _, _ = shoebox.bbox
    xsize, ysize = x1 - x0, y1 - y0

    mask = flex.int(shoebox.mask.accessor(), MaskCode.Valid)
    for j in range(ysize):
        for i in range(xsize):
            value = min(dxy[j, i], dxy[j + 1, i], dxy[j, i + 1], dxy[j + 1, i + 1])
            mask[0, j, i] |= (
                MaskCode.Foreground if value <= 1.0 else MaskCode.Background
            )
    return mask


def test_mask_calculator_2d_matches_single_coordinate_conversion():
    from dials.algorithms.profile_model.gaussian_rs import MaskCalculator2D

    detector, beam = _stills_detector_and_beam()
    shoeboxes, s1, panel = _stills_shoeboxes(detector, beam)
    delta_b = 0.02

    MaskCalculator2D(beam, detector, delta_b, 0.0)(
        shoeboxes, s1, flex.double(len(shoeboxes), 0), panel
    )

    for index, shoebox in enumerate(shoeboxes):
        reference = _reference_mask(detector, beam, shoebox, s1[index], delta_b)
        assert list(shoebox.mask) == list(reference), shoebox.bbox

    foreground = sum(
        int(value & MaskCode.Foreground != 0) for s in shoeboxes for value in s.mask
    )
    assert foreground > 0


def test_mask_calculator_2d_array_matches_one_shoebox_at_a_time():
    from dials.algorithms.profile_model.gaussian_rs import MaskCalculator2D

    detector, beam = _stills_detector_and_beam()
    delta_b = 0.02
    calculator = MaskCalculator2D(beam, detector, delta_b, 0.0)

    batched, s1, panel = _stills_shoeboxes(detector, beam)
    calculator(batched, s1, flex.double(len(batched), 0), panel)

    one_at_a_time, _, _ = _stills_shoeboxes(detector, beam)
    for index, shoebox in enumerate(one_at_a_time):
        calculator(shoebox, s1[index], 0.0, panel[index], False)

    for a, b in zip(batched, one_at_a_time):
        assert list(a.mask) == list(b.mask), a.bbox


def test_mask_calculator_2d_volume_matches_single_coordinate_conversion():
    """The image volume path shares the corner loop but not the stills_process path."""
    from dials.algorithms.profile_model.gaussian_rs import MaskCalculator2D
    from dials.model.data import ImageVolume, MultiPanelImageVolume

    detector, beam = _stills_detector_and_beam()
    delta_b = 0.02
    width, height = detector[0].get_image_size()

    # The second box straddles the panel edge, so its fraction is not trivially zero.
    bbox = flex.int6([(200, 212, 150, 162, 0, 1), (-4, 8, 300, 312, 0, 1)])
    s0_length = matrix.col(beam.get_s0()).length()
    s1 = flex.vec3_double()
    for x0, x1, y0, y1, _, _ in bbox:
        centre = detector[0].get_pixel_lab_coord((0.5 * (x0 + x1), 0.5 * (y0 + y1)))
        s1.append(tuple(matrix.col(centre).normalize() * s0_length))
    panel = flex.size_t(len(bbox), 0)

    volume = MultiPanelImageVolume()
    volume.add(ImageVolume(0, 1, height, width))

    fraction = MaskCalculator2D(beam, detector, delta_b, 0.0)(
        volume, bbox, s1, flex.double(len(bbox), 0), panel
    )

    for index in range(len(bbox)):
        x0, x1, y0, y1, _, _ = bbox[index]
        dxy = _reference_dxy(detector, beam, bbox[index], s1[index], delta_b)
        inside = outside_foreground = 0
        for j in range(y1 - y0):
            for i in range(x1 - x0):
                value = min(dxy[j, i], dxy[j + 1, i], dxy[j, i + 1], dxy[j + 1, i + 1])
                if 0 <= y0 + j < height and 0 <= x0 + i < width:
                    inside += 1
                elif value <= 1.0:
                    outside_foreground += 1
        assert fraction[index] == outside_foreground / (inside + outside_foreground)

    assert fraction[0] == 0.0
    assert fraction[1] > 0.0

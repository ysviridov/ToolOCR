import numpy as np
import pytest

from ocr.app.camera_calibration import (
    CameraCalibrationError,
    build_plane_calibration,
    measure_quad_mm_consensus,
    select_calibrations_for_frame,
)
from ocr.app.gost_r_51506_99 import ENVELOPE_SPECS, GOST_ID, EnvelopeFormat


def _calibration(fmt, image_w, image_h, quad):
    return build_plane_calibration(
        np.asarray(quad, dtype=np.float32),
        image_width_px=image_w,
        image_height_px=image_h,
        spec=ENVELOPE_SPECS[fmt],
        standard=GOST_ID,
    )


def test_multifov_set_selects_only_compatible_calibration():
    c5 = _calibration(
        EnvelopeFormat.C5,
        1600,
        1200,
        [[180, 180], [1420, 180], [1420, 1030], [180, 1030]],
    )
    dl = _calibration(
        EnvelopeFormat.DL,
        1000,
        1100,
        [[100, 300], [900, 300], [900, 700], [100, 700]],
    )

    selected = select_calibrations_for_frame(
        (c5, dl),
        image_width_px=1000,
        image_height_px=1100,
    )
    assert [item.reference_format for item in selected] == ["DL"]

    consensus = measure_quad_mm_consensus(
        (c5, dl),
        np.asarray([[100, 300], [900, 300], [900, 700], [100, 700]], dtype=np.float32),
        image_width_px=1000,
        image_height_px=1100,
    )
    assert consensus.reference_formats == ("DL",)
    assert abs(consensus.measurement.width_mm - 220.0) < 0.5
    assert abs(consensus.measurement.height_mm - 110.0) < 0.5


def test_multifov_set_rejects_frame_without_matching_geometry():
    c5 = _calibration(
        EnvelopeFormat.C5,
        1600,
        1200,
        [[180, 180], [1420, 180], [1420, 1030], [180, 1030]],
    )

    with pytest.raises(CameraCalibrationError, match="Нет калибровки для FOV"):
        measure_quad_mm_consensus(
            (c5,),
            np.asarray([[100, 300], [900, 300], [900, 700], [100, 700]], dtype=np.float32),
            image_width_px=1000,
            image_height_px=1100,
        )

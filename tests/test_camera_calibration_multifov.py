import numpy as np

from ocr.app.camera_calibration import (
    build_plane_calibration,
    measure_quad_mm_consensus,
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


def _axis_aligned_quad(x, y, width, height):
    return np.asarray(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        dtype=np.float32,
    )


def test_different_jpeg_crops_with_same_pixel_scale_use_all_calibrations():
    # Два эталона сняты с разным размером полного JPEG, но само содержимое имеет
    # одинаковый масштаб 4 px/mm. Это модель разного количества черного фона.
    c5 = _calibration(
        EnvelopeFormat.C5,
        1600,
        1200,
        _axis_aligned_quad(250, 180, 229 * 4, 162 * 4),
    )
    dl = _calibration(
        EnvelopeFormat.DL,
        1180,
        1100,
        _axis_aligned_quad(120, 420, 220 * 4, 110 * 4),
    )

    runtime_dl = _axis_aligned_quad(80, 500, 220 * 4, 110 * 4)
    consensus = measure_quad_mm_consensus(
        (c5, dl),
        runtime_dl,
        # Намеренно третий размер полного JPEG: metric mode его игнорирует.
        image_width_px=1370,
        image_height_px=1100,
    )

    assert set(consensus.reference_formats) == {"C5", "DL"}
    assert consensus.consistent is True
    assert consensus.width_spread_mm < 0.1
    assert consensus.height_spread_mm < 0.1
    assert abs(consensus.measurement.width_mm - 220.0) < 0.1
    assert abs(consensus.measurement.height_mm - 110.0) < 0.1


def test_real_upstream_resize_is_detected_as_inconsistent_scale():
    # Здесь DL был реально resize-нут до 3 px/mm, а C5 остался 4 px/mm.
    # Это уже не изменение черного crop и такой набор нельзя молча усреднять.
    c5 = _calibration(
        EnvelopeFormat.C5,
        1600,
        1200,
        _axis_aligned_quad(250, 180, 229 * 4, 162 * 4),
    )
    dl_resized = _calibration(
        EnvelopeFormat.DL,
        1000,
        900,
        _axis_aligned_quad(140, 300, 220 * 3, 110 * 3),
    )

    runtime_c5 = _axis_aligned_quad(100, 250, 229 * 4, 162 * 4)
    consensus = measure_quad_mm_consensus(
        (c5, dl_resized),
        runtime_c5,
        image_width_px=1500,
        image_height_px=1200,
    )

    assert consensus.consistent is False
    assert consensus.width_spread_mm > 12.0 or consensus.height_spread_mm > 12.0

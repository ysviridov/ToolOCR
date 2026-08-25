import cv2
import numpy as np
import pytest

from ocr.app.camera_calibration import (
    CameraCalibrationError,
    build_plane_calibration,
    match_format_by_metric,
    measure_quad_mm,
)
from ocr.app.gost_r_51506_99 import ENVELOPE_SPECS, GOST_ID, EnvelopeFormat


IMAGE_W = 1600
IMAGE_H = 1200


def _calibration_c4():
    quad = np.array(
        [
            [200.0, 180.0],
            [1400.0, 210.0],
            [1360.0, 1050.0],
            [180.0, 1020.0],
        ],
        dtype=np.float32,
    )
    return build_plane_calibration(
        quad,
        image_width_px=IMAGE_W,
        image_height_px=IMAGE_H,
        spec=ENVELOPE_SPECS[EnvelopeFormat.C4],
        standard=GOST_ID,
    )


def _world_rect_to_image(calibration, x_mm, y_mm, width_mm, height_mm, *, scale=1.0):
    world = np.array(
        [
            [x_mm, y_mm],
            [x_mm + width_mm, y_mm],
            [x_mm + width_mm, y_mm + height_mm],
            [x_mm, y_mm + height_mm],
        ],
        dtype=np.float64,
    )
    inverse = np.linalg.inv(calibration.homography_norm_to_mm)
    normalized = cv2.perspectiveTransform(
        world.reshape(1, 4, 2),
        inverse,
    )[0]
    result = normalized.copy()
    result[:, 0] *= IMAGE_W * scale
    result[:, 1] *= IMAGE_H * scale
    return result.astype(np.float32)


def test_metric_calibration_distinguishes_c5_from_c4_and_b4():
    calibration = _calibration_c4()
    c5 = ENVELOPE_SPECS[EnvelopeFormat.C5]
    quad = _world_rect_to_image(calibration, 40.0, 30.0, c5.width_mm, c5.height_mm)

    measurement = measure_quad_mm(
        calibration,
        quad,
        image_width_px=IMAGE_W,
        image_height_px=IMAGE_H,
    )
    decision = match_format_by_metric(measurement)

    assert abs(measurement.width_mm - 229.0) < 0.5
    assert abs(measurement.height_mm - 162.0) < 0.5
    assert decision.status == "resolved"
    assert decision.format == EnvelopeFormat.C5
    assert decision.margin > 0.5


def test_partial_bottom_c4_is_resolved_by_full_width():
    calibration = _calibration_c4()
    # Нижняя физическая часть C4 не попала в кадр: наблюдаем только 180 мм
    # высоты, но полная ширина 324 мм сохранилась.
    quad = _world_rect_to_image(calibration, 0.0, 0.0, 324.0, 180.0)

    measurement = measure_quad_mm(
        calibration,
        quad,
        image_width_px=IMAGE_W,
        image_height_px=IMAGE_H,
        frame_contact_sides=("bottom",),
    )
    decision = match_format_by_metric(measurement)

    assert measurement.width_exact is True
    assert measurement.height_exact is False
    assert abs(measurement.width_mm - 324.0) < 0.5
    assert decision.status == "resolved"
    assert decision.format == EnvelopeFormat.C4


def test_same_fov_at_double_resolution_keeps_metric_size():
    calibration = _calibration_c4()
    c5 = ENVELOPE_SPECS[EnvelopeFormat.C5]
    quad = _world_rect_to_image(
        calibration,
        40.0,
        30.0,
        c5.width_mm,
        c5.height_mm,
        scale=2.0,
    )

    measurement = measure_quad_mm(
        calibration,
        quad,
        image_width_px=IMAGE_W * 2,
        image_height_px=IMAGE_H * 2,
    )

    assert abs(measurement.width_mm - 229.0) < 0.5
    assert abs(measurement.height_mm - 162.0) < 0.5


def test_changed_frame_aspect_ratio_rejects_calibration():
    calibration = _calibration_c4()
    quad = _world_rect_to_image(calibration, 0.0, 0.0, 229.0, 162.0)

    with pytest.raises(CameraCalibrationError):
        measure_quad_mm(
            calibration,
            quad,
            image_width_px=1600,
            image_height_px=1000,
        )

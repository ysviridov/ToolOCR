import cv2
import numpy as np

from ocr.app.gost_r_51506_99 import EnvelopeFormat, candidate_formats_by_aspect_ratio
from ocr.app.layout import detect_envelope_quad, rectify_envelope
from ocr.app.profiles import DOMESTIC_PROFILES


def _synthetic_envelope() -> np.ndarray:
    image = np.zeros((1000, 1400, 3), dtype=np.uint8)
    quad = np.array(
        [
            [180, 140],
            [1210, 170],
            [1170, 870],
            [150, 850],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(image, quad, (245, 245, 245))

    # Несколько внутренних линий не должны перехватить роль внешнего контура.
    cv2.line(image, (620, 400), (1080, 410), (80, 80, 80), 5)
    cv2.line(image, (620, 470), (1080, 480), (80, 80, 80), 5)
    return image


def test_detects_external_quad_before_internal_lines():
    image = _synthetic_envelope()
    detection = detect_envelope_quad(image)

    assert detection.method == "contour_approx"
    assert detection.confidence > 0.70
    assert detection.area_ratio > 0.40
    assert detection.rectangularity > 0.85
    assert detection.points.shape == (4, 2)


def test_rectification_returns_landscape_envelope():
    image = _synthetic_envelope()
    detection = detect_envelope_quad(image)
    rectified = rectify_envelope(image, detection.points)

    assert rectified.width_px > rectified.height_px
    assert rectified.width_px > 900
    assert rectified.height_px > 600


def test_180_degree_input_keeps_same_format_candidates():
    image = _synthetic_envelope()
    rotated = cv2.rotate(image, cv2.ROTATE_180)

    first = rectify_envelope(image, detect_envelope_quad(image).points)
    second = rectify_envelope(rotated, detect_envelope_quad(rotated).points)

    first_formats = {
        item.format
        for item in candidate_formats_by_aspect_ratio(
            first.width_px,
            first.height_px,
            max_relative_error=0.08,
        )
    }
    second_formats = {
        item.format
        for item in candidate_formats_by_aspect_ratio(
            second.width_px,
            second.height_px,
            max_relative_error=0.08,
        )
    }

    assert first_formats == second_formats
    assert EnvelopeFormat.C6 in first_formats


def test_domestic_profile_catalog_contains_all_gost_format_layout_pairs():
    assert len(DOMESTIC_PROFILES) == 16

    by_format = {}
    for profile in DOMESTIC_PROFILES:
        by_format.setdefault(profile.format, []).append(profile)

    assert len(by_format[EnvelopeFormat.C6]) == 4
    assert len(by_format[EnvelopeFormat.DL]) == 4
    assert len(by_format[EnvelopeFormat.C5]) == 4
    assert len(by_format[EnvelopeFormat.C4]) == 2
    assert len(by_format[EnvelopeFormat.B4]) == 2

    assert all(not profile.window for profile in by_format[EnvelopeFormat.C4])
    assert all(not profile.window for profile in by_format[EnvelopeFormat.B4])

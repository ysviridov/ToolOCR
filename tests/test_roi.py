import cv2
import numpy as np

from ocr.app.gost_r_51506_99 import EnvelopeFormat
from ocr.app.roi import canonicalize_rectified, detect_simple_mail_rois


def _draw_postcode_stencil(
    image: np.ndarray,
    *,
    x: int,
    y: int,
    bar_width: int = 42,
    include_start_marker: bool = True,
    bar_count: int = 7,
) -> None:
    """Синтетический '= + 6 цифр' для regression-тестов ROI."""

    bar_height = max(8, round(bar_width * 0.28))
    step = round(bar_width * 1.25)

    for index in range(bar_count):
        bx = x + index * step
        cv2.rectangle(image, (bx, y), (bx + bar_width, y + bar_height), (0, 0, 0), -1)

    if include_start_marker:
        cv2.rectangle(
            image,
            (x, y + round(bar_height * 1.35)),
            (x + bar_width, y + round(bar_height * 1.85)),
            (0, 0, 0),
            -1,
        )

    digit_count = max(0, min(6, bar_count - 1))
    for index in range(digit_count):
        bx = x + (index + 1) * step + round(bar_width * 0.10)
        top = y + round(bar_width * 0.75)
        bottom = y + round(bar_width * 2.45)
        cv2.rectangle(image, (bx, top), (bx + round(bar_width * 0.70), bottom), (0, 0, 0), 3)
        cv2.line(image, (bx + 4, bottom - 5), (bx + round(bar_width * 0.62), top + 5), (0, 0, 0), 3)


def _postcode_region(image: np.ndarray, envelope_format: EnvelopeFormat):
    result = detect_simple_mail_rois(image, envelope_format)
    return next(item for item in result.regions if item.kind == "recipient_postcode")


def test_canonicalization_rotates_resolved_180():
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    image[2:8, 4:12] = (10, 120, 240)

    result = canonicalize_rectified(
        image,
        orientation_status="resolved",
        orientation_deg=180,
    )

    assert result.reliable is True
    assert result.status == "canonical"
    assert result.rotation_applied_deg == 180
    assert np.array_equal(result.image, cv2.rotate(image, cv2.ROTATE_180))


def test_canonicalization_does_not_guess_ambiguous_orientation():
    image = np.full((30, 60, 3), 255, dtype=np.uint8)

    result = canonicalize_rectified(
        image,
        orientation_status="ambiguous",
        orientation_deg=None,
    )

    assert result.reliable is False
    assert result.status == "orientation_unresolved"
    assert result.rotation_applied_deg == 0
    assert result.image is image


def test_dl_simple_mail_roi_detects_postcode_stencil_strict():
    image = np.full((700, 1400, 3), 235, dtype=np.uint8)

    for row, width in enumerate((360, 390, 330, 280)):
        y = 260 + row * 44
        for x in range(760, 760 + width, 38):
            cv2.rectangle(image, (x, y), (x + 20, y + 20), (20, 20, 20), -1)

    _draw_postcode_stencil(image, x=55, y=555, bar_width=38)
    postcode = _postcode_region(image, EnvelopeFormat.DL)

    assert postcode.status == "stencil_detected"
    assert postcode.detector == "postcode_stencil"
    assert postcode.detected_bbox is not None
    assert postcode.component_count == 7
    assert postcode.features is not None
    assert postcode.features["digit_count"] == 6
    assert postcode.features["start_marker_score"] >= 0.4
    assert postcode.features["confirmation_mode"] == "strict_start_marker"
    assert postcode.features["rejection_reason"] is None


def test_c4_postcode_stencil_is_found_close_to_left_and_bottom_edges():
    image = np.full((900, 1300, 3), 205, dtype=np.uint8)

    gradient = np.linspace(0, 35, image.shape[1], dtype=np.uint8)
    image[:] = np.clip(image.astype(np.int16) - gradient[None, :, None], 0, 255).astype(np.uint8)

    _draw_postcode_stencil(image, x=8, y=760, bar_width=42)
    postcode = _postcode_region(image, EnvelopeFormat.C4)

    assert postcode.status == "stencil_detected"
    assert postcode.detected_bbox is not None
    assert postcode.detected_bbox.x <= 20
    assert postcode.detected_bbox.y >= 720
    assert postcode.confidence >= 0.78
    assert postcode.features is not None
    assert postcode.features["bar_count"] == 7
    assert postcode.features["row_y_norm"] >= 0.70


def test_seven_bar_rescue_accepts_full_regular_row_without_start_marker():
    image = np.full((900, 1300, 3), 225, dtype=np.uint8)
    _draw_postcode_stencil(
        image,
        x=42,
        y=760,
        bar_width=42,
        include_start_marker=False,
    )

    postcode = _postcode_region(image, EnvelopeFormat.C4)

    assert postcode.status == "stencil_detected"
    assert postcode.detected_bbox is not None
    assert postcode.features is not None
    assert postcode.features["confirmation_mode"] == "seven_bar_rescue"
    assert postcode.features["bar_count"] == 7
    assert postcode.features["start_marker_score"] < 0.40
    assert postcode.features["width_cv"] <= 0.12
    assert postcode.features["spacing_error"] <= 0.08
    assert postcode.features["alignment_error"] <= 0.55
    assert postcode.features["row_y_norm"] >= 0.70


def test_seven_bar_rescue_rejects_regular_row_too_high_in_letter():
    image = np.full((900, 1300, 3), 225, dtype=np.uint8)
    _draw_postcode_stencil(
        image,
        x=42,
        y=500,
        bar_width=42,
        include_start_marker=False,
    )

    postcode = _postcode_region(image, EnvelopeFormat.C4)

    assert postcode.status == "stencil_not_found"
    assert postcode.detected_bbox is None
    assert postcode.features is not None
    assert postcode.features["confirmation_mode"] == "none"
    assert postcode.features["rejection_reason"] == "row_not_low_enough"


def test_seven_bar_rescue_does_not_accept_six_bars_without_start_marker():
    image = np.full((900, 1300, 3), 225, dtype=np.uint8)
    _draw_postcode_stencil(
        image,
        x=42,
        y=760,
        bar_width=42,
        include_start_marker=False,
        bar_count=6,
    )

    postcode = _postcode_region(image, EnvelopeFormat.C4)

    assert postcode.status == "stencil_not_found"
    assert postcode.detected_bbox is None
    assert postcode.features is not None
    assert postcode.features["confirmation_mode"] == "none"
    assert postcode.features["rejection_reason"] == "insufficient_top_bars"


def test_postcode_does_not_fallback_to_arbitrary_dark_content():
    image = np.full((700, 1400, 3), 230, dtype=np.uint8)

    for x in range(80, 360, 12):
        cv2.rectangle(image, (x, 510), (x + 4, 620), (0, 0, 0), -1)
    cv2.putText(image, "POST 123456", (90, 665), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    postcode = _postcode_region(image, EnvelopeFormat.C4)

    assert postcode.status == "stencil_not_found"
    assert postcode.detected_bbox is None
    assert postcode.detector == "postcode_stencil"


def test_roi_unsupported_format_remains_explicit():
    image = np.full((500, 1000, 3), 255, dtype=np.uint8)

    result = detect_simple_mail_rois(image, EnvelopeFormat.B4)

    assert result.status == "unsupported_format"
    assert result.regions == ()

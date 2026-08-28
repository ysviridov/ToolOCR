import cv2
import numpy as np

from ocr.app.postcode_digit_cells import PostcodeDigitCell, PostcodeDigitGeometry
from ocr.app.postcode_recognizer import (
    DigitRecognition,
    _normalize_digit_crop,
    _normalize_digit_crop_with_debug,
    _suppress_stencil_dots,
    postcode_recognition_to_dict,
    recognize_postcode_digits,
)
from ocr.app.roi import PixelRect


def _geometry() -> PostcodeDigitGeometry:
    cells = tuple(
        PostcodeDigitCell(
            index=index,
            bbox=PixelRect(10 + (index - 1) * 20, 10, 18, 40),
            center_x_px=19 + (index - 1) * 20,
        )
        for index in range(1, 7)
    )
    return PostcodeDigitGeometry(
        status="ready",
        reason=None,
        source="stencil_upper_bar_geometry",
        cells=cells,
        bar_width_px=16.0,
        bar_step_px=20.0,
        row_center_y_px=8.0,
        start_anchor_center_x_px=-1.0,
    )


def test_stencil_dot_suppression_removes_small_template_components():
    binary = np.full((80, 42), 255, dtype=np.uint8)

    # Рукописный штрих: крупная связная компонента.
    cv2.line(binary, (9, 12), (30, 52), 0, 3)
    cv2.line(binary, (30, 12), (30, 62), 0, 3)

    # Регулярные точки печатного шаблона.
    for y in range(8, 56, 10):
        for x in (7, 18, 29):
            cv2.rectangle(binary, (x, y), (x + 1, y + 1), 0, -1)

    cleaned, debug = _suppress_stencil_dots(binary)

    assert debug["status"] == "applied"
    assert debug["suppressed_components"] > 0
    assert debug["ink_pixels_after"] < debug["ink_pixels_before"]
    assert debug["suppressed_ink_ratio"] > 0

    # Основной рукописный штрих должен сохраниться.
    assert np.count_nonzero(cleaned[10:64, 24:34] < 128) > 20


def test_digit_normalization_produces_fixed_canvas_and_debug():
    image = np.full((80, 160, 3), 235, dtype=np.uint8)
    cell = _geometry().cells[0]
    x1, y1 = cell.bbox.x + 5, cell.bbox.y + 5
    image[y1:y1 + 28, x1:x1 + 7] = 20

    normalized, debug = _normalize_digit_crop_with_debug(image, cell)

    assert normalized is not None
    assert normalized.shape == (128, 96)
    assert normalized.dtype == np.uint8
    assert np.min(normalized) < 128
    assert np.max(normalized) == 255
    assert debug["method"] == "stencil_dot_suppression_v1"

    # Совместимый wrapper по-прежнему возвращает только canvas.
    wrapper = _normalize_digit_crop(image, cell)
    assert wrapper is not None
    assert wrapper.shape == (128, 96)


def _fake_preprocess(monkeypatch):
    monkeypatch.setattr(
        "ocr.app.postcode_recognizer._normalize_digit_crop_with_debug",
        lambda image, cell: (
            np.full((128, 96), 255, dtype=np.uint8),
            {
                "method": "stencil_dot_suppression_v1",
                "status": "applied",
                "suppressed_components": 12,
            },
        ),
    )


def test_recognizer_assembles_six_independent_digits(monkeypatch):
    expected = "123456"
    _fake_preprocess(monkeypatch)

    def fake_recognize(crop, index):
        return DigitRecognition(
            index=index,
            status="recognized",
            digit=expected[index - 1],
            confidence=0.90 + index * 0.01,
        )

    monkeypatch.setattr("ocr.app.postcode_recognizer._recognize_digit_crop", fake_recognize)

    result = recognize_postcode_digits(np.zeros((80, 160, 3), dtype=np.uint8), _geometry())

    assert result.status == "recognized"
    assert result.text == expected
    assert result.postcode == expected
    assert result.structurally_valid is True
    assert result.confidence == 0.935
    assert result.min_digit_confidence == 0.91
    assert [item.digit for item in result.digits] == list(expected)
    assert result.engine == "tesseract_single_digit+stencil_dot_suppression_v1"
    assert result.digits[0].preprocess["suppressed_components"] == 12

    payload = postcode_recognition_to_dict(result)
    assert payload["postcode"] == expected
    assert payload["digits"][0]["index"] == 1
    assert payload["digits"][-1]["index"] == 6
    assert payload["digits"][0]["preprocess"]["method"] == "stencil_dot_suppression_v1"


def test_recognizer_keeps_incomplete_result_visible(monkeypatch):
    expected = ["1", "2", None, "4", "5", "6"]
    _fake_preprocess(monkeypatch)

    def fake_recognize(crop, index):
        digit = expected[index - 1]
        if digit is None:
            return DigitRecognition(
                index=index,
                status="unrecognized",
                digit=None,
                confidence=None,
                reason="no_single_digit_candidate",
            )
        return DigitRecognition(index=index, status="recognized", digit=digit, confidence=0.9)

    monkeypatch.setattr("ocr.app.postcode_recognizer._recognize_digit_crop", fake_recognize)

    result = recognize_postcode_digits(np.zeros((80, 160, 3), dtype=np.uint8), _geometry())

    assert result.status == "incomplete"
    assert result.text == "12?456"
    assert result.postcode is None
    assert result.confidence is None
    assert result.structurally_valid is False
    assert result.reason == "one_or_more_digits_unrecognized"
    assert result.digits[2].preprocess["status"] == "applied"


def test_leading_zero_is_preserved_but_marked_structurally_invalid(monkeypatch):
    expected = "012345"
    _fake_preprocess(monkeypatch)

    def fake_recognize(crop, index):
        return DigitRecognition(
            index=index,
            status="recognized",
            digit=expected[index - 1],
            confidence=0.95,
        )

    monkeypatch.setattr("ocr.app.postcode_recognizer._recognize_digit_crop", fake_recognize)

    result = recognize_postcode_digits(np.zeros((80, 160, 3), dtype=np.uint8), _geometry())

    assert result.status == "recognized"
    assert result.postcode == expected
    assert result.text == expected
    assert result.structurally_valid is False

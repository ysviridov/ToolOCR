import numpy as np

from ocr.app.postcode_digit_cells import (
    PostcodeDigitCell,
    PostcodeDigitGeometry,
)
from ocr.app.postcode_preprocess_preview import append_postcode_preprocess_strip
from ocr.app.postcode_recognizer import DigitRecognition, PostcodeRecognition
from ocr.app.roi import PixelRect


def _geometry() -> PostcodeDigitGeometry:
    return PostcodeDigitGeometry(
        status="ready",
        reason=None,
        source="stencil_upper_bar_geometry",
        cells=tuple(
            PostcodeDigitCell(
                index=index,
                bbox=PixelRect(20 + (index - 1) * 40, 40, 34, 70),
                center_x_px=37 + (index - 1) * 40,
            )
            for index in range(1, 7)
        ),
        bar_width_px=30.0,
        bar_step_px=40.0,
        row_center_y_px=30.0,
        start_anchor_center_x_px=0.0,
    )


def _recognition() -> PostcodeRecognition:
    digits = tuple(
        DigitRecognition(
            index=index,
            status="recognized",
            digit=str(index % 10),
            confidence=0.9,
            preprocess={
                "method": "stencil_dot_suppression_v1",
                "status": "applied",
                "suppressed_components": 10 + index,
                "restored_components": 1,
                "suppressed_ink_ratio": 0.12,
            },
        )
        for index in range(1, 7)
    )
    return PostcodeRecognition(
        status="recognized",
        text="123456",
        postcode="123456",
        confidence=0.9,
        min_digit_confidence=0.9,
        structurally_valid=True,
        reason=None,
        engine="tesseract_single_digit+stencil_dot_suppression_v1",
        digits=digits,
    )


def test_preprocess_strip_appends_six_final_ocr_canvases(monkeypatch):
    source = np.full((300, 900, 3), 230, dtype=np.uint8)
    overlay = source.copy()

    calls = []

    def fake_preprocess(image, cell):
        calls.append(cell.index)
        canvas = np.full((128, 96), 255, dtype=np.uint8)
        canvas[20:110, 35:60] = 0
        return canvas, {
            "method": "stencil_dot_suppression_v1",
            "status": "applied",
            "suppressed_components": 12,
            "restored_components": 1,
            "suppressed_ink_ratio": 0.15,
        }

    monkeypatch.setattr(
        "ocr.app.postcode_preprocess_preview._normalize_digit_crop_with_debug",
        fake_preprocess,
    )

    preview = append_postcode_preprocess_strip(
        overlay,
        source,
        _geometry(),
        _recognition(),
    )

    assert calls == [1, 2, 3, 4, 5, 6]
    assert preview.shape[1] == overlay.shape[1]
    assert preview.shape[0] > overlay.shape[0]
    assert np.array_equal(preview[: overlay.shape[0]], overlay)
    assert np.min(preview[overlay.shape[0] :]) < 30
    assert np.max(preview[overlay.shape[0] :]) == 255


def test_preprocess_strip_does_not_change_image_without_digit_geometry():
    image = np.full((200, 400, 3), 230, dtype=np.uint8)
    geometry = PostcodeDigitGeometry(
        status="unavailable",
        reason="postcode_not_detected",
        source="stencil_upper_bar_geometry",
        cells=(),
    )

    result = append_postcode_preprocess_strip(
        image,
        image,
        geometry,
        _recognition(),
    )

    assert result is image

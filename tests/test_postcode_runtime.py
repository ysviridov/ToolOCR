import numpy as np

from ocr.app.postcode_digit_cells import PostcodeDigitCell, PostcodeDigitGeometry
from ocr.app.postcode_onnx import OnnxDigitPrediction, PostcodeOnnxError, predict_digit_onnx
from ocr.app.postcode_recognizer import DigitRecognition
from ocr.app.postcode_runtime import postcode_recognition_to_dict, recognize_postcode_digits
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


def _fake_preprocess(monkeypatch):
    monkeypatch.setattr(
        "ocr.app.postcode_runtime._normalize_digit_crop_with_debug",
        lambda image, cell: (
            np.full((128, 96), 255, dtype=np.uint8),
            {
                "method": "stencil_dot_suppression_v1",
                "status": "applied",
                "suppressed_components": 7,
            },
        ),
    )


def test_predict_digit_onnx_exposes_probabilities_and_top3():
    class FakeNet:
        def __init__(self):
            self.input_shape = None

        def setInput(self, value):
            self.input_shape = value.shape

        def forward(self):
            return np.asarray([[0.0, 0.2, 0.5, 0.1, -0.5, 0.3, 0.4, 4.0, 1.5, -1.0]], dtype=np.float32)

    net = FakeNet()
    prediction = predict_digit_onnx(
        np.full((128, 96), 255, dtype=np.uint8),
        net=net,
    )

    assert net.input_shape == (1, 1, 128, 96)
    assert prediction.digit == "7"
    assert prediction.confidence > 0.7
    assert [item[0] for item in prediction.top3] == ["7", "8", "2"]
    assert prediction.top3[0][1] == prediction.confidence


def test_runtime_uses_onnx_as_primary_and_exposes_top3(monkeypatch):
    expected = "123456"
    _fake_preprocess(monkeypatch)
    monkeypatch.setenv("POSTCODE_RECOGNIZER_ENGINE", "onnx")
    monkeypatch.setenv("POSTCODE_ONNX_MODEL", "/tmp/postcode_digit_v1.onnx")
    monkeypatch.setenv("POSTCODE_ONNX_FALLBACK_TESSERACT", "1")
    monkeypatch.setattr("ocr.app.postcode_runtime.load_configured_onnx_net", lambda: object())

    calls = {"value": 0}

    def fake_predict(canvas, *, net=None):
        calls["value"] += 1
        digit = expected[calls["value"] - 1]
        confidence = 0.90 + calls["value"] * 0.01
        return OnnxDigitPrediction(
            digit=digit,
            confidence=confidence,
            top3=((digit, confidence), ("0", 0.04), ("9", 0.02)),
        )

    monkeypatch.setattr("ocr.app.postcode_runtime.predict_digit_onnx", fake_predict)
    monkeypatch.setattr(
        "ocr.app.postcode_runtime._recognize_digit_crop_tesseract",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Tesseract не должен вызываться")),
    )

    result = recognize_postcode_digits(
        np.zeros((80, 160, 3), dtype=np.uint8),
        _geometry(),
    )

    assert result.status == "recognized"
    assert result.postcode == expected
    assert result.structurally_valid is True
    assert result.engine == "onnx_postcode_digit_v1+stencil_dot_suppression_v1"
    assert result.min_digit_confidence == 0.91
    assert result.geometric_mean_confidence is not None
    assert result.geometric_mean_confidence <= result.confidence
    assert all(item.engine == "onnx" for item in result.digits)

    payload = postcode_recognition_to_dict(result)
    assert payload["postcode"] == expected
    assert payload["model_path"] == "/tmp/postcode_digit_v1.onnx"
    assert payload["digits"][0]["top3"][0] == {
        "digit": "1",
        "probability": 0.91,
    }


def test_runtime_falls_back_to_tesseract_if_model_cannot_load(monkeypatch):
    expected = "123456"
    _fake_preprocess(monkeypatch)
    monkeypatch.setenv("POSTCODE_RECOGNIZER_ENGINE", "onnx")
    monkeypatch.setenv("POSTCODE_ONNX_MODEL", "/tmp/missing.onnx")
    monkeypatch.setenv("POSTCODE_ONNX_FALLBACK_TESSERACT", "1")

    def fail_load():
        raise PostcodeOnnxError("model missing")

    monkeypatch.setattr("ocr.app.postcode_runtime.load_configured_onnx_net", fail_load)

    def fake_tesseract(crop, index):
        return DigitRecognition(
            index=index,
            status="recognized",
            digit=expected[index - 1],
            confidence=0.8,
        )

    monkeypatch.setattr(
        "ocr.app.postcode_runtime._recognize_digit_crop_tesseract",
        fake_tesseract,
    )

    result = recognize_postcode_digits(
        np.zeros((80, 160, 3), dtype=np.uint8),
        _geometry(),
    )

    assert result.status == "recognized"
    assert result.postcode == expected
    assert result.engine == "tesseract_fallback_from_onnx+stencil_dot_suppression_v1"
    assert all(item.engine == "tesseract_single_digit" for item in result.digits)
    assert all(len(item.top3) == 1 for item in result.digits)


def test_runtime_can_disable_tesseract_fallback(monkeypatch):
    _fake_preprocess(monkeypatch)
    monkeypatch.setenv("POSTCODE_RECOGNIZER_ENGINE", "onnx")
    monkeypatch.setenv("POSTCODE_ONNX_MODEL", "/tmp/missing.onnx")
    monkeypatch.setenv("POSTCODE_ONNX_FALLBACK_TESSERACT", "0")

    def fail_load():
        raise PostcodeOnnxError("model missing")

    monkeypatch.setattr("ocr.app.postcode_runtime.load_configured_onnx_net", fail_load)

    result = recognize_postcode_digits(
        np.zeros((80, 160, 3), dtype=np.uint8),
        _geometry(),
    )

    assert result.status == "error"
    assert result.postcode is None
    assert result.text == "??????"
    assert all(item.status == "error" for item in result.digits)
    assert all("onnx_model_unavailable" in (item.reason or "") for item in result.digits)

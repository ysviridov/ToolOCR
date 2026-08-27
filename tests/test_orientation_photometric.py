import cv2
import numpy as np

import ocr.app.profile_scoring as profile_scoring
from ocr.app.gost_r_51506_99 import EnvelopeFormat
from ocr.app.orientation_photometric import (
    _prepare_orientation_work_photometric,
    prepare_photometric_gray,
)
from ocr.app.profiles import profiles_for_format


def _dark_uneven_letter() -> np.ndarray:
    height, width = 500, 1000
    x = np.linspace(135, 178, width, dtype=np.float32)
    gray = np.repeat(x[np.newaxis, :], height, axis=0)

    # Медленная вертикальная тень, похожая на неравномерность production C4.
    y_shadow = 18.0 * np.exp(-((np.arange(height) - 300.0) / 150.0) ** 2)
    gray -= y_shadow[:, np.newaxis]
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Несколько реальных тёмных элементов должны остаться тёмными после коррекции.
    cv2.rectangle(gray, (650, 80), (900, 105), 30, -1)
    cv2.rectangle(gray, (120, 360), (500, 372), 35, -1)
    return gray


def test_dark_uneven_letter_gets_photometric_normalization():
    gray = _dark_uneven_letter()
    normalized, diagnostics = prepare_photometric_gray(gray)

    assert diagnostics.status == "applied"
    assert "dark_background" in diagnostics.reasons
    assert diagnostics.background_p50 < 205
    assert np.median(normalized) > 220

    # До коррекции фиксированный threshold считает почти всю бумагу чернилами.
    assert float(np.mean(gray < 185)) > 0.70
    # После коррекции threshold снова в основном выделяет реальный контент.
    assert float(np.mean(normalized < 185)) < 0.20


def test_bright_letter_keeps_original_cv_path():
    gray = np.full((500, 1000), 240, dtype=np.uint8)
    cv2.rectangle(gray, (100, 100), (400, 112), 35, -1)
    cv2.rectangle(gray, (550, 330), (880, 342), 35, -1)

    prepared, diagnostics = prepare_photometric_gray(gray)

    assert diagnostics.status == "not_needed"
    assert diagnostics.reasons == ()
    assert prepared is gray


def test_barcode_uses_raw_gray_but_threshold_features_use_normalized(monkeypatch):
    gray = _dark_uneven_letter()
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    seen = {"barcode": [], "postage": [], "address": []}

    def barcode_signal(value):
        seen["barcode"].append(float(np.median(value)))
        return 0.0

    def postage_signal(value, _profile, _sides):
        seen["postage"].append(float(np.median(value)))
        return 0.0

    def address_signal(value):
        seen["address"].append(float(np.median(value)))
        return 0.0

    monkeypatch.setattr(profile_scoring, "_barcode_layout_signal", barcode_signal)
    monkeypatch.setattr(profile_scoring, "_postage", postage_signal)
    monkeypatch.setattr(profile_scoring, "_address_layout_signal", address_signal)
    monkeypatch.setattr(profile_scoring, "_code_stamp", lambda *_args: 0.0)
    monkeypatch.setattr(profile_scoring, "_line_signal", lambda *_args: 0.0)
    monkeypatch.setattr(profile_scoring, "_window_signal", lambda *_args: 0.0)

    _prepare_orientation_work_photometric(
        image,
        tuple(profiles_for_format(EnvelopeFormat.DL)),
        (),
        False,
    )

    assert seen["barcode"]
    assert seen["postage"]
    assert seen["address"]
    assert seen["barcode"][0] < 190
    assert seen["postage"][0] > 220
    assert seen["address"][0] > 220

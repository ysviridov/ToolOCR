import cv2
import numpy as np

from ocr.app.profile_scoring import score_gost_profiles
from ocr.app.profiles import DOMESTIC_PROFILES


def _synthetic_c4_layout_i() -> np.ndarray:
    """Синтетический C4 с якорями ГОСТ, без копирования изображения стандарта."""
    width_mm = 324
    height_mm = 229
    px_per_mm = 4
    image = np.full(
        (height_mm * px_per_mm, width_mm * px_per_mm, 3),
        235,
        dtype=np.uint8,
    )

    # П. 6.1.2.5: поле знака оплаты 40x25 мм у верхней правой угловой метки.
    x1 = (width_mm - 15 - 40) * px_per_mm
    x2 = (width_mm - 15) * px_per_mm
    y1 = 15 * px_per_mm
    y2 = (15 + 25) * px_per_mm
    for index in range(4):
        cv2.rectangle(
            image,
            (x1 + index * 35, y1 + 5),
            (x1 + 25 + index * 35, y2 - 5),
            (40, 40, 40),
            -1,
        )

    # Приложение Д, рисунок Д.1: шесть верхних элементов 7x2 мм с шагом 9 мм.
    code_x = 10 * px_per_mm
    code_y = image.shape[0] - 25 * px_per_mm
    for index in range(6):
        x = code_x + index * 9 * px_per_mm
        cv2.rectangle(
            image,
            (x, code_y),
            (x + 7 * px_per_mm - 1, code_y + 2 * px_per_mm - 1),
            (20, 20, 20),
            -1,
        )

    # Исполнение I: длинные направляющие линии. Координаты здесь тестовые;
    # они не используются как production ROI, проверяется только сам тип сигнала.
    for y in (70, 90, 110, 130):
        cv2.line(image, (60, y), (600, y), (80, 80, 80), 2)
    for y in (600, 630, 660, 690, 720, 750):
        cv2.line(image, (620, y), (1180, y), (80, 80, 80), 2)

    return image


def test_scoring_resolves_c4_layout_i_and_zero_orientation():
    result = score_gost_profiles(_synthetic_c4_layout_i(), DOMESTIC_PROFILES)

    assert result.orientation.status == "resolved"
    assert result.orientation.value_deg == 0
    assert result.orientation.margin > 0.12
    assert result.profile.status == "resolved"
    assert result.profile.profile_id == "vn-c4-i"


def test_scoring_resolves_180_degree_input():
    image = cv2.rotate(_synthetic_c4_layout_i(), cv2.ROTATE_180)
    result = score_gost_profiles(image, DOMESTIC_PROFILES)

    assert result.orientation.status == "resolved"
    assert result.orientation.value_deg == 180
    assert result.profile.profile_id == "vn-c4-i"


def test_blank_face_does_not_force_orientation():
    image = np.full((900, 1280, 3), 235, dtype=np.uint8)
    result = score_gost_profiles(image, DOMESTIC_PROFILES)

    assert result.orientation.status == "ambiguous"
    assert result.orientation.value_deg is None
    assert result.profile.status == "ambiguous"

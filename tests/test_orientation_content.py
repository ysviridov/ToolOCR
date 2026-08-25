import cv2
import numpy as np

from ocr.app.gost_r_51506_99 import EnvelopeFormat
from ocr.app.profile_scoring import (
    _address_layout_signal,
    _barcode_layout_signal,
    _orientation_signal,
    score_gost_profiles,
)
from ocr.app.profiles import profiles_for_format


def _synthetic_machine_dl() -> np.ndarray:
    """Машинно оформленный DL без марок и шестизначного кодового штампа."""
    height = 500
    width = 1000
    image = np.full((height, width, 3), 240, dtype=np.uint8)

    # Крупный линейный barcode в левой части.
    for index in range(38):
        x = 90 + index * 4
        cv2.rectangle(image, (x, 80), (x + 1, 135), (20, 20, 20), -1)

    # Небольшой блок отправителя сверху слева.
    for index, line_width in enumerate((180, 210, 160)):
        y = 180 + index * 18
        cv2.rectangle(image, (80, y), (80 + line_width, y + 6), (40, 40, 40), -1)

    # Более крупный блок получателя снизу справа.
    for index, line_width in enumerate((320, 360, 290, 340, 260)):
        y = 300 + index * 22
        cv2.rectangle(image, (520, y), (520 + line_width, y + 8), (40, 40, 40), -1)

    return image


def test_barcode_layout_is_asymmetric_under_180_rotation():
    image = cv2.cvtColor(_synthetic_machine_dl(), cv2.COLOR_BGR2GRAY)
    rotated = cv2.rotate(image, cv2.ROTATE_180)

    assert _barcode_layout_signal(image) > 0.7
    assert _barcode_layout_signal(image) > _barcode_layout_signal(rotated) + 0.4


def test_address_layout_prefers_sender_upper_left_recipient_lower_right():
    image = cv2.cvtColor(_synthetic_machine_dl(), cv2.COLOR_BGR2GRAY)
    rotated = cv2.rotate(image, cv2.ROTATE_180)

    assert _address_layout_signal(image) > 0.6
    assert _address_layout_signal(image) > _address_layout_signal(rotated) + 0.1


def test_content_orientation_resolves_machine_dl_without_gost_code_stamp():
    image = _synthetic_machine_dl()
    result = score_gost_profiles(image, profiles_for_format(EnvelopeFormat.DL))

    assert result.orientation.status == "resolved"
    assert result.orientation.value_deg == 0
    assert result.orientation.margin > 0.12
    evidence = {item.orientation_deg: item for item in result.orientation_evidence}
    assert evidence[0].content_orientation > evidence[180].content_orientation
    assert evidence[0].barcode_layout > evidence[180].barcode_layout


def test_content_orientation_resolves_rotated_machine_dl():
    image = cv2.rotate(_synthetic_machine_dl(), cv2.ROTATE_180)
    result = score_gost_profiles(image, profiles_for_format(EnvelopeFormat.DL))

    assert result.orientation.status == "resolved"
    assert result.orientation.value_deg == 180
    assert result.orientation.margin > 0.12


def test_single_false_postage_anchor_cannot_force_orientation():
    assert _orientation_signal(1.0, 0.0, 0.0) < 0.30
    assert _orientation_signal(0.0, 1.0, 0.0) < 0.30

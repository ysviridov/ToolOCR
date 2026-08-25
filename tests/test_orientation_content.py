import cv2
import numpy as np

from ocr.app.gost_r_51506_99 import EnvelopeFormat
from ocr.app.profile_scoring import (
    OrientationEvidence,
    _address_layout_signal,
    _barcode_layout_signal,
    _contrast_aware_fusion,
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


def _evidence(
    degree: int,
    *,
    postage: float,
    code_stamp: float,
    barcode: float,
    address: float,
    text: float,
    base_score: float,
) -> OrientationEvidence:
    return OrientationEvidence(
        orientation_deg=degree,
        postage=postage,
        code_stamp=code_stamp,
        barcode_layout=barcode,
        address_layout=address,
        text_direction=text,
        content_orientation=0.0,
        base_score=base_score,
        postage_delta=0.0,
        code_stamp_delta=0.0,
        barcode_delta=0.0,
        address_delta=0.0,
        text_delta=0.0,
        contrast_bonus=0.0,
        agreement_bonus=0.0,
        agreement_channels=0,
        score=base_score,
    )


def _fused_pair(zero: OrientationEvidence, one_eighty: OrientationEvidence):
    return _contrast_aware_fusion(
        {0: zero.base_score, 180: one_eighty.base_score},
        {0: zero, 180: one_eighty},
    )


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


def test_nearly_symmetric_barcode_has_negligible_contrast_bonus():
    fused = _fused_pair(
        _evidence(
            0,
            postage=0.0,
            code_stamp=0.0,
            barcode=0.7195,
            address=0.0,
            text=0.0,
            base_score=0.3,
        ),
        _evidence(
            180,
            postage=0.0,
            code_stamp=0.0,
            barcode=0.7070,
            address=0.0,
            text=0.0,
            base_score=0.3,
        ),
    )

    assert fused[0].barcode_delta == pytest.approx(0.0125, abs=1e-6)
    assert fused[0].contrast_bonus < 0.001
    assert fused[0].agreement_bonus == 0.0


def test_contrast_and_agreement_resolve_four_regression_cases():
    cases = [
        # FE8D: 0° — postage + barcode.
        (
            0,
            _evidence(0, postage=1.0, code_stamp=0.0, barcode=1.0, address=0.5331, text=0.0, base_score=0.5840),
            _evidence(180, postage=0.9218, code_stamp=0.0, barcode=0.6790, address=0.5129, text=0.0, base_score=0.4923),
        ),
        # FE86: 0° — barcode + OSD text.
        (
            0,
            _evidence(0, postage=0.0152, code_stamp=0.0, barcode=0.7989, address=0.5684, text=0.6440, base_score=0.3139),
            _evidence(180, postage=0.0402, code_stamp=0.0, barcode=0.6740, address=0.6151, text=0.0, base_score=0.3050),
        ),
        # FE8B: 180° — postage + barcode + text.
        (
            180,
            _evidence(0, postage=0.0176, code_stamp=0.0, barcode=0.7785, address=0.4436, text=0.0, base_score=0.2792),
            _evidence(180, postage=0.3991, code_stamp=0.0, barcode=1.0, address=0.3278, text=0.1873, base_score=0.3888),
        ),
        # FE7F: 0° — postage + text, barcode почти симметричен.
        (
            0,
            _evidence(0, postage=0.9894, code_stamp=0.0, barcode=0.7195, address=0.3351, text=0.4493, base_score=0.4729),
            _evidence(180, postage=0.7232, code_stamp=0.0, barcode=0.7070, address=0.4546, text=0.0, base_score=0.4361),
        ),
    ]

    for expected, zero, one_eighty in cases:
        fused = _fused_pair(zero, one_eighty)
        winner = max(fused, key=lambda degree: fused[degree].score)
        margin = fused[winner].score - fused[180 if winner == 0 else 0].score
        assert winner == expected
        assert margin >= 0.12
        assert fused[winner].contrast_bonus > 0.0
        assert fused[winner].agreement_bonus > 0.0

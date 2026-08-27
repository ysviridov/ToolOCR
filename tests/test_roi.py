import cv2
import numpy as np

from ocr.app.gost_r_51506_99 import EnvelopeFormat
from ocr.app.roi import canonicalize_rectified, detect_simple_mail_rois


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


def test_dl_simple_mail_roi_detects_address_and_postcode_content():
    image = np.full((500, 1000, 3), 255, dtype=np.uint8)

    # Имитация нескольких строк адреса в правой части canonical DL.
    for row, width in enumerate((300, 340, 280, 250)):
        y = 190 + row * 38
        for x in range(560, 560 + width, 34):
            cv2.rectangle(image, (x, y), (x + 18, y + 18), (0, 0, 0), -1)

    # Имитация шести рукописных цифр в нижнем левом кодовом штампе.
    for index in range(6):
        x = 110 + index * 55
        cv2.rectangle(image, (x, 385), (x + 24, 435), (0, 0, 0), 4)
        cv2.line(image, (x + 5, 425), (x + 20, 395), (0, 0, 0), 3)

    result = detect_simple_mail_rois(image, EnvelopeFormat.DL)
    regions = {item.kind: item for item in result.regions}

    assert result.status == "detected"
    assert regions["recipient_address"].status == "detected"
    assert regions["recipient_postcode"].status == "detected"
    assert regions["recipient_address"].detected_bbox is not None
    assert regions["recipient_postcode"].detected_bbox is not None
    assert regions["recipient_address"].component_count > 0
    assert regions["recipient_postcode"].component_count > 0


def test_c4_simple_mail_roi_detects_address_and_postcode_content():
    image = np.full((700, 1000, 3), 255, dtype=np.uint8)

    # C4: адрес получателя в правой нижней части canonical-листа.
    for row, width in enumerate((330, 360, 310, 280)):
        y = 285 + row * 42
        for x in range(555, 555 + width, 38):
            cv2.rectangle(image, (x, y), (x + 20, y + 20), (0, 0, 0), -1)

    # Шестизначный кодовый штамп в нижней левой части.
    for index in range(6):
        x = 110 + index * 54
        cv2.rectangle(image, (x, 535), (x + 23, 585), (0, 0, 0), 4)
        cv2.line(image, (x + 4, 575), (x + 19, 545), (0, 0, 0), 3)

    result = detect_simple_mail_rois(image, EnvelopeFormat.C4)
    regions = {item.kind: item for item in result.regions}

    assert result.status == "detected"
    assert regions["recipient_address"].status == "detected"
    assert regions["recipient_postcode"].status == "detected"
    assert regions["recipient_address"].detected_bbox is not None
    assert regions["recipient_postcode"].detected_bbox is not None
    assert regions["recipient_address"].component_count > 0
    assert regions["recipient_postcode"].component_count > 0


def test_roi_unsupported_format_remains_explicit():
    image = np.full((500, 1000, 3), 255, dtype=np.uint8)

    result = detect_simple_mail_rois(image, EnvelopeFormat.B4)

    assert result.status == "unsupported_format"
    assert result.regions == ()

import cv2
import numpy as np

from ocr.app.gost_r_51506_99 import EnvelopeFormat
from ocr.app.postcode_digit_cells import (
    derive_postcode_digit_geometry,
    postcode_digit_cells_to_dict,
)
from ocr.app.roi import detect_simple_mail_rois


def _draw_stencil_with_known_digit_bounds(
    image: np.ndarray,
    *,
    x: int,
    y: int,
    bar_width: int,
) -> list[tuple[int, int, int, int]]:
    bar_height = max(8, round(bar_width * 0.28))
    step = round(bar_width * 1.25)

    for index in range(7):
        bx = x + index * step
        cv2.rectangle(
            image,
            (bx, y),
            (bx + bar_width, y + bar_height),
            (0, 0, 0),
            -1,
        )

    # Нижняя половинная плашка '=' для strict confirmation.
    cv2.rectangle(
        image,
        (x, y + round(bar_height * 1.35)),
        (x + bar_width, y + round(bar_height * 1.85)),
        (0, 0, 0),
        -1,
    )

    digit_bounds: list[tuple[int, int, int, int]] = []
    for digit_index in range(1, 7):
        bx = x + digit_index * step + round(bar_width * 0.10)
        left = bx
        right = bx + round(bar_width * 0.70)
        top = y + round(bar_width * 0.75)
        bottom = y + round(bar_width * 2.45)

        # Контур цифры намеренно почти заполняет полезную высоту ячейки.
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 0), 3)
        cv2.line(
            image,
            (left + 4, bottom - 5),
            (right - 3, top + 5),
            (0, 0, 0),
            3,
        )
        digit_bounds.append((left, top, right, bottom))

    return digit_bounds


def test_six_digit_cells_fully_contain_known_digit_bounds():
    image = np.full((900, 1300, 3), 230, dtype=np.uint8)
    digit_bounds = _draw_stencil_with_known_digit_bounds(
        image,
        x=42,
        y=760,
        bar_width=42,
    )

    roi = detect_simple_mail_rois(image, EnvelopeFormat.C5)
    postcode = next(item for item in roi.regions if item.kind == "recipient_postcode")
    assert postcode.status == "stencil_detected"

    geometry = derive_postcode_digit_geometry(
        roi,
        image_width=image.shape[1],
        image_height=image.shape[0],
    )

    assert geometry.status == "ready"
    assert geometry.reason is None
    assert len(geometry.cells) == 6
    assert [cell.index for cell in geometry.cells] == [1, 2, 3, 4, 5, 6]

    for cell, (digit_left, digit_top, digit_right, digit_bottom) in zip(
        geometry.cells,
        digit_bounds,
        strict=True,
    ):
        assert cell.bbox.x <= digit_left
        assert cell.bbox.x2 >= digit_right
        assert cell.bbox.y <= digit_top
        assert cell.bbox.y2 >= digit_bottom

    # Здесь намеренно 5 соседних пар из 6 ячеек, поэтому strict=True неприменим.
    for left_cell, right_cell in zip(geometry.cells, geometry.cells[1:]):
        assert left_cell.bbox.x < right_cell.bbox.x
        assert left_cell.bbox.x2 <= right_cell.bbox.x + 1

    payload = postcode_digit_cells_to_dict(geometry)
    assert len(payload) == 6
    assert payload[0]["index"] == 1
    assert payload[-1]["index"] == 6


def test_digit_cells_are_unavailable_when_postcode_stencil_is_not_detected():
    image = np.full((900, 1300, 3), 230, dtype=np.uint8)
    roi = detect_simple_mail_rois(image, EnvelopeFormat.C4)

    geometry = derive_postcode_digit_geometry(
        roi,
        image_width=image.shape[1],
        image_height=image.shape[0],
    )

    assert geometry.status == "unavailable"
    assert geometry.reason == "postcode_not_detected"
    assert geometry.cells == ()

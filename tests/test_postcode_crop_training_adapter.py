from pathlib import Path

import cv2
import numpy as np

from scripts.postcode_crop_training_adapter import (
    build_training_postcode_canvas,
    extract_postcode_crop_cells,
)


def _draw_realistic_tight_postcode_crop(
    *,
    angle_deg: float = 0.0,
    width: int = 840,
    height: int = 264,
    faded_start_bar: bool = False,
) -> np.ndarray:
    # Масштаб близок к реальным crop: ~840x264, плашки ~75x21.
    image = np.full((height, width, 3), 220, dtype=np.uint8)
    x0 = 79
    y0 = 52
    bar_width = 76
    bar_height = 21
    step = 97

    for index in range(7):
        x = x0 + index * step
        value = 165 if faded_start_bar and index == 0 else 15
        cv2.rectangle(
            image,
            (x, y0),
            (x + bar_width - 1, y0 + bar_height - 1),
            (value, value, value),
            -1,
        )

    # Нижняя половинная плашка '='.
    cv2.rectangle(
        image,
        (x0, y0 + 32),
        (x0 + bar_width - 1, y0 + 42),
        (15, 15, 15),
        -1,
    )

    # Шесть простых glyph-контуров внутри будущих digit-cell.
    for index in range(1, 7):
        center_x = x0 + index * step + bar_width // 2
        top = y0 + 48
        bottom = min(height - 12, y0 + 145)
        cv2.line(image, (center_x - 16, top), (center_x + 16, top + 8), (20, 20, 20), 5)
        cv2.line(image, (center_x + 16, top + 8), (center_x - 14, bottom), (20, 20, 20), 5)

    if abs(angle_deg) < 1e-6:
        return image

    matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0),
        angle_deg,
        1.0,
    )
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(220, 220, 220),
    )


def test_virtual_canvas_uses_anchor_scale_for_tight_crop():
    crop = _draw_realistic_tight_postcode_crop()
    canvas, search, debug = build_training_postcode_canvas(crop)

    assert crop.shape[:2] == (264, 840)
    assert canvas.shape[1] > crop.shape[1] * 2
    assert canvas.shape[0] > crop.shape[0] * 4
    assert search.width == 840
    assert search.height == 264
    assert search.y > 0
    assert debug["adapter"] == "postcode_crop_virtual_canonical_v2"
    assert debug["scale_source"] == "anchor_bar_width"
    assert 0.038 <= debug["effective_bar_width_ratio"] <= 0.046
    assert debug["deskew"]["status"] == "not_needed"
    assert debug["deskew"]["anchor_before"]["matched_count"] == 7


def test_skewed_tight_crop_is_deskewed_and_produces_six_digit_canvases(
    tmp_path: Path,
):
    path = tmp_path / "letter_crop.jpg"
    assert cv2.imwrite(
        str(path),
        _draw_realistic_tight_postcode_crop(angle_deg=-1.4),
    )

    cells, debug = extract_postcode_crop_cells(path)

    assert debug["adapter"] == "postcode_crop_virtual_canonical_v2"
    assert debug["deskew"]["status"] == "applied"
    assert abs(debug["deskew"]["rotation_applied_deg"]) >= 1.0
    assert abs(debug["deskew"]["anchor_after"]["angle_deg"]) <= 0.20
    assert debug["digit_cell_count"] == 6
    assert [index for index, _, _ in cells] == [1, 2, 3, 4, 5, 6]
    assert all(canvas.shape == (128, 96) for _, canvas, _ in cells)


def test_dynamic_scale_is_stable_when_operator_crop_width_changes():
    normal = _draw_realistic_tight_postcode_crop()
    wider = cv2.copyMakeBorder(
        normal,
        0,
        0,
        0,
        140,
        cv2.BORDER_CONSTANT,
        value=(220, 220, 220),
    )

    normal_canvas, _, normal_debug = build_training_postcode_canvas(normal)
    wider_canvas, _, wider_debug = build_training_postcode_canvas(wider)

    assert wider.shape[1] > normal.shape[1]
    assert normal_debug["scale_source"] == "anchor_bar_width"
    assert wider_debug["scale_source"] == "anchor_bar_width"
    assert abs(normal_canvas.shape[1] - wider_canvas.shape[1]) <= 3
    assert abs(
        normal_debug["effective_bar_width_ratio"]
        - wider_debug["effective_bar_width_ratio"]
    ) <= 0.002


def test_faded_start_bar_is_available_to_training_anchor_detector(tmp_path: Path):
    path = tmp_path / "faded_start_crop.jpg"
    assert cv2.imwrite(
        str(path),
        _draw_realistic_tight_postcode_crop(faded_start_bar=True),
    )

    cells, debug = extract_postcode_crop_cells(path)

    assert debug["deskew"]["anchor_before"]["matched_count"] == 7
    assert debug["digit_cell_count"] == 6
    assert len(cells) == 6

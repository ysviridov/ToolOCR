from pathlib import Path

import cv2
import numpy as np

from scripts.postcode_crop_training_adapter import (
    build_training_postcode_canvas,
    extract_postcode_crop_cells,
)


def _draw_realistic_tight_postcode_crop() -> np.ndarray:
    # Масштаб близок к реальному присланному crop: 840x264, плашки ~75x21.
    image = np.full((264, 840, 3), 220, dtype=np.uint8)
    x0 = 79
    y0 = 52
    bar_width = 76
    bar_height = 21
    step = 97

    for index in range(7):
        x = x0 + index * step
        cv2.rectangle(
            image,
            (x, y0),
            (x + bar_width - 1, y0 + bar_height - 1),
            (15, 15, 15),
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
        bottom = y0 + 145
        cv2.line(image, (center_x - 16, top), (center_x + 16, top + 8), (20, 20, 20), 5)
        cv2.line(image, (center_x + 16, top + 8), (center_x - 14, bottom), (20, 20, 20), 5)

    return image


def test_virtual_canvas_restores_full_frame_scale_for_tight_crop():
    crop = _draw_realistic_tight_postcode_crop()
    canvas, search, debug = build_training_postcode_canvas(crop)

    assert crop.shape[:2] == (264, 840)
    assert canvas.shape[1] > crop.shape[1] * 2
    assert canvas.shape[0] > crop.shape[0] * 4
    assert search.width == 840
    assert search.height == 264
    assert search.y > 0
    assert debug["adapter"] == "postcode_crop_virtual_canonical_v1"


def test_realistic_tight_crop_produces_six_digit_canvases(tmp_path: Path):
    path = tmp_path / "letter_crop.jpg"
    assert cv2.imwrite(str(path), _draw_realistic_tight_postcode_crop())

    cells, debug = extract_postcode_crop_cells(path)

    assert debug["adapter"] == "postcode_crop_virtual_canonical_v1"
    assert debug["digit_cell_count"] == 6
    assert [index for index, _, _ in cells] == [1, 2, 3, 4, 5, 6]
    assert all(canvas.shape == (128, 96) for _, canvas, _ in cells)

from __future__ import annotations

import cv2
import numpy as np

from .postcode_digit_cells import PostcodeDigitGeometry
from .postcode_recognizer import (
    PostcodeRecognition,
    _normalize_digit_crop_with_debug,
)


_PANEL_BG = (245, 247, 250)
_TILE_BORDER = (120, 130, 145)
_TEXT = (35, 40, 48)
_MUTED = (90, 98, 110)


def _recognized_by_index(recognition: PostcodeRecognition) -> dict[int, object]:
    return {item.index: item for item in recognition.digits}


def append_postcode_preprocess_strip(
    image: np.ndarray,
    source_image: np.ndarray,
    geometry: PostcodeDigitGeometry,
    recognition: PostcodeRecognition,
) -> np.ndarray:
    """Добавляет снизу диагностическую полосу с финальным входом OCR.

    Каждый tile — это тот же 96x128 canvas, который получается после
    `stencil_dot_suppression_v1`, tight glyph crop и нормализации и затем
    передаётся в Tesseract. Никакие промежуточные изображения не сохраняются.
    """

    if geometry.status != "ready" or len(geometry.cells) != 6:
        return image

    image_height, image_width = image.shape[:2]
    gap = max(14, round(image_width * 0.007))
    side_margin = gap
    usable_width = max(1, image_width - side_margin * 2 - gap * 5)
    tile_width = max(120, min(280, usable_width // 6))
    tile_height = round(tile_width * 128 / 96)
    header_height = max(76, round(tile_height * 0.24))
    footer_height = max(66, round(tile_height * 0.22))
    panel_height = header_height + tile_height + footer_height + gap * 2

    panel = np.full((panel_height, image_width, 3), _PANEL_BG, dtype=np.uint8)
    recognized = _recognized_by_index(recognition)

    title_scale = max(0.72, image_width / 2600.0)
    item_scale = max(0.50, image_width / 3900.0)
    small_scale = max(0.42, image_width / 4700.0)
    title_thickness = max(1, round(title_scale * 2.0))
    item_thickness = max(1, round(item_scale * 1.8))

    title = "OCR PREP AFTER stencil_dot_suppression_v1"
    subtitle = f"OCR RESULT: {recognition.text}"
    cv2.putText(
        panel,
        title,
        (side_margin, max(28, round(header_height * 0.42))),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        _TEXT,
        title_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        subtitle,
        (side_margin, max(56, round(header_height * 0.82))),
        cv2.FONT_HERSHEY_SIMPLEX,
        item_scale,
        _MUTED,
        item_thickness,
        cv2.LINE_AA,
    )

    total_tiles_width = tile_width * 6 + gap * 5
    start_x = max(side_margin, (image_width - total_tiles_width) // 2)
    tile_y = header_height + gap

    for position, cell in enumerate(geometry.cells):
        canvas, debug = _normalize_digit_crop_with_debug(source_image, cell)
        x = start_x + position * (tile_width + gap)
        y = tile_y

        if canvas is None:
            tile = np.full((tile_height, tile_width, 3), 255, dtype=np.uint8)
            cv2.putText(
                tile,
                "NO FOREGROUND",
                (8, max(30, tile_height // 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                _MUTED,
                1,
                cv2.LINE_AA,
            )
        else:
            enlarged = cv2.resize(
                canvas,
                (tile_width, tile_height),
                interpolation=cv2.INTER_NEAREST,
            )
            tile = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)

        panel[y:y + tile_height, x:x + tile_width] = tile
        cv2.rectangle(
            panel,
            (x, y),
            (x + tile_width - 1, y + tile_height - 1),
            _TILE_BORDER,
            max(1, round(image_width / 2200.0)),
        )

        recognized_digit = recognized.get(cell.index)
        digit = getattr(recognized_digit, "digit", None) or "?"
        confidence = getattr(recognized_digit, "confidence", None)
        label = f"D{cell.index} -> {digit}"
        if confidence is not None:
            label += f"  conf={confidence:.2f}"

        footer_y = y + tile_height + max(18, round(footer_height * 0.34))
        cv2.putText(
            panel,
            label,
            (x, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            item_scale,
            _TEXT,
            item_thickness,
            cv2.LINE_AA,
        )

        suppressed = debug.get("suppressed_components", 0)
        restored = debug.get("restored_components", 0)
        ink_ratio = debug.get("suppressed_ink_ratio")
        ink_text = "?" if not isinstance(ink_ratio, (int, float)) else f"{ink_ratio:.0%}"
        stats = f"removed={suppressed} restored={restored} ink=-{ink_text}"
        cv2.putText(
            panel,
            stats,
            (x, footer_y + max(18, round(footer_height * 0.34))),
            cv2.FONT_HERSHEY_SIMPLEX,
            small_scale,
            _MUTED,
            1,
            cv2.LINE_AA,
        )

    return np.vstack((image, panel))

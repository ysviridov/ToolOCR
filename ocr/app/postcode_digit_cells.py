from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2

from .roi import PixelRect, RoiDetection, RoiRegion, pixel_rect_to_dict


# Ячейки нужны как геометрический контракт перед подключением recognizer.
# Горизонтальная геометрия выводится из уже подтверждённой строки верхних
# плашек, а не делением общего postcode bbox на шесть равных частей.
_DIGIT_CELL_TOP_FROM_ROW_CENTER_BAR_WIDTH = 0.30
_DIGIT_CELL_BOTTOM_FROM_ROW_CENTER_BAR_WIDTH = 2.60
_DIGIT_CELL_COLOR_BGR = (255, 170, 0)


@dataclass(frozen=True, slots=True)
class PostcodeDigitCell:
    index: int
    bbox: PixelRect
    center_x_px: float


@dataclass(frozen=True, slots=True)
class PostcodeDigitGeometry:
    status: str
    reason: str | None
    source: str
    cells: tuple[PostcodeDigitCell, ...]
    bar_width_px: float | None = None
    bar_step_px: float | None = None
    row_center_y_px: float | None = None
    start_anchor_center_x_px: float | None = None


def _postcode_region(result: RoiDetection) -> RoiRegion | None:
    return next((item for item in result.regions if item.kind == "recipient_postcode"), None)


def _clip_axis_rect(
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> PixelRect | None:
    if image_width <= 0 or image_height <= 0:
        return None

    left = max(0, min(image_width - 1, int(round(x1))))
    top = max(0, min(image_height - 1, int(round(y1))))
    right = max(left + 1, min(image_width, int(round(x2))))
    bottom = max(top + 1, min(image_height, int(round(y2))))
    if right <= left or bottom <= top:
        return None
    return PixelRect(left, top, right - left, bottom - top)


def derive_postcode_digit_geometry(
    result: RoiDetection,
    *,
    image_width: int,
    image_height: int,
) -> PostcodeDigitGeometry:
    """Строит шесть digit-cell ROI из геометрии подтверждённого stencil.

    `detected_bbox.x` в текущем postcode detector начинается примерно за
    0.15 ширины плашки до первой верхней плашки. Поэтому центр стартовой
    плашки `=` восстанавливается как x + 0.65 * bar_width. Остальные шесть
    центров получаются регулярным шагом `bar_step_px`.

    Границы digit-cell по X проходят посередине между соседними anchors.
    По Y используется центр подтверждённого верхнего ряда (`row_y_norm`),
    чтобы верхняя плашка не попадала в будущий recognizer crop.
    """

    postcode = _postcode_region(result)
    if postcode is None:
        return PostcodeDigitGeometry(
            status="unavailable",
            reason="postcode_region_missing",
            source="stencil_upper_bar_geometry",
            cells=(),
        )
    if postcode.status != "stencil_detected" or postcode.detected_bbox is None:
        return PostcodeDigitGeometry(
            status="unavailable",
            reason="postcode_not_detected",
            source="stencil_upper_bar_geometry",
            cells=(),
        )

    features = postcode.features or {}
    confirmation_mode = features.get("confirmation_mode")
    if confirmation_mode not in {"strict_start_marker", "seven_bar_rescue"}:
        return PostcodeDigitGeometry(
            status="unavailable",
            reason="stencil_not_confirmed",
            source="stencil_upper_bar_geometry",
            cells=(),
        )

    try:
        bar_width = float(features["bar_width_px"])
        bar_step = float(features["bar_step_px"])
        row_y_norm = float(features["row_y_norm"])
    except (KeyError, TypeError, ValueError):
        return PostcodeDigitGeometry(
            status="unavailable",
            reason="stencil_geometry_missing",
            source="stencil_upper_bar_geometry",
            cells=(),
        )

    if bar_width <= 0 or bar_step <= 0 or not (0.0 <= row_y_norm <= 1.0):
        return PostcodeDigitGeometry(
            status="unavailable",
            reason="stencil_geometry_invalid",
            source="stencil_upper_bar_geometry",
            cells=(),
            bar_width_px=bar_width,
            bar_step_px=bar_step,
        )

    # Защитный sanity-check: штатный detector принимает step около 1.05..1.65
    # ширины плашки. Здесь не создаём второй detector, а лишь отбрасываем
    # явно повреждённую debug-геометрию.
    step_ratio = bar_step / bar_width
    if not (0.90 <= step_ratio <= 1.80):
        return PostcodeDigitGeometry(
            status="unavailable",
            reason="stencil_step_invalid",
            source="stencil_upper_bar_geometry",
            cells=(),
            bar_width_px=bar_width,
            bar_step_px=bar_step,
        )

    start_center_x = postcode.detected_bbox.x + 0.65 * bar_width
    row_center_y = row_y_norm * image_height
    digit_y1 = row_center_y + _DIGIT_CELL_TOP_FROM_ROW_CENTER_BAR_WIDTH * bar_width
    digit_y2 = row_center_y + _DIGIT_CELL_BOTTOM_FROM_ROW_CENTER_BAR_WIDTH * bar_width

    cells: list[PostcodeDigitCell] = []
    for digit_index in range(1, 7):
        center_x = start_center_x + digit_index * bar_step
        left = center_x - 0.5 * bar_step
        right = center_x + 0.5 * bar_step
        bbox = _clip_axis_rect(
            x1=left,
            y1=digit_y1,
            x2=right,
            y2=digit_y2,
            image_width=image_width,
            image_height=image_height,
        )
        if bbox is None:
            return PostcodeDigitGeometry(
                status="unavailable",
                reason="digit_cell_outside_image",
                source="stencil_upper_bar_geometry",
                cells=(),
                bar_width_px=bar_width,
                bar_step_px=bar_step,
                row_center_y_px=row_center_y,
                start_anchor_center_x_px=start_center_x,
            )
        cells.append(
            PostcodeDigitCell(
                index=digit_index,
                bbox=bbox,
                center_x_px=center_x,
            )
        )

    return PostcodeDigitGeometry(
        status="ready",
        reason=None,
        source="stencil_upper_bar_geometry",
        cells=tuple(cells),
        bar_width_px=bar_width,
        bar_step_px=bar_step,
        row_center_y_px=row_center_y,
        start_anchor_center_x_px=start_center_x,
    )


def postcode_digit_geometry_to_dict(geometry: PostcodeDigitGeometry) -> dict[str, Any]:
    return {
        "status": geometry.status,
        "reason": geometry.reason,
        "source": geometry.source,
        "cell_count": len(geometry.cells),
        "bar_width_px": None if geometry.bar_width_px is None else round(geometry.bar_width_px, 2),
        "bar_step_px": None if geometry.bar_step_px is None else round(geometry.bar_step_px, 2),
        "row_center_y_px": None if geometry.row_center_y_px is None else round(geometry.row_center_y_px, 2),
        "start_anchor_center_x_px": (
            None
            if geometry.start_anchor_center_x_px is None
            else round(geometry.start_anchor_center_x_px, 2)
        ),
    }


def postcode_digit_cells_to_dict(geometry: PostcodeDigitGeometry) -> list[dict[str, Any]]:
    return [
        {
            "index": cell.index,
            "bbox": pixel_rect_to_dict(cell.bbox),
            "center_x_px": round(cell.center_x_px, 2),
        }
        for cell in geometry.cells
    ]


def draw_postcode_digit_cells(
    image,
    geometry: PostcodeDigitGeometry,
) -> None:
    """Рисует 6 ячеек поверх ROI preview для corpus-validation."""

    if geometry.status != "ready":
        return

    scale = max(1.0, max(image.shape[:2]) / 1600.0)
    thickness = max(1, round(2.0 * scale))
    font_scale = max(0.45, 0.52 * scale)
    font_thickness = max(1, round(1.5 * scale))

    for cell in geometry.cells:
        rect = cell.bbox
        cv2.rectangle(
            image,
            (rect.x, rect.y),
            (rect.x2, rect.y2),
            _DIGIT_CELL_COLOR_BGR,
            thickness,
        )
        label_y = max(18, rect.y - max(4, round(4 * scale)))
        cv2.putText(
            image,
            f"D{cell.index}",
            (rect.x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            _DIGIT_CELL_COLOR_BGR,
            font_thickness,
            cv2.LINE_AA,
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .gost_r_51506_99 import EnvelopeFormat, RectNormalized


@dataclass(frozen=True, slots=True)
class CanonicalImage:
    image: np.ndarray
    status: str
    source_orientation_deg: int | None
    rotation_applied_deg: int
    reliable: bool


@dataclass(frozen=True, slots=True)
class PixelRect:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


@dataclass(frozen=True, slots=True)
class RoiRegion:
    kind: str
    status: str
    confidence: float
    search_bbox: PixelRect
    detected_bbox: PixelRect | None
    bbox: PixelRect
    component_count: int
    ink_density: float


@dataclass(frozen=True, slots=True)
class RoiDetection:
    status: str
    format: EnvelopeFormat
    coordinate_space: str
    mail_class: str
    regions: tuple[RoiRegion, ...]
    source_reference: str


# Это расширенные поисковые зоны, а не нормативные границы адресных зон.
# Они получены из компоновки обязательного приложения А ГОСТ Р 51506-99
# и намеренно содержат запас для реального рукописного текста. На первом
# этапе поддерживаются основные production-форматы простых писем: DL и C5.
_GOST_GUIDED_SEARCH_ZONES: dict[EnvelopeFormat, dict[str, RectNormalized]] = {
    EnvelopeFormat.DL: {
        "recipient_address": RectNormalized(0.50, 0.27, 0.47, 0.50),
        "recipient_postcode": RectNormalized(0.05, 0.66, 0.47, 0.31),
    },
    EnvelopeFormat.C5: {
        "recipient_address": RectNormalized(0.50, 0.28, 0.47, 0.48),
        "recipient_postcode": RectNormalized(0.05, 0.68, 0.47, 0.29),
    },
}

ROI_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "recipient_address": (70, 175, 70),
    "recipient_postcode": (0, 155, 255),
}

ROI_LABELS = {
    "recipient_address": "RECIPIENT",
    "recipient_postcode": "POSTCODE",
}


def canonicalize_rectified(
    image: np.ndarray,
    *,
    orientation_status: str | None,
    orientation_deg: int | None,
) -> CanonicalImage:
    """Приводит rectified-письмо к единой ориентации перед ROI/OCR.

    При ambiguous изображение не угадывается и возвращается без поворота,
    но reliable=False. ROI/OCR должны проверять этот флаг.
    """

    if image is None or image.size == 0:
        raise ValueError("Пустое rectified-изображение")

    if orientation_status != "resolved" or orientation_deg not in (0, 180):
        return CanonicalImage(
            image=image,
            status="orientation_unresolved",
            source_orientation_deg=orientation_deg,
            rotation_applied_deg=0,
            reliable=False,
        )

    if orientation_deg == 180:
        canonical = cv2.rotate(image, cv2.ROTATE_180)
        rotation = 180
    else:
        canonical = image
        rotation = 0

    return CanonicalImage(
        image=canonical,
        status="canonical",
        source_orientation_deg=orientation_deg,
        rotation_applied_deg=rotation,
        reliable=True,
    )


def _rect_to_pixels(rect: RectNormalized, width: int, height: int) -> PixelRect:
    x1 = max(0, min(width - 1, int(round(rect.x * width))))
    y1 = max(0, min(height - 1, int(round(rect.y * height))))
    x2 = max(x1 + 1, min(width, int(round((rect.x + rect.width) * width))))
    y2 = max(y1 + 1, min(height, int(round((rect.y + rect.height) * height))))
    return PixelRect(x1, y1, x2 - x1, y2 - y1)


def _clip_rect(rect: PixelRect, width: int, height: int) -> PixelRect:
    x1 = max(0, min(width - 1, rect.x))
    y1 = max(0, min(height - 1, rect.y))
    x2 = max(x1 + 1, min(width, rect.x2))
    y2 = max(y1 + 1, min(height, rect.y2))
    return PixelRect(x1, y1, x2 - x1, y2 - y1)


def _foreground_bbox(
    image: np.ndarray,
    search: PixelRect,
    *,
    kind: str,
) -> tuple[PixelRect | None, int, float, float]:
    crop = image[search.y:search.y2, search.x:search.x2]
    if crop.size == 0:
        return None, 0, 0.0, 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Убираем границу поисковой зоны, чтобы край конверта/рамки не стал ROI.
    pad_x = max(2, round(search.width * 0.012))
    pad_y = max(2, round(search.height * 0.018))
    binary[:pad_y, :] = 0
    binary[-pad_y:, :] = 0
    binary[:, :pad_x] = 0
    binary[:, -pad_x:] = 0

    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    crop_area = float(search.width * search.height)
    min_area = max(5, int(crop_area * (0.000018 if kind == "recipient_address" else 0.000012)))
    max_area = max(min_area + 1, int(crop_area * 0.075))
    min_h = max(3, int(search.height * 0.018))
    max_h = max(min_h + 1, int(search.height * 0.42))

    boxes: list[tuple[int, int, int, int, int]] = []
    ink_pixels = 0
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        if area < min_area or area > max_area:
            continue
        if h < min_h or h > max_h or w < 2:
            continue
        # Длинные тонкие направляющие/границы не считаем содержимым.
        if w / float(max(1, h)) > 28.0 and h < search.height * 0.07:
            continue
        boxes.append((x, y, w, h, area))
        ink_pixels += area

    if not boxes:
        return None, 0, 0.0, 0.0

    x1 = min(item[0] for item in boxes)
    y1 = min(item[1] for item in boxes)
    x2 = max(item[0] + item[2] for item in boxes)
    y2 = max(item[1] + item[3] for item in boxes)

    margin_x = max(4, int(search.width * 0.025))
    margin_y = max(4, int(search.height * 0.045))
    local = PixelRect(
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(search.width, x2 + margin_x) - max(0, x1 - margin_x),
        min(search.height, y2 + margin_y) - max(0, y1 - margin_y),
    )
    detected = PixelRect(
        search.x + local.x,
        search.y + local.y,
        local.width,
        local.height,
    )

    density = ink_pixels / crop_area if crop_area else 0.0
    component_score = min(1.0, len(boxes) / (9.0 if kind == "recipient_address" else 7.0))
    density_target = 0.022 if kind == "recipient_address" else 0.017
    density_score = min(1.0, density / density_target) if density_target else 0.0
    confidence = round(min(0.99, 0.25 + 0.45 * component_score + 0.30 * density_score), 4)
    return detected, len(boxes), round(density, 6), confidence


def detect_simple_mail_rois(image: np.ndarray, envelope_format: EnvelopeFormat) -> RoiDetection:
    """Находит ROI простого письма на canonical rectified изображении.

    Текущая цель Stage 2.2 — только рукописные/машинописные блоки простых
    писем: адрес получателя и шестизначный индекс места назначения.
    Штрихкоды заказных отправлений намеренно не распознаются здесь.
    """

    if image is None or image.size == 0:
        raise ValueError("Пустое canonical-изображение")

    zones = _GOST_GUIDED_SEARCH_ZONES.get(envelope_format)
    if zones is None:
        return RoiDetection(
            status="unsupported_format",
            format=envelope_format,
            coordinate_space="canonical_rectified",
            mail_class="simple",
            regions=(),
            source_reference="ГОСТ Р 51506-99, приложение А",
        )

    height, width = image.shape[:2]
    regions: list[RoiRegion] = []
    detected_count = 0
    for kind in ("recipient_address", "recipient_postcode"):
        search = _rect_to_pixels(zones[kind], width, height)
        detected, component_count, density, confidence = _foreground_bbox(
            image,
            search,
            kind=kind,
        )
        if detected is not None:
            detected = _clip_rect(detected, width, height)
            status = "detected"
            final_bbox = detected
            detected_count += 1
        else:
            status = "search_zone_only"
            final_bbox = search
            confidence = 0.15

        regions.append(
            RoiRegion(
                kind=kind,
                status=status,
                confidence=confidence,
                search_bbox=search,
                detected_bbox=detected,
                bbox=final_bbox,
                component_count=component_count,
                ink_density=density,
            )
        )

    return RoiDetection(
        status="detected" if detected_count == len(regions) else "partial",
        format=envelope_format,
        coordinate_space="canonical_rectified",
        mail_class="simple",
        regions=tuple(regions),
        source_reference="ГОСТ Р 51506-99, приложение А; расширенные поисковые зоны ToolOCR",
    )


def pixel_rect_to_dict(rect: PixelRect | None) -> dict[str, int] | None:
    if rect is None:
        return None
    return {
        "x": rect.x,
        "y": rect.y,
        "width": rect.width,
        "height": rect.height,
    }


def roi_detection_to_dict(result: RoiDetection) -> dict[str, Any]:
    return {
        "status": result.status,
        "mail_class": result.mail_class,
        "format": result.format.value,
        "coordinate_space": result.coordinate_space,
        "source_reference": result.source_reference,
        "regions": [
            {
                "kind": region.kind,
                "status": region.status,
                "confidence": region.confidence,
                "search_bbox": pixel_rect_to_dict(region.search_bbox),
                "detected_bbox": pixel_rect_to_dict(region.detected_bbox),
                "bbox": pixel_rect_to_dict(region.bbox),
                "component_count": region.component_count,
                "ink_density": region.ink_density,
            }
            for region in result.regions
        ],
    }


def _draw_dashed_rect(
    image: np.ndarray,
    rect: PixelRect,
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    dash: int = 14,
    gap: int = 9,
) -> None:
    step = max(2, dash + gap)
    for x in range(rect.x, rect.x2, step):
        cv2.line(image, (x, rect.y), (min(x + dash, rect.x2), rect.y), color, thickness)
        cv2.line(image, (x, rect.y2), (min(x + dash, rect.x2), rect.y2), color, thickness)
    for y in range(rect.y, rect.y2, step):
        cv2.line(image, (rect.x, y), (rect.x, min(y + dash, rect.y2)), color, thickness)
        cv2.line(image, (rect.x2, y), (rect.x2, min(y + dash, rect.y2)), color, thickness)


def draw_roi_overlay(image: np.ndarray, result: RoiDetection) -> np.ndarray:
    overlay = image.copy()
    scale = max(1.0, max(image.shape[:2]) / 1600.0)
    search_thickness = max(1, round(1.5 * scale))
    detected_thickness = max(2, round(3.0 * scale))
    font_scale = max(0.5, 0.62 * scale)

    for region in result.regions:
        color = ROI_COLORS_BGR[region.kind]
        _draw_dashed_rect(
            overlay,
            region.search_bbox,
            color,
            thickness=search_thickness,
            dash=max(8, round(14 * scale)),
            gap=max(5, round(9 * scale)),
        )
        if region.detected_bbox is not None:
            cv2.rectangle(
                overlay,
                (region.detected_bbox.x, region.detected_bbox.y),
                (region.detected_bbox.x2, region.detected_bbox.y2),
                color,
                detected_thickness,
            )

        label = f"{ROI_LABELS[region.kind]} {region.confidence:.2f}"
        anchor = region.detected_bbox or region.search_bbox
        text_y = max(22, anchor.y - max(7, round(7 * scale)))
        cv2.putText(
            overlay,
            label,
            (anchor.x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, round(2 * scale)),
            cv2.LINE_AA,
        )

    return overlay

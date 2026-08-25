from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .layout import QuadDetection, order_quad


@dataclass(frozen=True, slots=True)
class FrameNormalization:
    image: np.ndarray
    source_width_px: int
    source_height_px: int
    crop_x: int
    crop_y: int
    crop_width_px: int
    crop_height_px: int
    foreground_x: int
    foreground_y: int
    foreground_width_px: int
    foreground_height_px: int
    foreground_area_ratio: float
    bottom_anchored: bool
    status: str

    @property
    def used(self) -> bool:
        return self.status == "cropped"

    @property
    def crop_area_ratio(self) -> float:
        source_area = float(self.source_width_px * self.source_height_px)
        if source_area <= 0:
            return 1.0
        return float(self.crop_width_px * self.crop_height_px) / source_area

    def points_to_source(self, points: np.ndarray) -> np.ndarray:
        result = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
        result[:, 0] += float(self.crop_x)
        result[:, 1] += float(self.crop_y)
        return result


def _largest_light_component(image: np.ndarray) -> tuple[int, int, int, int, float] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Закрываем разрывы на границе письма и убираем мелкий светлый шум ленты.
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
        iterations=2,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = float(image.shape[0] * image.shape[1])
    best: tuple[float, int, int, int, int] | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        # Письмо должно быть крупным объектом, а не бликом/этикеткой.
        if area / frame_area < 0.08:
            continue
        rectangularity = area / max(1.0, float(width * height))
        score = area * (0.65 + 0.35 * min(1.0, rectangularity))
        if best is None or score > best[0]:
            best = (score, x, y, width, height)

    if best is None:
        return None

    _, x, y, width, height = best
    return x, y, width, height, float(width * height) / frame_area


def normalize_black_background(
    image: np.ndarray,
    *,
    margin_ratio: float = 0.015,
    min_saving_ratio: float = 0.04,
    bottom_anchor_ratio: float = 0.012,
) -> FrameNormalization:
    """Убирает черный фон вокруг письма, не теряя привязку к source-координатам.

    Crop нужен только для ускорения и стабилизации CV. Геометрия письма затем
    переводится обратно в координаты исходного кадра через `points_to_source()`.
    Поэтому физический размер не зависит от того, сколько черного поля было
    сохранено в конкретном JPEG.
    """

    if image is None or image.size == 0:
        raise ValueError("Передано пустое изображение")
    if margin_ratio < 0 or margin_ratio > 0.10:
        raise ValueError("margin_ratio должен быть в диапазоне 0..0.10")

    source_height, source_width = image.shape[:2]
    component = _largest_light_component(image)
    if component is None:
        return FrameNormalization(
            image=image,
            source_width_px=source_width,
            source_height_px=source_height,
            crop_x=0,
            crop_y=0,
            crop_width_px=source_width,
            crop_height_px=source_height,
            foreground_x=0,
            foreground_y=0,
            foreground_width_px=source_width,
            foreground_height_px=source_height,
            foreground_area_ratio=1.0,
            bottom_anchored=False,
            status="foreground_not_found",
        )

    x, y, width, height, area_ratio = component
    margin = max(12, int(round(min(source_width, source_height) * margin_ratio)))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(source_width, x + width + margin)
    y1 = min(source_height, y + height + margin)

    crop_width = max(1, x1 - x0)
    crop_height = max(1, y1 - y0)
    crop_area_ratio = float(crop_width * crop_height) / float(source_width * source_height)
    bottom_gap = max(0, source_height - (y + height))
    bottom_anchored = bottom_gap <= max(3, int(round(source_height * bottom_anchor_ratio)))

    # Если выигрыш практически отсутствует, оставляем исходный массив без copy.
    if crop_area_ratio >= 1.0 - min_saving_ratio:
        return FrameNormalization(
            image=image,
            source_width_px=source_width,
            source_height_px=source_height,
            crop_x=0,
            crop_y=0,
            crop_width_px=source_width,
            crop_height_px=source_height,
            foreground_x=x,
            foreground_y=y,
            foreground_width_px=width,
            foreground_height_px=height,
            foreground_area_ratio=round(area_ratio, 4),
            bottom_anchored=bottom_anchored,
            status="unchanged",
        )

    cropped = image[y0:y1, x0:x1]
    return FrameNormalization(
        image=cropped,
        source_width_px=source_width,
        source_height_px=source_height,
        crop_x=x0,
        crop_y=y0,
        crop_width_px=crop_width,
        crop_height_px=crop_height,
        foreground_x=x,
        foreground_y=y,
        foreground_width_px=width,
        foreground_height_px=height,
        foreground_area_ratio=round(area_ratio, 4),
        bottom_anchored=bottom_anchored,
        status="cropped",
    )


def detection_to_source(
    detection: QuadDetection,
    normalization: FrameNormalization,
) -> QuadDetection:
    """Переводит quad из crop-координат обратно в исходный JPEG."""

    if not normalization.used:
        return detection
    points = order_quad(normalization.points_to_source(detection.points))
    return QuadDetection(
        points=points,
        method=f"normalized:{detection.method}",
        confidence=detection.confidence,
        raw_confidence=detection.raw_confidence,
        area_ratio=detection.area_ratio,
        rectangularity=detection.rectangularity,
        angle_score=detection.angle_score,
        frame_contact_sides=detection.frame_contact_sides,
    )


def normalization_to_dict(normalization: FrameNormalization) -> dict:
    return {
        "status": normalization.status,
        "source": {
            "width_px": normalization.source_width_px,
            "height_px": normalization.source_height_px,
        },
        "crop": {
            "x": normalization.crop_x,
            "y": normalization.crop_y,
            "width_px": normalization.crop_width_px,
            "height_px": normalization.crop_height_px,
            "area_ratio": round(normalization.crop_area_ratio, 4),
        },
        "foreground_bbox": {
            "x": normalization.foreground_x,
            "y": normalization.foreground_y,
            "width_px": normalization.foreground_width_px,
            "height_px": normalization.foreground_height_px,
            "area_ratio": normalization.foreground_area_ratio,
        },
        "bottom_anchored": normalization.bottom_anchored,
    }

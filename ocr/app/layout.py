from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .gost_r_51506_99 import ENVELOPE_SPECS


class EnvelopeNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class QuadDetection:
    points: np.ndarray
    method: str
    confidence: float
    area_ratio: float
    rectangularity: float
    angle_score: float


@dataclass(frozen=True, slots=True)
class RectificationResult:
    image: np.ndarray
    width_px: int
    height_px: int
    source_quad: np.ndarray


def order_quad(points: np.ndarray) -> np.ndarray:
    """Возвращает точки в порядке TL, TR, BR, BL."""

    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    return np.array(
        [
            pts[np.argmin(sums)],
            pts[np.argmin(diffs)],
            pts[np.argmax(sums)],
            pts[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def _angle_score(quad: np.ndarray) -> float:
    ordered = order_quad(quad)
    scores: list[float] = []

    for index in range(4):
        point = ordered[index]
        previous = ordered[(index - 1) % 4]
        next_point = ordered[(index + 1) % 4]

        v1 = previous - point
        v2 = next_point - point
        denominator = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denominator <= 1e-9:
            return 0.0

        cosine = float(np.dot(v1, v2) / denominator)
        cosine = max(-1.0, min(1.0, cosine))
        angle = math.degrees(math.acos(cosine))
        scores.append(max(0.0, 1.0 - abs(angle - 90.0) / 45.0))

    return float(sum(scores) / len(scores))


def _quad_side_lengths(quad: np.ndarray) -> tuple[float, float]:
    tl, tr, br, bl = order_quad(quad)
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    return float(width), float(height)


def _aspect_score(quad: np.ndarray) -> float:
    width, height = _quad_side_lengths(quad)
    if width <= 0 or height <= 0:
        return 0.0

    observed = max(width, height) / min(width, height)
    nearest_error = min(
        abs(observed - spec.aspect_ratio) / spec.aspect_ratio
        for spec in ENVELOPE_SPECS.values()
    )
    return max(0.0, 1.0 - nearest_error / 0.12)


def _candidate_score(
    *,
    quad: np.ndarray,
    contour_area: float,
    frame_area: float,
) -> tuple[float, float, float, float]:
    area_ratio = contour_area / frame_area
    rect = cv2.minAreaRect(quad.astype(np.float32))
    rect_area = float(rect[1][0] * rect[1][1])
    rectangularity = 0.0 if rect_area <= 1e-9 else min(1.0, contour_area / rect_area)
    angle_score = _angle_score(quad)
    aspect_score = _aspect_score(quad)

    confidence = (
        0.45 * min(1.0, area_ratio / 0.65)
        + 0.25 * rectangularity
        + 0.20 * angle_score
        + 0.10 * aspect_score
    )

    return confidence, area_ratio, rectangularity, angle_score


def _foreground_mask_otsu(gray_blurred: np.ndarray) -> np.ndarray:
    """Строит маску светлого отправления на более тёмном фоне сортировщика.

    ГОСТ допускает белую и светлую бумагу, поэтому на фотографиях с тёмной
    транспортной лентой яркостная сегментация является сильным первым каналом
    detector-а. Текст, марки и штемпели образуют тёмные области внутри листа,
    но RETR_EXTERNAL сохраняет внешний контур.

    Canny остаётся вторым независимым каналом для кадров, где яркостное
    разделение письма и фона недостаточно.
    """

    _, foreground = cv2.threshold(
        gray_blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
        iterations=2,
    )
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return foreground


def _edge_mask(gray_blurred: np.ndarray) -> np.ndarray:
    median = float(np.median(gray_blurred))
    lower = max(20, int(0.66 * median))
    upper = min(255, max(lower + 30, int(1.33 * median)))

    edges = cv2.Canny(gray_blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)


def _contour_sources(gray_blurred: np.ndarray) -> list[tuple[str, list[np.ndarray]]]:
    """Возвращает независимые наборы контуров в порядке предпочтения."""

    foreground = _foreground_mask_otsu(gray_blurred)
    foreground_contours, _ = cv2.findContours(
        foreground,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    edges = _edge_mask(gray_blurred)
    edge_contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    return [
        (
            "foreground_otsu",
            sorted(foreground_contours, key=cv2.contourArea, reverse=True)[:20],
        ),
        (
            "contour_approx",
            sorted(edge_contours, key=cv2.contourArea, reverse=True)[:80],
        ),
    ]


def detect_envelope_quad(
    image: np.ndarray,
    *,
    max_detection_side: int = 1800,
    min_area_ratio: float = 0.15,
) -> QuadDetection:
    """Ищет внешний четырёхугольник полного письма.

    Используются два независимых CV-канала:
    1. foreground_otsu — светлый лист на более тёмном фоне транспортёра;
    2. contour_approx — границы Canny для кадров со слабым яркостным
       разделением.

    Детектор работает на уменьшенной копии изображения, затем возвращает
    координаты в системе исходной фотографии. В production crop-режим не
    используется: если внешний контур не найден, вызывающий код должен
    вернуть layout reject.
    """

    if image is None or image.size == 0:
        raise ValueError("Передано пустое изображение")
    if min_area_ratio <= 0.0 or min_area_ratio >= 1.0:
        raise ValueError("min_area_ratio должен находиться между 0 и 1")

    original_height, original_width = image.shape[:2]
    scale = min(1.0, max_detection_side / float(max(original_width, original_height)))

    if scale < 1.0:
        work = cv2.resize(
            image,
            (max(1, round(original_width * scale)), max(1, round(original_height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work = image

    height, width = work.shape[:2]
    frame_area = float(width * height)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    sources = _contour_sources(blurred)

    # confidence, quad, method, area_ratio, rectangularity, angle_score
    best: tuple[float, np.ndarray, str, float, float, float] | None = None

    for method, contours in sources:
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area / frame_area < min_area_ratio:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue

            for epsilon_factor in (0.005, 0.01, 0.015, 0.02, 0.025, 0.03):
                approx = cv2.approxPolyDP(
                    contour,
                    epsilon_factor * perimeter,
                    True,
                )
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue

                quad = order_quad(approx.reshape(4, 2))
                confidence, area_ratio, rectangularity, angle_score = _candidate_score(
                    quad=quad,
                    contour_area=area,
                    frame_area=frame_area,
                )

                if best is None or confidence > best[0]:
                    best = (
                        confidence,
                        quad,
                        method,
                        area_ratio,
                        rectangularity,
                        angle_score,
                    )

    if best is not None:
        confidence, quad, method, area_ratio, rectangularity, angle_score = best
        quad = quad / scale
        return QuadDetection(
            points=order_quad(quad),
            method=method,
            confidence=round(float(confidence), 4),
            area_ratio=round(float(area_ratio), 4),
            rectangularity=round(float(rectangularity), 4),
            angle_score=round(float(angle_score), 4),
        )

    # Резервный путь: минимальный ориентированный прямоугольник большого
    # контура. Проверяем оба канала, но confidence специально ограничен.
    fallback_best: tuple[float, np.ndarray, str, float, float, float] | None = None

    for source_method, contours in sources:
        for contour in contours:
            area = float(cv2.contourArea(contour))
            area_ratio = area / frame_area
            if area_ratio < min_area_ratio:
                continue

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.float32)
            box_area = float(rect[1][0] * rect[1][1])
            if box_area <= 1e-9:
                continue

            rectangularity = min(1.0, area / box_area)
            if rectangularity < 0.72:
                continue

            quad = order_quad(box)
            angle_score = _angle_score(quad)
            confidence = min(
                0.72,
                0.35 * min(1.0, area_ratio / 0.65)
                + 0.30 * rectangularity
                + 0.20 * angle_score
                + 0.15 * _aspect_score(quad),
            )

            method = f"{source_method}_min_area_rect"
            if fallback_best is None or confidence > fallback_best[0]:
                fallback_best = (
                    confidence,
                    quad,
                    method,
                    area_ratio,
                    rectangularity,
                    angle_score,
                )

    if fallback_best is not None:
        confidence, quad, method, area_ratio, rectangularity, angle_score = fallback_best
        quad = quad / scale
        return QuadDetection(
            points=order_quad(quad),
            method=method,
            confidence=round(float(confidence), 4),
            area_ratio=round(float(area_ratio), 4),
            rectangularity=round(float(rectangularity), 4),
            angle_score=round(float(angle_score), 4),
        )

    raise EnvelopeNotFoundError("Не найден внешний четырёхугольник полного письма")


def rectify_envelope(image: np.ndarray, quad: np.ndarray) -> RectificationResult:
    """Устраняет перспективу и возвращает лицевую сторону в landscape-виде.

    Поворот на 180 градусов здесь намеренно НЕ исправляется. После этого
    этапа остаются две допустимые гипотезы ориентации: 0° и 180°.
    """

    source = order_quad(quad)
    tl, tr, br, bl = source

    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    height_left = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))

    target_width = max(2, int(round(max(width_top, width_bottom))))
    target_height = max(2, int(round(max(height_left, height_right))))

    destination = np.array(
        [
            [0.0, 0.0],
            [target_width - 1.0, 0.0],
            [target_width - 1.0, target_height - 1.0],
            [0.0, target_height - 1.0],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(source.astype(np.float32), destination)
    rectified = cv2.warpPerspective(
        image,
        transform,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    if rectified.shape[0] > rectified.shape[1]:
        rectified = cv2.rotate(rectified, cv2.ROTATE_90_CLOCKWISE)

    height, width = rectified.shape[:2]
    return RectificationResult(
        image=rectified,
        width_px=int(width),
        height_px=int(height),
        source_quad=source,
    )


def draw_detection_overlay(image: np.ndarray, detection: QuadDetection) -> np.ndarray:
    overlay = image.copy()
    points = np.round(detection.points).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [points], True, (0, 220, 0), 4, cv2.LINE_AA)

    for index, point in enumerate(np.round(detection.points).astype(np.int32)):
        x, y = int(point[0]), int(point[1])
        cv2.circle(overlay, (x, y), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            ("TL", "TR", "BR", "BL")[index],
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    label = f"{detection.method} conf={detection.confidence:.3f}"
    cv2.putText(
        overlay,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 120, 0),
        2,
        cv2.LINE_AA,
    )
    return overlay

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
    detector: str = "foreground"
    features: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class RoiDetection:
    status: str
    format: EnvelopeFormat
    coordinate_space: str
    mail_class: str
    regions: tuple[RoiRegion, ...]
    source_reference: str


_GOST_GUIDED_SEARCH_ZONES: dict[EnvelopeFormat, dict[str, RectNormalized]] = {
    EnvelopeFormat.DL: {
        "recipient_address": RectNormalized(0.50, 0.27, 0.47, 0.50),
    },
    EnvelopeFormat.C5: {
        "recipient_address": RectNormalized(0.50, 0.28, 0.47, 0.48),
    },
    EnvelopeFormat.C4: {
        "recipient_address": RectNormalized(0.49, 0.30, 0.48, 0.48),
    },
}

# Трафаретный индекс — одинаковый технический объект для поддерживаемых
# форматов: стартовый знак '=' + шесть индивидуальных ячеек цифр.
_POSTCODE_STENCIL_SEARCH_ZONE = RectNormalized(0.0, 0.50, 0.62, 0.50)

# Strict path остаётся основным. Seven-bar rescue разрешает слабый нижний
# штрих '=' только при практически идеальной геометрии всей верхней строки.
_STENCIL_STRICT_SCORE_MIN = 0.78
_STENCIL_START_MARKER_MIN = 0.40
_STENCIL_RESCUE_BAR_COUNT = 7
_STENCIL_RESCUE_WIDTH_CV_MAX = 0.12
_STENCIL_RESCUE_SPACING_ERROR_MAX = 0.08
_STENCIL_RESCUE_ALIGNMENT_ERROR_MAX = 0.55
_STENCIL_RESCUE_ROW_Y_NORM_MIN = 0.70

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
    """Приводит rectified-письмо к единой ориентации перед ROI/OCR."""

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

    pad_x = max(2, round(search.width * 0.012))
    pad_y = max(2, round(search.height * 0.018))
    binary[:pad_y, :] = 0
    binary[-pad_y:, :] = 0
    binary[:, :pad_x] = 0
    binary[:, -pad_x:] = 0

    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    crop_area = float(search.width * search.height)
    min_area = max(5, int(crop_area * 0.000018))
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
    component_score = min(1.0, len(boxes) / 9.0)
    density_score = min(1.0, density / 0.022)
    confidence = round(min(0.99, 0.25 + 0.45 * component_score + 0.30 * density_score), 4)
    return detected, len(boxes), round(density, 6), confidence


def _normalize_stencil_gray(gray: np.ndarray) -> np.ndarray:
    """Компенсирует медленные перепады освещённости для stencil detector."""

    height, width = gray.shape[:2]
    sigma = max(12.0, min(height, width) * 0.025)
    kernel_size = min(151, max(31, int(round(sigma * 4.0)) * 2 + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    corrected = gray.astype(np.float32) * 235.0 / np.maximum(background.astype(np.float32), 40.0)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _confirmation_mode(candidate: dict[str, Any]) -> str | None:
    strict = (
        candidate["score"] >= _STENCIL_STRICT_SCORE_MIN
        and candidate["start_marker_score"] >= _STENCIL_START_MARKER_MIN
    )
    if strict:
        return "strict_start_marker"

    rescue = (
        candidate["bar_count"] == _STENCIL_RESCUE_BAR_COUNT
        and candidate["width_cv"] <= _STENCIL_RESCUE_WIDTH_CV_MAX
        and candidate["spacing_error"] <= _STENCIL_RESCUE_SPACING_ERROR_MAX
        and candidate["alignment_error"] <= _STENCIL_RESCUE_ALIGNMENT_ERROR_MAX
        and candidate["row_y_norm"] >= _STENCIL_RESCUE_ROW_Y_NORM_MIN
    )
    if rescue:
        return "seven_bar_rescue"
    return None


def _rejection_reason(candidate: dict[str, Any] | None) -> str:
    if candidate is None:
        return "no_structural_candidate"
    if candidate["bar_count"] < 7:
        return "insufficient_top_bars"
    if candidate["row_y_norm"] < _STENCIL_RESCUE_ROW_Y_NORM_MIN:
        return "row_not_low_enough"
    if candidate["width_cv"] > _STENCIL_RESCUE_WIDTH_CV_MAX:
        return "width_variation_too_high"
    if candidate["spacing_error"] > _STENCIL_RESCUE_SPACING_ERROR_MAX:
        return "spacing_error_too_high"
    if candidate["alignment_error"] > _STENCIL_RESCUE_ALIGNMENT_ERROR_MAX:
        return "alignment_error_too_high"
    if candidate["start_marker_score"] < _STENCIL_START_MARKER_MIN:
        return "start_marker_weak"
    return "structural_score_too_low"


def _postcode_stencil_bbox(
    image: np.ndarray,
    search: PixelRect,
) -> tuple[PixelRect | None, int, float, float, dict[str, Any]]:
    """Ищет '= + 6 цифр' по геометрии верхних плашек и стартового '='.

    Основная ветка требует нижнюю половинную плашку стартового '='. Rescue
    допускается без неё только для полной строки из семи практически
    одинаковых и регулярно расположенных верхних плашек в нижней части
    canonical-письма.
    """

    crop = image[search.y:search.y2, search.x:search.x2]
    if crop.size == 0:
        return None, 0, 0.0, 0.0, {
            "confirmation_mode": "none",
            "rejection_reason": "empty_search_zone",
        }

    full_height, full_width = image.shape[:2]
    crop_height, crop_width = crop.shape[:2]

    max_detector_side = 1200
    scale = min(1.0, max_detector_side / float(max(crop_height, crop_width)))
    if scale < 1.0:
        work = cv2.resize(
            crop,
            (max(1, int(round(crop_width * scale))), max(1, int(round(crop_height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work = crop

    work_height, work_width = work.shape[:2]
    scaled_full_width = full_width * scale
    scaled_full_height = full_height * scale

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
    gray = _normalize_stencil_gray(gray)
    otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_threshold = float(np.clip(otsu_threshold, 135.0, 190.0))
    binary = np.where(gray < dark_threshold, 255, 0).astype(np.uint8)

    kernel_width = max(9, int(round(scaled_full_width * 0.008)))
    kernel_height = max(2, int(round(scaled_full_height * 0.0015)))
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height)),
    )

    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        aspect = width / float(max(1, height))
        if not (scaled_full_width * 0.008 <= width <= scaled_full_width * 0.055):
            continue
        if not (scaled_full_height * 0.002 <= height <= scaled_full_height * 0.025):
            continue
        if aspect < 2.5:
            continue
        fill = float(np.mean(binary[y:y + height, x:x + width] > 0))
        if fill < 0.45:
            continue
        candidates.append((x, y, width, height, fill))

    top_bars = [
        item
        for item in candidates
        if item[2] >= scaled_full_width * 0.012
        and item[3] >= scaled_full_height * 0.005
        and item[4] >= 0.70
    ]

    structural: list[dict[str, Any]] = []
    for first_index, first in enumerate(top_bars):
        first_center = first[0] + first[2] / 2.0
        for second_index, second in enumerate(top_bars):
            if first_index == second_index:
                continue
            second_center = second[0] + second[2] / 2.0
            if second_center <= first_center:
                continue

            mean_width = (first[2] + second[2]) / 2.0
            step = second_center - first_center
            if not (1.05 * mean_width <= step <= 1.65 * mean_width):
                continue

            matched: list[tuple[int, int, int, int, float] | None] = []
            used: set[int] = set()
            for position in range(7):
                expected_x = first_center + position * step
                options = [
                    (
                        abs((candidate[0] + candidate[2] / 2.0) - expected_x),
                        index,
                        candidate,
                    )
                    for index, candidate in enumerate(top_bars)
                    if index not in used
                    and abs((candidate[0] + candidate[2] / 2.0) - expected_x) <= 0.28 * step
                ]
                if not options:
                    matched.append(None)
                    continue
                _, index, candidate = min(options, key=lambda item: item[0])
                used.add(index)
                matched.append(candidate)

            valid = [item for item in matched if item is not None]
            if len(valid) < 6:
                continue

            widths = np.asarray([item[2] for item in valid], dtype=np.float32)
            heights = np.asarray([item[3] for item in valid], dtype=np.float32)
            centers_x = np.asarray([item[0] + item[2] / 2.0 for item in valid], dtype=np.float32)
            centers_y = np.asarray([item[1] + item[3] / 2.0 for item in valid], dtype=np.float32)
            positions = np.asarray(
                [index for index, item in enumerate(matched) if item is not None],
                dtype=np.float32,
            )

            expected_x = first_center + positions * step
            spacing_error = float(
                np.sqrt(np.mean(np.square(centers_x - expected_x))) / max(step, 1.0)
            )
            width_cv = float(np.std(widths) / max(float(np.mean(widths)), 1.0))

            line = np.polyfit(centers_x, centers_y, 1)
            alignment_error = float(
                np.sqrt(np.mean(np.square(centers_y - np.polyval(line, centers_x))))
                / max(float(np.median(heights)), 1.0)
            )
            row_center_y_work = float(np.median(centers_y))
            row_y_norm = float(
                (search.y * scale + row_center_y_work) / max(scaled_full_height, 1.0)
            )

            start_bar = matched[0]
            start_marker_score = 0.0
            if start_bar is not None:
                for candidate in candidates:
                    if candidate == start_bar:
                        continue
                    same_x = abs(candidate[0] - start_bar[0]) <= 0.25 * start_bar[2]
                    same_width = abs(candidate[2] - start_bar[2]) <= 0.30 * start_bar[2]
                    below = (
                        start_bar[1] + 0.8 * start_bar[3]
                        <= candidate[1]
                        <= start_bar[1] + 2.5 * start_bar[3]
                    )
                    half_height = 0.30 * start_bar[3] <= candidate[3] <= 0.80 * start_bar[3]
                    if same_x and same_width and below and half_height:
                        start_marker_score = max(
                            start_marker_score,
                            1.0 - abs(candidate[3] / float(start_bar[3]) - 0.5),
                        )

            score = (
                (len(valid) / 7.0) * 0.45
                + max(0.0, 1.0 - width_cv / 0.20) * 0.15
                + max(0.0, 1.0 - spacing_error / 0.20) * 0.15
                + max(0.0, 1.0 - alignment_error / 1.0) * 0.10
                + start_marker_score * 0.15
            )

            structural.append(
                {
                    "score": float(score),
                    "matched": matched,
                    "bar_count": len(valid),
                    "start_marker_score": float(start_marker_score),
                    "width_cv": width_cv,
                    "spacing_error": spacing_error,
                    "alignment_error": alignment_error,
                    "row_y_norm": row_y_norm,
                    "bar_step_px_work": float(step),
                    "bar_width_px_work": float(np.median(widths)),
                }
            )

    strict_candidates = [
        item for item in structural if _confirmation_mode(item) == "strict_start_marker"
    ]
    rescue_candidates = [
        item for item in structural if _confirmation_mode(item) == "seven_bar_rescue"
    ]

    if strict_candidates:
        best = max(strict_candidates, key=lambda item: item["score"])
        confirmation_mode = "strict_start_marker"
    elif rescue_candidates:
        # В rescue приоритет геометрии, а не слабому случайному start-marker.
        best = min(
            rescue_candidates,
            key=lambda item: (
                item["width_cv"] + item["spacing_error"] + item["alignment_error"],
                -item["score"],
            ),
        )
        confirmation_mode = "seven_bar_rescue"
    else:
        best_failed = max(structural, key=lambda item: item["score"], default=None)
        return (
            None,
            0 if best_failed is None else int(best_failed["bar_count"]),
            0.0,
            0.0 if best_failed is None else round(float(best_failed["score"]), 4),
            {
                "confirmation_mode": "none",
                "rejection_reason": _rejection_reason(best_failed),
                "candidate_count": len(candidates),
                "top_bar_candidate_count": len(top_bars),
                "best_score": None if best_failed is None else round(float(best_failed["score"]), 4),
                "bar_count": None if best_failed is None else int(best_failed["bar_count"]),
                "start_marker_score": None if best_failed is None else round(float(best_failed["start_marker_score"]), 4),
                "width_cv": None if best_failed is None else round(float(best_failed["width_cv"]), 4),
                "spacing_error": None if best_failed is None else round(float(best_failed["spacing_error"]), 4),
                "alignment_error": None if best_failed is None else round(float(best_failed["alignment_error"]), 4),
                "row_y_norm": None if best_failed is None else round(float(best_failed["row_y_norm"]), 4),
                "threshold": round(dark_threshold, 2),
            },
        )

    matched_valid = [item for item in best["matched"] if item is not None]
    bar_width = float(np.median([item[2] for item in matched_valid]))
    local_x1 = max(0, int(round(min(item[0] for item in matched_valid) - 0.15 * bar_width)))
    local_x2 = min(
        work_width,
        int(round(max(item[0] + item[2] for item in matched_valid) + 0.15 * bar_width)),
    )
    local_y1 = max(0, int(round(min(item[1] for item in matched_valid) - 0.12 * bar_width)))
    top_bar_bottom = max(item[1] + item[3] for item in matched_valid)
    local_y2 = min(work_height, int(round(top_bar_bottom + 2.45 * bar_width)))

    inverse_scale = 1.0 / scale
    detected = PixelRect(
        search.x + int(round(local_x1 * inverse_scale)),
        search.y + int(round(local_y1 * inverse_scale)),
        max(1, int(round((local_x2 - local_x1) * inverse_scale))),
        max(1, int(round((local_y2 - local_y1) * inverse_scale))),
    )
    detected = _clip_rect(detected, full_width, full_height)

    scaled_local = PixelRect(
        local_x1,
        local_y1,
        max(1, local_x2 - local_x1),
        max(1, local_y2 - local_y1),
    )
    stencil_crop = binary[scaled_local.y:scaled_local.y2, scaled_local.x:scaled_local.x2]
    ink_density = float(np.mean(stencil_crop > 0)) if stencil_crop.size else 0.0

    features = {
        "confirmation_mode": confirmation_mode,
        "rejection_reason": None,
        "bar_count": int(best["bar_count"]),
        "expected_bar_count": 7,
        "digit_count": 6,
        "start_marker_score": round(float(best["start_marker_score"]), 4),
        "width_cv": round(float(best["width_cv"]), 4),
        "spacing_error": round(float(best["spacing_error"]), 4),
        "alignment_error": round(float(best["alignment_error"]), 4),
        "row_y_norm": round(float(best["row_y_norm"]), 4),
        "bar_width_px": round(float(best["bar_width_px_work"] * inverse_scale), 2),
        "bar_step_px": round(float(best["bar_step_px_work"] * inverse_scale), 2),
        "threshold": round(dark_threshold, 2),
        "detector_scale": round(scale, 4),
        "rescue_limits": {
            "bar_count": _STENCIL_RESCUE_BAR_COUNT,
            "width_cv_max": _STENCIL_RESCUE_WIDTH_CV_MAX,
            "spacing_error_max": _STENCIL_RESCUE_SPACING_ERROR_MAX,
            "alignment_error_max": _STENCIL_RESCUE_ALIGNMENT_ERROR_MAX,
            "row_y_norm_min": _STENCIL_RESCUE_ROW_Y_NORM_MIN,
        },
    }

    confidence = float(best["score"])
    if confirmation_mode == "seven_bar_rescue":
        # Rescue никогда не получает чрезмерно высокий confidence только из-за
        # отсутствующего start-marker, но остаётся выше detection threshold.
        confidence = max(0.78, min(0.94, confidence))

    return (
        detected,
        int(best["bar_count"]),
        round(ink_density, 6),
        round(min(0.99, confidence), 4),
        features,
    )


def detect_simple_mail_rois(image: np.ndarray, envelope_format: EnvelopeFormat) -> RoiDetection:
    """Находит адрес и трафаретный шестизначный индекс простого письма."""

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

    address_search = _rect_to_pixels(zones["recipient_address"], width, height)
    address_bbox, component_count, density, confidence = _foreground_bbox(
        image,
        address_search,
        kind="recipient_address",
    )
    if address_bbox is not None:
        address_bbox = _clip_rect(address_bbox, width, height)
        address_status = "detected"
        address_final = address_bbox
        detected_count += 1
    else:
        address_status = "search_zone_only"
        address_final = address_search
        confidence = 0.15

    regions.append(
        RoiRegion(
            kind="recipient_address",
            status=address_status,
            confidence=confidence,
            search_bbox=address_search,
            detected_bbox=address_bbox,
            bbox=address_final,
            component_count=component_count,
            ink_density=density,
            detector="foreground",
        )
    )

    postcode_search = _rect_to_pixels(_POSTCODE_STENCIL_SEARCH_ZONE, width, height)
    postcode_bbox, bar_count, density, confidence, stencil_features = _postcode_stencil_bbox(
        image,
        postcode_search,
    )
    if postcode_bbox is not None:
        postcode_status = "stencil_detected"
        postcode_final = postcode_bbox
        detected_count += 1
    else:
        postcode_status = "stencil_not_found"
        postcode_final = postcode_search
        confidence = min(0.25, confidence)

    regions.append(
        RoiRegion(
            kind="recipient_postcode",
            status=postcode_status,
            confidence=confidence,
            search_bbox=postcode_search,
            detected_bbox=postcode_bbox,
            bbox=postcode_final,
            component_count=bar_count,
            ink_density=density,
            detector="postcode_stencil",
            features=stencil_features,
        )
    )

    return RoiDetection(
        status="detected" if detected_count == len(regions) else "partial",
        format=envelope_format,
        coordinate_space="canonical_rectified",
        mail_class="simple",
        regions=tuple(regions),
        source_reference="ГОСТ Р 51506-99, приложение А и Д; postcode stencil detector ToolOCR",
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
                "detector": region.detector,
                "features": region.features,
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

        detector_suffix = " STENCIL" if region.detector == "postcode_stencil" else ""
        mode_suffix = ""
        if region.features and region.features.get("confirmation_mode") == "seven_bar_rescue":
            mode_suffix = " RESCUE"
        label = f"{ROI_LABELS[region.kind]}{detector_suffix}{mode_suffix} {region.confidence:.2f}"
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

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ocr.app.gost_r_51506_99 import EnvelopeFormat
from ocr.app.postcode_digit_cells import derive_postcode_digit_geometry
from ocr.app.postcode_recognizer import _normalize_digit_crop_with_debug
from ocr.app.roi import PixelRect, RoiDetection, RoiRegion, _postcode_stencil_bbox
from ocr.app.test_ui import _decode_image


# Production postcode detector нормирует размеры плашек относительно полного
# canonical-frame. Tight training crop нельзя подавать как полный кадр: плашки
# становятся искусственно слишком крупными. Adapter v2 сначала находит только
# верхнюю anchor-строку в самом crop, оценивает её небольшой наклон, deskew-ит
# crop, а затем строит виртуальный canonical-frame по реальной ширине плашек.
# Production thresholds/geometry при этом не меняются.
_TRAINING_CANVAS_FALLBACK_WIDTH_SCALE = 2.20
_TRAINING_CANVAS_WIDTH_SCALE_MIN = 1.65
_TRAINING_CANVAS_WIDTH_SCALE_MAX = 3.20
_TRAINING_CANVAS_ASPECT_RATIO = 1.42
_TRAINING_CANVAS_BOTTOM_MARGIN_RATIO = 0.01
_TRAINING_TARGET_BAR_WIDTH_RATIO = 0.042

# Training-only pre-detector anchors. Он намеренно мягче production detector:
# его задача — только геометрическая нормализация tight crop, а окончательное
# подтверждение stencil всё равно выполняет штатный _postcode_stencil_bbox().
_ANCHOR_DARK_THRESHOLD_MIN = 190.0
_ANCHOR_TOP_ZONE_MAX = 0.50
_ANCHOR_WIDTH_MIN_RATIO = 0.035
_ANCHOR_WIDTH_MAX_RATIO = 0.16
_ANCHOR_HEIGHT_MIN_RATIO = 0.012
_ANCHOR_HEIGHT_MAX_RATIO = 0.18
_ANCHOR_ASPECT_MIN = 2.3
_ANCHOR_FILL_MIN = 0.45
_ANCHOR_STEP_MIN_WIDTHS = 1.05
_ANCHOR_STEP_MAX_WIDTHS = 1.75
_ANCHOR_X_TOLERANCE_STEP = 0.30
_ANCHOR_Y_TOLERANCE_HEIGHT = 1.40
_ANCHOR_WIDTH_TOLERANCE = 0.40
_ANCHOR_MAX_ABS_ANGLE_DEG = 5.0
_ANCHOR_MIN_MATCHED = 6
_ANCHOR_ROTATE_MIN_DEG = 0.05

AnchorCandidate = tuple[int, int, int, int, float, float, float]


def _normalize_anchor_gray(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape[:2]
    sigma = max(8.0, min(height, width) * 0.03)
    kernel_size = min(101, max(21, int(round(sigma * 4.0)) * 2 + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    corrected = gray.astype(np.float32) * 235.0 / np.maximum(
        background.astype(np.float32), 40.0
    )
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _anchor_candidates(crop: np.ndarray) -> tuple[list[AnchorCandidate], float]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = _normalize_anchor_gray(gray)
    otsu_threshold, _ = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    # Faded/scanned crop может иметь светлую первую плашку '='. Для training
    # deskew она всё равно является полезным anchor, поэтому порог не даём
    # опускаться ниже 190. Production detector после deskew остаётся строгим.
    dark_threshold = float(max(_ANCHOR_DARK_THRESHOLD_MIN, otsu_threshold))
    dark_threshold = min(215.0, dark_threshold)
    binary = np.where(gray < dark_threshold, 255, 0).astype(np.uint8)

    height, width = binary.shape[:2]
    kernel_width = max(9, int(round(width * 0.012)))
    kernel_height = max(1, int(round(height * 0.004)))
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height)),
    )
    contours, _ = cv2.findContours(
        horizontal,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[AnchorCandidate] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        center_x = x + box_width / 2.0
        center_y = y + box_height / 2.0
        aspect = box_width / float(max(1, box_height))
        if not (
            width * _ANCHOR_WIDTH_MIN_RATIO
            <= box_width
            <= width * _ANCHOR_WIDTH_MAX_RATIO
        ):
            continue
        if not (
            height * _ANCHOR_HEIGHT_MIN_RATIO
            <= box_height
            <= height * _ANCHOR_HEIGHT_MAX_RATIO
        ):
            continue
        if aspect < _ANCHOR_ASPECT_MIN:
            continue
        if center_y > height * _ANCHOR_TOP_ZONE_MAX:
            continue
        fill = float(np.mean(binary[y : y + box_height, x : x + box_width] > 0))
        if fill < _ANCHOR_FILL_MIN:
            continue
        candidates.append(
            (x, y, box_width, box_height, fill, center_x, center_y)
        )

    candidates.sort(key=lambda item: (item[6], item[5]))
    return candidates, dark_threshold


def _fit_anchor_row(crop: np.ndarray) -> dict[str, Any] | None:
    height, width = crop.shape[:2]
    candidates, dark_threshold = _anchor_candidates(crop)
    best: tuple[tuple[int, float], list[AnchorCandidate | None], dict[str, Any]] | None = None

    for first_index, first in enumerate(candidates):
        for second_index, second in enumerate(candidates):
            if first_index == second_index or second[5] <= first[5]:
                continue

            step = second[5] - first[5]
            mean_width = (first[2] + second[2]) / 2.0
            if not (
                _ANCHOR_STEP_MIN_WIDTHS * mean_width
                <= step
                <= _ANCHOR_STEP_MAX_WIDTHS * mean_width
            ):
                continue

            initial_slope = (second[6] - first[6]) / max(step, 1.0)
            initial_angle = math.degrees(math.atan(initial_slope))
            if abs(initial_angle) > _ANCHOR_MAX_ABS_ANGLE_DEG:
                continue

            matched: list[AnchorCandidate | None] = []
            used: set[int] = set()
            for position in range(7):
                expected_x = first[5] + position * step
                expected_y = first[6] + initial_slope * (expected_x - first[5])
                options: list[tuple[float, int, AnchorCandidate]] = []
                for index, candidate in enumerate(candidates):
                    if index in used:
                        continue
                    x_error = abs(candidate[5] - expected_x) / max(step, 1.0)
                    y_error = abs(candidate[6] - expected_y) / max(candidate[3], 1.0)
                    width_error = abs(candidate[2] - mean_width) / max(mean_width, 1.0)
                    if x_error > _ANCHOR_X_TOLERANCE_STEP:
                        continue
                    if y_error > _ANCHOR_Y_TOLERANCE_HEIGHT:
                        continue
                    if width_error > _ANCHOR_WIDTH_TOLERANCE:
                        continue
                    options.append(
                        (x_error + 0.15 * y_error + 0.10 * width_error, index, candidate)
                    )

                if not options:
                    matched.append(None)
                    continue
                _, index, candidate = min(options, key=lambda item: item[0])
                used.add(index)
                matched.append(candidate)

            valid = [item for item in matched if item is not None]
            if len(valid) < _ANCHOR_MIN_MATCHED:
                continue

            centers_x = np.asarray([item[5] for item in valid], dtype=np.float32)
            centers_y = np.asarray([item[6] for item in valid], dtype=np.float32)
            positions = np.asarray(
                [index for index, item in enumerate(matched) if item is not None],
                dtype=np.float32,
            )
            heights = np.asarray([item[3] for item in valid], dtype=np.float32)
            widths = np.asarray([item[2] for item in valid], dtype=np.float32)

            slope, intercept = np.polyfit(centers_x, centers_y, 1)
            residuals = np.abs(centers_y - (slope * centers_x + intercept))
            median_height = float(np.median(heights))
            inliers = residuals <= max(2.5, 0.45 * median_height)
            if int(np.count_nonzero(inliers)) >= _ANCHOR_MIN_MATCHED:
                slope, intercept = np.polyfit(
                    centers_x[inliers],
                    centers_y[inliers],
                    1,
                )

            angle_deg = math.degrees(math.atan(float(slope)))
            if abs(angle_deg) > _ANCHOR_MAX_ABS_ANGLE_DEG:
                continue

            expected_x = first[5] + positions * step
            spacing_error = float(
                np.sqrt(np.mean(np.square(centers_x - expected_x))) / max(step, 1.0)
            )
            alignment_error = float(
                np.sqrt(
                    np.mean(
                        np.square(centers_y - (float(slope) * centers_x + float(intercept)))
                    )
                )
                / max(median_height, 1.0)
            )
            width_cv = float(np.std(widths) / max(float(np.mean(widths)), 1.0))
            median_y = float(np.median(centers_y))

            quality_penalty = (
                spacing_error
                + 0.50 * alignment_error
                + 0.50 * width_cv
                + 0.05 * (median_y / max(height, 1))
            )
            key = (len(valid), -quality_penalty)
            payload = {
                "candidate_count": len(candidates),
                "matched_count": len(valid),
                "angle_deg": round(angle_deg, 4),
                "median_bar_width_px": round(float(np.median(widths)), 3),
                "median_bar_height_px": round(median_height, 3),
                "bar_step_px": round(float(step), 3),
                "spacing_error": round(spacing_error, 6),
                "alignment_error": round(alignment_error, 6),
                "width_cv": round(width_cv, 6),
                "dark_threshold": round(dark_threshold, 3),
            }
            if best is None or key > best[0]:
                best = (key, matched, payload)

    return None if best is None else best[2]


def _rotate_crop(crop: np.ndarray, angle_deg: float) -> np.ndarray:
    height, width = crop.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0),
        angle_deg,
        1.0,
    )
    border_value: int | tuple[int, ...]
    if crop.ndim == 2:
        border_value = 255
    else:
        border_value = tuple(255 for _ in range(crop.shape[2]))
    return cv2.warpAffine(
        crop,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _prepare_training_crop(crop: np.ndarray) -> tuple[np.ndarray, dict[str, Any] | None, dict[str, Any]]:
    anchors_before = _fit_anchor_row(crop)
    angle_deg = float(anchors_before["angle_deg"]) if anchors_before is not None else 0.0

    if anchors_before is not None and abs(angle_deg) >= _ANCHOR_ROTATE_MIN_DEG:
        prepared = _rotate_crop(crop, angle_deg)
        deskew_status = "applied"
    else:
        prepared = crop.copy()
        deskew_status = "not_needed" if anchors_before is not None else "anchor_unavailable"

    anchors_after = _fit_anchor_row(prepared) if anchors_before is not None else None
    anchor_for_scale = anchors_after or anchors_before
    deskew_debug = {
        "status": deskew_status,
        "rotation_applied_deg": round(angle_deg, 4) if deskew_status == "applied" else 0.0,
        "anchor_before": anchors_before,
        "anchor_after": anchors_after,
    }
    return prepared, anchor_for_scale, deskew_debug


def build_training_postcode_canvas(
    crop: np.ndarray,
) -> tuple[np.ndarray, PixelRect, dict[str, Any]]:
    if crop is None or crop.size == 0:
        raise ValueError("Пустой postcode crop")

    source_height, source_width = crop.shape[:2]
    prepared, anchors, deskew_debug = _prepare_training_crop(crop)
    crop_height, crop_width = prepared.shape[:2]

    if anchors is not None:
        median_bar_width = float(anchors["median_bar_width_px"])
        desired_width = median_bar_width / _TRAINING_TARGET_BAR_WIDTH_RATIO
        min_width = crop_width * _TRAINING_CANVAS_WIDTH_SCALE_MIN
        max_width = crop_width * _TRAINING_CANVAS_WIDTH_SCALE_MAX
        canvas_width = int(round(float(np.clip(desired_width, min_width, max_width))))
        scale_source = "anchor_bar_width"
    else:
        median_bar_width = None
        canvas_width = int(round(crop_width * _TRAINING_CANVAS_FALLBACK_WIDTH_SCALE))
        scale_source = "fallback_crop_width"

    canvas_width = max(crop_width + 1, canvas_width)
    canvas_height = max(
        crop_height + 8,
        int(round(canvas_width / _TRAINING_CANVAS_ASPECT_RATIO)),
    )
    bottom_margin = max(
        4,
        int(round(canvas_height * _TRAINING_CANVAS_BOTTOM_MARGIN_RATIO)),
    )
    origin_x = 0
    origin_y = max(0, canvas_height - bottom_margin - crop_height)

    if prepared.ndim == 2:
        canvas = np.full((canvas_height, canvas_width), 255, dtype=prepared.dtype)
        canvas[origin_y : origin_y + crop_height, :crop_width] = prepared
    else:
        channels = prepared.shape[2]
        canvas = np.full((canvas_height, canvas_width, channels), 255, dtype=prepared.dtype)
        canvas[origin_y : origin_y + crop_height, :crop_width, :] = prepared

    search = PixelRect(origin_x, origin_y, crop_width, crop_height)
    debug = {
        "adapter": "postcode_crop_virtual_canonical_v2",
        "source_width_px": source_width,
        "source_height_px": source_height,
        "prepared_width_px": crop_width,
        "prepared_height_px": crop_height,
        "canvas_width_px": canvas_width,
        "canvas_height_px": canvas_height,
        "crop_origin_x_px": origin_x,
        "crop_origin_y_px": origin_y,
        "bottom_margin_px": bottom_margin,
        "canvas_aspect_ratio": _TRAINING_CANVAS_ASPECT_RATIO,
        "scale_source": scale_source,
        "target_bar_width_ratio": _TRAINING_TARGET_BAR_WIDTH_RATIO,
        "median_bar_width_px": median_bar_width,
        "effective_bar_width_ratio": (
            None
            if median_bar_width is None
            else round(median_bar_width / max(canvas_width, 1), 6)
        ),
        "deskew": deskew_debug,
    }
    return canvas, search, debug


def extract_postcode_crop_cells(
    image_path: Path,
) -> tuple[list[tuple[int, np.ndarray, dict[str, Any]]], dict[str, Any]]:
    crop = _decode_image(image_path.read_bytes())
    working_image, search, debug = build_training_postcode_canvas(crop)

    bbox, bar_count, density, confidence, features = _postcode_stencil_bbox(
        working_image,
        search,
    )
    if bbox is None:
        reason = features.get("rejection_reason") if isinstance(features, dict) else None
        raise RuntimeError(f"postcode_not_detected:{reason or 'unknown'}")

    region = RoiRegion(
        kind="recipient_postcode",
        status="stencil_detected",
        confidence=confidence,
        search_bbox=search,
        detected_bbox=bbox,
        bbox=bbox,
        component_count=bar_count,
        ink_density=density,
        detector="postcode_stencil",
        features=features,
    )
    roi = RoiDetection(
        status="detected",
        format=EnvelopeFormat.C4,
        coordinate_space="training_postcode_crop_virtual_canonical",
        mail_class="simple",
        regions=(region,),
        source_reference=(
            "ToolOCR postcode stencil detector; training crop virtual canonical adapter v2"
        ),
    )

    geometry = derive_postcode_digit_geometry(
        roi,
        image_width=int(working_image.shape[1]),
        image_height=int(working_image.shape[0]),
    )
    if geometry.status != "ready" or len(geometry.cells) != 6:
        raise RuntimeError(f"digit_geometry:{geometry.status}:{geometry.reason}")

    cells: list[tuple[int, np.ndarray, dict[str, Any]]] = []
    for cell in geometry.cells:
        canvas, preprocess = _normalize_digit_crop_with_debug(working_image, cell)
        if canvas is None:
            raise RuntimeError(f"digit_{cell.index}_preprocess:{preprocess.get('status')}")
        cells.append((cell.index, canvas, preprocess))

    debug["stencil_features"] = features
    debug["digit_cell_count"] = len(cells)
    return cells, debug

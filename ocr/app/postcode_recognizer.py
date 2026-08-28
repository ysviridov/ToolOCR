from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from .postcode_digit_cells import PostcodeDigitCell, PostcodeDigitGeometry


_TESSERACT_CONFIG = (
    "--oem 1 --psm 10 "
    "-c tessedit_char_whitelist=0123456789 "
    "-c classify_bln_numeric_mode=1"
)
_TESSERACT_TIMEOUT_SECONDS = 2.0
_CANVAS_WIDTH = 96
_CANVAS_HEIGHT = 128
_CANVAS_MARGIN = 12

# Печатная точечная сетка внутри индексной ячейки состоит из маленьких
# компактных connected-components. Рукописная цифра, напротив, образует
# существенно более крупные или вытянутые компоненты.
_DOT_MAX_AREA_RATIO = 0.0045
_STROKE_MIN_HEIGHT_RATIO = 0.12
_STROKE_MIN_WIDTH_RATIO = 0.18
_DOT_RESTORE_RADIUS_RATIO = 0.045
_DOT_SUPPRESSION_MIN_RETAINED_INK_RATIO = 0.20


@dataclass(frozen=True, slots=True)
class DigitRecognition:
    index: int
    status: str
    digit: str | None
    confidence: float | None
    reason: str | None = None
    preprocess: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PostcodeRecognition:
    status: str
    text: str
    postcode: str | None
    confidence: float | None
    min_digit_confidence: float | None
    structurally_valid: bool
    reason: str | None
    engine: str
    digits: tuple[DigitRecognition, ...]


def _suppress_stencil_dots(binary: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Удаляет печатные точки шаблона, сохраняя рукописные штрихи.

    Вход: белый фон (255), тёмный foreground (0).
    Сначала сохраняются крупные/вытянутые компоненты. Затем возвращаются
    маленькие компоненты, расположенные непосредственно рядом с сохранённым
    штрихом: это защищает слабые разрывы карандашной линии.
    """

    if binary is None or binary.size == 0:
        return binary, {
            "method": "stencil_dot_suppression_v1",
            "status": "empty",
        }

    height, width = binary.shape[:2]
    ink = (binary < 128).astype(np.uint8)
    ink_pixels_before = int(np.count_nonzero(ink))
    if ink_pixels_before == 0:
        return binary.copy(), {
            "method": "stencil_dot_suppression_v1",
            "status": "no_foreground",
            "components_before": 0,
            "components_after": 0,
            "suppressed_components": 0,
            "restored_components": 0,
            "ink_pixels_before": 0,
            "ink_pixels_after": 0,
            "suppressed_ink_ratio": 0.0,
        }

    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    cell_area = max(1, height * width)
    dot_max_area = max(8, int(round(cell_area * _DOT_MAX_AREA_RATIO)))
    stroke_min_height = max(4, int(round(height * _STROKE_MIN_HEIGHT_RATIO)))
    stroke_min_width = max(3, int(round(width * _STROKE_MIN_WIDTH_RATIO)))

    strong_labels: list[int] = []
    small_labels: list[int] = []
    for label in range(1, count):
        _, _, component_width, component_height, area = (
            int(value) for value in stats[label]
        )
        is_stroke = (
            area > dot_max_area
            or component_height >= stroke_min_height
            or component_width >= stroke_min_width
        )
        if is_stroke:
            strong_labels.append(label)
        else:
            small_labels.append(label)

    debug: dict[str, Any] = {
        "method": "stencil_dot_suppression_v1",
        "status": "applied",
        "components_before": max(0, count - 1),
        "strong_components": len(strong_labels),
        "dot_candidates": len(small_labels),
        "dot_max_area_px": dot_max_area,
        "stroke_min_height_px": stroke_min_height,
        "stroke_min_width_px": stroke_min_width,
        "ink_pixels_before": ink_pixels_before,
    }

    if not strong_labels:
        debug.update(
            {
                "status": "fallback_no_strong_components",
                "components_after": max(0, count - 1),
                "suppressed_components": 0,
                "restored_components": 0,
                "ink_pixels_after": ink_pixels_before,
                "suppressed_ink_ratio": 0.0,
            }
        )
        return binary.copy(), debug

    strong_mask = np.zeros_like(ink)
    for label in strong_labels:
        strong_mask[labels == label] = 1

    radius = max(1, int(round(min(height, width) * _DOT_RESTORE_RADIUS_RATIO)))
    kernel_size = radius * 2 + 1
    near_stroke = cv2.dilate(
        strong_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        iterations=1,
    )

    keep_labels = set(strong_labels)
    restored_components = 0
    for label in small_labels:
        component_mask = labels == label
        if np.any(near_stroke[component_mask] > 0):
            keep_labels.add(label)
            restored_components += 1

    cleaned_ink = np.zeros_like(ink)
    for label in keep_labels:
        cleaned_ink[labels == label] = 1

    ink_pixels_after = int(np.count_nonzero(cleaned_ink))
    retained_ratio = ink_pixels_after / float(max(ink_pixels_before, 1))
    if retained_ratio < _DOT_SUPPRESSION_MIN_RETAINED_INK_RATIO:
        debug.update(
            {
                "status": "fallback_excessive_suppression",
                "components_after": max(0, count - 1),
                "suppressed_components": 0,
                "restored_components": restored_components,
                "ink_pixels_after": ink_pixels_before,
                "suppressed_ink_ratio": 0.0,
                "retained_ink_ratio": round(retained_ratio, 4),
                "restore_radius_px": radius,
            }
        )
        return binary.copy(), debug

    cleaned = np.full_like(binary, 255)
    cleaned[cleaned_ink > 0] = 0
    suppressed_components = len(small_labels) - restored_components
    suppressed_ink_ratio = max(
        0.0,
        (ink_pixels_before - ink_pixels_after) / float(max(ink_pixels_before, 1)),
    )
    debug.update(
        {
            "components_after": len(keep_labels),
            "suppressed_components": suppressed_components,
            "restored_components": restored_components,
            "ink_pixels_after": ink_pixels_after,
            "suppressed_ink_ratio": round(suppressed_ink_ratio, 4),
            "retained_ink_ratio": round(retained_ratio, 4),
            "restore_radius_px": radius,
        }
    )
    return cleaned, debug


def _normalize_digit_crop_with_debug(
    image: np.ndarray,
    cell: PostcodeDigitCell,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Готовит одну stencil-ячейку и возвращает preprocessing diagnostics."""

    rect = cell.bbox
    crop = image[rect.y:rect.y2, rect.x:rect.x2]
    if crop.size == 0:
        return None, {
            "method": "stencil_dot_suppression_v1",
            "status": "empty_cell",
        }

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    height, width = gray.shape[:2]
    if height < 3 or width < 3:
        return None, {
            "method": "stencil_dot_suppression_v1",
            "status": "cell_too_small",
        }

    sigma = max(4.0, min(height, width) * 0.18)
    kernel = max(9, int(round(sigma * 4.0)) | 1)
    kernel = min(kernel, 101)
    if kernel % 2 == 0:
        kernel += 1
    background = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    corrected = gray.astype(np.float32) * 235.0 / np.maximum(
        background.astype(np.float32),
        45.0,
    )
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    corrected = cv2.GaussianBlur(corrected, (3, 3), 0)

    threshold_value, binary = cv2.threshold(
        corrected,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Убираем только самый внешний край ячейки. Полезную область не
    # подрезаем: corpus-validation уже подтвердила, что цифра внутри.
    edge = max(1, min(height, width) // 80)
    binary[:edge, :] = 255
    binary[-edge:, :] = 255
    binary[:, :edge] = 255
    binary[:, -edge:] = 255

    cleaned, suppression_debug = _suppress_stencil_dots(binary)
    suppression_debug["otsu_threshold"] = round(float(threshold_value), 2)
    suppression_debug["cell_width_px"] = int(width)
    suppression_debug["cell_height_px"] = int(height)

    ys, xs = np.where(cleaned < 128)
    if xs.size < 8 or ys.size < 8:
        suppression_debug["status"] = "insufficient_foreground_after_suppression"
        return None, suppression_debug

    x1 = max(0, int(xs.min()) - 2)
    x2 = min(width, int(xs.max()) + 3)
    y1 = max(0, int(ys.min()) - 2)
    y2 = min(height, int(ys.max()) + 3)
    glyph = cleaned[y1:y2, x1:x2]
    if glyph.size == 0:
        suppression_debug["status"] = "empty_glyph_after_suppression"
        return None, suppression_debug

    target_w = _CANVAS_WIDTH - 2 * _CANVAS_MARGIN
    target_h = _CANVAS_HEIGHT - 2 * _CANVAS_MARGIN
    scale = min(target_w / glyph.shape[1], target_h / glyph.shape[0])
    if scale <= 0:
        suppression_debug["status"] = "invalid_glyph_scale"
        return None, suppression_debug

    new_w = max(1, int(round(glyph.shape[1] * scale)))
    new_h = max(1, int(round(glyph.shape[0] * scale)))
    resized = cv2.resize(
        glyph,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )

    canvas = np.full((_CANVAS_HEIGHT, _CANVAS_WIDTH), 255, dtype=np.uint8)
    x = (_CANVAS_WIDTH - new_w) // 2
    y = (_CANVAS_HEIGHT - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    suppression_debug["glyph_bbox"] = {
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1,
    }
    suppression_debug["canvas_width_px"] = _CANVAS_WIDTH
    suppression_debug["canvas_height_px"] = _CANVAS_HEIGHT
    return canvas, suppression_debug


def _normalize_digit_crop(image: np.ndarray, cell: PostcodeDigitCell) -> np.ndarray | None:
    """Совместимый wrapper: возвращает только нормализованный canvas."""

    canvas, _ = _normalize_digit_crop_with_debug(image, cell)
    return canvas


def _recognize_digit_crop(crop: np.ndarray, index: int) -> DigitRecognition:
    try:
        data = pytesseract.image_to_data(
            crop,
            lang="eng",
            config=_TESSERACT_CONFIG,
            output_type=Output.DICT,
            timeout=_TESSERACT_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        return DigitRecognition(
            index=index,
            status="error",
            digit=None,
            confidence=None,
            reason=f"{type(exc).__name__}: {exc}",
        )

    candidates: list[tuple[float, str]] = []
    for text, confidence_raw in zip(
        data.get("text", []),
        data.get("conf", []),
        strict=False,
    ):
        text = str(text).strip()
        if len(text) != 1 or text not in "0123456789":
            continue
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            continue
        if confidence < 0:
            continue
        candidates.append((confidence, text))

    if not candidates:
        return DigitRecognition(
            index=index,
            status="unrecognized",
            digit=None,
            confidence=None,
            reason="no_single_digit_candidate",
        )

    confidence, digit = max(candidates, key=lambda item: item[0])
    return DigitRecognition(
        index=index,
        status="recognized",
        digit=digit,
        confidence=round(max(0.0, min(1.0, confidence / 100.0)), 4),
    )


def recognize_postcode_digits(
    image: np.ndarray,
    geometry: PostcodeDigitGeometry,
) -> PostcodeRecognition:
    """Распознаёт шесть digit-cell независимо и собирает сырой индекс."""

    if geometry.status != "ready" or len(geometry.cells) != 6:
        return PostcodeRecognition(
            status="unavailable",
            text="??????",
            postcode=None,
            confidence=None,
            min_digit_confidence=None,
            structurally_valid=False,
            reason=geometry.reason or "digit_geometry_unavailable",
            engine="tesseract_single_digit+stencil_dot_suppression_v1",
            digits=(),
        )

    digits: list[DigitRecognition] = []
    for cell in geometry.cells:
        normalized, preprocess = _normalize_digit_crop_with_debug(image, cell)
        if normalized is None:
            digits.append(
                DigitRecognition(
                    index=cell.index,
                    status="unrecognized",
                    digit=None,
                    confidence=None,
                    reason="empty_or_insufficient_foreground",
                    preprocess=preprocess,
                )
            )
            continue

        recognized = _recognize_digit_crop(normalized, cell.index)
        digits.append(replace(recognized, preprocess=preprocess))

    text = "".join(item.digit if item.digit is not None else "?" for item in digits)
    complete = len(digits) == 6 and all(item.digit is not None for item in digits)
    confidences = [item.confidence for item in digits if item.confidence is not None]
    mean_confidence = (
        round(float(sum(confidences) / len(confidences)), 4)
        if complete and len(confidences) == 6
        else None
    )
    min_confidence = (
        round(float(min(confidences)), 4)
        if complete and len(confidences) == 6
        else None
    )

    if any(item.status == "error" for item in digits):
        status = "error"
        reason = "digit_recognizer_error"
    elif not complete:
        status = "incomplete"
        reason = "one_or_more_digits_unrecognized"
    else:
        status = "recognized"
        reason = None

    postcode = text if complete else None
    structurally_valid = bool(
        complete
        and postcode is not None
        and len(postcode) == 6
        and postcode.isdigit()
        and postcode[0] != "0"
    )

    return PostcodeRecognition(
        status=status,
        text=text,
        postcode=postcode,
        confidence=mean_confidence,
        min_digit_confidence=min_confidence,
        structurally_valid=structurally_valid,
        reason=reason,
        engine="tesseract_single_digit+stencil_dot_suppression_v1",
        digits=tuple(digits),
    )


def postcode_recognition_to_dict(result: PostcodeRecognition) -> dict[str, Any]:
    return {
        "status": result.status,
        "text": result.text,
        "postcode": result.postcode,
        "confidence": result.confidence,
        "min_digit_confidence": result.min_digit_confidence,
        "structurally_valid": result.structurally_valid,
        "reason": result.reason,
        "engine": result.engine,
        "digits": [
            {
                "index": item.index,
                "status": item.status,
                "digit": item.digit,
                "confidence": item.confidence,
                "reason": item.reason,
                "preprocess": item.preprocess,
            }
            for item in result.digits
        ],
    }


def postcode_digit_overlay_labels(result: PostcodeRecognition) -> dict[int, str]:
    return {
        item.index: f"D{item.index}={item.digit if item.digit is not None else '?'}"
        for item in result.digits
    }


def draw_postcode_recognition_summary(
    image: np.ndarray,
    geometry: PostcodeDigitGeometry,
    recognition: PostcodeRecognition,
) -> None:
    """Рисует крупный итоговый OCR postcode в правом нижнем углу ROI."""

    if geometry.status != "ready" or not geometry.cells:
        return

    image_height, image_width = image.shape[:2]
    scale = max(1.0, max(image.shape[:2]) / 1600.0)
    font_scale = max(1.05, 1.00 * scale)
    thickness = max(2, round(3.0 * scale))
    margin = max(18, round(26 * scale))

    if recognition.confidence is None:
        label = f"POSTCODE OCR: {recognition.text}"
    else:
        label = (
            f"POSTCODE OCR: {recognition.text}  "
            f"conf={recognition.confidence:.2f}"
        )

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    x = max(margin, image_width - margin - text_width)
    y = max(text_height + margin, image_height - margin - baseline)

    cv2.putText(
        image,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (20, 20, 20),
        thickness + max(2, round(2.0 * scale)),
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 90, 0),
        thickness,
        cv2.LINE_AA,
    )

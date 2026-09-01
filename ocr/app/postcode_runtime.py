from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .postcode_digit_cells import PostcodeDigitGeometry
from .postcode_onnx import (
    PostcodeOnnxError,
    configured_onnx_model_path,
    configured_recognizer_engine,
    load_configured_onnx_net,
    onnx_engine_label,
    predict_digit_onnx,
    tesseract_fallback_enabled,
)
from .postcode_recognizer import (
    _normalize_digit_crop_with_debug,
    _recognize_digit_crop as _recognize_digit_crop_tesseract,
)


_TESSERACT_ENGINE = "tesseract_single_digit+stencil_dot_suppression_v1"


@dataclass(frozen=True, slots=True)
class RuntimeDigitRecognition:
    index: int
    status: str
    digit: str | None
    confidence: float | None
    reason: str | None = None
    preprocess: dict[str, Any] | None = None
    top3: tuple[tuple[str, float], ...] = ()
    engine: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimePostcodeRecognition:
    status: str
    text: str
    postcode: str | None
    confidence: float | None
    min_digit_confidence: float | None
    structurally_valid: bool
    reason: str | None
    engine: str
    digits: tuple[RuntimeDigitRecognition, ...]
    geometric_mean_confidence: float | None = None
    model_path: str | None = None


def _tesseract_digit(
    canvas: np.ndarray,
    index: int,
    preprocess: dict[str, Any],
) -> RuntimeDigitRecognition:
    recognized = _recognize_digit_crop_tesseract(canvas, index)
    top3: tuple[tuple[str, float], ...] = ()
    if recognized.digit is not None and recognized.confidence is not None:
        top3 = ((recognized.digit, float(recognized.confidence)),)
    return RuntimeDigitRecognition(
        index=index,
        status=recognized.status,
        digit=recognized.digit,
        confidence=recognized.confidence,
        reason=recognized.reason,
        preprocess=preprocess,
        top3=top3,
        engine="tesseract_single_digit",
    )


def _onnx_digit(
    canvas: np.ndarray,
    index: int,
    preprocess: dict[str, Any],
    *,
    net,
) -> RuntimeDigitRecognition:
    prediction = predict_digit_onnx(canvas, net=net)
    return RuntimeDigitRecognition(
        index=index,
        status="recognized",
        digit=prediction.digit,
        confidence=prediction.confidence,
        reason=None,
        preprocess=preprocess,
        top3=prediction.top3,
        engine="onnx",
    )


def _aggregate_engine(
    used_engines: set[str],
    *,
    model_path: Path,
    onnx_requested: bool,
    initial_onnx_error: str | None,
) -> str:
    if used_engines == {"onnx"}:
        return onnx_engine_label(model_path)
    if used_engines == {"tesseract_single_digit"}:
        if onnx_requested and initial_onnx_error:
            return "tesseract_fallback_from_onnx+stencil_dot_suppression_v1"
        return _TESSERACT_ENGINE
    if "onnx" in used_engines and "tesseract_single_digit" in used_engines:
        return (
            f"{onnx_engine_label(model_path)}"
            "+tesseract_fallback"
        )
    if onnx_requested:
        return onnx_engine_label(model_path)
    return _TESSERACT_ENGINE


def _confidence_metrics(
    digits: list[RuntimeDigitRecognition],
    *,
    complete: bool,
) -> tuple[float | None, float | None, float | None]:
    confidences = [item.confidence for item in digits if item.confidence is not None]
    if not complete or len(confidences) != 6:
        return None, None, None

    values = [max(0.0, min(1.0, float(value))) for value in confidences]
    arithmetic = round(float(sum(values) / len(values)), 6)
    minimum = round(float(min(values)), 6)
    if any(value <= 0.0 for value in values):
        geometric = 0.0
    else:
        geometric = round(
            float(math.exp(sum(math.log(value) for value in values) / len(values))),
            6,
        )
    return arithmetic, minimum, geometric


def recognize_postcode_digits(
    image: np.ndarray,
    geometry: PostcodeDigitGeometry,
) -> RuntimePostcodeRecognition:
    """Runtime recognizer: ONNX primary, Tesseract fallback/debug.

    Preprocessing и digit-cell geometry полностью общие с training pipeline.
    ONNX получает ровно те же grayscale 96x128 canvas после
    stencil_dot_suppression_v1.
    """

    mode = configured_recognizer_engine()
    model_path = configured_onnx_model_path()
    onnx_requested = mode in {"onnx", "auto"}

    if geometry.status != "ready" or len(geometry.cells) != 6:
        engine = onnx_engine_label(model_path) if onnx_requested else _TESSERACT_ENGINE
        return RuntimePostcodeRecognition(
            status="unavailable",
            text="??????",
            postcode=None,
            confidence=None,
            min_digit_confidence=None,
            geometric_mean_confidence=None,
            structurally_valid=False,
            reason=geometry.reason or "digit_geometry_unavailable",
            engine=engine,
            digits=(),
            model_path=str(model_path) if onnx_requested else None,
        )

    net = None
    initial_onnx_error: str | None = None
    fallback_enabled = tesseract_fallback_enabled()

    if onnx_requested:
        if mode == "auto" and not model_path.is_file():
            initial_onnx_error = f"ONNX model не найден: {model_path}"
        else:
            try:
                net = load_configured_onnx_net()
            except Exception as exc:
                initial_onnx_error = f"{type(exc).__name__}: {exc}"

    digits: list[RuntimeDigitRecognition] = []
    used_engines: set[str] = set()

    for cell in geometry.cells:
        normalized, preprocess = _normalize_digit_crop_with_debug(image, cell)
        if normalized is None:
            digits.append(
                RuntimeDigitRecognition(
                    index=cell.index,
                    status="unrecognized",
                    digit=None,
                    confidence=None,
                    reason="empty_or_insufficient_foreground",
                    preprocess=preprocess,
                    engine=None,
                )
            )
            continue

        if onnx_requested and net is not None:
            try:
                recognized = _onnx_digit(
                    normalized,
                    cell.index,
                    preprocess,
                    net=net,
                )
                used_engines.add("onnx")
                digits.append(recognized)
                continue
            except Exception as exc:
                if not fallback_enabled:
                    digits.append(
                        RuntimeDigitRecognition(
                            index=cell.index,
                            status="error",
                            digit=None,
                            confidence=None,
                            reason=f"onnx_inference_error:{type(exc).__name__}: {exc}",
                            preprocess=preprocess,
                            engine="onnx",
                        )
                    )
                    used_engines.add("onnx")
                    continue

        if onnx_requested and net is None and not fallback_enabled:
            digits.append(
                RuntimeDigitRecognition(
                    index=cell.index,
                    status="error",
                    digit=None,
                    confidence=None,
                    reason=f"onnx_model_unavailable:{initial_onnx_error or 'unknown'}",
                    preprocess=preprocess,
                    engine="onnx",
                )
            )
            used_engines.add("onnx")
            continue

        recognized = _tesseract_digit(normalized, cell.index, preprocess)
        digits.append(recognized)
        used_engines.add("tesseract_single_digit")

    text = "".join(item.digit if item.digit is not None else "?" for item in digits)
    complete = len(digits) == 6 and all(item.digit is not None for item in digits)
    mean_confidence, min_confidence, geometric_confidence = _confidence_metrics(
        digits,
        complete=complete,
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

    engine = _aggregate_engine(
        used_engines,
        model_path=model_path,
        onnx_requested=onnx_requested,
        initial_onnx_error=initial_onnx_error,
    )
    return RuntimePostcodeRecognition(
        status=status,
        text=text,
        postcode=postcode,
        confidence=mean_confidence,
        min_digit_confidence=min_confidence,
        geometric_mean_confidence=geometric_confidence,
        structurally_valid=structurally_valid,
        reason=reason,
        engine=engine,
        digits=tuple(digits),
        model_path=str(model_path) if onnx_requested else None,
    )


def postcode_recognition_to_dict(result: RuntimePostcodeRecognition) -> dict[str, Any]:
    return {
        "status": result.status,
        "text": result.text,
        "postcode": result.postcode,
        "confidence": result.confidence,
        "min_digit_confidence": result.min_digit_confidence,
        "geometric_mean_confidence": result.geometric_mean_confidence,
        "structurally_valid": result.structurally_valid,
        "reason": result.reason,
        "engine": result.engine,
        "model_path": result.model_path,
        "digits": [
            {
                "index": item.index,
                "status": item.status,
                "digit": item.digit,
                "confidence": item.confidence,
                "top3": [
                    {"digit": digit, "probability": probability}
                    for digit, probability in item.top3
                ],
                "reason": item.reason,
                "engine": item.engine,
                "preprocess": item.preprocess,
            }
            for item in result.digits
        ],
    }


def postcode_digit_overlay_labels(result: RuntimePostcodeRecognition) -> dict[int, str]:
    return {
        item.index: f"D{item.index}={item.digit if item.digit is not None else '?'}"
        for item in result.digits
    }


def draw_postcode_recognition_summary(
    image: np.ndarray,
    geometry: PostcodeDigitGeometry,
    recognition: RuntimePostcodeRecognition,
) -> None:
    """Runtime summary; визуально совместим с прежним Tesseract overlay."""

    if geometry.status != "ready" or not geometry.cells:
        return

    import cv2

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

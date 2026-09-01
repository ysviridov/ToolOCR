from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


POSTCODE_RECOGNIZER_ENGINE_ENV = "POSTCODE_RECOGNIZER_ENGINE"
POSTCODE_ONNX_MODEL_ENV = "POSTCODE_ONNX_MODEL"
POSTCODE_ONNX_FALLBACK_TESSERACT_ENV = "POSTCODE_ONNX_FALLBACK_TESSERACT"

DEFAULT_RECOGNIZER_ENGINE = "onnx"
DEFAULT_ONNX_MODEL_PATH = "/app/models/postcode_digit_v1.onnx"

_CANVAS_WIDTH = 96
_CANVAS_HEIGHT = 128
_NUM_CLASSES = 10
_INFERENCE_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class OnnxDigitPrediction:
    digit: str
    confidence: float
    top3: tuple[tuple[str, float], ...]


class PostcodeOnnxError(RuntimeError):
    pass


def configured_recognizer_engine() -> str:
    value = os.getenv(POSTCODE_RECOGNIZER_ENGINE_ENV, DEFAULT_RECOGNIZER_ENGINE)
    value = str(value or "").strip().lower()
    if value not in {"onnx", "tesseract", "auto"}:
        return DEFAULT_RECOGNIZER_ENGINE
    return value


def configured_onnx_model_path() -> Path:
    value = os.getenv(POSTCODE_ONNX_MODEL_ENV, DEFAULT_ONNX_MODEL_PATH)
    return Path(str(value or DEFAULT_ONNX_MODEL_PATH)).expanduser()


def tesseract_fallback_enabled() -> bool:
    value = str(
        os.getenv(POSTCODE_ONNX_FALLBACK_TESSERACT_ENV, "1") or "1"
    ).strip().lower()
    return value not in {"0", "false", "no", "off"}


def onnx_engine_label(model_path: Path | None = None) -> str:
    path = model_path or configured_onnx_model_path()
    name = path.stem or "postcode_digit"
    return f"onnx_{name}+stencil_dot_suppression_v1"


@lru_cache(maxsize=4)
def _load_onnx_net(path_value: str):
    path = Path(path_value)
    if not path.is_file():
        raise PostcodeOnnxError(f"ONNX model не найден: {path}")
    try:
        return cv2.dnn.readNetFromONNX(str(path))
    except Exception as exc:  # pragma: no cover - конкретный тип зависит от OpenCV
        raise PostcodeOnnxError(
            f"Не удалось загрузить ONNX model {path}: {type(exc).__name__}: {exc}"
        ) from exc


def load_configured_onnx_net():
    path = configured_onnx_model_path()
    return _load_onnx_net(str(path.resolve()))


def clear_onnx_model_cache() -> None:
    _load_onnx_net.cache_clear()


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size != _NUM_CLASSES:
        raise PostcodeOnnxError(
            f"ONNX logits имеют размер {values.size}, ожидалось {_NUM_CLASSES}"
        )
    if not np.all(np.isfinite(values)):
        raise PostcodeOnnxError("ONNX logits содержат NaN/Inf")
    values -= float(np.max(values))
    exp_values = np.exp(values)
    denominator = float(np.sum(exp_values))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise PostcodeOnnxError("Некорректная сумма softmax")
    return exp_values / denominator


def predict_digit_onnx(
    canvas: np.ndarray,
    *,
    net=None,
) -> OnnxDigitPrediction:
    if canvas is None or canvas.size == 0:
        raise PostcodeOnnxError("Пустой digit canvas")

    gray = canvas
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.shape != (_CANVAS_HEIGHT, _CANVAS_WIDTH):
        raise PostcodeOnnxError(
            f"Некорректный digit canvas {gray.shape}; ожидалось "
            f"({_CANVAS_HEIGHT}, {_CANVAS_WIDTH})"
        )

    tensor = ((255.0 - gray.astype(np.float32)) / 255.0)[None, None, :, :]
    current_net = net if net is not None else load_configured_onnx_net()

    try:
        # cv2.dnn.Net хранит input внутри объекта; один Net не должен получать
        # одновременные setInput/forward из разных запросов.
        with _INFERENCE_LOCK:
            current_net.setInput(tensor)
            logits = current_net.forward()
    except Exception as exc:
        raise PostcodeOnnxError(
            f"Ошибка ONNX inference: {type(exc).__name__}: {exc}"
        ) from exc

    probabilities = _softmax(logits)
    order = np.argsort(probabilities)[::-1][:3]
    top3 = tuple(
        (str(int(index)), round(float(probabilities[index]), 6))
        for index in order
    )
    best = int(order[0])
    return OnnxDigitPrediction(
        digit=str(best),
        confidence=round(float(probabilities[best]), 6),
        top3=top3,
    )

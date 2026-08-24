from __future__ import annotations

import os
import time
from functools import lru_cache
from collections import defaultdict
from enum import Enum
from typing import Any

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from pytesseract import Output

APP_VERSION = "2.0.0"
MAX_UPLOAD_MB = int(os.environ.get("OCR_MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_PIXELS = int(os.environ.get("OCR_MAX_PIXELS", "40000000"))
DEFAULT_LANG = os.environ.get("OCR_DEFAULT_LANG", "rus+eng")
DEFAULT_PSM = int(os.environ.get("OCR_DEFAULT_PSM", "11"))
ALLOWED_PSM = {3, 6, 11, 12}

app = FastAPI(
    title="ToolOCR OCR API",
    version=APP_VERSION,
    description=(
        "Stage 2: приём изображения, предобработка и базовое OCR. "
        "Baseline-движок — Tesseract 5 rus+eng."
    ),
)


class PreprocessMode(str, Enum):
    none = "none"
    gray = "gray"
    otsu = "otsu"
    adaptive = "adaptive"
    auto = "auto"


class BBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class OCRWord(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    block: int
    paragraph: int
    line: int


class OCRLine(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    block: int
    paragraph: int
    line: int


class OCRAttempt(BaseModel):
    preprocess: str
    confidence: float = Field(ge=0.0, le=1.0)
    text: str
    ocr_ms: float


class ImageInfo(BaseModel):
    filename: str | None
    content_type: str | None
    width: int
    height: int
    channels: int
    bytes_received: int
    deskew_angle: float


class OCRTiming(BaseModel):
    decode_ms: float
    preprocess_ms: float
    ocr_ms: float
    total_ms: float


class OCRResponse(BaseModel):
    engine: str
    engine_version: str
    language: str
    psm: int
    selected_preprocess: str
    confidence: float
    text: str
    image: ImageInfo
    timing: OCRTiming
    lines: list[OCRLine]
    words: list[OCRWord]
    alternatives: list[OCRAttempt] | None = None


@lru_cache(maxsize=1)
def _engine_version() -> str:
    return str(pytesseract.get_tesseract_version()).splitlines()[0]


@lru_cache(maxsize=1)
def _available_languages() -> tuple[str, ...]:
    return tuple(pytesseract.get_languages(config=""))


def _round_ms(value: float) -> float:
    return round(value, 3)


def _decode_image(raw: bytes) -> np.ndarray:
    encoded = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=415,
            detail="Не удалось декодировать изображение. Поддерживаются JPEG/PNG/BMP/TIFF.",
        )
    height, width = image.shape[:2]
    if height * width > MAX_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"Изображение слишком большое: {width}x{height}; максимум {MAX_PIXELS} пикселей",
        )
    return image


def _deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Осторожная коррекция небольшого наклона текста.

    Если оценка выглядит ненадёжной (> 15 градусов), изображение не вращается.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return image, 0.0

    # minAreaRect принимает координаты (x, y), а np.where возвращает (y, x).
    points = coords[:, ::-1].astype(np.float32)
    rect_angle = cv2.minAreaRect(points)[-1]
    if rect_angle < -45.0:
        angle = -(90.0 + rect_angle)
    else:
        angle = -rect_angle

    if not np.isfinite(angle) or abs(angle) < 0.2 or abs(angle) > 15.0:
        return image, 0.0

    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated, round(float(angle), 3)


def _preprocess(image: np.ndarray, mode: PreprocessMode) -> np.ndarray:
    if mode == PreprocessMode.none:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode == PreprocessMode.gray:
        return gray
    if mode == PreprocessMode.otsu:
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if mode == PreprocessMode.adaptive:
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,
            11,
        )
    raise ValueError(f"Неподдерживаемый режим предобработки: {mode}")


def _bbox_union(boxes: list[BBox]) -> BBox:
    x1 = min(box.x for box in boxes)
    y1 = min(box.y for box in boxes)
    x2 = max(box.x + box.width for box in boxes)
    y2 = max(box.y + box.height for box in boxes)
    return BBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def _recognize(image: np.ndarray, lang: str, psm: int) -> tuple[str, float, list[OCRWord], list[OCRLine]]:
    config = f"--oem 1 --psm {psm}"
    data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=Output.DICT)

    words: list[OCRWord] = []
    groups: dict[tuple[int, int, int], list[OCRWord]] = defaultdict(list)

    for idx, raw_text in enumerate(data.get("text", [])):
        text = (raw_text or "").strip()
        try:
            conf_raw = float(data["conf"][idx])
        except (TypeError, ValueError):
            conf_raw = -1.0
        if not text or conf_raw < 0:
            continue

        confidence = max(0.0, min(1.0, conf_raw / 100.0))
        word = OCRWord(
            text=text,
            confidence=round(confidence, 4),
            bbox=BBox(
                x=int(data["left"][idx]),
                y=int(data["top"][idx]),
                width=int(data["width"][idx]),
                height=int(data["height"][idx]),
            ),
            block=int(data["block_num"][idx]),
            paragraph=int(data["par_num"][idx]),
            line=int(data["line_num"][idx]),
        )
        words.append(word)
        groups[(word.block, word.paragraph, word.line)].append(word)

    lines: list[OCRLine] = []
    for (block, paragraph, line_num), line_words in sorted(groups.items()):
        text = " ".join(item.text for item in line_words)
        char_weight = sum(max(1, len(item.text)) for item in line_words)
        confidence = sum(item.confidence * max(1, len(item.text)) for item in line_words) / char_weight
        lines.append(
            OCRLine(
                text=text,
                confidence=round(confidence, 4),
                bbox=_bbox_union([item.bbox for item in line_words]),
                block=block,
                paragraph=paragraph,
                line=line_num,
            )
        )

    text = "\n".join(line.text for line in lines)
    if words:
        char_weight = sum(max(1, len(item.text)) for item in words)
        confidence = sum(item.confidence * max(1, len(item.text)) for item in words) / char_weight
    else:
        confidence = 0.0

    return text, round(confidence, 4), words, lines


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        version = _engine_version()
        languages = _available_languages()
    except Exception as exc:  # pragma: no cover - зависит от системного бинарника
        raise HTTPException(status_code=503, detail=f"Tesseract недоступен: {exc}") from exc

    required = {part for part in DEFAULT_LANG.split("+") if part}
    missing = sorted(required.difference(languages))
    if missing:
        raise HTTPException(status_code=503, detail=f"Не установлены OCR-языки: {', '.join(missing)}")

    return {
        "status": "ok",
        "stage": "2",
        "engine": "tesseract",
        "engine_version": version,
        "default_language": DEFAULT_LANG,
        "available_languages": languages,
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_pixels": MAX_PIXELS,
    }


@app.post("/v1/ocr/recognize", response_model=OCRResponse)
async def recognize(
    file: UploadFile = File(..., description="JPEG/PNG/BMP/TIFF"),
    preprocess: PreprocessMode = Query(default=PreprocessMode.auto),
    deskew: bool = Query(default=True),
    psm: int = Query(default=DEFAULT_PSM),
    lang: str = Query(default=DEFAULT_LANG, min_length=2, max_length=64),
    include_alternatives: bool = Query(default=False),
) -> OCRResponse:
    total_started = time.perf_counter()

    if psm not in ALLOWED_PSM:
        raise HTTPException(status_code=422, detail=f"Допустимые psm: {sorted(ALLOWED_PSM)}")

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Максимальный размер файла: {MAX_UPLOAD_MB} МБ")
    if not raw:
        raise HTTPException(status_code=400, detail="Передан пустой файл")

    decode_started = time.perf_counter()
    image = _decode_image(raw)
    decode_ms = (time.perf_counter() - decode_started) * 1000.0

    prep_started = time.perf_counter()
    if deskew:
        image, deskew_angle = _deskew(image)
    else:
        deskew_angle = 0.0

    modes = (
        [PreprocessMode.gray, PreprocessMode.otsu]
        if preprocess == PreprocessMode.auto
        else [preprocess]
    )
    prepared = [(mode, _preprocess(image, mode)) for mode in modes]
    preprocess_ms = (time.perf_counter() - prep_started) * 1000.0

    attempts: list[tuple[PreprocessMode, str, float, list[OCRWord], list[OCRLine], float]] = []
    ocr_total_ms = 0.0
    try:
        for mode, variant in prepared:
            ocr_started = time.perf_counter()
            text, confidence, words, lines = _recognize(variant, lang=lang, psm=psm)
            ocr_ms = (time.perf_counter() - ocr_started) * 1000.0
            ocr_total_ms += ocr_ms
            attempts.append((mode, text, confidence, words, lines, ocr_ms))
    except pytesseract.TesseractError as exc:
        raise HTTPException(status_code=422, detail=f"Ошибка Tesseract: {exc}") from exc

    # Сначала confidence, затем число распознанных символов как tie-breaker.
    selected = max(attempts, key=lambda item: (item[2], len(item[1])))
    selected_mode, text, confidence, words, lines, _ = selected

    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else int(image.shape[2])
    total_ms = (time.perf_counter() - total_started) * 1000.0

    alternatives = None
    if include_alternatives:
        alternatives = [
            OCRAttempt(
                preprocess=mode.value,
                confidence=conf,
                text=attempt_text,
                ocr_ms=_round_ms(attempt_ms),
            )
            for mode, attempt_text, conf, _, _, attempt_ms in attempts
        ]

    try:
        engine_version = _engine_version()
    except Exception:
        engine_version = "unknown"

    return OCRResponse(
        engine="tesseract",
        engine_version=engine_version,
        language=lang,
        psm=psm,
        selected_preprocess=selected_mode.value,
        confidence=confidence,
        text=text,
        image=ImageInfo(
            filename=file.filename,
            content_type=file.content_type,
            width=width,
            height=height,
            channels=channels,
            bytes_received=len(raw),
            deskew_angle=deskew_angle,
        ),
        timing=OCRTiming(
            decode_ms=_round_ms(decode_ms),
            preprocess_ms=_round_ms(preprocess_ms),
            ocr_ms=_round_ms(ocr_total_ms),
            total_ms=_round_ms(total_ms),
        ),
        lines=lines,
        words=words,
        alternatives=alternatives,
    )

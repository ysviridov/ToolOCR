from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .gost_r_51506_99 import ENVELOPE_SPECS, EnvelopeFormat, EnvelopeSpec
from .layout import order_quad


CALIBRATION_VERSION = 1
DEFAULT_SIZE_TOLERANCE_MM = 8.0


class CameraCalibrationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PlaneCalibration:
    version: int
    standard: str
    reference_format: str
    reference_width_mm: float
    reference_height_mm: float
    image_width_px: int
    image_height_px: int
    image_aspect_ratio: float
    homography_norm_to_mm: np.ndarray


@dataclass(frozen=True, slots=True)
class MetricMeasurement:
    width_mm: float
    height_mm: float
    top_width_mm: float
    bottom_width_mm: float
    left_height_mm: float
    right_height_mm: float
    width_exact: bool
    height_exact: bool


@dataclass(frozen=True, slots=True)
class MetricFormatCandidate:
    format: EnvelopeFormat
    score: float
    width_error_mm: float
    height_error_mm: float
    width_mode: str
    height_mode: str


@dataclass(frozen=True, slots=True)
class MetricFormatDecision:
    status: str
    format: EnvelopeFormat | None
    confidence: float
    margin: float
    measurement: MetricMeasurement
    candidates: tuple[MetricFormatCandidate, ...]


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _normalize_points(points: np.ndarray, width_px: int, height_px: int) -> np.ndarray:
    if width_px <= 0 or height_px <= 0:
        raise ValueError("Размер изображения должен быть положительным")
    normalized = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    normalized[:, 0] /= float(width_px)
    normalized[:, 1] /= float(height_px)
    return normalized


def build_plane_calibration(
    quad: np.ndarray,
    *,
    image_width_px: int,
    image_height_px: int,
    spec: EnvelopeSpec,
    standard: str,
) -> PlaneCalibration:
    """Строит homography image(normalized) -> mm по отправлению известного формата.

    Калибровочный конверт должен полностью попадать в кадр и лежать длинной
    стороной примерно горизонтально. Поскольку homography строится в
    нормализованных координатах изображения, одинаковый FOV можно использовать
    при другом разрешении с тем же отношением сторон.
    """

    ordered = order_quad(quad)
    tl, tr, br, bl = ordered
    top = _distance(tl, tr)
    bottom = _distance(bl, br)
    left = _distance(tl, bl)
    right = _distance(tr, br)
    observed_width = (top + bottom) / 2.0
    observed_height = (left + right) / 2.0
    if observed_width <= observed_height:
        raise CameraCalibrationError(
            "Калибровочный эталон должен лежать длинной стороной горизонтально"
        )

    source = _normalize_points(ordered, image_width_px, image_height_px)
    destination = np.array(
        [
            [0.0, 0.0],
            [spec.width_mm, 0.0],
            [spec.width_mm, spec.height_mm],
            [0.0, spec.height_mm],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source.astype(np.float32), destination)
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise CameraCalibrationError("Не удалось построить устойчивую homography")

    return PlaneCalibration(
        version=CALIBRATION_VERSION,
        standard=standard,
        reference_format=spec.code.value,
        reference_width_mm=spec.width_mm,
        reference_height_mm=spec.height_mm,
        image_width_px=int(image_width_px),
        image_height_px=int(image_height_px),
        image_aspect_ratio=float(image_width_px) / float(image_height_px),
        homography_norm_to_mm=matrix.astype(np.float64),
    )


def calibration_to_dict(calibration: PlaneCalibration) -> dict:
    return {
        "version": calibration.version,
        "standard": calibration.standard,
        "reference_format": calibration.reference_format,
        "reference_width_mm": round(calibration.reference_width_mm, 4),
        "reference_height_mm": round(calibration.reference_height_mm, 4),
        "image_width_px": calibration.image_width_px,
        "image_height_px": calibration.image_height_px,
        "image_aspect_ratio": round(calibration.image_aspect_ratio, 9),
        "homography_norm_to_mm": [
            [round(float(value), 12) for value in row]
            for row in calibration.homography_norm_to_mm
        ],
    }


def calibration_from_dict(payload: dict) -> PlaneCalibration:
    try:
        version = int(payload["version"])
        matrix = np.asarray(payload["homography_norm_to_mm"], dtype=np.float64).reshape(3, 3)
        calibration = PlaneCalibration(
            version=version,
            standard=str(payload["standard"]),
            reference_format=str(payload["reference_format"]),
            reference_width_mm=float(payload["reference_width_mm"]),
            reference_height_mm=float(payload["reference_height_mm"]),
            image_width_px=int(payload["image_width_px"]),
            image_height_px=int(payload["image_height_px"]),
            image_aspect_ratio=float(payload["image_aspect_ratio"]),
            homography_norm_to_mm=matrix,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CameraCalibrationError(f"Некорректный файл калибровки: {exc}") from exc

    if calibration.version != CALIBRATION_VERSION:
        raise CameraCalibrationError(
            f"Неподдерживаемая версия калибровки: {calibration.version}"
        )
    if not np.isfinite(calibration.homography_norm_to_mm).all():
        raise CameraCalibrationError("Матрица калибровки содержит NaN/Inf")
    if calibration.image_width_px <= 0 or calibration.image_height_px <= 0:
        raise CameraCalibrationError("Некорректное разрешение калибровки")
    return calibration


def load_plane_calibration(path: str | Path) -> PlaneCalibration:
    calibration_path = Path(path)
    try:
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CameraCalibrationError(f"Файл калибровки не найден: {calibration_path}") from exc
    except json.JSONDecodeError as exc:
        raise CameraCalibrationError(f"Некорректный JSON калибровки: {exc}") from exc
    return calibration_from_dict(payload)


def _validate_frame_geometry(
    calibration: PlaneCalibration,
    *,
    image_width_px: int,
    image_height_px: int,
    aspect_tolerance: float = 0.002,
) -> None:
    observed = float(image_width_px) / float(image_height_px)
    relative_error = abs(observed - calibration.image_aspect_ratio) / calibration.image_aspect_ratio
    if relative_error > aspect_tolerance:
        raise CameraCalibrationError(
            "FOV/отношение сторон кадра отличается от калибровочного: "
            f"cal={calibration.image_aspect_ratio:.6f}, current={observed:.6f}"
        )


def map_image_points_to_mm(
    calibration: PlaneCalibration,
    points: np.ndarray,
    *,
    image_width_px: int,
    image_height_px: int,
) -> np.ndarray:
    _validate_frame_geometry(
        calibration,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
    )
    normalized = _normalize_points(points, image_width_px, image_height_px)
    mapped = cv2.perspectiveTransform(
        normalized.reshape(1, -1, 2).astype(np.float32),
        calibration.homography_norm_to_mm.astype(np.float64),
    )[0]
    return mapped.astype(np.float64)


def measure_quad_mm(
    calibration: PlaneCalibration,
    quad: np.ndarray,
    *,
    image_width_px: int,
    image_height_px: int,
    frame_contact_sides: Sequence[str] = (),
) -> MetricMeasurement:
    points_mm = map_image_points_to_mm(
        calibration,
        order_quad(quad),
        image_width_px=image_width_px,
        image_height_px=image_height_px,
    )
    tl, tr, br, bl = points_mm
    top = _distance(tl, tr)
    bottom = _distance(bl, br)
    left = _distance(tl, bl)
    right = _distance(tr, br)

    sides = frozenset(str(side).lower() for side in frame_contact_sides)
    width_exact = not bool({"left", "right"} & sides)
    height_exact = not bool({"top", "bottom"} & sides)

    # При обрезке top/bottom ширина между боковыми кромками остаётся валидным
    # метрическим признаком. Аналогично для высоты при обрезке left/right.
    width = (top + bottom) / 2.0
    height = (left + right) / 2.0

    return MetricMeasurement(
        width_mm=round(width, 3),
        height_mm=round(height, 3),
        top_width_mm=round(top, 3),
        bottom_width_mm=round(bottom, 3),
        left_height_mm=round(left, 3),
        right_height_mm=round(right, 3),
        width_exact=width_exact,
        height_exact=height_exact,
    )


def _dimension_score(
    observed_mm: float,
    expected_mm: float,
    *,
    exact: bool,
    tolerance_mm: float,
) -> tuple[float, float, str]:
    error = abs(observed_mm - expected_mm)
    if exact:
        score = math.exp(-0.5 * (error / tolerance_mm) ** 2)
        return score, error, "exact"

    # При контакте с границей кадра измерение является нижней оценкой размера:
    # ожидаемый размер не может быть заметно меньше наблюдаемого. Более крупные
    # форматы не штрафуются по обрезанному измерению и должны различаться по
    # другой, полностью наблюдаемой стороне.
    if observed_mm <= expected_mm + tolerance_mm:
        return 1.0, max(0.0, observed_mm - expected_mm), "lower_bound"
    overshoot = observed_mm - expected_mm
    score = math.exp(-0.5 * (overshoot / tolerance_mm) ** 2)
    return score, overshoot, "lower_bound"


def match_format_by_metric(
    measurement: MetricMeasurement,
    *,
    specs: dict[EnvelopeFormat, EnvelopeSpec] = ENVELOPE_SPECS,
    tolerance_mm: float = DEFAULT_SIZE_TOLERANCE_MM,
    min_score: float = 0.72,
    min_margin: float = 0.18,
) -> MetricFormatDecision:
    if tolerance_mm <= 0:
        raise ValueError("tolerance_mm должен быть > 0")

    candidates: list[MetricFormatCandidate] = []
    for spec in specs.values():
        width_score, width_error, width_mode = _dimension_score(
            measurement.width_mm,
            spec.width_mm,
            exact=measurement.width_exact,
            tolerance_mm=tolerance_mm,
        )
        height_score, height_error, height_mode = _dimension_score(
            measurement.height_mm,
            spec.height_mm,
            exact=measurement.height_exact,
            tolerance_mm=tolerance_mm,
        )

        if measurement.width_exact and measurement.height_exact:
            score = math.sqrt(width_score * height_score)
        elif measurement.width_exact:
            score = width_score * (0.92 + 0.08 * height_score)
        elif measurement.height_exact:
            score = height_score * (0.92 + 0.08 * width_score)
        else:
            score = math.sqrt(width_score * height_score) * 0.55

        candidates.append(
            MetricFormatCandidate(
                format=spec.code,
                score=round(max(0.0, min(1.0, score)), 4),
                width_error_mm=round(width_error, 3),
                height_error_mm=round(height_error, 3),
                width_mode=width_mode,
                height_mode=height_mode,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    best = candidates[0]
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    margin = best.score - second_score
    resolved = best.score >= min_score and margin >= min_margin

    confidence = max(0.0, min(1.0, 0.72 * best.score + 0.28 * min(1.0, margin / 0.45)))
    return MetricFormatDecision(
        status="resolved" if resolved else "ambiguous",
        format=best.format if resolved else None,
        confidence=round(confidence, 4),
        margin=round(margin, 4),
        measurement=measurement,
        candidates=tuple(candidates),
    )

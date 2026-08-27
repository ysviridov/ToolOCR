from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass
from types import ModuleType
from typing import Any, Sequence

import cv2
import numpy as np

from . import profile_scoring
from .profiles import ShipmentProfile


@dataclass(frozen=True, slots=True)
class PhotometricDiagnostics:
    status: str
    reasons: tuple[str, ...]
    raw_p25: float
    raw_p50: float
    raw_p75: float
    raw_p95: float
    background_p10: float
    background_p25: float
    background_p50: float
    background_p75: float
    background_p90: float
    background_iqr: float
    sigma_px: float
    target_background: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_LAST_DIAGNOSTICS: ContextVar[PhotometricDiagnostics | None] = ContextVar(
    "toolocr_orientation_photometric_diagnostics",
    default=None,
)

_PATCH_INSTALLED = False
_LAYOUT_DEBUG_PATCHED = False

# Порог намеренно консервативный. Яркие DL/C5 из предыдущих regression-наборов
# должны идти по старому пути без преобразования. Коррекция включается, когда
# сглаженный фон письма действительно тёмный или имеет выраженную теневую зону.
DARK_BACKGROUND_MEDIAN = 205.0
SHADOW_BACKGROUND_P25 = 190.0
SHADOW_IQR_MIN = 22.0
TARGET_BACKGROUND = 235.0
MIN_BACKGROUND_VALUE = 24.0
BACKGROUND_SIGMA_RATIO = 0.045
BACKGROUND_SIGMA_MIN = 12.0


def _percentiles(gray: np.ndarray, values: Sequence[float]) -> tuple[float, ...]:
    return tuple(round(float(value), 2) for value in np.percentile(gray, values))


def prepare_photometric_gray(gray: np.ndarray) -> tuple[np.ndarray, PhotometricDiagnostics]:
    """Возвращает gray для threshold-based orientation-признаков.

    Barcode и Tesseract OSD эту копию не используют. Функция корректирует
    только низкочастотную неравномерность освещения/тёмный фон бумаги, чтобы
    фиксированные пороги старых CV-признаков снова означали «чернила», а не
    саму бумагу.
    """

    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("Ожидается непустое grayscale-изображение")

    h, w = gray.shape[:2]
    sigma = max(BACKGROUND_SIGMA_MIN, BACKGROUND_SIGMA_RATIO * min(h, w))
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )

    raw_p25, raw_p50, raw_p75, raw_p95 = _percentiles(gray, (25, 50, 75, 95))
    bg_p10, bg_p25, bg_p50, bg_p75, bg_p90 = _percentiles(
        background,
        (10, 25, 50, 75, 90),
    )
    background_iqr = round(bg_p75 - bg_p25, 2)

    reasons: list[str] = []
    if bg_p50 < DARK_BACKGROUND_MEDIAN:
        reasons.append("dark_background")
    if bg_p25 < SHADOW_BACKGROUND_P25 and background_iqr >= SHADOW_IQR_MIN:
        reasons.append("uneven_shadow")

    applied = bool(reasons)
    if applied:
        denominator = np.maximum(background.astype(np.float32), MIN_BACKGROUND_VALUE)
        normalized = gray.astype(np.float32) * TARGET_BACKGROUND / denominator
        feature_gray = np.clip(normalized, 0.0, 255.0).astype(np.uint8)
        status = "applied"
    else:
        feature_gray = gray
        status = "not_needed"

    diagnostics = PhotometricDiagnostics(
        status=status,
        reasons=tuple(reasons),
        raw_p25=raw_p25,
        raw_p50=raw_p50,
        raw_p75=raw_p75,
        raw_p95=raw_p95,
        background_p10=bg_p10,
        background_p25=bg_p25,
        background_p50=bg_p50,
        background_p75=bg_p75,
        background_p90=bg_p90,
        background_iqr=background_iqr,
        sigma_px=round(float(sigma), 2),
        target_background=TARGET_BACKGROUND,
    )
    return feature_gray, diagnostics


def get_last_photometric_diagnostics() -> PhotometricDiagnostics | None:
    return _LAST_DIAGNOSTICS.get()


def _prepare_orientation_work_photometric(
    scoring_image: np.ndarray,
    profile_list: tuple[ShipmentProfile, ...],
    frame_contact_sides: Sequence[str],
    partial: bool,
) -> dict[int, Any]:
    """Версия `_prepare_orientation_work` с раздельными gray-потоками.

    raw_gray:
      - barcode_layout
      - window_signal

    feature_gray (возможно нормализованный):
      - postage
      - code_stamp
      - line_signal
      - address_layout

    Все fusion-веса, production thresholds и OSD-path остаются неизменными.
    """

    raw_gray_0 = cv2.cvtColor(scoring_image, cv2.COLOR_BGR2GRAY)
    feature_gray_0, diagnostics = prepare_photometric_gray(raw_gray_0)
    _LAST_DIAGNOSTICS.set(diagnostics)

    work: dict[int, Any] = {}
    for orientation in (0, 180):
        if orientation == 0:
            oriented = scoring_image
            raw_gray = raw_gray_0
            feature_gray = feature_gray_0
        else:
            oriented = cv2.rotate(scoring_image, cv2.ROTATE_180)
            raw_gray = cv2.rotate(raw_gray_0, cv2.ROTATE_180)
            feature_gray = cv2.rotate(feature_gray_0, cv2.ROTATE_180)

        sides = profile_scoring._contacts(frame_contact_sides, orientation)
        format_features: dict[object, tuple[float, float, float, float]] = {}
        for profile in profile_list:
            if profile.format not in format_features:
                format_features[profile.format] = (
                    profile_scoring._aspect(profile, oriented, partial),
                    profile_scoring._postage(feature_gray, profile, sides),
                    profile_scoring._code_stamp(feature_gray, profile, sides),
                    profile_scoring._line_signal(feature_gray, profile),
                )

        work[orientation] = profile_scoring._OrientationWork(
            oriented=oriented,
            gray=feature_gray,
            sides=sides,
            window_signal=profile_scoring._window_signal(raw_gray),
            barcode_layout=profile_scoring._barcode_layout_signal(raw_gray),
            address_layout=profile_scoring._address_layout_signal(feature_gray),
            format_features=format_features,
        )

    return work


def install_photometric_orientation_preprocessing() -> None:
    """Устанавливает preprocessing один раз на процесс OCR worker."""

    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return
    profile_scoring._prepare_orientation_work = _prepare_orientation_work_photometric
    _PATCH_INSTALLED = True


def install_layout_debug_metadata(layout_api: ModuleType) -> None:
    """Добавляет photometric diagnostics в orientation evidence debug JSON.

    Serializer патчится после импорта layout_api. Это не меняет API decision
    contract: добавляется только диагностическое поле в debug evidence.
    """

    global _LAYOUT_DEBUG_PATCHED
    if _LAYOUT_DEBUG_PATCHED:
        return

    original = layout_api._orientation_evidence_to_dict

    def with_photometric(evidence: Any) -> dict[str, Any]:
        payload = original(evidence)
        diagnostics = get_last_photometric_diagnostics()
        payload["photometric_normalization"] = (
            diagnostics.to_dict() if diagnostics is not None else None
        )
        return payload

    layout_api._orientation_evidence_to_dict = with_photometric
    _LAYOUT_DEBUG_PATCHED = True

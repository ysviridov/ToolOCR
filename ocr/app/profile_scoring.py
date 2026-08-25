from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import cv2
import numpy as np

from .gost_r_51506_99 import DomesticLayout
from .profiles import ShipmentProfile


POSTAGE_FIELD_WIDTH_MM = 40.0
POSTAGE_FIELD_HEIGHT_MM = 25.0
POSTAGE_ANCHOR_MARGIN_MM = 15.0
CODE_BAR_WIDTH_MM = 7.0
CODE_BAR_PITCH_MM = 9.0
CODE_BAR_HEIGHT_MM = 2.0
CODE_BAR_COUNT = 6
CODE_REFERENCE_FROM_BOTTOM_MM = 25.0


@dataclass(frozen=True, slots=True)
class HypothesisComponents:
    aspect: float
    postage: float
    code_stamp: float
    layout: float
    window: float
    orientation_signal: float


@dataclass(frozen=True, slots=True)
class ProfileHypothesis:
    profile_id: str
    format: str
    layout: str
    window: bool
    orientation_deg: int
    score: float
    components: HypothesisComponents


@dataclass(frozen=True, slots=True)
class OrientationDecision:
    status: str
    value_deg: int | None
    confidence: float
    margin: float
    scores: tuple[tuple[int, float], ...]


@dataclass(frozen=True, slots=True)
class ProfileDecision:
    status: str
    profile_id: str | None
    confidence: float
    margin: float


@dataclass(frozen=True, slots=True)
class ProfileScoringResult:
    orientation: OrientationDecision
    profile: ProfileDecision
    hypotheses: tuple[ProfileHypothesis, ...]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _rotate(image: np.ndarray, orientation_deg: int) -> np.ndarray:
    if orientation_deg == 0:
        return image
    if orientation_deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    raise ValueError("Поддерживаются только ориентации 0 и 180 градусов")


def _rotated_contact_sides(sides: Sequence[str], orientation_deg: int) -> frozenset[str]:
    normalized = frozenset(str(side).lower() for side in sides)
    if orientation_deg == 0:
        return normalized
    if orientation_deg != 180:
        raise ValueError("Поддерживаются только ориентации 0 и 180 градусов")
    mapping = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    return frozenset(mapping.get(side, side) for side in normalized)


def _gray_and_ink(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return gray, ink


def _profile_scale(profile: ShipmentProfile, image: np.ndarray) -> tuple[float, float]:
    height, width = image.shape[:2]
    return width / profile.width_mm, height / profile.height_mm


def _rect_mm_to_px(
    profile: ShipmentProfile,
    image: np.ndarray,
    *,
    x_mm: float,
    y_mm: float,
    width_mm: float,
    height_mm: float,
) -> tuple[int, int, int, int]:
    sx, sy = _profile_scale(profile, image)
    h, w = image.shape[:2]
    x1 = max(0, min(w, int(round(x_mm * sx))))
    y1 = max(0, min(h, int(round(y_mm * sy))))
    x2 = max(x1, min(w, int(round((x_mm + width_mm) * sx))))
    y2 = max(y1, min(h, int(round((y_mm + height_mm) * sy))))
    return x1, y1, x2, y2


def _aspect_score(profile: ShipmentProfile, image: np.ndarray, *, partial_frame: bool) -> float:
    h, w = image.shape[:2]
    observed = max(w, h) / max(1.0, min(w, h))
    expected = profile.width_mm / profile.height_mm
    relative_error = abs(observed - expected) / expected
    tolerance = 0.11 if partial_frame else 0.055
    return _clamp01(1.0 - relative_error / tolerance)


def _postage_score(
    gray: np.ndarray,
    profile: ShipmentProfile,
    image: np.ndarray,
    *,
    contact_sides: frozenset[str],
) -> float:
    """Оценивает заполнение нормативного поля знака почтовой оплаты.

    ГОСТ Р 51506-99, п. 6.1.2.5 задаёт поле 40x25 мм. Рисунки А.1/А.2
    показывают угловую метку в верхней правой части с размером 15 мм. Для
    scoring используется поле, примыкающее к этой метке; это не OCR-ROI.
    """

    x_mm = max(0.0, profile.width_mm - POSTAGE_ANCHOR_MARGIN_MM - POSTAGE_FIELD_WIDTH_MM)
    y_mm = POSTAGE_ANCHOR_MARGIN_MM
    x1, y1, x2, y2 = _rect_mm_to_px(
        profile,
        image,
        x_mm=x_mm,
        y_mm=y_mm,
        width_mm=POSTAGE_FIELD_WIDTH_MM,
        height_mm=POSTAGE_FIELD_HEIGHT_MM,
    )
    region = gray[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0

    visibility = 0.45 if ({"top", "right"} & contact_sides) else 1.0
    dark_fraction = float(np.mean(region < 175))
    texture = _clamp01(float(np.std(region)) / 55.0)
    darkness = _clamp01((dark_fraction - 0.015) / 0.30)
    return _clamp01(visibility * (0.72 * darkness + 0.28 * texture))


def _code_stamp_score(
    gray: np.ndarray,
    profile: ShipmentProfile,
    image: np.ndarray,
    *,
    contact_sides: frozenset[str],
) -> float:
    """Ищет периодическую верхнюю гребёнку шестизначного кодового штампа.

    Приложение Д, рисунок Д.1: шесть позиций с шагом 9 мм (6x9=54),
    верхние чёрные элементы имеют ширину 7 мм и высоту 2 мм. Вертикальное
    положение ищется около размерной линии 25±2,5 мм от нижней кромки.
    Допуски detector-а намеренно шире нормативных, поскольку фотография может
    быть частично обрезана и содержать перспективные/печатные отклонения.
    """

    visibility = 0.55 if "bottom" in contact_sides else 1.0
    sx, sy = _profile_scale(profile, image)
    h, w = gray.shape[:2]

    y_min = max(0, int(round(h - 45.0 * sy)))
    x_max = min(w, int(round(min(profile.width_mm, 95.0) * sx)))
    if y_min >= h or x_max <= 0:
        return 0.0

    ink = (gray < 185).astype(np.uint8) * 255
    kernel_len = max(3, int(round(4.0 * sx)))
    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1)),
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats((horizontal > 0).astype(np.uint8), 8)
    components: list[tuple[float, float, float, float]] = []
    for stat in stats[1:count]:
        x, y, width, height, _ = (float(value) for value in stat)
        if x > x_max or y < y_min:
            continue
        if not (3.5 * sx <= width <= 11.0 * sx):
            continue
        if not (0.4 * sy <= height <= 5.0 * sy):
            continue
        components.append((x, y, width, height))

    if len(components) < 4:
        return 0.0

    components.sort(key=lambda item: (item[0], item[1]))
    best = 0.0
    expected_y = h - CODE_REFERENCE_FROM_BOTTOM_MM * sy

    for seed in components:
        x0, y0, _, _ = seed
        selected: list[tuple[float, float, float, float]] = []
        phase_errors: list[float] = []

        for index in range(CODE_BAR_COUNT):
            expected_x = x0 + index * CODE_BAR_PITCH_MM * sx
            candidate = min(
                components,
                key=lambda item: abs(item[0] - expected_x) + 2.0 * abs(item[1] - y0),
            )
            dx = abs(candidate[0] - expected_x) / max(1.0, 2.5 * sx)
            dy = abs(candidate[1] - y0) / max(1.0, 2.0 * sy)
            if dx <= 1.0 and dy <= 1.0 and candidate not in selected:
                selected.append(candidate)
                phase_errors.append(dx)

        if len(selected) < 4:
            continue

        widths = np.array([item[2] for item in selected], dtype=np.float32)
        heights = np.array([item[3] for item in selected], dtype=np.float32)
        y_centers = np.array([item[1] + item[3] / 2.0 for item in selected], dtype=np.float32)

        count_score = _clamp01(len(selected) / CODE_BAR_COUNT)
        width_score = math.exp(
            -abs(float(np.mean(widths)) - CODE_BAR_WIDTH_MM * sx) / max(1.0, 3.0 * sx)
        )
        height_score = math.exp(
            -abs(float(np.mean(heights)) - CODE_BAR_HEIGHT_MM * sy) / max(1.0, 2.0 * sy)
        )
        y_score = math.exp(
            -abs(float(np.mean(y_centers)) - expected_y) / max(1.0, 8.0 * sy)
        )
        phase_score = math.exp(-float(np.mean(phase_errors))) if phase_errors else 0.0

        score = count_score * (
            0.34 * width_score
            + 0.14 * height_score
            + 0.24 * y_score
            + 0.28 * phase_score
        )
        best = max(best, score)

    return _clamp01(best * visibility)


def _long_line_signal(gray: np.ndarray, profile: ShipmentProfile) -> float:
    """Глобальный сигнал исполнения I (направляющие линии) против II (углы)."""

    sx, sy = _profile_scale(profile, gray)
    ink = (gray < 195).astype(np.uint8) * 255
    closed = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(round(2.0 * sx))), 1)),
    )
    horizontal = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(round(22.0 * sx))), 1)),
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats((horizontal > 0).astype(np.uint8), 8)
    long_components = 0
    for stat in stats[1:count]:
        _, _, width, height, _ = (float(value) for value in stat)
        if width >= 25.0 * sx and height <= 7.0 * sy:
            long_components += 1

    return _clamp01((long_components - 2.0) / 8.0)


def _window_signal(gray: np.ndarray) -> float:
    """Ищет крупное прямоугольное окно без привязки к OCR.

    Приложение Б задаёт окно в правой части лицевой стороны. Здесь используется
    только структурный сигнал наличия крупного прямоугольного контура.
    """

    h, w = gray.shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(h * w)
    best = 0.0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
        area = float(cv2.contourArea(contour))
        area_ratio = area / frame_area
        if not (0.025 <= area_ratio <= 0.28):
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, width, height = cv2.boundingRect(approx)
        if x + width / 2.0 < 0.45 * w:
            continue
        ratio = width / max(1.0, float(height))
        if not (1.5 <= ratio <= 4.5):
            continue
        rectangularity = area / max(1.0, float(width * height))
        best = max(best, _clamp01(rectangularity))

    return best


def _profile_score(
    *,
    aspect: float,
    postage: float,
    code_stamp: float,
    layout: float,
    window: float,
    partial_frame: bool,
) -> tuple[float, float]:
    orientation_signal = _clamp01(0.58 * postage + 0.42 * code_stamp)

    if partial_frame:
        weights = (0.08, 0.27, 0.28, 0.23, 0.14)
    else:
        weights = (0.20, 0.23, 0.24, 0.20, 0.13)

    score = (
        weights[0] * aspect
        + weights[1] * postage
        + weights[2] * code_stamp
        + weights[3] * layout
        + weights[4] * window
    )
    return _clamp01(score), orientation_signal


def score_gost_profiles(
    image: np.ndarray,
    profiles: Iterable[ShipmentProfile],
    *,
    frame_contact_sides: Sequence[str] = (),
    orientation_min_signal: float = 0.30,
    orientation_min_margin: float = 0.12,
    profile_min_score: float = 0.48,
    profile_min_margin: float = 0.055,
) -> ProfileScoringResult:
    """Оценивает ГОСТ-профили для ориентаций 0° и 180°.

    Если структурные якоря не дают достаточного отрыва, функция возвращает
    ambiguous вместо принудительного выбора.
    """

    profile_list = tuple(profiles)
    if not profile_list:
        raise ValueError("Не переданы профили для scoring")

    partial_frame = bool(frame_contact_sides)
    hypotheses: list[ProfileHypothesis] = []
    orientation_best: dict[int, float] = {0: 0.0, 180: 0.0}

    for orientation_deg in (0, 180):
        oriented = _rotate(image, orientation_deg)
        gray, _ = _gray_and_ink(oriented)
        contact_sides = _rotated_contact_sides(frame_contact_sides, orientation_deg)
        global_window_signal = _window_signal(gray)

        for profile in profile_list:
            aspect = _aspect_score(profile, oriented, partial_frame=partial_frame)
            postage = _postage_score(gray, profile, oriented, contact_sides=contact_sides)
            code_stamp = _code_stamp_score(gray, profile, oriented, contact_sides=contact_sides)
            long_lines = _long_line_signal(gray, profile)
            layout = long_lines if profile.layout == DomesticLayout.LINES else 1.0 - long_lines
            window = global_window_signal if profile.window else 1.0 - global_window_signal

            score, orientation_signal = _profile_score(
                aspect=aspect,
                postage=postage,
                code_stamp=code_stamp,
                layout=layout,
                window=window,
                partial_frame=partial_frame,
            )
            orientation_best[orientation_deg] = max(orientation_best[orientation_deg], orientation_signal)
            hypotheses.append(
                ProfileHypothesis(
                    profile_id=profile.profile_id,
                    format=profile.format.value,
                    layout=profile.layout.value,
                    window=profile.window,
                    orientation_deg=orientation_deg,
                    score=round(score, 4),
                    components=HypothesisComponents(
                        aspect=round(aspect, 4),
                        postage=round(postage, 4),
                        code_stamp=round(code_stamp, 4),
                        layout=round(layout, 4),
                        window=round(window, 4),
                        orientation_signal=round(orientation_signal, 4),
                    ),
                )
            )

    orientation_pairs = sorted(orientation_best.items(), key=lambda item: item[1], reverse=True)
    best_orientation, best_signal = orientation_pairs[0]
    second_signal = orientation_pairs[1][1]
    orientation_margin = best_signal - second_signal

    if best_signal >= orientation_min_signal and orientation_margin >= orientation_min_margin:
        orientation_status = "resolved"
        orientation_value: int | None = best_orientation
    else:
        orientation_status = "ambiguous"
        orientation_value = None

    orientation_confidence = _clamp01(
        0.65 * best_signal + 0.35 * _clamp01(orientation_margin / 0.45)
    )
    orientation = OrientationDecision(
        status=orientation_status,
        value_deg=orientation_value,
        confidence=round(orientation_confidence, 4),
        margin=round(orientation_margin, 4),
        scores=tuple((deg, round(score, 4)) for deg, score in sorted(orientation_best.items())),
    )

    if orientation_value is None:
        profile_pool = sorted(hypotheses, key=lambda item: item.score, reverse=True)
    else:
        profile_pool = sorted(
            (item for item in hypotheses if item.orientation_deg == orientation_value),
            key=lambda item: item.score,
            reverse=True,
        )

    best_profile = profile_pool[0]
    second_profile_score = profile_pool[1].score if len(profile_pool) > 1 else 0.0
    profile_margin = best_profile.score - second_profile_score
    if (
        orientation_value is not None
        and best_profile.score >= profile_min_score
        and profile_margin >= profile_min_margin
    ):
        profile_status = "resolved"
        profile_id: str | None = best_profile.profile_id
    else:
        profile_status = "ambiguous"
        profile_id = None

    profile_confidence = _clamp01(
        0.72 * best_profile.score + 0.28 * _clamp01(profile_margin / 0.15)
    )
    profile_decision = ProfileDecision(
        status=profile_status,
        profile_id=profile_id,
        confidence=round(profile_confidence, 4),
        margin=round(profile_margin, 4),
    )

    return ProfileScoringResult(
        orientation=orientation,
        profile=profile_decision,
        hypotheses=tuple(sorted(hypotheses, key=lambda item: item.score, reverse=True)),
    )

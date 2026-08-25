from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import cv2
import numpy as np

from .gost_r_51506_99 import DomesticLayout
from .profiles import ShipmentProfile

# ГОСТ Р 51506-99: п. 6.1.2.5 и рисунки А.1/А.2.
POSTAGE_FIELD_MM = (40.0, 25.0)
POSTAGE_MARGIN_MM = 15.0
# Приложение Д, рисунок Д.1: 6×9=54 мм, элемент 7×2 мм.
CODE_BAR_COUNT = 6
CODE_BAR_PITCH_MM = 9.0
CODE_BAR_WIDTH_MM = 7.0
CODE_BAR_HEIGHT_MM = 2.0
CODE_FROM_BOTTOM_MM = 25.0
SCORING_MAX_SIDE = 1000


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


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _scale(profile: ShipmentProfile, image: np.ndarray) -> tuple[float, float]:
    h, w = image.shape[:2]
    return w / profile.width_mm, h / profile.height_mm


def _contacts(sides: Sequence[str], orientation: int) -> frozenset[str]:
    result = frozenset(str(side).lower() for side in sides)
    if orientation == 0:
        return result
    mapping = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    return frozenset(mapping.get(side, side) for side in result)


def _aspect(profile: ShipmentProfile, image: np.ndarray, partial: bool) -> float:
    h, w = image.shape[:2]
    observed = max(w, h) / max(1.0, min(w, h))
    expected = profile.width_mm / profile.height_mm
    tolerance = 0.11 if partial else 0.055
    return _clamp(1.0 - abs(observed - expected) / expected / tolerance)


def _postage(gray: np.ndarray, profile: ShipmentProfile, sides: frozenset[str]) -> float:
    """Поле 40×25 мм у верхней правой угловой метки, без OCR.

    `sides` намеренно не уменьшает уже обнаруженный visual-сигнал. Контакт
    конверта с краем кадра означает, что физическая граница может быть
    обрезана, но не означает, что содержимое нормативной зоны невидимо. На
    реальных кадрах сортировщика жёсткий штраф по frame_contact приводил к
    инверсии решения 0°/180°.
    """
    _ = sides
    h, w = gray.shape[:2]
    sx, sy = _scale(profile, gray)
    fw, fh = POSTAGE_FIELD_MM
    x1 = int(round(w - (POSTAGE_MARGIN_MM + fw) * sx))
    x2 = int(round(w - POSTAGE_MARGIN_MM * sx))
    y1 = int(round(POSTAGE_MARGIN_MM * sy))
    y2 = int(round((POSTAGE_MARGIN_MM + fh) * sy))
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = gray[y1:y2, x1:x2]
    dark = float(np.mean(region < 175))
    texture = _clamp(float(np.std(region)) / 55.0)
    score = 0.72 * _clamp((dark - 0.015) / 0.30) + 0.28 * texture
    return _clamp(score)


def _code_stamp(gray: np.ndarray, profile: ShipmentProfile, sides: frozenset[str]) -> float:
    """Периодическая гребёнка шестизначного штампа по рисунку Д.1.

    Как и для postage-якоря, frame_contact не является основанием снижать
    уже найденный структурный сигнал. Если штамп реально обрезан, это
    естественным образом уменьшит число/качество найденных элементов.
    """
    _ = sides
    h, w = gray.shape[:2]
    sx, sy = _scale(profile, gray)
    y_min = max(0, int(round(h - 45.0 * sy)))
    x_max = min(w, int(round(min(profile.width_mm, 95.0) * sx)))

    ink = (gray < 185).astype(np.uint8) * 255
    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(round(4.0 * sx))), 1)),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats((horizontal > 0).astype(np.uint8), 8)
    components: list[tuple[float, float, float, float]] = []
    for stat in stats[1:count]:
        x, y, ww, hh, _ = (float(v) for v in stat)
        if x <= x_max and y >= y_min and 3.5 * sx <= ww <= 11.0 * sx and 0.4 * sy <= hh <= 5.0 * sy:
            components.append((x, y, ww, hh))
    if len(components) < 4:
        return 0.0

    expected_y = h - CODE_FROM_BOTTOM_MM * sy
    best = 0.0
    for seed in components:
        x0, y0, _, _ = seed
        selected: list[tuple[float, float, float, float]] = []
        phase: list[float] = []
        for index in range(CODE_BAR_COUNT):
            expected_x = x0 + index * CODE_BAR_PITCH_MM * sx
            candidate = min(components, key=lambda c: abs(c[0] - expected_x) + 2.0 * abs(c[1] - y0))
            dx = abs(candidate[0] - expected_x) / max(1.0, 2.5 * sx)
            dy = abs(candidate[1] - y0) / max(1.0, 2.0 * sy)
            if dx <= 1.0 and dy <= 1.0 and candidate not in selected:
                selected.append(candidate)
                phase.append(dx)
        if len(selected) < 4:
            continue

        widths = np.array([c[2] for c in selected], dtype=np.float32)
        heights = np.array([c[3] for c in selected], dtype=np.float32)
        centers_y = np.array([c[1] + c[3] / 2.0 for c in selected], dtype=np.float32)
        count_score = _clamp(len(selected) / CODE_BAR_COUNT)
        width_score = math.exp(-abs(float(widths.mean()) - CODE_BAR_WIDTH_MM * sx) / max(1.0, 3.0 * sx))
        height_score = math.exp(-abs(float(heights.mean()) - CODE_BAR_HEIGHT_MM * sy) / max(1.0, 2.0 * sy))
        y_score = math.exp(-abs(float(centers_y.mean()) - expected_y) / max(1.0, 8.0 * sy))
        phase_score = math.exp(-float(np.mean(phase))) if phase else 0.0
        best = max(best, count_score * (0.34 * width_score + 0.14 * height_score + 0.24 * y_score + 0.28 * phase_score))

    return _clamp(best)


def _line_signal(gray: np.ndarray, profile: ShipmentProfile) -> float:
    """Исполнение I: много длинных направляющих; II: преимущественно угловые элементы."""
    sx, sy = _scale(profile, gray)
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
    long_lines = 0
    for stat in stats[1:count]:
        _, _, ww, hh, _ = (float(v) for v in stat)
        if ww >= 25.0 * sx and hh <= 7.0 * sy:
            long_lines += 1
    return _clamp((long_lines - 2.0) / 8.0)


def _window_signal(gray: np.ndarray) -> float:
    """Структурный сигнал крупного окна в правой части (приложение Б)."""
    h, w = gray.shape[:2]
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(h * w)
    best = 0.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
        area = float(cv2.contourArea(contour))
        if not 0.025 <= area / frame_area <= 0.28:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, _, ww, hh = cv2.boundingRect(approx)
        if x + ww / 2.0 < 0.45 * w or not 1.5 <= ww / max(1.0, float(hh)) <= 4.5:
            continue
        best = max(best, _clamp(area / max(1.0, float(ww * hh))))
    return best


def _resize_for_scoring(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, SCORING_MAX_SIDE / float(max(h, w)))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(2, int(round(w * scale))), max(2, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _orientation_signal(postage: float, code_stamp: float) -> float:
    """Согласованный сигнал ориентации по двум асимметричным ГОСТ-якорям.

    Один очень тёмный/текстурный участок в правом верхнем углу не должен
    самостоятельно решать ориентацию: перевёрнутый кодовый штамп или штрихкод
    может выглядеть как поле марки. Поэтому треть веса — это совместное
    присутствие обоих ожидаемых якорей (геометрическое среднее).
    """
    agreement = math.sqrt(max(0.0, postage * code_stamp))
    return _clamp(0.42 * postage + 0.38 * code_stamp + 0.20 * agreement)


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
    """Ранжирует ГОСТ-профили и консервативно выбирает 0°/180°."""
    profile_list = tuple(profiles)
    if not profile_list:
        raise ValueError("Не переданы профили для scoring")

    scoring_image = _resize_for_scoring(image)
    partial = bool(frame_contact_sides)
    hypotheses: list[ProfileHypothesis] = []
    orientation_best = {0: 0.0, 180: 0.0}

    for orientation in (0, 180):
        oriented = scoring_image if orientation == 0 else cv2.rotate(scoring_image, cv2.ROTATE_180)
        gray = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY)
        sides = _contacts(frame_contact_sides, orientation)
        window_signal = _window_signal(gray)

        # Тяжёлые признаки считаются один раз на физический формат, а не на
        # каждый вариант I/II и window/no-window.
        format_features: dict[object, tuple[float, float, float, float]] = {}
        for profile in profile_list:
            if profile.format not in format_features:
                format_features[profile.format] = (
                    _aspect(profile, oriented, partial),
                    _postage(gray, profile, sides),
                    _code_stamp(gray, profile, sides),
                    _line_signal(gray, profile),
                )

        for profile in profile_list:
            aspect, postage, code_stamp, line_signal = format_features[profile.format]
            layout = line_signal if profile.layout == DomesticLayout.LINES else 1.0 - line_signal
            window = window_signal if profile.window else 1.0 - window_signal
            orientation_signal = _orientation_signal(postage, code_stamp)
            orientation_best[orientation] = max(orientation_best[orientation], orientation_signal)
            weights = (0.08, 0.27, 0.28, 0.23, 0.14) if partial else (0.20, 0.23, 0.24, 0.20, 0.13)
            score = _clamp(
                weights[0] * aspect + weights[1] * postage + weights[2] * code_stamp
                + weights[3] * layout + weights[4] * window
            )
            hypotheses.append(
                ProfileHypothesis(
                    profile_id=profile.profile_id,
                    format=profile.format.value,
                    layout=profile.layout.value,
                    window=profile.window,
                    orientation_deg=orientation,
                    score=round(score, 4),
                    components=HypothesisComponents(
                        aspect=round(aspect, 4), postage=round(postage, 4),
                        code_stamp=round(code_stamp, 4), layout=round(layout, 4),
                        window=round(window, 4), orientation_signal=round(orientation_signal, 4),
                    ),
                )
            )

    orientation_rank = sorted(orientation_best.items(), key=lambda item: item[1], reverse=True)
    best_deg, best_signal = orientation_rank[0]
    margin = best_signal - orientation_rank[1][1]
    resolved = best_signal >= orientation_min_signal and margin >= orientation_min_margin
    orientation_decision = OrientationDecision(
        status="resolved" if resolved else "ambiguous",
        value_deg=best_deg if resolved else None,
        confidence=round(_clamp(0.65 * best_signal + 0.35 * _clamp(margin / 0.45)), 4),
        margin=round(margin, 4),
        scores=tuple((deg, round(value, 4)) for deg, value in sorted(orientation_best.items())),
    )

    pool = sorted(
        (h for h in hypotheses if not resolved or h.orientation_deg == best_deg),
        key=lambda h: h.score,
        reverse=True,
    )
    best = pool[0]
    second = pool[1].score if len(pool) > 1 else 0.0
    profile_margin = best.score - second
    profile_resolved = resolved and best.score >= profile_min_score and profile_margin >= profile_min_margin
    profile_decision = ProfileDecision(
        status="resolved" if profile_resolved else "ambiguous",
        profile_id=best.profile_id if profile_resolved else None,
        confidence=round(_clamp(0.72 * best.score + 0.28 * _clamp(profile_margin / 0.15)), 4),
        margin=round(profile_margin, 4),
    )
    return ProfileScoringResult(
        orientation=orientation_decision,
        profile=profile_decision,
        hypotheses=tuple(sorted(hypotheses, key=lambda h: h.score, reverse=True)),
    )

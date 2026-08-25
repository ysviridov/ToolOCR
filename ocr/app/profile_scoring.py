from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
from typing import Iterable, Sequence

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

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
TEXT_OSD_MAX_SIDE = max(1000, int(os.environ.get("ORIENTATION_TEXT_OSD_MAX_SIDE", "1800")))
TEXT_OSD_ENABLED = os.environ.get("ORIENTATION_TEXT_OSD", "1").strip().lower() not in {
    "0", "false", "no", "off",
}

# Contrast-aware fusion. Абсолютные сигналы сохраняются как основной score,
# а эти коэффициенты лишь увеличивают разрыв между 0°/180°, когда независимый
# признак действительно различает две гипотезы.
CONTRAST_CHANNEL_WEIGHTS = {
    "postage": 0.18,
    "code_stamp": 0.20,
    "barcode_layout": 0.20,
    "address_layout": 0.12,
    "text_direction": 0.30,
}
CONTRAST_BONUS_SCALE = 0.20
CONTRAST_BONUS_MAX = 0.08
AGREEMENT_DELTA_MIN = 0.06
AGREEMENT_BONUS_SCALE = 0.25
AGREEMENT_BONUS_MAX = 0.10


@dataclass(frozen=True, slots=True)
class HypothesisComponents:
    aspect: float
    postage: float
    code_stamp: float
    layout: float
    window: float
    barcode_layout: float
    address_layout: float
    text_direction: float
    content_orientation: float
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
class OrientationEvidence:
    orientation_deg: int
    postage: float
    code_stamp: float
    barcode_layout: float
    address_layout: float
    text_direction: float
    content_orientation: float
    base_score: float
    postage_delta: float
    code_stamp_delta: float
    barcode_delta: float
    address_delta: float
    text_delta: float
    contrast_bonus: float
    agreement_bonus: float
    agreement_channels: int
    score: float


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
    orientation_evidence: tuple[OrientationEvidence, ...]


@dataclass(frozen=True, slots=True)
class _OrientationWork:
    oriented: np.ndarray
    gray: np.ndarray
    sides: frozenset[str]
    window_signal: float
    barcode_layout: float
    address_layout: float
    format_features: dict[object, tuple[float, float, float, float]]


@dataclass(frozen=True, slots=True)
class _FusionAdjustment:
    postage_delta: float
    code_stamp_delta: float
    barcode_delta: float
    address_delta: float
    text_delta: float
    contrast_bonus: float
    agreement_bonus: float
    agreement_channels: int
    score: float


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
    обрезана, но не означает, что содержимое нормативной зоны невидимо.
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
    """Периодическая гребёнка шестизначного штампа по рисунку Д.1."""
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


def _barcode_layout_signal(gray: np.ndarray) -> float:
    """Ищет крупный линейный barcode и оценивает его положение слева.

    Это не декодирование штрихкода. Для production-кадров полезна только
    асимметрия расположения: после поворота на 180° левый barcode становится
    правым. Признак имеет ограниченный вес и не может один решать ориентацию.
    """
    h, w = gray.shape[:2]
    if h < 40 or w < 80:
        return 0.0

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    grad_x = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))
    gradient = cv2.subtract(grad_x, grad_y)
    gradient = cv2.GaussianBlur(gradient, (5, 5), 0)
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_w = max(9, int(round(w * 0.025)))
    close_h = max(3, int(round(h * 0.010)))
    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h)),
        iterations=2,
    )
    closed = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(h * w)
    best = 0.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
        x, y, ww, hh = cv2.boundingRect(contour)
        box_area = float(ww * hh)
        area_ratio = box_area / frame_area
        aspect = ww / max(1.0, float(hh))
        if not 0.0015 <= area_ratio <= 0.12:
            continue
        if not 1.4 <= aspect <= 14.0:
            continue
        if ww < 0.07 * w or hh < 0.018 * h:
            continue

        roi_grad = grad_x[y:y + hh, x:x + ww]
        edge_strength = _clamp(float(np.mean(roi_grad)) / 70.0)
        size_strength = _clamp(area_ratio / 0.018)
        center_x = (x + ww / 2.0) / float(w)
        left_preference = _clamp((0.72 - center_x) / 0.52)
        strength = (0.58 * edge_strength + 0.42 * size_strength) * left_preference
        best = max(best, strength)
    return _clamp(best)


def _address_layout_signal(gray: np.ndarray) -> float:
    """Оценивает каноническое расположение адресных строк без OCR.

    Плотные горизонтальные строки группируются морфологией. Каноническая
    гипотеза предпочитает небольшой блок отправителя сверху слева и более
    массивный блок получателя в нижней/правой части. При 180° эта асимметрия
    меняется местами.
    """
    h, w = gray.shape[:2]
    if h < 40 or w < 80:
        return 0.0

    ink = (gray < 190).astype(np.uint8) * 255
    join_w = max(5, int(round(w * 0.012)))
    joined = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (join_w, 3)),
        iterations=1,
    )
    joined = cv2.morphologyEx(
        joined,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)),
        iterations=1,
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats((joined > 0).astype(np.uint8), 8)
    samples: list[tuple[float, float, float]] = []
    total_mass = 0.0
    for stat in stats[1:count]:
        x, y, ww, hh, area = (float(v) for v in stat)
        aspect = ww / max(1.0, hh)
        if not 0.035 * w <= ww <= 0.65 * w:
            continue
        if not 0.006 * h <= hh <= 0.09 * h:
            continue
        if aspect < 2.0:
            continue
        cx = (x + ww / 2.0) / float(w)
        cy = (y + hh / 2.0) / float(h)
        mass = max(area, 0.35 * ww * hh)
        samples.append((cx, cy, mass))
        total_mass += mass

    if len(samples) < 2 or total_mass <= 0:
        return 0.0

    def gaussian(cx: float, cy: float, tx: float, ty: float, sx: float, sy: float) -> float:
        return math.exp(-0.5 * (((cx - tx) / sx) ** 2 + ((cy - ty) / sy) ** 2))

    preferred = 0.0
    opposite = 0.0
    for cx, cy, mass in samples:
        sender = gaussian(cx, cy, 0.27, 0.27, 0.27, 0.25)
        recipient = gaussian(cx, cy, 0.68, 0.67, 0.27, 0.27)
        wrong_upper_right = gaussian(cx, cy, 0.73, 0.27, 0.27, 0.25)
        wrong_lower_left = gaussian(cx, cy, 0.32, 0.67, 0.27, 0.27)
        preferred += mass * (0.34 * sender + 0.66 * recipient)
        opposite += mass * (0.34 * wrong_upper_right + 0.66 * wrong_lower_left)

    preferred /= total_mass
    opposite /= total_mass
    presence = _clamp(total_mass / max(1.0, 0.010 * h * w))
    contrast = _clamp((preferred - opposite + 0.18) / 0.48)
    return _clamp(presence * (0.55 * preferred + 0.45 * contrast))


def _resize_for_osd(image: np.ndarray) -> np.ndarray:
    """OSD требует заметно большего разрешения текста, чем быстрый CV scoring."""
    h, w = image.shape[:2]
    scale = min(1.0, TEXT_OSD_MAX_SIDE / float(max(h, w)))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(2, int(round(w * scale))), max(2, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _text_direction_scores(image: np.ndarray) -> dict[int, float]:
    """Определяет 0/180 по направлению текста через Tesseract OSD.

    OSD запускается максимум один раз на письмо и только для сложных случаев.
    Для OSD используется отдельная копия до 1800 px по длинной стороне (по
    умолчанию), а не 1000 px быстрых CV-признаков: на production-фотографиях
    уменьшение до 1000 px резко снижает orientation confidence Tesseract.
    """
    scores = {0: 0.0, 180: 0.0}
    if not TEXT_OSD_ENABLED:
        return scores
    try:
        osd_image = _resize_for_osd(image)
        result = pytesseract.image_to_osd(osd_image, output_type=Output.DICT)
        rotate = int(result.get("rotate", -1)) % 360
        confidence = float(result.get("orientation_conf", 0.0) or 0.0)
    except (pytesseract.TesseractError, pytesseract.TesseractNotFoundError, TypeError, ValueError):
        return scores

    if rotate not in (0, 180):
        return scores
    # В документации Tesseract ~15 считается уверенным orientation confidence.
    strength = _clamp(confidence / 15.0)
    if strength < 0.12:
        return scores
    scores[rotate] = strength
    return scores


def _content_orientation_signal(
    barcode_layout: float,
    address_layout: float,
    text_direction: float,
) -> float:
    """Третий независимый канал ориентации: barcode + адреса + направление текста."""
    cv_layout = _clamp(0.46 * barcode_layout + 0.54 * address_layout)
    # OSD orientation_conf уже нормирован в 0..1. После перехода на отдельное
    # разрешение до 1800 px сильный текстовый сигнал не ослабляем повторно.
    # При низком confidence он сам остаётся ниже CV-layout и не доминирует.
    return _clamp(max(cv_layout, text_direction, 0.58 * cv_layout + 0.42 * text_direction))


def _orientation_signal(
    postage: float,
    code_stamp: float,
    content_orientation: float,
) -> float:
    """Fusion трёх независимых каналов для 0°/180°.

    Один ложный postage или code-stamp теперь принципиально не проходит порог
    самостоятельно. Сильный content_orientation может решить машинное письмо,
    где ГОСТ-кодовый штамп отсутствует.
    """
    anchor_agreement = math.sqrt(max(0.0, postage * code_stamp))
    return _clamp(
        0.24 * postage
        + 0.20 * code_stamp
        + 0.46 * content_orientation
        + 0.10 * anchor_agreement
    )


def _contrast_aware_fusion(
    base_scores: dict[int, float],
    evidence: dict[int, OrientationEvidence],
) -> dict[int, _FusionAdjustment]:
    """Добавляет направленный contrast и agreement bonus к 0°/180°.

    Высокое абсолютное значение признака само по себе не считается полезным,
    если оно почти такое же у противоположной гипотезы. Для каждого канала
    используется только положительная разница current - opposite. Agreement
    начисляется, когда хотя бы два независимых канала имеют delta >= 0.06.

    Бонусы намеренно ограничены: contrast <= 0.08, agreement <= 0.10. Поэтому
    слой увеличивает margin у согласованных гипотез, но не заменяет исходные
    ГОСТ/content сигналы и не меняет пороги принятия решения.
    """
    result: dict[int, _FusionAdjustment] = {}
    for degree, opposite in ((0, 180), (180, 0)):
        current = evidence[degree]
        other = evidence[opposite]
        deltas = {
            "postage": max(0.0, current.postage - other.postage),
            "code_stamp": max(0.0, current.code_stamp - other.code_stamp),
            "barcode_layout": max(0.0, current.barcode_layout - other.barcode_layout),
            "address_layout": max(0.0, current.address_layout - other.address_layout),
            "text_direction": max(0.0, current.text_direction - other.text_direction),
        }
        weighted_delta = sum(
            CONTRAST_CHANNEL_WEIGHTS[name] * value
            for name, value in deltas.items()
        )
        contrast_bonus = min(CONTRAST_BONUS_MAX, CONTRAST_BONUS_SCALE * weighted_delta)

        supporting = sorted(
            (value for value in deltas.values() if value >= AGREEMENT_DELTA_MIN),
            reverse=True,
        )
        agreement_channels = len(supporting)
        agreement_bonus = 0.0
        if agreement_channels >= 2:
            agreement_bonus = min(
                AGREEMENT_BONUS_MAX,
                AGREEMENT_BONUS_SCALE * math.sqrt(supporting[0] * supporting[1]),
            )

        final_score = _clamp(base_scores[degree] + contrast_bonus + agreement_bonus)
        result[degree] = _FusionAdjustment(
            postage_delta=deltas["postage"],
            code_stamp_delta=deltas["code_stamp"],
            barcode_delta=deltas["barcode_layout"],
            address_delta=deltas["address_layout"],
            text_delta=deltas["text_direction"],
            contrast_bonus=contrast_bonus,
            agreement_bonus=agreement_bonus,
            agreement_channels=agreement_channels,
            score=final_score,
        )
    return result


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


def _prepare_orientation_work(
    scoring_image: np.ndarray,
    profile_list: tuple[ShipmentProfile, ...],
    frame_contact_sides: Sequence[str],
    partial: bool,
) -> dict[int, _OrientationWork]:
    work: dict[int, _OrientationWork] = {}
    for orientation in (0, 180):
        oriented = scoring_image if orientation == 0 else cv2.rotate(scoring_image, cv2.ROTATE_180)
        gray = cv2.cvtColor(oriented, cv2.COLOR_BGR2GRAY)
        sides = _contacts(frame_contact_sides, orientation)
        format_features: dict[object, tuple[float, float, float, float]] = {}
        for profile in profile_list:
            if profile.format not in format_features:
                format_features[profile.format] = (
                    _aspect(profile, oriented, partial),
                    _postage(gray, profile, sides),
                    _code_stamp(gray, profile, sides),
                    _line_signal(gray, profile),
                )
        work[orientation] = _OrientationWork(
            oriented=oriented,
            gray=gray,
            sides=sides,
            window_signal=_window_signal(gray),
            barcode_layout=_barcode_layout_signal(gray),
            address_layout=_address_layout_signal(gray),
            format_features=format_features,
        )
    return work


def _best_orientation_evidence(
    work: dict[int, _OrientationWork],
    text_scores: dict[int, float],
) -> tuple[dict[int, float], dict[int, OrientationEvidence]]:
    """Строит лучший абсолютный score каждой ориентации до contrast fusion."""
    base_scores = {0: 0.0, 180: 0.0}
    evidence: dict[int, OrientationEvidence] = {}

    for degree in (0, 180):
        item = work[degree]
        text_direction = text_scores.get(degree, 0.0)
        content_orientation = _content_orientation_signal(
            item.barcode_layout,
            item.address_layout,
            text_direction,
        )
        for _, postage, code_stamp, _ in item.format_features.values():
            score = _orientation_signal(postage, code_stamp, content_orientation)
            if score > base_scores[degree] or degree not in evidence:
                base_scores[degree] = score
                evidence[degree] = OrientationEvidence(
                    orientation_deg=degree,
                    postage=postage,
                    code_stamp=code_stamp,
                    barcode_layout=item.barcode_layout,
                    address_layout=item.address_layout,
                    text_direction=text_direction,
                    content_orientation=content_orientation,
                    base_score=score,
                    postage_delta=0.0,
                    code_stamp_delta=0.0,
                    barcode_delta=0.0,
                    address_delta=0.0,
                    text_delta=0.0,
                    contrast_bonus=0.0,
                    agreement_bonus=0.0,
                    agreement_channels=0,
                    score=score,
                )

        if degree not in evidence:
            evidence[degree] = OrientationEvidence(
                orientation_deg=degree,
                postage=0.0,
                code_stamp=0.0,
                barcode_layout=item.barcode_layout,
                address_layout=item.address_layout,
                text_direction=text_direction,
                content_orientation=content_orientation,
                base_score=0.0,
                postage_delta=0.0,
                code_stamp_delta=0.0,
                barcode_delta=0.0,
                address_delta=0.0,
                text_delta=0.0,
                contrast_bonus=0.0,
                agreement_bonus=0.0,
                agreement_channels=0,
                score=0.0,
            )

    return base_scores, evidence


def _apply_orientation_fusion(
    base_scores: dict[int, float],
    evidence: dict[int, OrientationEvidence],
) -> tuple[dict[int, float], dict[int, OrientationEvidence]]:
    fusion = _contrast_aware_fusion(base_scores, evidence)
    scores: dict[int, float] = {}
    fused: dict[int, OrientationEvidence] = {}
    for degree in (0, 180):
        adjustment = fusion[degree]
        scores[degree] = adjustment.score
        fused[degree] = replace(
            evidence[degree],
            base_score=round(base_scores[degree], 4),
            postage=round(evidence[degree].postage, 4),
            code_stamp=round(evidence[degree].code_stamp, 4),
            barcode_layout=round(evidence[degree].barcode_layout, 4),
            address_layout=round(evidence[degree].address_layout, 4),
            text_direction=round(evidence[degree].text_direction, 4),
            content_orientation=round(evidence[degree].content_orientation, 4),
            postage_delta=round(adjustment.postage_delta, 4),
            code_stamp_delta=round(adjustment.code_stamp_delta, 4),
            barcode_delta=round(adjustment.barcode_delta, 4),
            address_delta=round(adjustment.address_delta, 4),
            text_delta=round(adjustment.text_delta, 4),
            contrast_bonus=round(adjustment.contrast_bonus, 4),
            agreement_bonus=round(adjustment.agreement_bonus, 4),
            agreement_channels=adjustment.agreement_channels,
            score=round(adjustment.score, 4),
        )
    return scores, fused


def _orientation_is_resolved(
    scores: dict[int, float],
    *,
    min_signal: float,
    min_margin: float,
) -> bool:
    rank = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return rank[0][1] >= min_signal and rank[0][1] - rank[1][1] >= min_margin


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
    """Ранжирует ГОСТ-профили и консервативно выбирает 0°/180°.

    Сначала выполняется полный дешёвый CV-only fusion. Tesseract OSD запускается
    только если после contrast/agreement итог всё ещё не проходит те же
    production-пороги orientation_min_signal/orientation_min_margin.
    """
    profile_list = tuple(profiles)
    if not profile_list:
        raise ValueError("Не переданы профили для scoring")

    scoring_image = _resize_for_scoring(image)
    partial = bool(frame_contact_sides)
    work = _prepare_orientation_work(
        scoring_image,
        profile_list,
        frame_contact_sides,
        partial,
    )

    # FAST PATH: сначала считаем все CV-признаки и contrast-aware fusion без OSD.
    zero_text_scores = {0: 0.0, 180: 0.0}
    cv_base, cv_evidence = _best_orientation_evidence(work, zero_text_scores)
    cv_scores, _ = _apply_orientation_fusion(cv_base, cv_evidence)
    cv_resolved = _orientation_is_resolved(
        cv_scores,
        min_signal=orientation_min_signal,
        min_margin=orientation_min_margin,
    )

    # SLOW PATH: Tesseract запускается только для реально неоднозначного CV.
    # Он по-прежнему получает исходный rectified image и ограничивает его до
    # TEXT_OSD_MAX_SIDE внутри _text_direction_scores().
    text_scores = zero_text_scores if cv_resolved else _text_direction_scores(image)

    base_orientation_best, raw_evidence = _best_orientation_evidence(work, text_scores)
    orientation_best, evidence_best = _apply_orientation_fusion(
        base_orientation_best,
        raw_evidence,
    )

    hypotheses: list[ProfileHypothesis] = []
    for orientation in (0, 180):
        item = work[orientation]
        text_direction = text_scores[orientation]
        content_orientation = _content_orientation_signal(
            item.barcode_layout,
            item.address_layout,
            text_direction,
        )

        for profile in profile_list:
            aspect, postage, code_stamp, line_signal = item.format_features[profile.format]
            layout = line_signal if profile.layout == DomesticLayout.LINES else 1.0 - line_signal
            window = item.window_signal if profile.window else 1.0 - item.window_signal
            orientation_signal = _orientation_signal(postage, code_stamp, content_orientation)

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
                        aspect=round(aspect, 4),
                        postage=round(postage, 4),
                        code_stamp=round(code_stamp, 4),
                        layout=round(layout, 4),
                        window=round(window, 4),
                        barcode_layout=round(item.barcode_layout, 4),
                        address_layout=round(item.address_layout, 4),
                        text_direction=round(text_direction, 4),
                        content_orientation=round(content_orientation, 4),
                        orientation_signal=round(orientation_signal, 4),
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
        orientation_evidence=tuple(evidence_best[degree] for degree in (0, 180)),
    )
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


GOST_ID = "ГОСТ Р 51506-99"


class EnvelopeFormat(str, Enum):
    C6 = "C6"
    DL = "DL"
    C5 = "C5"
    C4 = "C4"
    B4 = "B4"


class DomesticLayout(str, Enum):
    """Исполнение адресных зон для внутренних отправлений.

    Значения соответствуют п. 4.1.4 ГОСТ Р 51506-99:
    I  — направляющие линии;
    II — угловые элементы, ограничивающие адресные зоны.
    """

    LINES = "I"
    CORNERS = "II"


@dataclass(frozen=True, slots=True)
class EnvelopeSpec:
    code: EnvelopeFormat
    height_mm: float
    width_mm: float
    window_allowed: bool

    @property
    def aspect_ratio(self) -> float:
        return self.width_mm / self.height_mm


# Таблица 1 ГОСТ Р 51506-99, п. 4.1.1.
ENVELOPE_SPECS: dict[EnvelopeFormat, EnvelopeSpec] = {
    EnvelopeFormat.C6: EnvelopeSpec(EnvelopeFormat.C6, 114.0, 162.0, True),
    EnvelopeFormat.DL: EnvelopeSpec(EnvelopeFormat.DL, 110.0, 220.0, True),
    EnvelopeFormat.C5: EnvelopeSpec(EnvelopeFormat.C5, 162.0, 229.0, True),
    EnvelopeFormat.C4: EnvelopeSpec(EnvelopeFormat.C4, 229.0, 324.0, False),
    EnvelopeFormat.B4: EnvelopeSpec(EnvelopeFormat.B4, 250.0, 353.0, False),
}


@dataclass(frozen=True, slots=True)
class FormatCandidate:
    format: EnvelopeFormat
    ratio_error: float


@dataclass(frozen=True, slots=True)
class RectMM:
    """Прямоугольник в системе координат лицевой стороны конверта, мм.

    Начало координат: левый верхний угол после перспективного выравнивания.
    X направлена вправо, Y — вниз.
    """

    x: float
    y: float
    width: float
    height: float


# П. 6.1.2.5: поле под изображение марки/знака оплаты — 40x25 мм.
STAMP_KEEP_OUT_SIZE_MM = (40.0, 25.0)


# Рисунки А.1/А.2 и таблица А.1 являются источником координат адресных зон
# для внутренних отправлений без окна. Координаты зон намеренно не
# зашиваются приблизительно: они должны быть перенесены из размерных цепочек
# ГОСТ и проверены тестами для каждого формата/исполнения.
DOMESTIC_NO_WINDOW_FIGURES = {
    DomesticLayout.LINES: "А.1",
    DomesticLayout.CORNERS: "А.2",
}


# Для конвертов с окном используются обязательные рисунки приложения Б.
DOMESTIC_WINDOW_FIGURES = {
    DomesticLayout.LINES: "Б.1",
    DomesticLayout.CORNERS: "Б.2",
}


def candidate_formats_by_aspect_ratio(
    width_px: int,
    height_px: int,
    *,
    max_relative_error: float = 0.035,
) -> list[FormatCandidate]:
    """Возвращает все форматы ГОСТ, совместимые с отношением сторон.

    Важно: функция НЕ выбирает единственный формат. C6/C5/C4/B4 имеют
    близкие отношения сторон, поэтому окончательная идентификация должна
    выполняться по геометрии стандартных элементов после rectification.
    """

    if width_px <= 0 or height_px <= 0:
        raise ValueError("Размеры изображения должны быть положительными")
    if max_relative_error <= 0:
        raise ValueError("max_relative_error должен быть > 0")

    long_side = float(max(width_px, height_px))
    short_side = float(min(width_px, height_px))
    observed = long_side / short_side

    candidates: list[FormatCandidate] = []
    for spec in ENVELOPE_SPECS.values():
        error = abs(observed - spec.aspect_ratio) / spec.aspect_ratio
        if error <= max_relative_error:
            candidates.append(FormatCandidate(spec.code, round(error, 6)))

    return sorted(candidates, key=lambda item: item.ratio_error)


def mm_to_normalized(rect: RectMM, spec: EnvelopeSpec) -> tuple[float, float, float, float]:
    """Переводит прямоугольник из миллиметров ГОСТ в нормализованные 0..1 координаты."""

    if rect.x < 0 or rect.y < 0 or rect.width <= 0 or rect.height <= 0:
        raise ValueError("Некорректный прямоугольник")
    if rect.x + rect.width > spec.width_mm or rect.y + rect.height > spec.height_mm:
        raise ValueError("Прямоугольник выходит за границы конверта")

    return (
        rect.x / spec.width_mm,
        rect.y / spec.height_mm,
        rect.width / spec.width_mm,
        rect.height / spec.height_mm,
    )

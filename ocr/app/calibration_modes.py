from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class CalibrationMode(str, Enum):
    """Режим проверки эталонного кадра при калибровке камеры."""

    STRICT = "strict"
    SCALE_REFERENCE = "scale_reference"


@dataclass(frozen=True, slots=True)
class CalibrationFrameValidation:
    accepted: bool
    contact_sides: tuple[str, ...]
    allowed_contact_sides: tuple[str, ...]
    reason: str | None


def validate_calibration_frame(
    mode: CalibrationMode,
    frame_contact_sides: Iterable[str],
) -> CalibrationFrameValidation:
    """Проверяет допустимость контакта эталона с границей source-кадра.

    strict:
        эталон не должен касаться ни одной границы.

    scale_reference:
        допускается только нижняя граница. Это штатный режим сортировщика,
        где письмо лежит у нижнего края кадра при неизменных camera/FOV.
        Контакт с top/left/right остаётся блокирующим.
    """

    sides = tuple(dict.fromkeys(str(side).lower() for side in frame_contact_sides))
    known = {"top", "right", "bottom", "left"}
    unknown = tuple(side for side in sides if side not in known)
    if unknown:
        return CalibrationFrameValidation(
            accepted=False,
            contact_sides=sides,
            allowed_contact_sides=(),
            reason=f"Неизвестные стороны контакта: {', '.join(unknown)}",
        )

    if not sides:
        return CalibrationFrameValidation(
            accepted=True,
            contact_sides=(),
            allowed_contact_sides=(),
            reason=None,
        )

    if mode is CalibrationMode.SCALE_REFERENCE:
        allowed = ("bottom",)
        forbidden = tuple(side for side in sides if side not in allowed)
        if not forbidden:
            return CalibrationFrameValidation(
                accepted=True,
                contact_sides=sides,
                allowed_contact_sides=allowed,
                reason=None,
            )
        return CalibrationFrameValidation(
            accepted=False,
            contact_sides=sides,
            allowed_contact_sides=allowed,
            reason=(
                "В scale_reference допускается контакт только с нижней границей; "
                f"обнаружено: {', '.join(sides)}"
            ),
        )

    return CalibrationFrameValidation(
        accepted=False,
        contact_sides=sides,
        allowed_contact_sides=(),
        reason=(
            "В strict-режиме калибровочный эталон не должен касаться границ кадра; "
            f"обнаружено: {', '.join(sides)}"
        ),
    )

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from .gost_r_51506_99 import ENVELOPE_SPECS, EnvelopeFormat


class FormatMode(str, Enum):
    """Режим выбора физического формата письма."""

    AUTO = "auto"
    SESSION = "session"
    FIXED = "fixed"


@dataclass(frozen=True, slots=True)
class FormatDecision:
    format: EnvelopeFormat | None
    status: str
    validation: dict[str, Any]


def require_expected_format(
    format_mode: FormatMode,
    expected_format: EnvelopeFormat | None,
) -> EnvelopeFormat | None:
    """Проверяет контракт mode/expected_format.

    В auto expected_format игнорируется. Для session/fixed формат обязателен.
    """

    if format_mode in {FormatMode.SESSION, FormatMode.FIXED} and expected_format is None:
        raise ValueError(f"Для format_mode={format_mode.value} требуется expected_format")
    return expected_format if format_mode is not FormatMode.AUTO else None


def expected_aspect_error(
    width_px: int,
    height_px: int,
    expected_format: EnvelopeFormat,
) -> float:
    observed = max(width_px, height_px) / float(min(width_px, height_px))
    expected = ENVELOPE_SPECS[expected_format].aspect_ratio
    return abs(observed - expected) / expected


def _resolved_metric_format(metric_decision: Any | None) -> EnvelopeFormat | None:
    if metric_decision is None or getattr(metric_decision, "status", None) != "resolved":
        return None
    value = getattr(metric_decision, "format", None)
    return value if isinstance(value, EnvelopeFormat) else None


def decide_format(
    *,
    format_mode: FormatMode,
    expected_format: EnvelopeFormat | None,
    metric_decision: Any | None,
    selected_profile: Any | None,
    format_candidates: Iterable[Any],
    rectified_width_px: int,
    rectified_height_px: int,
    partial_frame: bool,
    ratio_tolerance: float = 0.08,
) -> FormatDecision:
    """Формирует окончательное решение о формате и его validation metadata."""

    expected = require_expected_format(format_mode, expected_format)
    metric_format = _resolved_metric_format(metric_decision)
    candidates = tuple(format_candidates)

    if format_mode is FormatMode.AUTO:
        if metric_format is not None:
            status = (
                "resolved_by_camera_calibration_partial_frame"
                if partial_frame
                else "resolved_by_camera_calibration"
            )
            resolved = metric_format
        elif selected_profile is not None:
            status = (
                "resolved_by_profile_scoring_partial_frame"
                if partial_frame
                else "resolved_by_profile_scoring"
            )
            resolved = selected_profile.format
        elif partial_frame:
            status = "unreliable_partial_frame"
            resolved = None
        elif not candidates:
            status = "unknown"
            resolved = None
        elif len(candidates) == 1:
            status = "resolved_by_ratio"
            resolved = candidates[0].format
        else:
            status = "ambiguous_by_ratio"
            resolved = None

        return FormatDecision(
            format=resolved,
            status=status,
            validation={
                "status": "auto",
                "expected_format": None,
                "metric_observed_format": metric_format.value if metric_format is not None else None,
                "blocking": False,
                "reasons": [],
                "warnings": [],
            },
        )

    assert expected is not None
    aspect_error = expected_aspect_error(
        rectified_width_px,
        rectified_height_px,
        expected,
    )
    profile_matches = bool(
        selected_profile is not None and selected_profile.format == expected
    )
    metric_matches = metric_format == expected
    reasons: list[str] = []

    if metric_format is not None and metric_format != expected:
        reasons.append(
            f"metric_format={metric_format.value} противоречит expected_format={expected.value}"
        )
    if aspect_error > ratio_tolerance:
        reasons.append(
            f"aspect_error={aspect_error:.4f} превышает допустимые {ratio_tolerance:.4f}"
        )

    validation_common = {
        "expected_format": expected.value,
        "metric_observed_format": metric_format.value if metric_format is not None else None,
        "aspect_error": round(aspect_error, 6),
        "aspect_tolerance": ratio_tolerance,
        "profile_matches_expected": profile_matches,
        "metric_matches_expected": metric_matches,
    }

    if format_mode is FormatMode.FIXED:
        return FormatDecision(
            format=expected,
            status="fixed",
            validation={
                **validation_common,
                "status": "fixed",
                "blocking": False,
                "reasons": [],
                "warnings": reasons,
            },
        )

    # SESSION: expected_format — сильный constraint, но независимые сильные
    # противоречия блокируют дальнейшее использование формата.
    if reasons:
        return FormatDecision(
            format=None,
            status="session_mismatch",
            validation={
                **validation_common,
                "status": "mismatch",
                "blocking": True,
                "reasons": reasons,
                "warnings": [],
            },
        )

    if metric_matches or profile_matches:
        validation_status = "confirmed"
        status = "session_confirmed"
    else:
        validation_status = "plausible"
        status = "session_plausible"

    return FormatDecision(
        format=expected,
        status=status,
        validation={
            **validation_common,
            "status": validation_status,
            "blocking": False,
            "reasons": [],
            "warnings": [],
        },
    )

from types import SimpleNamespace

import pytest

from ocr.app.format_modes import FormatMode, decide_format, require_expected_format
from ocr.app.gost_r_51506_99 import EnvelopeFormat


def _candidate(fmt: EnvelopeFormat, ratio_error: float = 0.01):
    return SimpleNamespace(format=fmt, ratio_error=ratio_error)


def _metric(fmt: EnvelopeFormat | None, status: str = "resolved"):
    return SimpleNamespace(format=fmt, status=status)


def _profile(fmt: EnvelopeFormat):
    return SimpleNamespace(format=fmt)


def test_session_and_fixed_require_expected_format():
    assert require_expected_format(FormatMode.AUTO, None) is None
    with pytest.raises(ValueError):
        require_expected_format(FormatMode.SESSION, None)
    with pytest.raises(ValueError):
        require_expected_format(FormatMode.FIXED, None)


def test_auto_preserves_metric_resolution():
    decision = decide_format(
        format_mode=FormatMode.AUTO,
        expected_format=None,
        metric_decision=_metric(EnvelopeFormat.DL),
        selected_profile=None,
        format_candidates=(_candidate(EnvelopeFormat.DL),),
        rectified_width_px=2000,
        rectified_height_px=1000,
        partial_frame=False,
    )

    assert decision.format is EnvelopeFormat.DL
    assert decision.status == "resolved_by_camera_calibration"
    assert decision.validation["status"] == "auto"


def test_auto_ratio_only_resolution_returns_format_value():
    decision = decide_format(
        format_mode=FormatMode.AUTO,
        expected_format=None,
        metric_decision=None,
        selected_profile=None,
        format_candidates=(_candidate(EnvelopeFormat.DL),),
        rectified_width_px=2000,
        rectified_height_px=1000,
        partial_frame=False,
    )

    assert decision.format is EnvelopeFormat.DL
    assert decision.status == "resolved_by_ratio"


def test_session_confirms_expected_format_from_profile():
    decision = decide_format(
        format_mode=FormatMode.SESSION,
        expected_format=EnvelopeFormat.C5,
        metric_decision=None,
        selected_profile=_profile(EnvelopeFormat.C5),
        format_candidates=(
            _candidate(EnvelopeFormat.C6),
            _candidate(EnvelopeFormat.C5),
            _candidate(EnvelopeFormat.C4),
        ),
        rectified_width_px=1414,
        rectified_height_px=1000,
        partial_frame=False,
    )

    assert decision.format is EnvelopeFormat.C5
    assert decision.status == "session_confirmed"
    assert decision.validation["status"] == "confirmed"
    assert decision.validation["blocking"] is False


def test_session_blocks_strong_aspect_mismatch():
    decision = decide_format(
        format_mode=FormatMode.SESSION,
        expected_format=EnvelopeFormat.C5,
        metric_decision=None,
        selected_profile=None,
        format_candidates=(_candidate(EnvelopeFormat.DL),),
        rectified_width_px=2000,
        rectified_height_px=1000,
        partial_frame=False,
    )

    assert decision.format is None
    assert decision.status == "session_mismatch"
    assert decision.validation["status"] == "mismatch"
    assert decision.validation["blocking"] is True
    assert decision.validation["reasons"]


def test_session_blocks_metric_contradiction_even_when_aspect_is_similar():
    decision = decide_format(
        format_mode=FormatMode.SESSION,
        expected_format=EnvelopeFormat.C5,
        metric_decision=_metric(EnvelopeFormat.C6),
        selected_profile=_profile(EnvelopeFormat.C5),
        format_candidates=(_candidate(EnvelopeFormat.C5), _candidate(EnvelopeFormat.C6)),
        rectified_width_px=1414,
        rectified_height_px=1000,
        partial_frame=False,
    )

    assert decision.format is None
    assert decision.status == "session_mismatch"
    assert decision.validation["metric_observed_format"] == "C6"


def test_fixed_keeps_expected_format_and_reports_warnings():
    decision = decide_format(
        format_mode=FormatMode.FIXED,
        expected_format=EnvelopeFormat.C5,
        metric_decision=_metric(EnvelopeFormat.DL),
        selected_profile=None,
        format_candidates=(_candidate(EnvelopeFormat.DL),),
        rectified_width_px=2000,
        rectified_height_px=1000,
        partial_frame=False,
    )

    assert decision.format is EnvelopeFormat.C5
    assert decision.status == "fixed"
    assert decision.validation["status"] == "fixed"
    assert decision.validation["blocking"] is False
    assert len(decision.validation["warnings"]) >= 1

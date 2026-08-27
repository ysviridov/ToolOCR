from types import SimpleNamespace

import numpy as np

from ocr.app import test_ui_library_preview as ui_status


def _patch_roi(monkeypatch, *, status: str, confirmation_mode: str | None, confidence: float = 0.9):
    monkeypatch.setattr(ui_status, "_load_metadata", lambda file_id: {"id": file_id})
    monkeypatch.setattr(
        ui_status,
        "_image_path",
        lambda meta: SimpleNamespace(read_bytes=lambda: b"image"),
    )
    monkeypatch.setattr(
        ui_status,
        "_decode_image",
        lambda raw: np.zeros((20, 40, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        ui_status,
        "_canonical_from_analysis",
        lambda analysis, image: SimpleNamespace(image=image),
    )

    postcode = SimpleNamespace(
        kind="recipient_postcode",
        status=status,
        confidence=confidence,
        features={
            "confirmation_mode": confirmation_mode,
            "rejection_reason": "start_marker_weak" if status == "stencil_not_found" else None,
        },
    )
    monkeypatch.setattr(
        ui_status,
        "detect_simple_mail_rois",
        lambda image, envelope_format: SimpleNamespace(regions=(postcode,)),
    )


def _result() -> dict:
    return {
        "id": "a" * 32,
        "ok": True,
        "orientation_status": "resolved",
        "format": "C4",
        "debug": {},
    }


def test_strict_postcode_colors_roi_green(monkeypatch):
    _patch_roi(
        monkeypatch,
        status="stencil_detected",
        confirmation_mode="strict_start_marker",
        confidence=0.96,
    )

    summary = ui_status._postcode_summary_for_result(_result())

    assert summary["postcode_roi_color"] == "green"
    assert summary["postcode_confirmation_mode"] == "strict_start_marker"
    assert summary["postcode_confidence"] == 0.96
    assert summary["input_quality_status"] == "ok"


def test_seven_bar_rescue_colors_roi_yellow(monkeypatch):
    _patch_roi(
        monkeypatch,
        status="stencil_detected",
        confirmation_mode="seven_bar_rescue",
        confidence=0.84,
    )

    summary = ui_status._postcode_summary_for_result(_result())

    assert summary["postcode_roi_color"] == "yellow"
    assert summary["postcode_confirmation_mode"] == "seven_bar_rescue"
    assert summary["input_quality_status"] == "ok"


def test_stencil_not_found_without_input_evidence_stays_red(monkeypatch):
    _patch_roi(
        monkeypatch,
        status="stencil_not_found",
        confirmation_mode="none",
        confidence=0.25,
    )

    summary = ui_status._postcode_summary_for_result(_result())

    assert summary["postcode_roi_color"] == "red"
    assert summary["postcode_rejection_reason"] == "start_marker_weak"
    assert summary["input_quality_status"] == "ok"


def test_failed_postcode_with_partial_bottom_frame_becomes_gray(monkeypatch):
    _patch_roi(
        monkeypatch,
        status="stencil_not_found",
        confirmation_mode="none",
        confidence=0.25,
    )
    result = _result()
    result["debug"] = {
        "detector": {
            "frame_status": "partial_frame",
            "frame_contact_sides": ["bottom"],
        }
    }

    summary = ui_status._postcode_summary_for_result(result)

    assert summary["postcode_roi_color"] == "gray"
    assert summary["input_quality_status"] == "partial_crop_suspected"
    assert "partial_frame:bottom" in summary["input_quality_reasons"]


def test_failed_postcode_with_full_frame_but_metric_shortfall_becomes_gray(monkeypatch):
    _patch_roi(
        monkeypatch,
        status="stencil_not_found",
        confirmation_mode="none",
        confidence=0.25,
    )
    result = _result()
    result["debug"] = {
        "detector": {
            "frame_status": "full_frame",
            "frame_contact_sides": [],
        },
        "metric_format": {
            "status": "unknown",
            "measurement": {
                "width_mm": 303.0,
                "height_mm": 216.0,
                "left_height_mm": 231.0,
                "right_height_mm": 202.0,
                "top_width_mm": 304.0,
                "bottom_width_mm": 302.0,
            },
        },
    }

    summary = ui_status._postcode_summary_for_result(result)

    assert summary["postcode_roi_color"] == "gray"
    assert summary["input_quality_status"] == "partial_crop_suspected"
    assert any(reason.startswith("width_shortfall:") for reason in summary["input_quality_reasons"])
    assert any(reason.startswith("height_shortfall:") for reason in summary["input_quality_reasons"])
    assert any(reason.startswith("side_height_asymmetry:") for reason in summary["input_quality_reasons"])


def test_successful_postcode_is_not_downgraded_by_partial_frame(monkeypatch):
    _patch_roi(
        monkeypatch,
        status="stencil_detected",
        confirmation_mode="strict_start_marker",
        confidence=0.97,
    )
    result = _result()
    result["debug"] = {
        "detector": {
            "frame_status": "partial_frame",
            "frame_contact_sides": ["bottom"],
        }
    }

    summary = ui_status._postcode_summary_for_result(result)

    assert summary["postcode_roi_color"] == "green"
    assert summary["input_quality_status"] == "ok"
    assert summary["input_quality_reasons"] == []


def test_unresolved_orientation_keeps_roi_white():
    result = _result()
    result["orientation_status"] = "ambiguous"

    summary = ui_status._postcode_summary_for_result(result)

    assert summary["postcode_roi_color"] == "white"
    assert summary["postcode_roi_status"] == "not_evaluated"
    assert summary["postcode_roi_note"] == "orientation unresolved"
    assert summary["input_quality_status"] == "not_evaluated"

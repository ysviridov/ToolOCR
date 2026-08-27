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


def test_stencil_not_found_colors_roi_red(monkeypatch):
    _patch_roi(
        monkeypatch,
        status="stencil_not_found",
        confirmation_mode="none",
        confidence=0.25,
    )

    summary = ui_status._postcode_summary_for_result(_result())

    assert summary["postcode_roi_color"] == "red"
    assert summary["postcode_rejection_reason"] == "start_marker_weak"


def test_unresolved_orientation_keeps_roi_white():
    result = _result()
    result["orientation_status"] = "ambiguous"

    summary = ui_status._postcode_summary_for_result(result)

    assert summary["postcode_roi_color"] == "white"
    assert summary["postcode_roi_status"] == "not_evaluated"
    assert summary["postcode_roi_note"] == "orientation unresolved"

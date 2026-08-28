import asyncio
from types import SimpleNamespace

from ocr.app import test_ui_library_preview as ui_status


def _summary() -> dict:
    return {
        "postcode_roi_status": "stencil_detected",
        "postcode_confirmation_mode": "strict_start_marker",
        "postcode_confidence": 0.97,
        "postcode_rejection_reason": None,
        "postcode_roi_color": "green",
        "postcode_roi_note": None,
        "postcode_ocr_status": "recognized",
        "postcode_ocr_text": "167420",
        "postcode_ocr_postcode": "167420",
        "postcode_ocr_confidence": 0.91,
        "postcode_ocr_min_digit_confidence": 0.82,
        "postcode_ocr_structurally_valid": True,
        "postcode_ocr_reason": None,
        "postcode_ocr_digits": [
            {"index": 1, "status": "recognized", "digit": "1", "confidence": 0.96, "reason": None},
            {"index": 2, "status": "recognized", "digit": "6", "confidence": 0.94, "reason": None},
            {"index": 3, "status": "recognized", "digit": "7", "confidence": 0.92, "reason": None},
            {"index": 4, "status": "recognized", "digit": "4", "confidence": 0.90, "reason": None},
            {"index": 5, "status": "recognized", "digit": "2", "confidence": 0.91, "reason": None},
            {"index": 6, "status": "recognized", "digit": "0", "confidence": 0.82, "reason": None},
        ],
        "input_quality_status": "ok",
        "input_quality_reasons": [],
        "input_quality_metrics": {},
    }


def test_batch_debug_contains_recognized_postcode(monkeypatch):
    async def fake_base_run_test(request):
        return {
            "results": [
                {
                    "id": "a" * 32,
                    "debug": {},
                }
            ]
        }

    monkeypatch.setattr(ui_status, "_base_run_test", fake_base_run_test)
    monkeypatch.setattr(ui_status, "_postcode_summary_for_result", lambda result: _summary())

    payload = asyncio.run(
        ui_status.run_test_with_postcode_roi_status(SimpleNamespace())
    )

    result = payload["results"][0]
    assert result["postcode_ocr_text"] == "167420"
    assert result["postcode_ocr_postcode"] == "167420"
    assert result["postcode_ocr_structurally_valid"] is True

    debug_ocr = result["debug"]["test_ui_postcode_ocr"]
    assert debug_ocr["status"] == "recognized"
    assert debug_ocr["text"] == "167420"
    assert debug_ocr["postcode"] == "167420"
    assert debug_ocr["confidence"] == 0.91
    assert debug_ocr["min_digit_confidence"] == 0.82
    assert len(debug_ocr["digits"]) == 6

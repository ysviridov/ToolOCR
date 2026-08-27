from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from .gost_r_51506_99 import EnvelopeFormat
from .roi import detect_simple_mail_rois
from .roi_test_ui import SUPPORTED_ROI_FORMATS, _canonical_from_analysis
from .test_ui import (
    TestRunRequest,
    _decode_image,
    _image_path,
    _load_metadata,
    run_test as _base_run_test,
)


router = APIRouter(tags=["test-ui"])
HTML_PATH = Path(__file__).with_name("test_ui.html")
SUPPORTED_ROI_FORMAT_VALUES = {item.value for item in SUPPORTED_ROI_FORMATS}


_ROI_STATUS_STYLE = r"""
<style>
  button.roi-status-white {
    background:#fff !important;
    border-color:#aeb7c6 !important;
    color:var(--text) !important;
  }
  button.roi-status-red {
    background:#fdecea !important;
    border-color:#df8d84 !important;
    color:#9f1c12 !important;
  }
  button.roi-status-yellow {
    background:#fff5d6 !important;
    border-color:#d8b54b !important;
    color:#7a5400 !important;
  }
  button.roi-status-green {
    background:#e7f4ea !important;
    border-color:#78b98a !important;
    color:#0d6b2f !important;
  }
</style>
"""


_PREVIEW_SCRIPT = r"""
<script>
(() => {
  const libraryBody = document.getElementById('libraryBody');
  const resultBody = document.getElementById('resultBody');
  const dialog = document.getElementById('cropDialog');
  const title = document.getElementById('cropTitle');
  const image = document.getElementById('cropImage');
  if (!libraryBody || !resultBody || !dialog || !title || !image) return;

  function addPreviewButtons() {
    libraryBody.querySelectorAll('tr').forEach(row => {
      const pick = row.querySelector('input.pick');
      const nameCell = row.querySelector('td.name');
      if (!pick || !nameCell || nameCell.querySelector('.library-preview-button')) return;

      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'library-preview-button';
      button.textContent = 'Предпросмотр';
      button.style.marginLeft = '10px';
      button.style.padding = '4px 8px';
      button.style.fontSize = '12px';
      button.title = 'Открыть исходный файл';

      button.addEventListener('click', event => {
        event.preventDefault();
        event.stopPropagation();
        const fileId = pick.value;
        const fileName = nameCell.getAttribute('title') || fileId;
        title.textContent = `Предпросмотр — ${fileName}`;
        image.alt = `Предпросмотр — ${fileName}`;
        image.src = `/v1/test-ui/images/${fileId}/original?t=${Date.now()}`;
        dialog.showModal();
      });

      nameCell.appendChild(button);
    });
  }

  function roiButtonTitle(item) {
    const confidence = typeof item.postcode_confidence === 'number'
      ? ` · confidence ${item.postcode_confidence.toFixed(2)}`
      : '';

    if (item.postcode_roi_color === 'green') {
      return `Postcode ROI: strict_start_marker${confidence}`;
    }
    if (item.postcode_roi_color === 'yellow') {
      return `Postcode ROI: seven_bar_rescue${confidence}`;
    }
    if (item.postcode_roi_color === 'red') {
      const reason = item.postcode_rejection_reason
        ? ` · ${item.postcode_rejection_reason}`
        : '';
      return `Postcode ROI: stencil_not_found${reason}`;
    }
    const reason = item.postcode_roi_note ? ` · ${item.postcode_roi_note}` : '';
    return `Postcode ROI: не оценён${reason}`;
  }

  function applyRoiButtonColors() {
    resultBody.querySelectorAll('tr').forEach(row => {
      const roiButton = [...row.querySelectorAll('button')]
        .find(button => button.textContent.trim() === 'ROI');
      if (!roiButton) return;

      const onclick = roiButton.getAttribute('onclick') || '';
      const match = onclick.match(/showRoi\('([0-9a-f]+)'\)/i);
      if (!match) return;

      const item = state.results.get(match[1]);
      if (!item) return;

      roiButton.classList.remove(
        'roi-status-white',
        'roi-status-red',
        'roi-status-yellow',
        'roi-status-green',
      );
      const color = ['red', 'yellow', 'green'].includes(item.postcode_roi_color)
        ? item.postcode_roi_color
        : 'white';
      roiButton.classList.add(`roi-status-${color}`);
      roiButton.title = roiButtonTitle(item);
    });
  }

  new MutationObserver(addPreviewButtons).observe(libraryBody, { childList: true, subtree: true });
  new MutationObserver(applyRoiButtonColors).observe(resultBody, { childList: true, subtree: true });
  addPreviewButtons();
  applyRoiButtonColors();
})();
</script>
"""


def _empty_postcode_summary(note: str) -> dict[str, Any]:
    return {
        "postcode_roi_status": "not_evaluated",
        "postcode_confirmation_mode": None,
        "postcode_confidence": None,
        "postcode_rejection_reason": None,
        "postcode_roi_color": "white",
        "postcode_roi_note": note,
    }


def _postcode_summary_for_result(result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("ok"):
        return _empty_postcode_summary("layout error")

    if result.get("orientation_status") != "resolved":
        return _empty_postcode_summary("orientation unresolved")

    format_value = result.get("format")
    if format_value not in SUPPORTED_ROI_FORMAT_VALUES:
        return _empty_postcode_summary("ROI format unsupported")

    analysis = result.get("debug")
    if not isinstance(analysis, dict):
        return _empty_postcode_summary("layout debug unavailable")

    try:
        meta = _load_metadata(str(result["id"]))
        raw = _image_path(meta).read_bytes()
        image = _decode_image(raw)
        canonical = _canonical_from_analysis(analysis, image)
        roi = detect_simple_mail_rois(canonical.image, EnvelopeFormat(str(format_value)))
        postcode = next(
            region for region in roi.regions if region.kind == "recipient_postcode"
        )
    except Exception as exc:
        summary = _empty_postcode_summary(f"ROI evaluation error: {type(exc).__name__}")
        summary["postcode_rejection_reason"] = str(exc)
        return summary

    features = postcode.features or {}
    confirmation_mode = features.get("confirmation_mode")
    rejection_reason = features.get("rejection_reason")

    if postcode.status == "stencil_detected" and confirmation_mode == "strict_start_marker":
        color = "green"
    elif postcode.status == "stencil_detected" and confirmation_mode == "seven_bar_rescue":
        color = "yellow"
    elif postcode.status == "stencil_not_found":
        color = "red"
    else:
        color = "white"

    return {
        "postcode_roi_status": postcode.status,
        "postcode_confirmation_mode": confirmation_mode,
        "postcode_confidence": postcode.confidence,
        "postcode_rejection_reason": rejection_reason,
        "postcode_roi_color": color,
        "postcode_roi_note": None,
    }


@router.post("/v1/test-ui/run")
async def run_test_with_postcode_roi_status(request: TestRunRequest) -> dict[str, Any]:
    """Расширяет обычный batch-test кратким postcode ROI summary для UI."""

    payload = await _base_run_test(request)
    for result in payload.get("results", []):
        summary = _postcode_summary_for_result(result)
        result.update(summary)

        debug = result.get("debug")
        if isinstance(debug, dict):
            debug["test_ui_postcode_roi"] = {
                "status": summary["postcode_roi_status"],
                "confirmation_mode": summary["postcode_confirmation_mode"],
                "confidence": summary["postcode_confidence"],
                "rejection_reason": summary["postcode_rejection_reason"],
                "color": summary["postcode_roi_color"],
                "note": summary["postcode_roi_note"],
            }

    return payload


@router.get("/test-ui", response_class=HTMLResponse, include_in_schema=False)
def test_ui_page_with_library_preview() -> HTMLResponse:
    """Расширенный Test UI: library preview, C4 ROI и цвет postcode ROI."""

    try:
        html = HTML_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Не найден шаблон test-ui") from exc

    if "</head>" not in html or "</body>" not in html:
        raise HTTPException(status_code=500, detail="Некорректный шаблон test-ui")

    html = html.replace(
        "!['DL','C5'].includes(item.format)",
        "!['DL','C5','C4'].includes(item.format)",
        1,
    )
    html = html.replace("</head>", _ROI_STATUS_STYLE + "\n</head>", 1)
    html = html.replace("</body>", _PREVIEW_SCRIPT + "\n</body>", 1)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})

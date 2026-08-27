from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from .gost_r_51506_99 import ENVELOPE_SPECS, EnvelopeFormat
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

# Это UI-диагностика, а не новый layout hard-gate. Порогами намеренно
# помечаем только выраженную геометрическую аномалию входа.
_INPUT_WIDTH_SHORTFALL_RATIO = 0.03
_INPUT_HEIGHT_SHORTFALL_RATIO = 0.03
_INPUT_SIDE_HEIGHT_ASYMMETRY_RATIO = 0.05
_INPUT_TOP_BOTTOM_WIDTH_ASYMMETRY_RATIO = 0.03


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
  button.roi-status-gray {
    background:#eef1f5 !important;
    border-color:#9ca6b5 !important;
    color:#4f5968 !important;
  }
  .input-quality-badge {
    display:inline-block;
    margin-left:7px;
    border-radius:999px;
    padding:2px 7px;
    font-size:11px;
    line-height:1.35;
    white-space:nowrap;
  }
  .input-quality-badge.suspect {
    background:#eef1f5;
    border:1px solid #aab2bf;
    color:#4f5968;
  }
  .roi-status-legend {
    display:flex;
    flex-wrap:wrap;
    gap:7px 12px;
    align-items:center;
    margin:0 0 12px;
    color:var(--muted);
    font-size:12px;
  }
  .roi-status-legend span::before {
    content:'';
    display:inline-block;
    width:11px;
    height:11px;
    margin-right:5px;
    border-radius:3px;
    border:1px solid #aeb7c6;
    vertical-align:-1px;
  }
  .roi-status-legend .legend-green::before { background:#e7f4ea; border-color:#78b98a; }
  .roi-status-legend .legend-yellow::before { background:#fff5d6; border-color:#d8b54b; }
  .roi-status-legend .legend-gray::before { background:#eef1f5; border-color:#9ca6b5; }
  .roi-status-legend .legend-red::before { background:#fdecea; border-color:#df8d84; }
  .roi-status-legend .legend-white::before { background:#fff; }
</style>
"""


_PREVIEW_SCRIPT = r"""
<script>
(() => {
  const libraryBody = document.getElementById('libraryBody');
  const resultBody = document.getElementById('resultBody');
  const resultStats = document.getElementById('resultStats');
  const dialog = document.getElementById('cropDialog');
  const title = document.getElementById('cropTitle');
  const image = document.getElementById('cropImage');
  if (!libraryBody || !resultBody || !dialog || !title || !image) return;

  function ensureLegend() {
    if (document.getElementById('roiStatusLegend')) return;
    const tableWrap = resultBody.closest('.table-wrap');
    if (!tableWrap) return;
    const legend = document.createElement('div');
    legend.id = 'roiStatusLegend';
    legend.className = 'roi-status-legend';
    legend.innerHTML = `
      <strong>ROI postcode:</strong>
      <span class="legend-green">strict</span>
      <span class="legend-yellow">seven-bar rescue</span>
      <span class="legend-gray">возможен дефект/crop входа</span>
      <span class="legend-red">detector failure</span>
      <span class="legend-white">не оценён</span>`;
    tableWrap.parentNode.insertBefore(legend, tableWrap);
  }

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

  function inputQualityText(item) {
    const reasons = Array.isArray(item.input_quality_reasons)
      ? item.input_quality_reasons.join(', ')
      : '';
    return reasons ? `Вход: partial/crop suspected · ${reasons}` : 'Вход: partial/crop suspected';
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
    if (item.postcode_roi_color === 'gray') {
      const detectorReason = item.postcode_rejection_reason
        ? ` · postcode: ${item.postcode_rejection_reason}`
        : '';
      return `${inputQualityText(item)}${detectorReason}`;
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

  function applyInputQualityBadge(row, item) {
    const cells = row.querySelectorAll('td');
    if (cells.length < 7) return;
    const normalizationCell = cells[6];
    let badge = normalizationCell.querySelector('.input-quality-badge');

    if (item.input_quality_status !== 'partial_crop_suspected') {
      if (badge) badge.remove();
      return;
    }

    if (!badge) {
      badge = document.createElement('span');
      badge.className = 'input-quality-badge suspect';
      normalizationCell.appendChild(badge);
    }
    badge.textContent = 'INPUT: crop?';
    badge.title = inputQualityText(item);
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
        'roi-status-gray',
      );
      const color = ['red', 'yellow', 'green', 'gray'].includes(item.postcode_roi_color)
        ? item.postcode_roi_color
        : 'white';
      roiButton.classList.add(`roi-status-${color}`);
      roiButton.title = roiButtonTitle(item);
      applyInputQualityBadge(row, item);
    });
  }

  new MutationObserver(addPreviewButtons).observe(libraryBody, { childList: true, subtree: true });
  new MutationObserver(applyRoiButtonColors).observe(resultBody, { childList: true, subtree: true });
  addPreviewButtons();
  ensureLegend();
  applyRoiButtonColors();
})();
</script>
"""


def _empty_input_quality(status: str = "not_evaluated") -> dict[str, Any]:
    return {
        "input_quality_status": status,
        "input_quality_reasons": [],
        "input_quality_metrics": {},
    }


def _empty_postcode_summary(note: str) -> dict[str, Any]:
    summary = {
        "postcode_roi_status": "not_evaluated",
        "postcode_confirmation_mode": None,
        "postcode_confidence": None,
        "postcode_rejection_reason": None,
        "postcode_roi_color": "white",
        "postcode_roi_note": note,
    }
    summary.update(_empty_input_quality())
    return summary


def _safe_ratio(numerator: float | None, denominator: float) -> float | None:
    if numerator is None or denominator <= 0:
        return None
    return float(numerator) / denominator


def _input_quality_for_failed_postcode(
    analysis: dict[str, Any],
    envelope_format: EnvelopeFormat,
) -> dict[str, Any]:
    """Диагностирует вероятный физический crop/деформацию входа.

    Функция используется только после `stencil_not_found` и не влияет на
    layout/ROI решения. Статус `partial_crop_suspected` — предупреждение UI,
    а не доказательство повреждения изображения.
    """

    reasons: list[str] = []
    metrics: dict[str, Any] = {}

    detector = analysis.get("detector") or {}
    frame_status = detector.get("frame_status")
    contact_sides = [str(item) for item in (detector.get("frame_contact_sides") or [])]
    metrics["frame_status"] = frame_status
    metrics["frame_contact_sides"] = contact_sides

    relevant_contacts = [side for side in contact_sides if side in {"left", "bottom"}]
    if frame_status == "partial_frame" and relevant_contacts:
        reasons.append("partial_frame:" + "+".join(relevant_contacts))

    metric_format = analysis.get("metric_format") or {}
    measurement = metric_format.get("measurement") or {}
    spec = ENVELOPE_SPECS[envelope_format]

    width_mm = measurement.get("width_mm")
    height_mm = measurement.get("height_mm")
    left_height_mm = measurement.get("left_height_mm")
    right_height_mm = measurement.get("right_height_mm")
    top_width_mm = measurement.get("top_width_mm")
    bottom_width_mm = measurement.get("bottom_width_mm")

    width_shortfall = (
        max(0.0, (spec.width_mm - float(width_mm)) / spec.width_mm)
        if isinstance(width_mm, (int, float))
        else None
    )
    height_shortfall = (
        max(0.0, (spec.height_mm - float(height_mm)) / spec.height_mm)
        if isinstance(height_mm, (int, float))
        else None
    )
    side_height_asymmetry = (
        abs(float(left_height_mm) - float(right_height_mm)) / spec.height_mm
        if isinstance(left_height_mm, (int, float)) and isinstance(right_height_mm, (int, float))
        else None
    )
    top_bottom_width_asymmetry = (
        abs(float(top_width_mm) - float(bottom_width_mm)) / spec.width_mm
        if isinstance(top_width_mm, (int, float)) and isinstance(bottom_width_mm, (int, float))
        else None
    )

    metrics.update(
        {
            "expected_width_mm": spec.width_mm,
            "expected_height_mm": spec.height_mm,
            "measured_width_mm": width_mm,
            "measured_height_mm": height_mm,
            "width_shortfall_ratio": None if width_shortfall is None else round(width_shortfall, 4),
            "height_shortfall_ratio": None if height_shortfall is None else round(height_shortfall, 4),
            "side_height_asymmetry_ratio": None if side_height_asymmetry is None else round(side_height_asymmetry, 4),
            "top_bottom_width_asymmetry_ratio": None if top_bottom_width_asymmetry is None else round(top_bottom_width_asymmetry, 4),
            "metric_status": metric_format.get("status"),
        }
    )

    if width_shortfall is not None and width_shortfall >= _INPUT_WIDTH_SHORTFALL_RATIO:
        reasons.append(f"width_shortfall:{width_shortfall:.1%}")
    if height_shortfall is not None and height_shortfall >= _INPUT_HEIGHT_SHORTFALL_RATIO:
        reasons.append(f"height_shortfall:{height_shortfall:.1%}")
    if (
        side_height_asymmetry is not None
        and side_height_asymmetry >= _INPUT_SIDE_HEIGHT_ASYMMETRY_RATIO
    ):
        reasons.append(f"side_height_asymmetry:{side_height_asymmetry:.1%}")
    if (
        top_bottom_width_asymmetry is not None
        and top_bottom_width_asymmetry >= _INPUT_TOP_BOTTOM_WIDTH_ASYMMETRY_RATIO
    ):
        reasons.append(f"top_bottom_width_asymmetry:{top_bottom_width_asymmetry:.1%}")

    return {
        "input_quality_status": "partial_crop_suspected" if reasons else "ok",
        "input_quality_reasons": reasons,
        "input_quality_metrics": metrics,
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
        envelope_format = EnvelopeFormat(str(format_value))
        roi = detect_simple_mail_rois(canonical.image, envelope_format)
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
        input_quality = _empty_input_quality("ok")
    elif postcode.status == "stencil_detected" and confirmation_mode == "seven_bar_rescue":
        color = "yellow"
        input_quality = _empty_input_quality("ok")
    elif postcode.status == "stencil_not_found":
        input_quality = _input_quality_for_failed_postcode(analysis, envelope_format)
        color = "gray" if input_quality["input_quality_status"] == "partial_crop_suspected" else "red"
    else:
        color = "white"
        input_quality = _empty_input_quality()

    summary = {
        "postcode_roi_status": postcode.status,
        "postcode_confirmation_mode": confirmation_mode,
        "postcode_confidence": postcode.confidence,
        "postcode_rejection_reason": rejection_reason,
        "postcode_roi_color": color,
        "postcode_roi_note": None,
    }
    summary.update(input_quality)
    return summary


@router.post("/v1/test-ui/run")
async def run_test_with_postcode_roi_status(request: TestRunRequest) -> dict[str, Any]:
    """Расширяет batch-test postcode ROI и диагностикой качества входа."""

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
            debug["test_ui_input_quality"] = {
                "status": summary["input_quality_status"],
                "reasons": summary["input_quality_reasons"],
                "metrics": summary["input_quality_metrics"],
            }

    return payload


@router.get("/test-ui", response_class=HTMLResponse, include_in_schema=False)
def test_ui_page_with_library_preview() -> HTMLResponse:
    """Test UI: preview, postcode ROI colors и input-quality diagnostics."""

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

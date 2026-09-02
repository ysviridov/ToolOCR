from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter

from . import postcode_runtime as _postcode_runtime
from . import test_ui_library_preview as _legacy_test_ui
from .postcode_onnx import (
    configured_onnx_model_path,
    configured_recognizer_engine,
    onnx_engine_label,
)
from .test_ui import TestRunRequest


router = APIRouter(tags=["test-ui"])

# Batch Test UI исторически импортировал Tesseract-only postcode_recognizer
# напрямую. ROI preview уже был переключён на ONNX-primary runtime отдельно,
# из-за чего overlay и debug могли показывать разные результаты одного письма.
# Здесь batch route использует ровно те же runtime-функции.
_legacy_test_ui.recognize_postcode_digits = _postcode_runtime.recognize_postcode_digits
_legacy_test_ui.postcode_recognition_to_dict = _postcode_runtime.postcode_recognition_to_dict


_META_BUTTON_SCRIPT = r"""
<script data-toolocr-postcode-meta="v1">
(() => {
  const resultBody = document.getElementById('resultBody');
  if (!resultBody) return;

  function addMetaButtons() {
    resultBody.querySelectorAll('tr').forEach(row => {
      if (row.querySelector('.postcode-meta-button')) return;
      const roiButton = [...row.querySelectorAll('button')]
        .find(button => button.textContent.trim() === 'ROI');
      if (!roiButton) return;

      const onclick = roiButton.getAttribute('onclick') || '';
      const match = onclick.match(/showRoi\('([0-9a-f]+)'\)/i);
      if (!match) return;

      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'postcode-meta-button';
      button.textContent = 'META';
      button.title = 'Открыть актуальный /roi/meta JSON';
      button.style.marginLeft = '6px';
      button.addEventListener('click', event => {
        event.preventDefault();
        event.stopPropagation();
        const url = `/v1/test-ui/images/${match[1]}/roi/meta`;
        window.open(url, '_blank', 'noopener');
      });
      roiButton.insertAdjacentElement('afterend', button);
    });
  }

  new MutationObserver(addMetaButtons).observe(resultBody, { childList: true, subtree: true });
  addMetaButtons();
})();
</script>
"""

# test_ui_page_with_library_preview читает _PREVIEW_SCRIPT при каждом запросе,
# поэтому можно безопасно добавить кнопку без копирования HTML-шаблона.
if 'data-toolocr-postcode-meta="v1"' not in _legacy_test_ui._PREVIEW_SCRIPT:
    _legacy_test_ui._PREVIEW_SCRIPT += "\n" + _META_BUTTON_SCRIPT


def _engine_from_digits(digits: list[dict[str, Any]]) -> tuple[str, str | None]:
    mode = configured_recognizer_engine()
    model_path = configured_onnx_model_path()
    engines = {
        str(item.get("engine"))
        for item in digits
        if isinstance(item, dict) and item.get("engine")
    }

    if engines == {"onnx"}:
        return onnx_engine_label(model_path), str(model_path)
    if engines == {"tesseract_single_digit"}:
        if mode in {"onnx", "auto"}:
            return "tesseract_fallback_from_onnx+stencil_dot_suppression_v1", str(model_path)
        return "tesseract_single_digit+stencil_dot_suppression_v1", None
    if "onnx" in engines and "tesseract_single_digit" in engines:
        return f"{onnx_engine_label(model_path)}+tesseract_fallback", str(model_path)

    if mode in {"onnx", "auto"}:
        return onnx_engine_label(model_path), str(model_path)
    return "tesseract_single_digit+stencil_dot_suppression_v1", None


def _geometric_mean_confidence(digits: list[dict[str, Any]]) -> float | None:
    if len(digits) != 6:
        return None
    values: list[float] = []
    for item in digits:
        value = item.get("confidence") if isinstance(item, dict) else None
        if not isinstance(value, (int, float)):
            return None
        values.append(max(0.0, min(1.0, float(value))))
    if any(value <= 0.0 for value in values):
        return 0.0
    return round(float(math.exp(sum(math.log(value) for value in values) / len(values))), 6)


def _enrich_runtime_debug(result: dict[str, Any]) -> None:
    debug = result.get("debug")
    if not isinstance(debug, dict):
        return
    ocr = debug.get("test_ui_postcode_ocr")
    if not isinstance(ocr, dict):
        return

    digits = ocr.get("digits")
    if not isinstance(digits, list):
        digits = []
    engine, model_path = _engine_from_digits(digits)
    geometric = _geometric_mean_confidence(digits)
    file_id = str(result.get("id") or "")
    roi_meta_url = (
        f"/v1/test-ui/images/{file_id}/roi/meta"
        if file_id
        else None
    )

    ocr["engine"] = engine
    ocr["geometric_mean_confidence"] = geometric
    ocr["model_path"] = model_path
    ocr["roi_meta_url"] = roi_meta_url

    # Дублируем ключевые runtime-поля на уровень результата: UI/tooling может
    # читать их без разбора большого debug payload.
    result["postcode_ocr_engine"] = engine
    result["postcode_ocr_geometric_mean_confidence"] = geometric
    result["postcode_ocr_model_path"] = model_path
    result["postcode_roi_meta_url"] = roi_meta_url
    debug["test_ui_roi_meta"] = {
        "url": roi_meta_url,
        "source": "live_endpoint",
        "note": "Полный актуальный ROI/CNN payload доступен по этому endpoint",
    }


@router.post("/v1/test-ui/run")
async def run_test_with_current_postcode_runtime(request: TestRunRequest) -> dict[str, Any]:
    """Batch Test UI с тем же ONNX-primary recognizer, что и /roi/meta."""

    payload = await _legacy_test_ui.run_test_with_postcode_roi_status(request)
    for result in payload.get("results", []):
        if isinstance(result, dict):
            _enrich_runtime_debug(result)
    return payload

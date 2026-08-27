from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse


router = APIRouter(tags=["test-ui"])
HTML_PATH = Path(__file__).with_name("test_ui.html")

_PREVIEW_SCRIPT = r"""
<script>
(() => {
  const body = document.getElementById('libraryBody');
  const dialog = document.getElementById('cropDialog');
  const title = document.getElementById('cropTitle');
  const image = document.getElementById('cropImage');
  if (!body || !dialog || !title || !image) return;

  function addPreviewButtons() {
    body.querySelectorAll('tr').forEach(row => {
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

  new MutationObserver(addPreviewButtons).observe(body, { childList: true, subtree: true });
  addPreviewButtons();
})();
</script>
"""


@router.get("/test-ui", response_class=HTMLResponse, include_in_schema=False)
def test_ui_page_with_library_preview() -> HTMLResponse:
    """Расширенный Test UI: library preview и ROI для DL/C5/C4.

    JPEG отображается исходными байтами через существующий /original endpoint.
    TIFF/BMP там же транскодируются в JPEG только для браузерного preview;
    постоянный файл в Docker volume не изменяется.

    Базовый test_ui.html пока содержит список DL/C5 для кнопки ROI. Здесь C4
    добавляется при выдаче HTML, чтобы не дублировать большой шаблон ради
    одного feature-flag. Backend ROI API независимо проверяет поддержку C4.
    """

    try:
        html = HTML_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Не найден шаблон test-ui") from exc

    if "</body>" not in html:
        raise HTTPException(status_code=500, detail="Некорректный шаблон test-ui")

    html = html.replace(
        "!['DL','C5'].includes(item.format)",
        "!['DL','C5','C4'].includes(item.format)",
        1,
    )
    html = html.replace("</body>", _PREVIEW_SCRIPT + "\n</body>", 1)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})

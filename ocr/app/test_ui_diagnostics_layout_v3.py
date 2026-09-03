from __future__ import annotations

from . import test_ui_library_preview as _test_ui


_LAYOUT_V3_STYLE = r"""
<style data-toolocr-debug-diagnostics-layout="v3">
  /* На широких мониторах диагностика использует почти весь viewport и больше
     не ограничена фиксированными 1320/1540 px. Пользователь также может
     вручную изменить размер dialog за нижний правый угол. */
  #debugDialog[open] {
    width:calc(100vw - 24px);
    max-width:none;
    max-height:calc(100vh - 24px);
    min-width:min(900px, calc(100vw - 24px));
    min-height:420px;
    resize:both;
    overflow:hidden;
    display:flex;
    flex-direction:column;
  }
  #debugDialog > .modal-head { flex:0 0 auto; }
  #debugDialog > .modal-body {
    flex:1 1 auto;
    min-height:0;
    max-height:none;
    overflow:auto;
  }

  .diag-special-row > td:first-child { width:22% !important; }
  .diag-special-row > td:nth-child(2) { width:auto !important; }

  .diag-entity-grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
    gap:10px;
    width:100%;
    min-width:0;
  }
  .diag-entity-grid.compact {
    grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
  }
  .diag-entity-card {
    border:1px solid #dfe4ec;
    border-radius:9px;
    background:#fff;
    overflow:hidden;
    min-width:0;
  }
  .diag-entity-card.best { border-color:#9ab5e8; box-shadow:0 0 0 1px #d8e4fb inset; }
  .diag-entity-head {
    display:flex;
    align-items:center;
    gap:8px;
    padding:8px 10px;
    background:#f7f8fa;
    border-bottom:1px solid #e5e9f0;
  }
  .diag-entity-title { font-weight:760; font-size:14px; overflow-wrap:anywhere; }
  .diag-entity-rank {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    min-width:24px;
    height:24px;
    padding:0 6px;
    border-radius:999px;
    background:#eef1f5;
    color:#526072;
    font-size:11px;
    font-weight:800;
  }
  .diag-entity-head .diag-badge,
  .diag-entity-head .diag-entity-score { margin-left:auto; }
  .diag-entity-score { font-variant-numeric:tabular-nums; font-weight:760; }
  .diag-entity-meta {
    display:grid;
    grid-template-columns:minmax(92px,auto) 1fr;
    gap:5px 9px;
    padding:8px 10px;
    font-size:12px;
  }
  .diag-entity-meta dt { color:var(--muted); }
  .diag-entity-meta dd { margin:0; min-width:0; overflow-wrap:anywhere; font-variant-numeric:tabular-nums; }
  .diag-channel-list { padding:7px 10px 9px; border-top:1px solid #edf0f4; }
  .diag-channel-row {
    display:grid;
    grid-template-columns:minmax(105px,1.2fr) minmax(84px,2fr) 72px;
    gap:8px;
    align-items:center;
    margin:5px 0;
    font-size:11px;
  }
  .diag-channel-name { color:#596579; overflow-wrap:anywhere; }
  .diag-meter { height:7px; border-radius:999px; background:#eef1f5; overflow:hidden; }
  .diag-meter > span { display:block; height:100%; background:#7b96c8; border-radius:999px; }
  .diag-channel-value { text-align:right; font-variant-numeric:tabular-nums; }
  .diag-mini-details { border-top:1px solid #edf0f4; }
  .diag-mini-details > summary { cursor:pointer; padding:6px 10px; color:#536074; font-size:12px; }
  .diag-mini-details-body { padding:0 10px 8px; }
  .diag-kv-line { display:flex; justify-content:space-between; gap:12px; padding:2px 0; font-size:11px; }
  .diag-kv-line span:first-child { color:var(--muted); }
  .diag-chip-list { display:flex; flex-wrap:wrap; gap:5px; padding:8px 10px; border-top:1px solid #edf0f4; }
  .diag-data-chip { border-radius:999px; padding:3px 7px; background:#eef1f5; font-size:11px; white-space:nowrap; }
  .diag-point-grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(135px,1fr));
    gap:7px;
    width:100%;
  }
  .diag-point-card { border:1px solid #e2e6ed; border-radius:7px; padding:7px 8px; background:#fafbfc; }
  .diag-point-title { color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.03em; }
  .diag-point-value { margin-top:2px; font:12px/1.35 ui-monospace,SFMono-Regular,Consolas,monospace; }

  @media (min-width:1900px) {
    .diag-entity-grid { grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }
    .diag-digit-grid { grid-template-columns:repeat(auto-fit,minmax(285px,1fr)); }
  }
  @media (max-width:920px) {
    #debugDialog[open] {
      width:calc(100vw - 10px);
      min-width:0;
      max-height:calc(100vh - 10px);
      min-height:320px;
      resize:none;
    }
    .diag-entity-grid,.diag-entity-grid.compact { grid-template-columns:1fr; }
  }
</style>
"""


_LAYOUT_V3_SCRIPT = r"""
<script data-toolocr-debug-diagnostics-layout="v3">
(() => {
  const esc = value => escapeHtml(value === null || value === undefined ? '—' : String(value));

  function fmt(value, digits=4) {
    const number = Number(value);
    if (!Number.isFinite(number)) return value === null || value === undefined ? '—' : String(value);
    if (Number.isInteger(number)) return String(number);
    if (Math.abs(number) >= 100) return number.toFixed(2).replace(/0+$/,'').replace(/\.$/,'');
    if (Math.abs(number) >= 0.01) return number.toFixed(digits).replace(/0+$/,'').replace(/\.$/,'');
    return number.toExponential(3);
  }

  function meter(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return '<div class="diag-meter"><span style="width:0%"></span></div>';
    const percent = Math.max(0, Math.min(100, number * 100));
    return `<div class="diag-meter"><span style="width:${percent.toFixed(1)}%"></span></div>`;
  }

  function rowForPath(root, path) {
    const node = [...root.querySelectorAll('.diag-param-code')].find(item => item.textContent.trim() === path);
    return node ? node.closest('tr') : null;
  }

  function replaceRow(root, path, html) {
    const row = rowForPath(root,path);
    if (!row || row.dataset.layoutV3Refined === '1') return false;
    row.dataset.layoutV3Refined = '1';
    row.classList.add('diag-special-row');
    const cells = row.children;
    if (cells.length < 2) return false;
    cells[1].innerHTML = html;
    if (cells.length >= 3) {
      cells[1].colSpan = 2;
      cells[2].remove();
    }
    return true;
  }

  function renderFormatCandidates(items) {
    if (!Array.isArray(items) || !items.length) return '<div class="diag-empty">Кандидаты формата отсутствуют</div>';
    return `<div class="diag-entity-grid compact">${items.map((item,index)=>`<article class="diag-entity-card ${index===0?'best':''}"><div class="diag-entity-head"><span class="diag-entity-rank">#${index+1}</span><span class="diag-entity-title">${esc(item?.format)}</span><span class="diag-entity-score">err=${esc(fmt(item?.ratio_error,6))}</span></div><dl class="diag-entity-meta"><dt>Размер ГОСТ</dt><dd>${esc(item?.width_mm)} × ${esc(item?.height_mm)} мм</dd><dt>Aspect ratio</dt><dd>${esc(fmt(item?.aspect_ratio,6))}</dd><dt>Ratio error</dt><dd>${esc(fmt(item?.ratio_error,6))}</dd></dl></article>`).join('')}</div>`;
  }

  function renderOrientationScores(items) {
    if (!Array.isArray(items) || !items.length) return '<div class="diag-empty">Orientation scores отсутствуют</div>';
    const sorted = [...items].sort((a,b)=>(Number(b?.score)||0)-(Number(a?.score)||0));
    return `<div class="diag-entity-grid compact">${sorted.map((item,index)=>`<article class="diag-entity-card ${index===0?'best':''}"><div class="diag-entity-head"><span class="diag-entity-title">${esc(item?.orientation_deg)}°</span><span class="diag-entity-score">score ${esc(fmt(item?.score,6))}</span></div><div class="diag-channel-list"><div class="diag-channel-row"><span class="diag-channel-name">Итоговый signal</span>${meter(item?.score)}<span class="diag-channel-value">${esc(fmt(item?.score,6))}</span></div></div></article>`).join('')}</div>`;
  }

  const evidenceChannels = [
    ['postage','Postage'], ['code_stamp','Code stamp'], ['barcode_layout','Barcode layout'],
    ['address_layout','Address layout'], ['text_direction','Text direction'],
    ['content_orientation','Content orientation'], ['base_score','Base score'], ['score','Итоговый score']
  ];

  function renderPairs(object) {
    if (!object || typeof object !== 'object') return '<span class="muted">—</span>';
    return Object.entries(object).map(([key,value])=>`<div class="diag-kv-line"><span>${esc(key.replaceAll('_',' '))}</span><span>${esc(Array.isArray(value)?value.join(', '):fmt(value,6))}</span></div>`).join('');
  }

  function renderOrientationEvidence(items) {
    if (!Array.isArray(items) || !items.length) return '<div class="diag-empty">Orientation evidence отсутствует</div>';
    const sorted = [...items].sort((a,b)=>(Number(b?.score)||0)-(Number(a?.score)||0));
    return `<div class="diag-entity-grid">${sorted.map((item,index)=>`<article class="diag-entity-card ${index===0?'best':''}"><div class="diag-entity-head"><span class="diag-entity-title">Гипотеза ${esc(item?.orientation_deg)}°</span><span class="diag-entity-score">${esc(fmt(item?.score,6))}</span></div><div class="diag-channel-list">${evidenceChannels.filter(([key])=>Object.prototype.hasOwnProperty.call(item||{},key)).map(([key,label])=>`<div class="diag-channel-row"><span class="diag-channel-name">${esc(label)}</span>${meter(item?.[key])}<span class="diag-channel-value">${esc(fmt(item?.[key],6))}</span></div>`).join('')}</div>${item?.contrast?`<details class="diag-mini-details"><summary>Contrast</summary><div class="diag-mini-details-body">${renderPairs(item.contrast)}</div></details>`:''}${item?.agreement?`<details class="diag-mini-details"><summary>Agreement</summary><div class="diag-mini-details-body">${renderPairs(item.agreement)}</div></details>`:''}</article>`).join('')}</div>`;
  }

  const hypothesisComponents = [
    ['aspect','Aspect'], ['postage','Postage'], ['code_stamp','Code stamp'], ['layout','Layout'], ['window','Window'],
    ['barcode_layout','Barcode'], ['address_layout','Address'], ['text_direction','Text'],
    ['content_orientation','Content'], ['orientation_signal','Orientation']
  ];

  function renderProfileHypotheses(items) {
    if (!Array.isArray(items) || !items.length) return '<div class="diag-empty">Profile hypotheses отсутствуют</div>';
    return `<div class="diag-entity-grid">${items.map((item,index)=>`<article class="diag-entity-card ${index===0?'best':''}"><div class="diag-entity-head"><span class="diag-entity-rank">#${index+1}</span><span class="diag-entity-title">${esc(item?.profile_id)}</span><span class="diag-entity-score">${esc(fmt(item?.score,6))}</span></div><dl class="diag-entity-meta"><dt>Формат</dt><dd>${esc(item?.format)}</dd><dt>Layout</dt><dd>${esc(item?.layout)}</dd><dt>Ориентация</dt><dd>${esc(item?.orientation_deg)}°</dd><dt>Window</dt><dd>${esc(item?.window)}</dd></dl><div class="diag-channel-list">${hypothesisComponents.filter(([key])=>Object.prototype.hasOwnProperty.call(item?.components||{},key)).map(([key,label])=>`<div class="diag-channel-row"><span class="diag-channel-name">${esc(label)}</span>${meter(item?.components?.[key])}<span class="diag-channel-value">${esc(fmt(item?.components?.[key],6))}</span></div>`).join('')}</div></article>`).join('')}</div>`;
  }

  function renderQuad(items, labels) {
    if (!Array.isArray(items) || !items.length) return '<div class="diag-empty">Quad отсутствует</div>';
    return `<div class="diag-point-grid">${items.map((point,index)=>`<div class="diag-point-card"><div class="diag-point-title">${esc(labels?.[index] || `P${index+1}`)}</div><div class="diag-point-value">x=${esc(fmt(point?.x,2))} · y=${esc(fmt(point?.y,2))}</div></div>`).join('')}</div>`;
  }

  function renderProfileSelected(item) {
    if (!item || typeof item !== 'object') return '<span class="muted">—</span>';
    const values = [
      ['ID',item.profile_id], ['Формат',item.format], ['Layout',item.layout], ['Window',item.window],
      ['Figure',item.figure], ['Размер',item.width_mm!=null&&item.height_mm!=null?`${item.width_mm} × ${item.height_mm} мм`:null]
    ].filter(([,value])=>value!==null&&value!==undefined&&value!=='');
    return `<div class="diag-chip-list" style="border-top:0;padding:0">${values.map(([label,value])=>`<span class="diag-data-chip"><strong>${esc(label)}:</strong> ${esc(value)}</span>`).join('')}</div>`;
  }

  function refineComplexSections(id) {
    const root = document.getElementById('debugDiagnosticBody');
    const item = state.results.get(id);
    const debug = item?.debug;
    if (!root || !debug || typeof debug !== 'object') return;

    replaceRow(root,'format_candidates',renderFormatCandidates(debug.format_candidates));
    replaceRow(root,'orientation.scores',renderOrientationScores(debug.orientation?.scores));
    replaceRow(root,'orientation.evidence',renderOrientationEvidence(debug.orientation?.evidence));
    replaceRow(root,'profile_scoring.selected',renderProfileSelected(debug.profile_scoring?.selected));
    replaceRow(root,'profile_scoring.top_hypotheses',renderProfileHypotheses(debug.profile_scoring?.top_hypotheses));
    replaceRow(root,'detector.quad',renderQuad(debug.detector?.quad,debug.detector?.quad_order));
  }

  const previousShowDebug = window.showDebug;
  if (typeof previousShowDebug === 'function') {
    window.showDebug = id => {
      previousShowDebug(id);
      refineComplexSections(id);
    };
  }
})();
</script>
"""


if 'data-toolocr-debug-diagnostics-layout="v3"' not in _test_ui._ROI_STATUS_STYLE:
    _test_ui._ROI_STATUS_STYLE += "\n" + _LAYOUT_V3_STYLE

if 'data-toolocr-debug-diagnostics-layout="v3"' not in _test_ui._PREVIEW_SCRIPT:
    _test_ui._PREVIEW_SCRIPT += "\n" + _LAYOUT_V3_SCRIPT

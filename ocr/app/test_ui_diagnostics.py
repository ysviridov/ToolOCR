from __future__ import annotations

from . import test_ui_library_preview as _test_ui


_DIAGNOSTIC_STYLE = r"""
<style data-toolocr-debug-diagnostics="v1">
  #debugDialog { width:min(1320px,96vw); }
  #debugJsonDialog { width:min(1320px,96vw); }
  .debug-modal-actions { display:flex; gap:7px; flex-wrap:wrap; align-items:center; }
  .diag-body { background:#f8f9fb; }
  .diag-summary-grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
    gap:9px;
    margin-bottom:14px;
  }
  .diag-summary-card {
    border:1px solid var(--line);
    border-radius:9px;
    padding:9px 11px;
    background:#fff;
    min-width:0;
  }
  .diag-summary-label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.025em; }
  .diag-summary-value { margin-top:3px; font-size:16px; font-weight:720; overflow-wrap:anywhere; }
  .diag-section {
    border:1px solid var(--line);
    border-radius:10px;
    background:#fff;
    margin:10px 0;
    overflow:hidden;
  }
  .diag-section > summary {
    list-style:none;
    cursor:pointer;
    padding:11px 13px;
    background:#f7f8fa;
    border-bottom:1px solid transparent;
    display:flex;
    gap:9px;
    align-items:center;
    font-weight:720;
  }
  .diag-section[open] > summary { border-bottom-color:var(--line); }
  .diag-section > summary::-webkit-details-marker { display:none; }
  .diag-section > summary::before { content:'▸'; color:var(--muted); width:12px; }
  .diag-section[open] > summary::before { content:'▾'; }
  .diag-section-note { margin-left:auto; color:var(--muted); font-size:12px; font-weight:500; }
  .diag-table-wrap { overflow:auto; }
  table.diag-table { width:100%; min-width:720px; table-layout:fixed; border-collapse:collapse; }
  .diag-table th { position:static; top:auto; }
  .diag-table th:nth-child(1) { width:31%; }
  .diag-table th:nth-child(2) { width:27%; }
  .diag-table th:nth-child(3) { width:42%; }
  .diag-table td { vertical-align:top; overflow-wrap:anywhere; word-break:break-word; }
  .diag-param { font-weight:650; }
  .diag-param-code { display:block; margin-top:2px; color:var(--muted); font:11px/1.35 ui-monospace,SFMono-Regular,Consolas,monospace; }
  .diag-info {
    display:inline-flex;
    width:17px;
    height:17px;
    margin-left:5px;
    align-items:center;
    justify-content:center;
    border:1px solid #aeb7c6;
    border-radius:50%;
    color:#586274;
    background:#fff;
    font-size:11px;
    font-weight:800;
    cursor:help;
    vertical-align:1px;
  }
  .diag-value { font-variant-numeric:tabular-nums; }
  .diag-value code { font:12px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace; }
  .diag-badge { display:inline-block; border-radius:999px; padding:2px 8px; background:#eef1f5; }
  .diag-badge.ok { color:var(--ok); background:#e7f4ea; }
  .diag-badge.warn { color:var(--warn); background:#fff5d6; }
  .diag-badge.err { color:var(--danger); background:#fdecea; }
  .diag-nested {
    border:1px solid #e1e5ec;
    border-radius:7px;
    background:#fafbfc;
    overflow:hidden;
  }
  .diag-nested > summary { cursor:pointer; padding:6px 8px; color:#475164; font-weight:650; }
  .diag-nested-body { border-top:1px solid #e1e5ec; padding:7px; background:#fff; }
  .diag-array-item { border:1px solid #e6e9ef; border-radius:7px; margin:7px 0; overflow:hidden; }
  .diag-array-item > summary { cursor:pointer; padding:6px 8px; background:#f7f8fa; font-weight:650; }
  .diag-empty { padding:20px; text-align:center; color:var(--muted); }
  .diag-source-note { color:var(--muted); font-size:12px; margin:0 0 10px; }
  #debugJsonPre { max-height:74vh; }
  @media (max-width:700px) {
    #debugDialog,#debugJsonDialog { width:98vw; }
    .debug-modal-actions { width:100%; }
    .debug-modal-actions button { flex:1; }
    table.diag-table { min-width:620px; }
  }
</style>
"""


_DIAGNOSTIC_SCRIPT = r"""
<script data-toolocr-debug-diagnostics="v1">
(() => {
  const baseDoc = 'docs/stage-2.2-test-ui-diagnostics.md';
  let currentDebugId = null;

  const sectionMeta = {
    input: ['Входное изображение', 'Имя, MIME type, размер и исходное разрешение входного файла.'],
    frame_normalization: ['Нормализация кадра', 'Выделение светлого письма на чёрном фоне и рабочий crop перед detector.'],
    detector: ['Контур письма', 'Результат external quad detector и качество найденного четырёхугольника.'],
    format_validation: ['Проверка формата', 'Согласованность expected format с metric/aspect/profile evidence для AUTO/SESSION/FIXED.'],
    metric_format: ['Метрическая оценка формата', 'Диагностика физического формата по camera calibration и pixel-scale.'],
    rectified: ['Perspective rectification', 'Размер изображения после выпрямления перспективы.'],
    orientation: ['Ориентация 0°/180°', 'Решение orientation scorer и свидетельства для канонической ориентации.'],
    format_candidates: ['Кандидаты формата', 'Форматы, совместимые с aspect ratio rectified изображения.'],
    profile_candidates: ['Кандидаты профилей', 'Набор ГОСТ-профилей, допущенных к profile scoring.'],
    profile_scoring: ['Profile scoring', 'Выбранный ГОСТ-профиль, confidence/margin и альтернативные гипотезы.'],
    timing: ['Время выполнения', 'Длительность отдельных этапов pipeline и полное время обработки.'],
    test_ui_postcode_roi: ['ROI почтового индекса', 'Диагностика postcode stencil detector в Test UI.'],
    test_ui_postcode_ocr: ['CNN почтового индекса', 'Результат актуального ONNX-primary postcode runtime, включая confidence и top-k цифр.'],
    test_ui_input_quality: ['Качество входа', 'Неблокирующая диагностика возможного физического crop/деформации входного кадра.'],
    test_ui_roi_meta: ['ROI metadata endpoint', 'Ссылка на полный актуальный /roi/meta payload для этого файла.'],
    debug_images: ['Debug images', 'Опциональные base64 debug-изображения. В диагностической таблице содержимое не разворачивается.'],
  };

  const labels = {
    stage:'Стадия', standard:'Стандарт', layout_status:'Layout status', format_mode:'Режим формата',
    expected_format:'Ожидаемый формат', format_status:'Статус формата', format:'Формат', profile_scope:'Область профилей',
    filename:'Имя файла', content_type:'MIME type', bytes_received:'Получено, байт', width_px:'Ширина, px', height_px:'Высота, px',
    status:'Статус', source:'Источник', crop:'Рабочий crop', foreground_bbox:'Foreground bbox', bottom_anchored:'Привязка к нижнему краю',
    x:'X', y:'Y', area_ratio:'Доля площади', method:'Метод', confidence:'Confidence', raw_confidence:'Raw confidence',
    frame_status:'Статус кадра', frame_contact_sides:'Контакт с краями кадра', rectangularity:'Прямоугольность', angle_score:'Оценка углов',
    quad_order:'Порядок вершин', quad:'Quad', validation:'Проверка', blocking:'Блокирующее решение', reasons:'Причины', warnings:'Предупреждения',
    metric_observed_format:'Формат по metric', aspect_error:'Ошибка aspect ratio', aspect_tolerance:'Допуск aspect ratio',
    profile_matches_expected:'Profile совпадает с expected', metric_matches_expected:'Metric совпадает с expected',
    calibration:'Калибровка', consensus:'Consensus', consistent:'Согласованность', measurement:'Измерение',
    width_mm:'Ширина, мм', height_mm:'Высота, мм', landscape:'Landscape', value_deg:'Угол, °', margin:'Margin', scores:'Scores', evidence:'Evidence',
    orientation_deg:'Гипотеза, °', score:'Score', profile_id:'Profile ID', selected:'Выбранный профиль', top_hypotheses:'Лучшие гипотезы',
    decode_ms:'Decode, ms', normalization_ms:'Normalization, ms', detect_ms:'Detector, ms', metric_ms:'Metric, ms', rectify_ms:'Rectify, ms',
    candidate_ms:'Candidates, ms', profile_scoring_ms:'Profile scoring, ms', profile_ms:'Profile total, ms', total_ms:'Всего, ms',
    confirmation_mode:'Режим подтверждения', rejection_reason:'Причина отклонения', color:'Цвет статуса', note:'Примечание',
    text:'Распознанный текст', postcode:'Почтовый индекс', min_digit_confidence:'Минимальный confidence цифры',
    geometric_mean_confidence:'Геометрическое среднее confidence', structurally_valid:'Структурно валиден', reason:'Причина', engine:'Движок',
    model_path:'Путь к модели', digits:'Цифры', index:'Позиция', digit:'Цифра', top3:'TOP-3', probability:'Probability', preprocess:'Preprocessing',
    roi_meta_url:'URL /roi/meta', url:'URL', metrics:'Метрики', input_quality_status:'Статус качества входа', input_quality_reasons:'Причины качества входа',
    input_quality_metrics:'Метрики качества входа', suppressed_components:'Удалено компонентов', restored_components:'Восстановлено компонентов',
    ink_pixels_before:'Ink до suppression', ink_pixels_after:'Ink после suppression', suppressed_ink_ratio:'Доля удалённого ink', retained_ink_ratio:'Доля сохранённого ink',
    otsu_threshold:'Порог Otsu', cell_width_px:'Ширина ячейки, px', cell_height_px:'Высота ячейки, px', glyph_bbox:'BBox цифры',
    canvas_width_px:'Ширина canvas, px', canvas_height_px:'Высота canvas, px', source_orientation_deg:'Исходная ориентация, °',
    rotation_applied_deg:'Применённый поворот, °', coordinate_space:'Система координат', search_bbox:'Область поиска', detected_bbox:'Найденный bbox',
    bbox:'BBox', component_count:'Компоненты', ink_density:'Плотность ink', detector:'Detector', features:'Признаки', digit_geometry:'Геометрия цифр', digit_cells:'Ячейки цифр', recognition:'Распознавание'
  };

  const keyDocs = {
    stage:['Версия диагностического этапа pipeline.','stage-2.1-test-ui.md'],
    standard:['Нормативный профиль/стандарт, относительно которого выполняется layout analysis.','stage-2.1-gost-layout.md'],
    layout_status:['Итог детекции контура: detected для полного кадра или partial_frame при контакте письма с границей.','stage-2.1-gost-layout.md'],
    format_mode:['AUTO определяет формат автоматически; SESSION проверяет заранее ожидаемый формат; FIXED жёстко принимает expected_format.','stage-2.1-format-modes.md'],
    expected_format:['Формат, заданный внешним контекстом для SESSION/FIXED. В AUTO не используется как hard constraint.','stage-2.1-format-modes.md'],
    format_status:['Итоговый статус решения по физическому формату.','stage-2.1-format-modes.md'],
    format:['Итоговый физический формат письма. Может быть null при неразрешённом или блокирующем mismatch.','stage-2.1-format-modes.md'],
    status:['Статус соответствующего диагностического этапа или объекта. Конкретные допустимые значения зависят от секции.','stage-2.2-test-ui-diagnostics.md'],
    confidence:['Нормированная уверенность соответствующего detector/classifier. Не является универсальной вероятностью безошибочного решения.','stage-2.2-test-ui-diagnostics.md'],
    margin:['Разница между лучшей и альтернативной гипотезой; используется как мера разделимости решения.','stage-2.1-profile-scoring.md'],
    area_ratio:['Доля площади, занимаемая найденной областью относительно указанной системы координат.','stage-2.1-frame-normalization.md'],
    bottom_anchored:['Признак того, что foreground письма заканчивается у нижней границы исходного кадра.','stage-2.1-frame-normalization.md'],
    frame_contact_sides:['Стороны кадра, с которыми соприкасается найденное письмо; используется для диагностики partial frame.','stage-2.1-gost-layout.md'],
    quad:['Четыре вершины контура письма, используемые для perspective rectification и метрической геометрии.','stage-2.1-gost-layout.md'],
    blocking:['Показывает, блокирует ли обнаруженное противоречие итоговое решение по формату.','stage-2.1-format-modes.md'],
    warnings:['Диагностические противоречия, которые сохранены, но не обязательно блокируют результат.','stage-2.1-format-modes.md'],
    reasons:['Причины текущего решения или отклонения, используемые для диагностики.','stage-2.1-format-modes.md'],
    value_deg:['Разрешённая ориентация письма: 0° или 180°. 90°/270° в production pipeline не применяются.','stage-2.1-orientation-content.md'],
    scores:['Оценки альтернативных orientation/profile гипотез.','stage-2.1-orientation-content.md'],
    evidence:['Набор независимых свидетельств, участвующих в решении ориентации.','stage-2.1-orientation-content.md'],
    profile_id:['Идентификатор выбранного ГОСТ layout-профиля.','stage-2.1-profile-scoring.md'],
    top_hypotheses:['Лучшие альтернативные profile hypotheses, отсортированные по score.','stage-2.1-profile-scoring.md'],
    total_ms:['Полное время обработки файла для данного pipeline.','stage-2.1-test-ui.md'],
    postcode:['Шестизначный индекс, собранный из шести независимо распознанных digit-cell.','stage-2.2-postcode-cnn-runtime.md'],
    min_digit_confidence:['Минимальная softmax confidence среди шести выбранных цифр. Пока не используется как hard gate.','stage-2.2-postcode-cnn-runtime.md'],
    geometric_mean_confidence:['Геометрическое среднее confidence шести цифр; дополнительная диагностическая метрика.','stage-2.2-postcode-cnn-runtime.md'],
    structurally_valid:['Проверка базовой структуры российского индекса: шесть цифр и первая цифра не равна нулю.','stage-2.2-postcode-cnn-runtime.md'],
    engine:['Фактически использованный recognizer: ONNX, Tesseract fallback либо смешанный вариант.','stage-2.2-postcode-cnn-runtime.md'],
    model_path:['Файл ONNX-модели, использованный runtime.','stage-2.2-postcode-cnn-runtime.md'],
    top3:['Три наиболее вероятных класса CNN для данной digit-cell с softmax probability.','stage-2.2-postcode-cnn-runtime.md'],
    probability:['Softmax output CNN для класса цифры. Не следует трактовать как гарантированную вероятность корректной сортировки.','stage-2.2-postcode-cnn-runtime.md'],
    preprocess:['Диагностика подготовки одной digit-cell: background correction, Otsu, stencil-dot suppression и формирование 96×128 canvas.','stage-2.2-postcode-cnn-runtime.md'],
    confirmation_mode:['Способ структурного подтверждения postcode stencil: strict marker либо rescue-режим.','stage-2.2-postcode-cnn-runtime.md'],
    rejection_reason:['Причина, по которой detector не подтвердил ROI/stencil candidate.','stage-2.2-test-ui-diagnostics.md'],
    roi_meta_url:['Живой endpoint с полным ROI/CNN metadata для файла.','stage-2.2-postcode-cnn-runtime.md'],
  };

  const pathDocs = {
    'frame_normalization.status':['cropped — чёрный фон существенно удалён; unchanged — crop почти равен исходнику; foreground_not_found — fallback на исходный кадр.','stage-2.1-frame-normalization.md'],
    'orientation.status':['resolved означает уверенное решение 0°/180°; ambiguous — evidence недостаточно.','stage-2.1-orientation-content.md'],
    'orientation.confidence':['Уверенность orientation decision, рассчитанная из доступных каналов evidence.','stage-2.1-orientation-content.md'],
    'format_validation.status':['SESSION: confirmed/plausible/mismatch; FIXED: fixed. Mismatch блокирует SESSION format.','stage-2.1-format-modes.md'],
    'test_ui_postcode_ocr.confidence':['Средняя softmax confidence шести выбранных цифр CNN. Пока диагностическая, не hard gate.','stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.status':['recognized — все шесть цифр получены; incomplete/error/unavailable — распознавание не завершено.','stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.digits[].confidence':['Softmax confidence выбранного класса одной digit-cell.','stage-2.2-postcode-cnn-runtime.md'],
  };

  const generalKeys = new Set(['stage','standard','layout_status','format_mode','expected_format','format_status','format','profile_scope']);
  const sectionOrder = [
    'input','frame_normalization','detector','format_validation','metric_format','rectified','orientation',
    'format_candidates','profile_candidates','profile_scoring','test_ui_postcode_roi','test_ui_postcode_ocr',
    'test_ui_input_quality','test_ui_roi_meta','timing','debug_images'
  ];

  function normalizePath(path) {
    return String(path || '').replace(/\[\d+\]/g, '[]');
  }

  function docFor(path, key) {
    const normalized = normalizePath(path);
    const entry = pathDocs[normalized] || keyDocs[key];
    if (entry) {
      return `${entry[0]}\nДокументация: docs/${entry[1]}`;
    }
    return `Параметр ${path || key} из диагностического payload ToolOCR. Значение показывается без изменения семантики backend.\nДокументация: ${baseDoc}`;
  }

  function labelFor(key) {
    return labels[key] || String(key || '').replaceAll('_',' ');
  }

  function escapeAttr(value) {
    return escapeHtml(value).replace(/\n/g, '&#10;');
  }

  function statusKind(value) {
    const text = String(value ?? '').toLowerCase();
    if (!text) return '';
    if (/(error|reject|mismatch|invalid|not_found|stencil_not_found|unavailable|failed|failure)/.test(text)) return 'err';
    if (/(partial|plausible|ambiguous|rescue|warning|warn|incomplete|suspected|not_evaluated)/.test(text)) return 'warn';
    if (/(ok|ready|recognized|resolved|detected|confirmed|cropped|canonical|strict|applied|valid)/.test(text)) return 'ok';
    return '';
  }

  function formatPrimitive(value) {
    if (value === null || value === undefined) return '<span class="muted">—</span>';
    if (typeof value === 'boolean') return `<span class="diag-badge ${value ? 'ok' : ''}">${value ? 'да' : 'нет'}</span>`;
    if (typeof value === 'number') return `<code>${escapeHtml(Number.isInteger(value) ? String(value) : String(value))}</code>`;
    const text = String(value);
    const kind = statusKind(text);
    if (kind) return `<span class="diag-badge ${kind}">${escapeHtml(text)}</span>`;
    return `<span class="diag-value">${escapeHtml(text)}</span>`;
  }

  function complexSummary(value) {
    if (Array.isArray(value)) return `${value.length} элемент(ов)`;
    if (value && typeof value === 'object') return `${Object.keys(value).length} полей`;
    return 'значение';
  }

  function compactArray(value) {
    if (!Array.isArray(value)) return null;
    if (value.every(item => item === null || ['string','number','boolean'].includes(typeof item))) {
      return value.length ? value.map(item => String(item)).join(', ') : '—';
    }
    return null;
  }

  function renderNestedObject(value, path) {
    if (Array.isArray(value)) {
      const compact = compactArray(value);
      if (compact !== null) return `<span class="diag-value">${escapeHtml(compact)}</span>`;
      return `<details class="diag-nested"><summary>${escapeHtml(complexSummary(value))}</summary><div class="diag-nested-body">${value.map((item,index)=>{
        if (item && typeof item === 'object') {
          return `<details class="diag-array-item"><summary>Элемент ${index + 1}</summary><div class="diag-nested-body"><table class="diag-table"><tbody>${renderRows(item, `${path}[${index}]`)}</tbody></table></div></details>`;
        }
        return `<div>${formatPrimitive(item)}</div>`;
      }).join('')}</div></details>`;
    }
    if (value && typeof value === 'object') {
      return `<details class="diag-nested"><summary>${escapeHtml(complexSummary(value))}</summary><div class="diag-nested-body"><table class="diag-table"><tbody>${renderRows(value,path)}</tbody></table></div></details>`;
    }
    return formatPrimitive(value);
  }

  function renderRows(object, basePath='') {
    if (!object || typeof object !== 'object') return '';
    return Object.entries(object).map(([key,value]) => {
      const path = basePath ? `${basePath}.${key}` : key;
      const description = docFor(path,key);
      let rendered;
      if (typeof value === 'string' && key.endsWith('_jpeg_base64')) {
        rendered = `<span class="muted">binary/base64 скрыт · ${value.length.toLocaleString('ru-RU')} символов</span>`;
      } else if (value && typeof value === 'object') {
        rendered = renderNestedObject(value,path);
      } else {
        rendered = formatPrimitive(value);
      }
      return `<tr><td><span class="diag-param" title="${escapeAttr(description)}">${escapeHtml(labelFor(key))}<span class="diag-info" title="${escapeAttr(description)}">i</span></span><span class="diag-param-code">${escapeHtml(path)}</span></td><td>${rendered}</td><td class="muted">${escapeHtml(description.split('\n')[0])}</td></tr>`;
    }).join('');
  }

  function renderSection(key, value, open=false) {
    const meta = sectionMeta[key] || [labelFor(key), docFor(key,key).split('\n')[0]];
    const bodyObject = value && typeof value === 'object' && !Array.isArray(value) ? value : {[key]: value};
    return `<details class="diag-section" ${open ? 'open' : ''}><summary title="${escapeAttr(meta[1])}"><span>${escapeHtml(meta[0])}</span><span class="diag-section-note">${escapeHtml(meta[1])}</span></summary><div class="diag-table-wrap"><table class="diag-table"><thead><tr><th>Параметр</th><th>Значение</th><th>Описание</th></tr></thead><tbody>${renderRows(bodyObject, Array.isArray(value) ? '' : key)}</tbody></table></div></details>`;
  }

  function summaryCard(label,value,kind='') {
    return `<div class="diag-summary-card"><div class="diag-summary-label">${escapeHtml(label)}</div><div class="diag-summary-value ${kind}">${formatPrimitive(value)}</div></div>`;
  }

  function renderDiagnostics(item) {
    const debug = item && item.debug && typeof item.debug === 'object' ? item.debug : {};
    const postcode = debug.test_ui_postcode_ocr?.postcode ?? debug.test_ui_postcode_ocr?.text ?? null;
    const engine = debug.test_ui_postcode_ocr?.engine ?? item.postcode_ocr_engine ?? null;
    const general = {};
    generalKeys.forEach(key => { if (Object.prototype.hasOwnProperty.call(debug,key)) general[key]=debug[key]; });

    const cards = [
      summaryCard('Layout', item.layout_status ?? debug.layout_status ?? '—'),
      summaryCard('Формат', item.format ?? debug.format ?? '—'),
      summaryCard('Ориентация', item.orientation_status === 'resolved' ? `${item.orientation_deg}°` : (item.orientation_status ?? debug.orientation?.status ?? '—')),
      summaryCard('Normalization', item.normalization_status ?? debug.frame_normalization?.status ?? '—'),
      summaryCard('POSTCODE', postcode ?? '—'),
      summaryCard('OCR engine', engine ?? '—'),
      summaryCard('Время', typeof item.total_ms === 'number' ? `${item.total_ms} ms` : (debug.timing?.total_ms ?? '—')),
    ].join('');

    const sections = [];
    if (Object.keys(general).length) sections.push(renderSection('Общее', general, true));
    const consumed = new Set([...generalKeys]);
    sectionOrder.forEach((key,index) => {
      if (!Object.prototype.hasOwnProperty.call(debug,key)) return;
      consumed.add(key);
      const open = ['frame_normalization','detector','format_validation','orientation','test_ui_postcode_roi','test_ui_postcode_ocr'].includes(key);
      sections.push(renderSection(key,debug[key],open));
    });
    const extras = Object.fromEntries(Object.entries(debug).filter(([key]) => !consumed.has(key)));
    if (Object.keys(extras).length) sections.push(renderSection('Дополнительные параметры',extras,false));

    return `<div class="diag-source-note">Таблица формируется из исходного debug payload без изменения backend-значений. Наведите курсор на <strong>i</strong> возле параметра для подсказки. Полный JSON доступен отдельными кнопками.</div><div class="diag-summary-grid">${cards}</div>${sections.join('') || '<div class="diag-empty">Debug payload пуст</div>'}`;
  }

  function singleDebugPayload(item) {
    return {
      schema:'toolocr.test-ui.debug-item.v1',
      exported_at:new Date().toISOString(),
      test_started_at:state.lastRunAt,
      result:{
        id:item.id,
        filename:item.name,
        folder_id:item.folder_id ?? null,
        folder_name:item.folder_name ?? null,
        format_mode:item.format_mode ?? null,
        expected_format:item.expected_format ?? null,
        format_validation_status:item.format_validation_status ?? null,
        ok:item.ok,
        layout_status:item.layout_status,
        format:item.format,
        format_status:item.format_status,
        orientation_status:item.orientation_status,
        orientation_deg:item.orientation_deg,
        normalization_status:item.normalization_status,
        total_ms:item.total_ms,
        error:item.error ?? null,
        debug:item.debug,
      }
    };
  }

  function currentItem() {
    return currentDebugId ? state.results.get(currentDebugId) : null;
  }

  function safeFilename(value) {
    return String(value || 'image').replace(/[^A-Za-z0-9А-Яа-яЁё._-]+/g,'_').slice(0,120);
  }

  function showCurrentJson() {
    const item = currentItem();
    if (!item) return;
    $('debugJsonTitle').textContent = `Debug JSON — ${item.name}`;
    $('debugJsonPre').innerHTML = highlightedJson(singleDebugPayload(item));
    $('debugJsonDialog').showModal();
  }

  function downloadCurrentJson() {
    const item = currentItem();
    if (!item) return;
    const payload = singleDebugPayload(item);
    const blob = new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json;charset=utf-8'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `toolocr-debug-${safeFilename(item.name)}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }

  function install() {
    const dialog = $('debugDialog');
    if (!dialog || dialog.dataset.diagnosticsInstalled === '1') return;
    dialog.dataset.diagnosticsInstalled = '1';
    dialog.innerHTML = `<div class="modal-head"><strong id="debugTitle">Диагностика</strong><div class="debug-modal-actions"><button id="debugShowJson">Показать JSON</button><button id="debugDownloadJson">Скачать JSON</button><button id="debugClose">Закрыть</button></div></div><div id="debugDiagnosticBody" class="modal-body diag-body"></div>`;

    let jsonDialog = $('debugJsonDialog');
    if (!jsonDialog) {
      jsonDialog = document.createElement('dialog');
      jsonDialog.id = 'debugJsonDialog';
      jsonDialog.innerHTML = `<div class="modal-head"><strong id="debugJsonTitle">Debug JSON</strong><div class="debug-modal-actions"><button id="debugJsonDownload">Скачать JSON</button><button id="debugJsonClose">Закрыть</button></div></div><div class="modal-body"><pre id="debugJsonPre"></pre></div>`;
      document.body.appendChild(jsonDialog);
    }

    $('debugClose').onclick = () => dialog.close();
    $('debugShowJson').onclick = showCurrentJson;
    $('debugDownloadJson').onclick = downloadCurrentJson;
    $('debugJsonClose').onclick = () => jsonDialog.close();
    $('debugJsonDownload').onclick = downloadCurrentJson;

    window.showDebug = id => {
      const item = state.results.get(id);
      if (!item) return;
      currentDebugId = id;
      $('debugTitle').textContent = `Диагностика — ${item.name}`;
      $('debugDiagnosticBody').innerHTML = renderDiagnostics(item);
      dialog.showModal();
    };
  }

  install();
})();
</script>
"""


# test_ui_page_with_library_preview читает эти строки при каждом запросе.
# Диагностический слой добавляется только в UI и не меняет API/debug payload.
if 'data-toolocr-debug-diagnostics="v1"' not in _test_ui._ROI_STATUS_STYLE:
    _test_ui._ROI_STATUS_STYLE += "\n" + _DIAGNOSTIC_STYLE

if 'data-toolocr-debug-diagnostics="v1"' not in _test_ui._PREVIEW_SCRIPT:
    _test_ui._PREVIEW_SCRIPT += "\n" + _DIAGNOSTIC_SCRIPT

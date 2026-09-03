from __future__ import annotations

from . import test_ui_library_preview as _test_ui


_REFINEMENT_STYLE = r"""
<style data-toolocr-debug-diagnostics-refinements="v2">
  #debugDialog { width:min(1540px,98vw); }

  /* Верхний уровень остаётся предсказуемой таблицей, но значение получает
     больше места. Вложенные таблицы больше не наследуют жёсткие 31/27/42%. */
  .diag-section > .diag-table-wrap > table.diag-table {
    min-width:980px;
    table-layout:fixed;
  }
  .diag-section > .diag-table-wrap > table.diag-table > thead > tr > th:nth-child(1),
  .diag-section > .diag-table-wrap > table.diag-table > tbody > tr > td:nth-child(1) { width:26%; }
  .diag-section > .diag-table-wrap > table.diag-table > thead > tr > th:nth-child(2),
  .diag-section > .diag-table-wrap > table.diag-table > tbody > tr > td:nth-child(2) { width:34%; }
  .diag-section > .diag-table-wrap > table.diag-table > thead > tr > th:nth-child(3),
  .diag-section > .diag-table-wrap > table.diag-table > tbody > tr > td:nth-child(3) { width:40%; }

  .diag-nested { max-width:100%; overflow:auto; }
  .diag-nested-body { overflow:auto; }
  .diag-nested-body table.diag-table {
    width:100%;
    min-width:640px;
    table-layout:auto;
  }
  .diag-nested-body table.diag-table th:nth-child(n),
  .diag-nested-body table.diag-table td:nth-child(n) { width:auto; }
  .diag-nested-body table.diag-table td:first-child { min-width:190px; }
  .diag-nested-body table.diag-table td:nth-child(2) { min-width:220px; }
  .diag-nested-body table.diag-table td:nth-child(3) { min-width:280px; }

  /* CNN digits имеют собственное плоское представление без каскада
     digits -> item -> top3 -> item. */
  .diag-digits-row > td:first-child { width:24% !important; }
  .diag-digit-grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(255px,1fr));
    gap:10px;
    width:100%;
    min-width:0;
  }
  .diag-digit-card {
    border:1px solid #dfe4ec;
    border-radius:9px;
    background:#fff;
    overflow:hidden;
    min-width:0;
  }
  .diag-digit-head {
    display:flex;
    align-items:center;
    gap:8px;
    padding:8px 10px;
    background:#f7f8fa;
    border-bottom:1px solid #e5e9f0;
  }
  .diag-digit-title { font-weight:760; font-size:14px; }
  .diag-digit-head .diag-badge { margin-left:auto; }
  .diag-digit-meta {
    display:grid;
    grid-template-columns:auto 1fr;
    gap:5px 9px;
    padding:8px 10px;
    font-size:12px;
  }
  .diag-digit-meta dt { color:var(--muted); }
  .diag-digit-meta dd { margin:0; min-width:0; overflow-wrap:anywhere; }
  .diag-top3-mini {
    width:100%;
    min-width:0;
    border-collapse:collapse;
    font-size:12px;
  }
  .diag-top3-mini th,
  .diag-top3-mini td { padding:5px 8px; border-top:1px solid #edf0f4; }
  .diag-top3-mini th { position:static; background:#fbfcfd; color:var(--muted); text-transform:none; letter-spacing:0; }
  .diag-top3-mini td:nth-child(1), .diag-top3-mini th:nth-child(1) { width:42px; text-align:center; }
  .diag-top3-mini td:nth-child(2), .diag-top3-mini th:nth-child(2) { width:54px; text-align:center; }
  .diag-top3-mini td:nth-child(3), .diag-top3-mini th:nth-child(3) { text-align:right; font-variant-numeric:tabular-nums; }
  .diag-digit-preprocess { border-top:1px solid #edf0f4; }
  .diag-digit-preprocess > summary { cursor:pointer; padding:6px 10px; color:#536074; font-size:12px; }
  .diag-digit-preprocess-body { padding:0 10px 8px; }
  .diag-digit-preprocess-line { display:flex; justify-content:space-between; gap:12px; padding:2px 0; font-size:11px; }
  .diag-digit-preprocess-line span:first-child { color:var(--muted); }

  .diag-description-doc { display:block; margin-top:3px; color:#7b8492; font-size:11px; }

  @media (max-width:800px) {
    #debugDialog { width:99vw; }
    .diag-section > .diag-table-wrap > table.diag-table { min-width:860px; }
    .diag-digit-grid { grid-template-columns:1fr; }
  }
</style>
"""


_REFINEMENT_SCRIPT = r"""
<script data-toolocr-debug-diagnostics-refinements="v2">
(() => {
  const richPathDocs = {
    'test_ui_postcode_ocr.status': ['Состояние распознавания индекса: recognized — получены все 6 цифр; incomplete — часть цифр не распознана; unavailable — pipeline не дошёл до CNN; error — ошибка recognizer.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.text': ['Сырая шестисимвольная строка результата. Неизвестная позиция обозначается знаком «?».', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.postcode': ['Итоговый шестизначный индекс. Заполняется только когда распознаны все шесть позиций.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.confidence': ['Среднее softmax-confidence шести выбранных CNN-классов. Это диагностическая уверенность модели, а не вероятность правильной сортировки.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.min_digit_confidence': ['Минимальный confidence среди шести цифр. Помогает быстро увидеть самую сомнительную позицию индекса.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.geometric_mean_confidence': ['Геометрическое среднее confidence шести цифр. Сильнее реагирует на одну слабую позицию, чем обычное среднее.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.structurally_valid': ['Базовая структурная проверка российского индекса: ровно 6 цифр, первая цифра не равна нулю.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.reason': ['Причина, по которой индекс не был полностью распознан. При успешном status=recognized обычно равна null.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.engine': ['Фактически использованный движок распознавания: ONNX, Tesseract fallback либо смешанный режим.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.model_path': ['Путь к ONNX-модели, которую runtime использовал для классификации цифр.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.digits': ['Шесть независимых digit-cell почтового индекса. Для каждой позиции показаны выбранная цифра, confidence, TOP-3 и preprocessing.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.digits[].index': ['Номер позиции цифры в индексе: от 1 до 6 слева направо.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.digits[].status': ['Статус распознавания конкретной digit-cell.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.digits[].digit': ['Цифра 0–9, выбранная CNN как наиболее вероятный класс для этой позиции.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.digits[].confidence': ['Softmax-confidence выбранного класса этой digit-cell. Высокое значение не гарантирует отсутствие ошибки.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.digits[].top3': ['Три наиболее вероятных класса CNN. Полезно для анализа похожих начертаний и будущей адресной проверки кандидатов.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.digits[].engine': ['Recognizer, который фактически обработал эту digit-cell. В штатном CNN runtime ожидается onnx.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.digits[].reason': ['Причина ошибки/неполного результата конкретной digit-cell. При успешном распознавании обычно null.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_ocr.digits[].preprocess': ['Диагностика подготовки digit-cell перед CNN: выравнивание фона, Otsu, suppression точек stencil и формирование canvas 96×128.', 'stage-2.2-postcode-cnn-runtime.md'],
    'test_ui_postcode_roi.status': ['Итог поиска области почтового индекса и stencil. detected означает подтверждённую структуру; stencil_not_found — шаблон не найден.', 'stage-2.2-test-ui-diagnostics.md'],
    'test_ui_postcode_roi.confirmation_mode': ['Способ подтверждения stencil-кандидата: строгий structural match либо rescue-ветка.', 'stage-2.2-test-ui-diagnostics.md'],
    'test_ui_postcode_roi.rejection_reason': ['Причина, по которой лучший stencil-кандидат был отклонён.', 'stage-2.2-test-ui-diagnostics.md'],
    'frame_normalization.status': ['Результат удаления чёрного поля: cropped — фон существенно обрезан; unchanged — crop почти равен исходнику; foreground_not_found — письмо не выделено и использован безопасный fallback.', 'stage-2.1-frame-normalization.md'],
    'frame_normalization.bottom_anchored': ['Показывает, касается ли светлая компонента письма нижней части кадра. Это диагностический признак механики подачи, а не определитель формата.', 'stage-2.1-frame-normalization.md'],
    'detector.confidence': ['Итоговая уверенность external quad detector в найденном контуре письма.', 'stage-2.1-gost-layout.md'],
    'detector.raw_confidence': ['Сырая оценка detector до дополнительных корректировок/ограничений качества.', 'stage-2.1-gost-layout.md'],
    'detector.rectangularity': ['Насколько найденный контур близок к прямоугольнику. Используется как один из признаков качества quad.', 'stage-2.1-gost-layout.md'],
    'detector.angle_score': ['Оценка близости углов найденного quad к геометрии прямоугольного отправления.', 'stage-2.1-gost-layout.md'],
    'format_validation.status': ['Результат проверки формата: SESSION даёт confirmed/plausible/mismatch; FIXED — fixed.', 'stage-2.1-format-modes.md'],
    'format_validation.blocking': ['Если true, обнаруженное противоречие блокирует использование выбранного формата downstream.', 'stage-2.1-format-modes.md'],
    'orientation.status': ['resolved — orientation scorer уверенно выбрал 0° или 180°; ambiguous — свидетельств недостаточно.', 'stage-2.1-orientation-content.md'],
    'orientation.value_deg': ['Выбранная ориентация лицевой стороны письма: только 0° или 180°. Повороты 90°/270° production pipeline не использует.', 'stage-2.1-orientation-content.md'],
    'orientation.confidence': ['Уверенность решения orientation scorer с учётом доступных каналов evidence.', 'stage-2.1-orientation-content.md'],
    'orientation.margin': ['Разница между лучшей и второй orientation-гипотезой. Чем больше margin, тем лучше разделены 0° и 180°.', 'stage-2.1-orientation-content.md'],
    'profile_scoring.profile_id': ['Идентификатор ГОСТ layout-профиля с лучшим score.', 'stage-2.1-profile-scoring.md'],
    'profile_scoring.confidence': ['Нормированная уверенность выбора layout-профиля.', 'stage-2.1-profile-scoring.md'],
    'profile_scoring.margin': ['Разница между лучшей и следующей profile-гипотезой.', 'stage-2.1-profile-scoring.md'],
    'timing.total_ms': ['Полное время обработки данного изображения всеми этапами текущего pipeline.', 'stage-2.1-test-ui.md'],
  };

  const richKeyDocs = {
    filename:['Исходное имя тестового файла.','stage-2.1-test-ui.md'],
    content_type:['MIME type загруженного изображения.','stage-2.1-test-ui.md'],
    bytes_received:['Размер входного файла, полученный backend, в байтах.','stage-2.1-test-ui.md'],
    width_px:['Ширина соответствующего изображения/области в пикселях.','stage-2.1-gost-layout.md'],
    height_px:['Высота соответствующего изображения/области в пикселях.','stage-2.1-gost-layout.md'],
    status:['Состояние конкретного этапа. Точное значение читается в контексте раздела, в котором находится параметр.','stage-2.2-test-ui-diagnostics.md'],
    source:['Источник данных или геометрии, из которого получено текущее диагностическое значение.','stage-2.2-test-ui-diagnostics.md'],
    method:['Алгоритм или метод, использованный на этом этапе обработки.','stage-2.2-test-ui-diagnostics.md'],
    reason:['Причина текущего статуса, отклонения или невозможности продолжить этап. При успешном результате может быть null.','stage-2.2-test-ui-diagnostics.md'],
    confidence:['Нормированная уверенность текущего detector/classifier. Смысл и калибровка зависят от раздела.','stage-2.2-test-ui-diagnostics.md'],
    margin:['Разница между лучшей и альтернативной гипотезой; характеризует разделимость решения.','stage-2.1-profile-scoring.md'],
    score:['Числовая оценка гипотезы/признака: большее значение означает более сильное соответствие модели этого этапа.','stage-2.2-test-ui-diagnostics.md'],
    format:['Физический формат отправления, выбранный или проверяемый на данном этапе.','stage-2.1-format-modes.md'],
    expected_format:['Формат, заранее заданный оператором/сессией для режимов SESSION или FIXED.','stage-2.1-format-modes.md'],
    warnings:['Неблокирующие диагностические противоречия, которые важно видеть при анализе результата.','stage-2.1-format-modes.md'],
    reasons:['Список причин, объясняющих текущее решение или отклонение.','stage-2.1-format-modes.md'],
    bbox:['Прямоугольная область в указанной системе координат: x, y, width и height.','stage-2.2-test-ui-diagnostics.md'],
    search_bbox:['Область изображения, внутри которой выполнялся поиск соответствующего ROI/признака.','stage-2.2-test-ui-diagnostics.md'],
    detected_bbox:['Фактически найденная область объекта после работы detector.','stage-2.2-test-ui-diagnostics.md'],
    component_count:['Количество connected-components/элементов foreground, использованных или найденных на этом этапе.','stage-2.2-test-ui-diagnostics.md'],
    ink_density:['Доля тёмных foreground-пикселей внутри анализируемой области.','stage-2.2-test-ui-diagnostics.md'],
    engine:['Движок/реализация, фактически выполнившая соответствующую операцию.','stage-2.2-postcode-cnn-runtime.md'],
    model_path:['Путь к модели, используемой recognizer во время этого запуска.','stage-2.2-postcode-cnn-runtime.md'],
    preprocess:['Диагностика преобразований, применённых к данным перед распознаванием.','stage-2.2-postcode-cnn-runtime.md'],
    top3:['Три лучших класса модели с их softmax-confidence.','stage-2.2-postcode-cnn-runtime.md'],
    probability:['Softmax-confidence конкретного класса CNN. Это не гарантированная вероятность правильной сортировки.','stage-2.2-postcode-cnn-runtime.md'],
    otsu_threshold:['Порог яркости, автоматически выбранный методом Otsu для бинаризации digit-cell.','stage-2.2-postcode-cnn-runtime.md'],
    suppressed_components:['Количество малых компонент stencil/dot pattern, удалённых preprocessing.','stage-2.2-postcode-cnn-runtime.md'],
    restored_components:['Количество малых компонент, возвращённых как вероятная часть рукописного штриха.','stage-2.2-postcode-cnn-runtime.md'],
    retained_ink_ratio:['Доля исходного тёмного foreground, оставшаяся после stencil-dot suppression.','stage-2.2-postcode-cnn-runtime.md'],
    suppressed_ink_ratio:['Доля тёмного foreground, удалённая stencil-dot suppression.','stage-2.2-postcode-cnn-runtime.md'],
  };

  const sectionNames = {
    input:'входного изображения', frame_normalization:'нормализации кадра', detector:'детектора контура',
    format_validation:'проверки формата', metric_format:'метрической оценки формата', rectified:'выпрямления перспективы',
    orientation:'определения ориентации', format_candidates:'кандидатов формата', profile_candidates:'кандидатов профиля',
    profile_scoring:'profile scoring', test_ui_postcode_roi:'ROI почтового индекса', test_ui_postcode_ocr:'CNN почтового индекса',
    test_ui_input_quality:'качества входного изображения', test_ui_roi_meta:'ROI metadata', timing:'измерения времени', debug_images:'debug-изображений'
  };

  function normalizePath(path) {
    return String(path || '').replace(/\[\d+\]/g,'[]');
  }

  function keyFromPath(path) {
    const clean = String(path || '').replace(/\[\d+\]/g,'');
    return clean.split('.').pop() || clean;
  }

  function humanFallbackDescription(path,key) {
    const normalized = normalizePath(path);
    const section = normalized.split('.')[0] || '';
    const sectionName = sectionNames[section] || 'диагностического этапа';
    const readable = String(key || '').replaceAll('_',' ');

    if (key.endsWith('_ms')) return [`Время выполнения операции «${readable.replace(/ ms$/,'')}» в миллисекундах.`, 'stage-2.1-test-ui.md'];
    if (key.endsWith('_px')) return [`Размер или координата «${readable.replace(/ px$/,'')}» в пикселях в системе координат этого раздела.`, 'stage-2.2-test-ui-diagnostics.md'];
    if (key.endsWith('_ratio')) return [`Нормированное отношение «${readable.replace(/ ratio$/,'')}», используемое для оценки ${sectionName}.`, 'stage-2.2-test-ui-diagnostics.md'];
    if (key.endsWith('_count')) return [`Количество элементов «${readable.replace(/ count$/,'')}», найденных на этапе ${sectionName}.`, 'stage-2.2-test-ui-diagnostics.md'];
    if (key.endsWith('_status')) return [`Статус «${readable.replace(/ status$/,'')}» на этапе ${sectionName}; показывает, завершён ли соответствующий шаг и каким результатом.`, 'stage-2.2-test-ui-diagnostics.md'];
    if (key.endsWith('_score')) return [`Числовая оценка «${readable.replace(/ score$/,'')}» на этапе ${sectionName}; используется для сравнения альтернативных гипотез.`, 'stage-2.2-test-ui-diagnostics.md'];
    if (key.endsWith('_confidence')) return [`Уверенность «${readable.replace(/ confidence$/,'')}» на этапе ${sectionName}. Значение диагностическое и не является универсальной вероятностью ошибки.`, 'stage-2.2-test-ui-diagnostics.md'];
    if (key.endsWith('_error')) return [`Величина ошибки «${readable.replace(/ error$/,'')}» на этапе ${sectionName}; чем меньше значение, тем ближе наблюдение к ожидаемой геометрии/модели.`, 'stage-2.2-test-ui-diagnostics.md'];
    if (key.endsWith('_bbox')) return [`Прямоугольная область «${readable.replace(/ bbox$/,'')}» в координатах изображения этого этапа.`, 'stage-2.2-test-ui-diagnostics.md'];

    return [`Диагностический признак «${readable}» раздела ${sectionName}. Используется для понимания решения этого этапа и поиска причины отклонений.`, 'stage-2.2-test-ui-diagnostics.md'];
  }

  function descriptionFor(path) {
    const normalized = normalizePath(path);
    const key = keyFromPath(path);
    return richPathDocs[normalized] || richKeyDocs[key] || humanFallbackDescription(normalized,key);
  }

  function tooltipText(entry) {
    return `${entry[0]}\nДокументация: docs/${entry[1]}`;
  }

  function refineDescriptions(root) {
    root.querySelectorAll('tr').forEach(row => {
      const code = row.querySelector(':scope > td:first-child .diag-param-code');
      const param = row.querySelector(':scope > td:first-child .diag-param');
      const info = row.querySelector(':scope > td:first-child .diag-info');
      if (!code || !param) return;
      const entry = descriptionFor(code.textContent.trim());
      const tooltip = tooltipText(entry);
      param.title = tooltip;
      if (info) info.title = tooltip;
      const cells = row.children;
      if (cells.length >= 3 && cells[2].classList.contains('muted')) {
        cells[2].textContent = entry[0];
        const doc = document.createElement('span');
        doc.className = 'diag-description-doc';
        doc.textContent = `docs/${entry[1]}`;
        cells[2].appendChild(doc);
      }
    });
  }

  function esc(value) {
    return escapeHtml(String(value ?? '—'));
  }

  function statusClass(value) {
    const text = String(value || '').toLowerCase();
    if (/(error|unavailable|failed|unrecognized)/.test(text)) return 'err';
    if (/(incomplete|fallback)/.test(text)) return 'warn';
    if (/(recognized|ready|ok)/.test(text)) return 'ok';
    return '';
  }

  function fmtProbability(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
    if (value >= 0.9995) return value.toFixed(6);
    if (value >= 0.01) return value.toFixed(6).replace(/0+$/,'').replace(/\.$/,'');
    return value.toExponential(3);
  }

  function renderTop3(items) {
    if (!Array.isArray(items) || !items.length) return '<div class="muted" style="padding:7px 10px">TOP-3 недоступен</div>';
    return `<table class="diag-top3-mini"><thead><tr><th>#</th><th>Цифра</th><th>Confidence</th></tr></thead><tbody>${items.map((item,index)=>`<tr><td>${index+1}</td><td><strong>${esc(item?.digit)}</strong></td><td>${esc(fmtProbability(item?.probability))}</td></tr>`).join('')}</tbody></table>`;
  }

  function renderPreprocess(preprocess) {
    if (!preprocess || typeof preprocess !== 'object') return '';
    const fields = [
      ['method','Метод'], ['status','Статус'], ['otsu_threshold','Otsu'],
      ['suppressed_components','Удалено точек'], ['restored_components','Восстановлено'],
      ['retained_ink_ratio','Сохранено ink'], ['suppressed_ink_ratio','Удалено ink'],
      ['canvas_width_px','Canvas W'], ['canvas_height_px','Canvas H']
    ].filter(([key]) => Object.prototype.hasOwnProperty.call(preprocess,key));
    if (!fields.length) return '';
    return `<details class="diag-digit-preprocess"><summary>Preprocessing · ${fields.length} параметров</summary><div class="diag-digit-preprocess-body">${fields.map(([key,label])=>`<div class="diag-digit-preprocess-line"><span>${esc(label)}</span><span>${esc(preprocess[key])}</span></div>`).join('')}</div></details>`;
  }

  function renderDigitCards(digits) {
    if (!Array.isArray(digits) || !digits.length) return '<div class="diag-empty">Digit-cell отсутствуют</div>';
    return `<div class="diag-digit-grid">${digits.map((digit,index)=>{
      const position = digit?.index ?? index + 1;
      const selected = digit?.digit ?? '?';
      const status = digit?.status ?? '—';
      return `<article class="diag-digit-card"><div class="diag-digit-head"><span class="diag-digit-title">D${esc(position)} → ${esc(selected)}</span><span class="diag-badge ${statusClass(status)}">${esc(status)}</span></div><dl class="diag-digit-meta"><dt>Confidence</dt><dd><strong>${esc(fmtProbability(digit?.confidence))}</strong></dd><dt>Engine</dt><dd>${esc(digit?.engine)}</dd>${digit?.reason ? `<dt>Причина</dt><dd>${esc(digit.reason)}</dd>` : ''}</dl>${renderTop3(digit?.top3)}${renderPreprocess(digit?.preprocess)}</article>`;
    }).join('')}</div>`;
  }

  function refineDigits(root,item) {
    const digits = item?.debug?.test_ui_postcode_ocr?.digits;
    if (!Array.isArray(digits)) return;
    const code = [...root.querySelectorAll('.diag-param-code')].find(node => node.textContent.trim() === 'test_ui_postcode_ocr.digits');
    if (!code) return;
    const row = code.closest('tr');
    if (!row || row.dataset.digitsRefined === '1') return;
    row.dataset.digitsRefined = '1';
    row.classList.add('diag-digits-row');
    const cells = row.children;
    if (cells.length < 2) return;
    cells[1].innerHTML = renderDigitCards(digits);
    if (cells.length >= 3) {
      cells[1].colSpan = 2;
      cells[2].remove();
    }
  }

  function refineCurrent(id) {
    const root = document.getElementById('debugDiagnosticBody');
    const item = state.results.get(id);
    if (!root || !item) return;
    refineDescriptions(root);
    refineDigits(root,item);
  }

  const previousShowDebug = window.showDebug;
  if (typeof previousShowDebug === 'function') {
    window.showDebug = id => {
      previousShowDebug(id);
      refineCurrent(id);
    };
  }
})();
</script>
"""


if 'data-toolocr-debug-diagnostics-refinements="v2"' not in _test_ui._ROI_STATUS_STYLE:
    _test_ui._ROI_STATUS_STYLE += "\n" + _REFINEMENT_STYLE

if 'data-toolocr-debug-diagnostics-refinements="v2"' not in _test_ui._PREVIEW_SCRIPT:
    _test_ui._PREVIEW_SCRIPT += "\n" + _REFINEMENT_SCRIPT

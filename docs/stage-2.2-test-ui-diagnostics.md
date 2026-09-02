# Stage 2.2 — диагностическое окно Test UI

## Назначение

Кнопка `Показать debug` в Test UI предназначена прежде всего для визуального контроля человеком. Полный JSON `analyze_layout()` со временем стал слишком большим для ежедневной проверки, поэтому основной debug-интерфейс разделён на два слоя:

```text
Показать debug
    ↓
человеко-читаемая диагностическая таблица
    ├── Показать JSON
    └── Скачать JSON
```

Backend payload при этом не изменяется. Табличная панель интерпретирует уже существующий `debug` в браузере, а JSON остаётся исходным техническим представлением для автоматического анализа и передачи в ИИ.

## Общие правила отображения

Каждый параметр показывается как строка:

```text
Параметр | Значение | Описание
```

Рядом с названием параметра находится значок `i`. Hover над названием или значком показывает расширенную подсказку и ссылку на документ, из которого взята семантика поля.

UI не пересчитывает backend-значения и не заменяет их собственными решениями. Цветные badges используются только для более быстрого визуального чтения известных статусов:

- зелёный — обычный успешный/готовый статус;
- жёлтый — неоднозначный, partial, rescue, warning или incomplete;
- красный — error, reject, mismatch, invalid, not-found или unavailable;
- нейтральный — значение без статусной семантики.

Если в будущей версии backend появляется новое поле, которого ещё нет в специализированном glossary, UI всё равно показывает его с полным JSON path. Tooltip в этом случае объясняет, что поле отображается без изменения backend-семантики и должно сверяться с актуальной документацией.

## Секции

Диагностика группируется в следующие секции.

### Общее

Поля верхнего уровня:

- `stage` — стадия pipeline;
- `standard` — используемый стандарт/нормативный профиль;
- `layout_status` — итог contour/layout detection;
- `format_mode` — `AUTO / SESSION / FIXED`;
- `expected_format` — внешний expected format;
- `format_status` — итоговый статус решения по формату;
- `format` — выбранный физический формат;
- `profile_scope` — область профилей, допущенных к scoring.

Семантика режимов формата определена в `docs/stage-2.1-format-modes.md`.

### Входное изображение

Блок `input` содержит имя файла, MIME type, количество полученных байт и разрешение исходного изображения.

### Нормализация кадра

Блок `frame_normalization` соответствует `docs/stage-2.1-frame-normalization.md`.

Основные поля:

- `status` — `cropped`, `unchanged` или `foreground_not_found`;
- `source` — исходное разрешение;
- `crop` — рабочая область CV;
- `foreground_bbox` — bbox светлого foreground;
- `area_ratio` — доля площади соответствующей области;
- `bottom_anchored` — письмо заканчивается у нижней границы исходного кадра.

Crop используется как рабочая область CV, но физическая метрическая геометрия не переводится в независимую растянутую систему координат.

### Контур письма

Блок `detector` содержит результат external quad detector:

- `method`;
- `confidence`;
- `raw_confidence`;
- `frame_status`;
- `frame_contact_sides`;
- `area_ratio`;
- `rectangularity`;
- `angle_score`;
- `quad_order`;
- `quad`.

`quad` — четыре вершины контура письма, используемые для perspective rectification и метрической геометрии.

### Проверка формата

Блок `format_validation` описан в `docs/stage-2.1-format-modes.md`.

Для SESSION ключевые состояния:

```text
confirmed
plausible
mismatch
```

`blocking=true` означает, что обнаруженное противоречие блокирует итоговое решение. `warnings` сохраняют диагностические противоречия, которые не обязательно блокируют результат.

### Метрическая оценка формата

`metric_format` содержит результат camera-calibration/pixel-scale проверки физического формата. Поля внутри этого блока могут развиваться вместе с calibration pipeline, поэтому UI показывает их рекурсивно и сохраняет полный JSON path.

### Perspective rectification

`rectified` содержит геометрию изображения после perspective rectification:

- `width_px`;
- `height_px`;
- `landscape`.

### Ориентация 0°/180°

Блок `orientation` описан в `docs/stage-2.1-orientation-content.md`.

Основные поля:

- `status` — `resolved` либо `ambiguous`;
- `value_deg` — решение `0` или `180`;
- `confidence` — уверенность orientation decision;
- `margin` — разделимость лучшей и альтернативной гипотез;
- `scores` — оценки 0°/180°;
- `evidence` — независимые свидетельства orientation scorer.

Production pipeline ToolOCR не поворачивает сортируемое письмо на 90°/270°.

### Profile scoring

`profile_scoring` содержит:

- `status`;
- `profile_id`;
- `confidence`;
- `margin`;
- `selected`;
- `top_hypotheses`.

`top_hypotheses` отображается сворачиваемым массивом, чтобы не перегружать окно.

### ROI почтового индекса

`test_ui_postcode_roi` показывает Test UI summary postcode stencil detector:

- `status`;
- `confirmation_mode`;
- `confidence`;
- `rejection_reason`;
- UI color/note.

Это диагностический summary; полный живой payload доступен через `/roi/meta`.

### CNN почтового индекса

`test_ui_postcode_ocr` соответствует актуальному ONNX-primary runtime и описан в `docs/stage-2.2-postcode-cnn-runtime.md`.

Основные поля:

- `status` — `recognized`, `incomplete`, `error` или `unavailable`;
- `text` — шесть символов с `?` для отсутствующих цифр;
- `postcode` — полный индекс, если распознаны все digit-cell;
- `confidence` — средняя softmax confidence шести выбранных цифр;
- `min_digit_confidence` — минимальная confidence;
- `geometric_mean_confidence` — геометрическое среднее confidence;
- `structurally_valid` — базовая структурная проверка российского индекса;
- `engine` — фактически использованный recognizer;
- `model_path` — ONNX model path;
- `digits` — результаты шести digit-cell.

Для каждой digit-cell доступны `digit`, `confidence`, `top3`, `engine` и `preprocess`.

`confidence`/`probability` являются softmax-выходами модели и не должны интерпретироваться как гарантированная вероятность правильной сортировки. До накопления production-корпуса они не используются как hard gate.

### Качество входа

`test_ui_input_quality` — неблокирующая диагностика возможного физического crop/деформации входа. Это предупреждение для оператора/разработчика, а не самостоятельное доказательство дефекта письма.

### ROI metadata endpoint

`test_ui_roi_meta.url` содержит живой URL:

```text
/v1/test-ui/images/{file_id}/roi/meta
```

Он используется для получения полного актуального ROI/CNN payload.

### Timing

`timing` показывает длительность этапов:

- `decode_ms`;
- `normalization_ms`;
- `detect_ms`;
- `metric_ms`;
- `rectify_ms`;
- `candidate_ms`;
- `profile_scoring_ms`;
- `profile_ms`;
- `total_ms`.

## Сворачиваемые тяжёлые значения

Объекты и массивы не разворачиваются в длинную JSON-простыню. Для них используется nested table с `details/summary`.

В частности, сворачиваются:

- orientation `evidence`;
- `top_hypotheses` profile scoring;
- CNN `digits`;
- `top3`;
- preprocessing diagnostics;
- прочие массивы объектов.

Поля `*_jpeg_base64` никогда не вставляются целиком в диагностическую таблицу. UI показывает только факт наличия binary/base64 payload и его размер в символах. Полное значение остаётся в JSON.

## Показать JSON

Кнопка `Показать JSON` открывает отдельное модальное окно с исходным JSON одного результата. Формат single-file payload:

```json
{
  "schema": "toolocr.test-ui.debug-item.v1",
  "exported_at": "...",
  "test_started_at": "...",
  "result": {
    "id": "...",
    "filename": "...",
    "format_mode": "auto",
    "ok": true,
    "layout_status": "detected",
    "format": "C5",
    "orientation_status": "resolved",
    "total_ms": 123.4,
    "debug": {}
  }
}
```

В `debug` находится исходный backend payload без преобразования.

## Скачать JSON

Кнопка `Скачать JSON` сохраняет single-file payload вида:

```text
toolocr-debug-<filename>.json
```

Такой файл предназначен для машинного анализа, прикладывания к issue и передачи в ИИ без копирования большого JSON вручную из браузера.

Глобальная кнопка `Выгрузить debug JSON` над таблицей результатов сохраняет прежнюю функцию: экспортирует результаты всего последнего batch-run одним файлом.

## Реализация

Диагностический UI подключён отдельным frontend-layer `ocr/app/test_ui_diagnostics.py` через существующий расширенный Test UI. Он не добавляет backend endpoint и не меняет схему `/v1/test-ui/run`.

Это позволяет модернизировать представление debug независимо от frozen layout/orientation/postcode detector pipeline.

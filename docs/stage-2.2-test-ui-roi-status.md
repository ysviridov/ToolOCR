# Stage 2.2 — цветовой статус ROI в Test UI

В таблице «Результаты последнего теста» кнопка `ROI` показывает краткий итог postcode detector сразу после batch-прогона.

Цвет относится только к `recipient_postcode` и не зависит от текущего качества `recipient_address`.

```text
белый   — postcode ROI не оценён или неприменим
красный — stencil_not_found
жёлтый  — stencil_detected + seven_bar_rescue
зелёный — stencil_detected + strict_start_marker
```

Кнопка сохраняет текст `ROI`. Подробности доступны в tooltip:

- для зелёной: `strict_start_marker` и confidence;
- для жёлтой: `seven_bar_rescue` и confidence;
- для красной: `stencil_not_found` и `rejection_reason`;
- для белой: причина, почему ROI не оценивался.

## Batch API

Расширенный `POST /v1/test-ui/run` после обычного layout-анализа выполняет postcode ROI на уже полученном canonical geometry и добавляет в каждый результат:

```json
{
  "postcode_roi_status": "stencil_detected",
  "postcode_confirmation_mode": "seven_bar_rescue",
  "postcode_confidence": 0.84,
  "postcode_rejection_reason": null,
  "postcode_roi_color": "yellow",
  "postcode_roi_note": null
}
```

При `orientation != resolved`, неподдерживаемом формате или ошибке layout результат остаётся белым с `postcode_roi_status=not_evaluated`.

Краткий summary также записывается в `debug.test_ui_postcode_roi`, поэтому он попадает в существующий debug JSON export без изменения схемы основного layout API.

ROI preview продолжает строиться по запросу и на диск не сохраняется.

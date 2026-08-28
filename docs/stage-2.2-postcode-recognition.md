# Stage 2.2 — распознавание шестизначного почтового индекса

## Цель

После стабилизации `postcode_stencil` и шести индивидуальных `digit_cell` ToolOCR распознаёт каждую цифру отдельно и собирает сырой шестизначный индекс.

Распознавание выполняется только после успешного построения canonical-изображения, подтверждения postcode stencil и получения шести digit-cell. Orientation и stencil detector этой итерацией не меняются.

## Pipeline

```text
SOURCE
  -> frame normalization
  -> perspective rectification
  -> orientation 0/180
  -> canonical image
  -> postcode stencil
  -> six digit-cell geometry
  -> normalize each digit-cell
  -> single-character OCR 0..9
  -> raw postcode text
  -> structural validation
  -> address DB validation (следующий этап)
```

Каждая digit-cell распознаётся независимо. Текущий baseline использует установленный в OCR-контейнере Tesseract с `PSM 10`, `OEM 1` и whitelist `0123456789`.

## Нормализация digit-cell

Для каждой ячейки:

1. Берётся crop из `canonical_rectified`.
2. Выполняется локальная компенсация медленного перепада освещения.
3. Otsu формирует чёрную цифру на белом фоне.
4. Самый внешний край ячейки очищается от случайных соседних артефактов.
5. Foreground цифры центрируется на фиксированном canvas `96x128` с сохранением aspect ratio.
6. Tesseract решает задачу single-character OCR.

Если foreground отсутствует или Tesseract не возвращает ровно одну цифру, для ячейки используется статус `unrecognized`.

## Результат

Полный результат содержит:

```json
{
  "status": "recognized",
  "text": "167420",
  "postcode": "167420",
  "confidence": 0.91,
  "min_digit_confidence": 0.82,
  "structurally_valid": true,
  "reason": null,
  "engine": "tesseract_single_digit",
  "digits": [
    {"index": 1, "status": "recognized", "digit": "1", "confidence": 0.97},
    {"index": 2, "status": "recognized", "digit": "6", "confidence": 0.93}
  ]
}
```

Если одна или несколько цифр не распознаны, `text` сохраняет диагностические `?`, например `16?420`, `postcode=null`, а `status=incomplete`.

## Structural validation

Распознанный OCR-результат не исправляется скрыто. Если Tesseract вернул шесть цифр, они сохраняются как raw postcode.

`structurally_valid=true` только если:

```text
- распознаны все 6 цифр;
- строка содержит только 0..9;
- первая цифра не равна 0.
```

Если OCR вернул `012345`, результат сохраняется как `012345`, но `structurally_valid=false`. Это позволяет отличать ошибку recognizer от последующей DB validation/correction.

## ROI overlay

`GET /v1/test-ui/images/{id}/roi` теперь наносит:

```text
D1=<digit> <confidence>
...
D6=<digit> <confidence>

POSTCODE OCR: <text> conf=<mean confidence>
```

При нераспознанной цифре используется `?`.

Рамки digit-cell остаются теми же, что были проверены corpus-validation; recognizer их геометрию не изменяет.

## Debug JSON

`GET /v1/test-ui/images/{id}/roi/meta` возвращает полный блок `postcode_recognition`, а также копию `recognition` внутри региона `recipient_postcode`.

Batch Test UI добавляет в каждый успешный result компактные поля:

```text
postcode_ocr_status
postcode_ocr_text
postcode_ocr_postcode
postcode_ocr_confidence
postcode_ocr_min_digit_confidence
postcode_ocr_structurally_valid
postcode_ocr_reason
postcode_ocr_digits
```

В существующий debug export дополнительно записывается:

```text
debug.test_ui_postcode_ocr
```

Блок содержит итоговый raw postcode, confidence и данные по каждой из шести digit-cell. Это предназначено для прямого сравнения с `ground_truth.csv`.

## Производительность

Текущий baseline делает шесть независимых single-character Tesseract-вызовов на письмо. Это намеренно оставляет максимально прозрачную диагностику по каждой ячейке. После corpus-run нужно отдельно измерить OCR latency и при необходимости оптимизировать выполнение без изменения контракта результата.

# Stage 2.1 — режимы определения физического формата

## Назначение

Layout API поддерживает три режима выбора физического формата письма:

```text
format_mode=auto
format_mode=session
format_mode=fixed
```

Параметр `expected_format` принимает один из форматов ГОСТ Р 51506-99:

```text
C6 | DL | C5 | C4 | B4
```

Для `session` и `fixed` параметр `expected_format` обязателен. В `auto` он игнорируется.

## AUTO

`auto` — универсальный режим без внешнего знания о текущей сортировочной пачке.

Система использует доступную метрическую калибровку, aspect ratio и profile scoring и самостоятельно выбирает формат. Если C6/C5/C4/B4 геометрически не удаётся различить, результат может остаться неоднозначным.

Этот режим сохраняет прежнюю семантику API и является значением по умолчанию.

Пример:

```text
POST /v1/layout/analyze?format_mode=auto
```

## SESSION

`session` предназначен для production-сортировки, когда оператор или внешняя система заранее задаёт ожидаемый формат текущей пачки.

Пример:

```text
POST /v1/layout/analyze?format_mode=session&expected_format=C5
```

В этом режиме:

1. profile scoring ограничивается профилями `expected_format`;
2. из файла camera calibration выбирается только запись соответствующего формата;
3. калибровки других форматов не участвуют в consensus и не могут сделать его `inconsistent`;
4. выполняется независимый sanity-check по aspect ratio и диагностическая metric-проверка;
5. сильное aspect-противоречие блокирует формат с `format_status=session_mismatch`;
6. одиночное metric-противоречие пока сохраняется как warning, а не blocking condition.

Последнее правило связано с текущей нестабильностью pixel-scale при изменении FOV/масштаба входных снимков. До стабилизации camera calibration нельзя считать одиночный metric mismatch достаточным основанием отклонить заведомо заданную сортировочную пачку. При этом совпадающий metric результат остаётся независимым подтверждением expected format.

Возможные состояния `format_validation.status`:

```text
confirmed  — ожидаемый формат подтверждён metric и/или profile evidence
plausible  — сильных противоречий нет, но независимого подтверждения недостаточно
mismatch   — aspect ratio явно противоречит ожидаемому формату; format=null
```

Это особенно надёжно отделяет DL (`aspect≈2.0`) от семейства C6/C5/C4/B4 (`aspect≈1.41`). Различение C6/C5/C4/B4 по одному aspect ratio невозможно и требует стабильной метрической шкалы либо дополнительных profile-признаков.

`session` не является слепым hard-lock: при mismatch downstream не должен строить ROI как будто ожидаемый формат подтверждён.

## FIXED

`fixed` — жёсткий внешний constraint:

```text
POST /v1/layout/analyze?format_mode=fixed&expected_format=DL
```

Система всегда возвращает заданный формат, а profile scoring ограничивается его профилями. Независимые противоречия не блокируют результат, но сохраняются в:

```json
{
  "format_validation": {
    "status": "fixed",
    "blocking": false,
    "warnings": ["..."]
  }
}
```

Этот режим следует использовать только когда внешний источник формата является авторитетным и ответственность за корректность hard-lock лежит на вызывающей системе.

## Поля ответа

`/v1/layout/analyze` теперь явно возвращает:

```json
{
  "format_mode": "session",
  "expected_format": "C5",
  "format_validation": {
    "status": "confirmed",
    "expected_format": "C5",
    "metric_observed_format": "C6",
    "aspect_error": 0.0123,
    "aspect_tolerance": 0.08,
    "profile_matches_expected": true,
    "metric_matches_expected": false,
    "blocking": false,
    "reasons": [],
    "warnings": [
      "metric_format=C6 противоречит expected_format=C5"
    ]
  },
  "format_status": "session_confirmed",
  "format": "C5"
}
```

Для `session_mismatch` поле `format` равно `null`, а `expected_format` остаётся в ответе для диагностики.

## Camera calibration в SESSION/FIXED

В `auto` по-прежнему используется consensus доступных калибровок.

В `session` и `fixed` выбирается только калибровка `expected_format`. Это устраняет ложную зависимость C5-сессии от DL-калибровки и наоборот.

Если нужной записи нет, `metric_format.calibration.status` принимает `reference_missing`; SESSION при этом может остаться `plausible` или получить подтверждение от profile scoring.

Важно: выбор только своей калибровки устраняет конфликт между reference entries, но сам по себе не делает pixel-scale корректным при смене масштаба/FOV. Поэтому metric mismatch сейчас диагностический, пока не будет стабилизирован camera plane/FOV contract.

## Test UI

Перед запуском теста UI позволяет выбрать:

```text
Режим формата: AUTO | SESSION | FIXED
Ожидаемый формат: C6 | DL | C5 | C4 | B4
```

Для AUTO поле ожидаемого формата отключено. Для SESSION/FIXED запуск без выбранного формата блокируется в браузере.

В таблице результатов показывается режим и `format_validation.status`; `mismatch` выделяется как ошибка проверки формата. Сводная статистика содержит распределение `Проверка формата`.

Экспорт debug JSON сохраняет режим и ожидаемый формат как на уровне всего запуска, так и для каждого результата.

## Rectify endpoint

`/v1/layout/rectify` принимает те же параметры `format_mode` и `expected_format`. Они используются для ограничения profile scoring и выбора соответствующей camera calibration. В HTTP headers возвращаются:

```text
X-ToolOCR-Format-Mode
X-ToolOCR-Expected-Format
X-ToolOCR-Format-Status
X-ToolOCR-Format
```

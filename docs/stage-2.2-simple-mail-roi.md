# Stage 2.2 — ROI простых писем

## Область задачи

Stage 2.2 сфокусирован на простых внутренних письмах с рукописным или машинописным индексом и адресом получателя.

Заказные отправления в этот OCR-маршрут не входят. Их трек-номер/штрихкод должен считываться специализированным оборудованием; ToolOCR не тратит OCR на декодирование штрихкода.

ROI реализованы для `DL`, `C5` и `C4`. Текущая работа сфокусирована на надёжной локализации технического шестизначного postcode-блока.

## Pipeline

```text
SOURCE
  -> frame normalization
  -> perspective rectification
  -> orientation 0/180
  -> canonicalization
       resolved 0   -> без поворота
       resolved 180 -> rotate 180
       ambiguous    -> не угадывать, ROI/OCR блокировать
  -> CANONICAL RECTIFIED
  -> ROI detection
       recipient_address
       recipient_postcode
  -> handwriting OCR (следующий этап)
  -> address parsing / validation
```

ROI всегда живут в системе координат `canonical_rectified`, где верх письма находится сверху независимо от исходного положения под камерой.

Orientation считается зафиксированной подсистемой. Настройка ROI не должна менять orientation weights, thresholds или photometric preprocessing.

## ROI текущей итерации

### `recipient_address`

Рукописный/машинописный адресный блок получателя. Пока определяется по расширенной format-specific search zone с foreground refinement.

### `recipient_postcode`

Технический шестизначный индекс места назначения определяется отдельным структурным detector `postcode_stencil`.

Вместо поиска произвольного тёмного содержимого используется инвариант печатного трафаретного блока:

```text
=  [digit1] [digit2] [digit3] [digit4] [digit5] [digit6]
```

Характерные признаки:

- семь верхних чёрных прямоугольных плашек: одна над стартовым знаком `=` и шесть над цифрами;
- плашки имеют близкую ширину и высоту;
- шаг между ними почти регулярный;
- центры плашек лежат почти на одной прямой;
- под первой верхней плашкой штатно находится вторая плашка той же ширины, приблизительно вдвое тоньше;
- число цифр фиксировано: шесть.

Этот pattern не зависит от формата конверта и устойчивее к рукописи, штрихкодам, печатным адресам и декоративным изображениям.

## Поиск postcode stencil

Поиск выполняется в расширенном левом нижнем секторе canonical-письма:

```text
x=0.00 y=0.50 w=0.62 h=0.50
```

Это только зона поиска anchor-pattern, а не итоговый ROI.

Перед выделением плашек выполняется локальная компенсация медленного перепада освещения. Затем horizontal morphology выделяет широкие чёрные элементы.

### Row-first association

На реальных C5 обнаружился класс false negative, где detector видел семь верхних плашек, но старый greedy matching связывал элементы только по X. Если под той же X-координатой находился нижний штрих `=` или горизонтальный штрих заполненной цифры, он мог попасть в одну из семи позиций верхней строки. После этого `alignment_error` становился большим и настоящий postcode отклонялся.

Теперь association выполняется в два этапа:

```text
horizontal candidates
        ->
Y-clustering / rows
        ->
X-regularity внутри одного row
        ->
7 upper bars
        ->
отдельный поиск нижнего штриха '=' под первой верхней плашкой
```

Кандидаты верхних плашек сначала объединяются в горизонтальные Y-ряды. Допуск центра ряда равен `0.75 × median_bar_height`, но не менее `2 px` в detector-scale. В X-matching допускаются только элементы выбранного ряда. Нижняя половинная плашка `=` не участвует в seven-position matching и используется только как strict-подтверждение после выбора верхней строки.

Это сохраняет прежние strict/rescue thresholds и устраняет первопричину `alignment_error_too_high` без расширения допустимой геометрической ошибки.

В debug добавлены:

```text
association_mode = row_first
row_cluster_count
row_tolerance_px
selected_row_index
selected_row_size
row_y_spread_px
```

Остальные структурные признаки остаются прежними:

```text
bar_count
start_marker_score
width_cv
spacing_error
alignment_error
row_y_norm
```

### Strict confirmation

Основной путь остаётся прежним:

```text
structural score >= 0.78
start_marker_score >= 0.40
-> confirmation_mode = strict_start_marker
```

То есть штатно detector требует подтверждение нижней половинной плашки стартового `=`.

### Seven-bar rescue

На реальных C4 нижняя плашка `=` иногда печатается/снимается хуже верхней строки. Чтобы не терять такой настоящий индекс, добавлен второй независимый путь. Он не ослабляет strict-порог и применяется только при полной и очень регулярной верхней строке:

```text
bar_count == 7
width_cv <= 0.12
spacing_error <= 0.08
alignment_error <= 0.55
row_y_norm >= 0.70
-> confirmation_mode = seven_bar_rescue
```

Шесть плашек rescue не активируют. Строка выше 70% высоты canonical-письма также не может пройти rescue. Поэтому отсутствие нижней плашки `=` компенсируется одновременно четырьмя сильными структурными ограничениями и геометрическим положением блока.

Strict-кандидат всегда имеет приоритет над rescue-кандидатом.

## Debug metadata

Подтверждённый stencil возвращает, например:

```json
{
  "kind": "recipient_postcode",
  "status": "stencil_detected",
  "detector": "postcode_stencil",
  "confidence": 0.86,
  "features": {
    "confirmation_mode": "seven_bar_rescue",
    "rejection_reason": null,
    "association_mode": "row_first",
    "bar_count": 7,
    "expected_bar_count": 7,
    "digit_count": 6,
    "start_marker_score": 0.12,
    "width_cv": 0.03,
    "spacing_error": 0.02,
    "alignment_error": 0.08,
    "row_y_norm": 0.91,
    "row_cluster_count": 2,
    "selected_row_size": 7,
    "row_y_spread_px": 2.0,
    "bar_width_px": 78.0,
    "bar_step_px": 98.0
  }
}
```

Для strict-path `confirmation_mode=strict_start_marker`.

Если pattern не подтверждён, generic foreground fallback намеренно не используется. Регион получает `status=stencil_not_found`, `detected_bbox=null`, а `features` содержит:

```text
confirmation_mode = none
rejection_reason =
  no_structural_candidate |
  insufficient_top_bars |
  row_not_low_enough |
  width_variation_too_high |
  spacing_error_too_high |
  alignment_error_too_high |
  start_marker_weak |
  structural_score_too_low
```

Это позволяет разбирать оставшиеся false negative без изменения orientation или общего foreground detector.

## ГОСТ

Источником общей компоновки является ГОСТ Р 51506-99, обязательное приложение А. Оформление шестизначного кодового штампа приведено в обязательном приложении Д.

Format-specific search zones продолжают использоваться для `recipient_address`, но postcode detector опирается прежде всего на структуру самого технического блока.

## Debug overlay

В Test UI доступны:

- `Canonical` — rectified-письмо после обязательного поворота 0/180;
- `ROI` — canonical-письмо с разметкой ROI.

Цвета:

- зелёный — `recipient_address`;
- оранжевый — `recipient_postcode`.

Strict detection подписывается как `POSTCODE STENCIL <confidence>`, rescue — как `POSTCODE STENCIL RESCUE <confidence>`.

Пунктирная оранжевая рамка показывает широкий сектор поиска, толстая сплошная — фактически найденный трафаретный блок. Preview генерируется по запросу и на диск не сохраняется.

## Test UI API

```text
GET /v1/test-ui/images/{id}/canonical
GET /v1/test-ui/images/{id}/roi
GET /v1/test-ui/images/{id}/roi/meta
```

`/roi/meta` возвращает `detector`, `features`, `search_bbox`, `detected_bbox` и confidence для postcode.

## Ограничения текущей итерации

- ROI поддерживаются для `DL`, `C5` и `C4`;
- распознавание самих шести цифр ещё не выполняется;
- распознавание текста адреса ещё не выполняется;
- штрихкоды заказных отправлений не декодируются;
- при `orientation=ambiguous` ROI не строятся;
- row-first association нужно проверить сначала на двух известных C5 `alignment_error_too_high`, затем на полном C5/C4 корпусе и отдельно на DL.

После corpus-validation отдельно анализируются false negative (`stencil_not_found`) и false positive. Для успешных детекций проверяется tightness bbox относительно семи плашек и шести цифр. Orientation при этом не меняется.

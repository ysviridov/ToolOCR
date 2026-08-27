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

## Canonicalization

Модуль `ocr/app/roi.py` содержит `canonicalize_rectified()`.

Если orientation имеет статус `resolved` и значение `180`, rectified-изображение поворачивается на 180 градусов. При `ambiguous` изображение не поворачивается наугад, а `reliable=false`; ROI preview для такого кадра возвращает ошибку.

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
- центры плашек лежат почти на одной прямой, допускается небольшой остаточный наклон после rectification;
- под первой верхней плашкой находится вторая плашка той же ширины, приблизительно вдвое тоньше;
- число цифр фиксировано: шесть.

Этот pattern не зависит от формата конверта и устойчивее к рукописи, штрихкодам, печатным адресам и декоративным изображениям.

## Поиск postcode stencil

Поиск выполняется в расширенном левом нижнем секторе canonical-письма:

```text
x=0.00 y=0.50 w=0.62 h=0.50
```

Это только зона поиска anchor-pattern, а не итоговый ROI.

Перед выделением плашек выполняется локальная компенсация медленного перепада освещения. Затем горизонтальная morphology выделяет широкие чёрные элементы. Кандидаты группируются в последовательность из семи позиций и оцениваются по:

```text
bar_count
start_marker_score
width_cv
spacing_error
alignment_error
```

Подтверждённый stencil возвращается как:

```json
{
  "kind": "recipient_postcode",
  "status": "stencil_detected",
  "detector": "postcode_stencil",
  "confidence": 0.96,
  "features": {
    "bar_count": 7,
    "expected_bar_count": 7,
    "digit_count": 6,
    "start_marker_score": 0.95,
    "width_cv": 0.02,
    "spacing_error": 0.03,
    "alignment_error": 0.04,
    "bar_width_px": 78.0,
    "bar_step_px": 98.0
  }
}
```

Итоговый `detected_bbox` строится от найденной последовательности плашек и включает шесть трафаретных цифр под ними.

Если pattern не подтверждён, generic foreground fallback намеренно не используется. Регион получает:

```text
status = stencil_not_found
detected_bbox = null
```

Это сделано специально: для postcode false negative безопаснее, чем уверенный bbox на чужом адресе, штрихкоде или иллюстрации.

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

Для postcode подпись выглядит как `POSTCODE STENCIL <confidence>`.

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
- stencil detector нужно провалидировать на полном 50-файловом C4-корпусе и затем отдельно на DL/C5.

После corpus-validation следует анализировать отдельно false negative (`stencil_not_found`) и false positive. Для успешных детекций проверяется tightness bbox относительно семи плашек и шести цифр. Orientation при этом не меняется.

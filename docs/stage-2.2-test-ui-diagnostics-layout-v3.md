# Stage 2.2 — широкое диагностическое окно Test UI, layout v3

## Назначение

Этот слой развивает `docs/stage-2.2-test-ui-diagnostics.md` и не меняет backend payload. Цель — использовать доступную площадь широкоформатного монитора и убрать каскады вложенных таблиц там, где данные естественно читаются как набор карточек или сравнение гипотез.

## Размер окна

`Диагностика` больше не ограничивается фиксированным `1320/1540 px`.

На desktop окно открывается примерно на всю доступную область браузера:

```text
width      = viewport - 24 px
max-width  = none
max-height = viewport - 24 px
```

Также включён CSS `resize: both`: при необходимости окно можно вручную уменьшить или растянуть. На узких экранах ручной resize отключается, а окно занимает практически весь viewport.

## Специализированные представления

### CNN digits

Слой v2 уже отображает `test_ui_postcode_ocr.digits` как шесть карточек `D1...D6` с confidence, engine, TOP-3 и сворачиваемым preprocessing.

### Кандидаты формата

`format_candidates` показываются как отдельные карточки. Для каждого кандидата видны:

- формат;
- физический размер ГОСТ;
- aspect ratio;
- ratio error;
- место кандидата в списке.

Это заменяет массив объектов с несколькими уровнями nested-table.

### Orientation scores

`orientation.scores` отображает 0°/180° как отдельные сравниваемые карточки с итоговым score. Лучшая гипотеза визуально выделяется рамкой.

### Orientation evidence

`orientation.evidence` отображается по одной карточке на orientation-гипотезу. Основные каналы показываются как компактные строки с числом и индикатором силы сигнала:

- postage;
- code stamp;
- barcode layout;
- address layout;
- text direction;
- content orientation;
- base score;
- итоговый score.

Внутренние `contrast` и `agreement` остаются доступны через компактные сворачиваемые блоки.

### Profile scoring

`profile_scoring.top_hypotheses` показывается как ранжированный набор карточек. В карточке видны:

- rank;
- profile ID;
- format;
- layout;
- orientation;
- window;
- итоговый score;
- компоненты scoring.

`profile_scoring.selected` показывается компактной строкой chips вместо отдельной nested-table.

### Quad detector

`detector.quad` показывается как четыре точки `TL/TR/BR/BL` с координатами `x/y`. Порядок берётся из `detector.quad_order`.

## Инварианты

UI не изменяет значения backend и не пересчитывает решения pipeline. Карточки являются только представлением исходного debug payload.

Полный JSON по-прежнему доступен через кнопки `Показать JSON` и `Скачать JSON`. Поэтому специализированное представление можно изменять независимо от формата технического экспорта.

## Реализация

Слой реализован в:

```text
ocr/app/test_ui_diagnostics_layout_v3.py
```

и подключается после базового диагностического слоя и refinement v2. Он оборачивает существующий `window.showDebug`, сначала вызывает уже установленный renderer, а затем заменяет только конкретные тяжёлые строки на специализированные карточки.

Backend API, frozen orientation, postcode stencil detector и CNN runtime этим изменением не затрагиваются.

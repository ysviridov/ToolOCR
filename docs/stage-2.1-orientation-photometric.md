# Stage 2.1 — фотометрическая нормализация orientation-признаков

## Причина

На части production C4 фон бумаги существенно темнее прежних DL/C5 тестов. Старые CV-признаки используют фиксированные grayscale thresholds (`175/185/190/195`). При тёмной бумаге большая часть конверта ошибочно становится `ink`, из-за чего:

- `postage` насыщается почти одинаково в обеих ориентациях;
- `code_stamp` перестаёт выделять периодическую гребёнку;
- `address_layout` теряет отдельные текстовые компоненты;
- orientation часто остаётся `ambiguous`, а затем безрезультатно уходит в Tesseract OSD slow-path.

## Решение

Перед threshold-based orientation-признаками добавлен отдельный photometric preprocessing в `ocr/app/orientation_photometric.py`.

Pipeline scoring теперь разделяет два grayscale-потока:

```text
rectified
  -> resize_for_scoring (max side 1000)
  -> raw_gray
       -> barcode_layout
       -> window_signal

  -> background estimation (large Gaussian)
  -> conditional illumination normalization
  -> feature_gray
       -> postage
       -> code_stamp
       -> line_signal
       -> address_layout
```

Tesseract OSD получает прежний исходный rectified image и не использует photometric-копию.

## Условие включения

Коррекция не применяется ко всем письмам. Для сглаженного background оцениваются percentiles. Статус становится `applied`, если выполняется хотя бы одно условие:

```text
background_p50 < 205
```

или обнаружена выраженная теневая область:

```text
background_p25 < 190
and background_iqr >= 22
```

На ярких письмах возвращается `not_needed`, и threshold-based признаки получают прежний grayscale без изменения пикселей.

## Коррекция

Низкочастотный фон оценивается Gaussian blur с масштабом:

```text
sigma = max(12 px, 0.045 * min(width, height))
```

После этого:

```text
feature_gray = gray * 235 / max(background, 24)
```

с ограничением результата в `0..255`.

Таким образом локальный фон бумаги приближается к 235, а реальные тёмные штрихи/печать остаются ниже фиксированных thresholds.

## Что НЕ изменено

Фиксируются без изменений:

- `CONTRAST_CHANNEL_WEIGHTS`;
- `CONTRAST_BONUS_SCALE/MAX`;
- `AGREEMENT_*`;
- базовая формула orientation signal;
- `orientation_min_signal = 0.30`;
- `orientation_min_margin = 0.12`;
- CV-first / OSD-fallback порядок.

Изменён только вход threshold-based CV-признаков.

## Debug

В каждый элемент `orientation.evidence[]` добавляется:

```json
"photometric_normalization": {
  "status": "applied",
  "reasons": ["dark_background", "uneven_shadow"],
  "raw_p25": 135.0,
  "raw_p50": 156.0,
  "raw_p75": 173.0,
  "raw_p95": 197.0,
  "background_p10": 110.0,
  "background_p25": 132.0,
  "background_p50": 151.0,
  "background_p75": 171.0,
  "background_p90": 184.0,
  "background_iqr": 39.0,
  "sigma_px": 31.5,
  "target_background": 235.0
}
```

Поле диагностическое и не меняет decision contract API.

## Проверка

После deployment рекомендуется повторить:

1. тот же batch из 50 C4;
2. предыдущий независимый DL/C5 regression batch.

Для C4 проверяем:

- долю `orientation_status=resolved`;
- правильность 0/180, а не только уменьшение `ambiguous`;
- `code_stamp` и `address_layout` contrast;
- сколько кадров имеют `photometric_normalization.status=applied`;
- снижение количества OSD slow-path по `profile_scoring_ms`.

Для DL/C5 основной regression-критерий — отсутствие ухудшения уже подтверждённой ориентации. На ярких кадрах ожидается `photometric_normalization.status=not_needed`.

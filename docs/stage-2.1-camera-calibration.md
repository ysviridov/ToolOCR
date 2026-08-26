# Stage 2.1 — метрическое определение формата по калиброванной камере

## Зачем нужна калибровка

Форматы C6, C5, C4 и B4 имеют близкое отношение сторон, поэтому по одному `width_px / height_px` их надёжно различить нельзя. Физические размеры при этом различаются существенно:

| Формат | Ширина, мм | Высота, мм |
|---|---:|---:|
| C6 | 162 | 114 |
| DL | 220 | 110 |
| C5 | 229 | 162 |
| C4 | 324 | 229 |
| B4 | 353 | 250 |

## Важное свойство production-кадров

Полный JPEG содержит чёрный фон транспортёра, причём его размер может немного различаться. Для C5, например, встречаются `3720x2888`, `3736x2888`, `3744x2888`, `3752x2888`.

Поэтому полный `image_width_px/image_height_px` больше **не является метрическим инвариантом**.

Перед layout detector выполняется frame normalization: светлое письмо выделяется на чёрном фоне, рабочая область обрезается, а найденный quad затем переводится обратно в source-координаты. Подробнее: `docs/stage-2.1-frame-normalization.md`.

## Metric mode: quad_pixel_scale

Начиная с текущей версии физический формат определяется по масштабу самого эталонного письма:

```text
px_per_mm_x = reference_width_px  / reference_width_mm
px_per_mm_y = reference_height_px / reference_height_mm
```

Для нового production-письма измеряется его внешний quad в пикселях и переводится в миллиметры через эти коэффициенты.

Это означает, что изменение только количества чёрного поля вокруг письма не влияет на результат.

Projective homography `homography_norm_to_mm` продолжает храниться в записи для обратной совместимости и перспективной геометрии, но основной метрический режим Stage 2.1 теперь:

```json
"metric_mode": "quad_pixel_scale"
```

## Набор калибровок

`config/camera-calibration.json` хранит несколько независимых эталонов:

```json
{
  "version": 2,
  "standard": "ГОСТ Р 51506-99",
  "calibrations": {
    "C5": {
      "version": 1,
      "reference_format": "C5",
      "reference_width_mm": 229.0,
      "reference_height_mm": 162.0,
      "reference_width_px": 2315.4,
      "reference_height_px": 1638.1,
      "px_per_mm_x": 10.1109,
      "px_per_mm_y": 10.1117,
      "metric_mode": "quad_pixel_scale",
      "homography_norm_to_mm": [[...], [...], [...]]
    },
    "DL": {
      "version": 1,
      "reference_format": "DL",
      "reference_width_mm": 220.0,
      "reference_height_mm": 110.0,
      "reference_width_px": 2224.1,
      "reference_height_px": 1112.4,
      "px_per_mm_x": 10.1095,
      "px_per_mm_y": 10.1127,
      "metric_mode": "quad_pixel_scale",
      "homography_norm_to_mm": [[...], [...], [...]]
    }
  }
}
```

Числа в примере иллюстративные.

## Калибровка

Эталон должен иметь точно известный формат и лежать в той же production-плоскости, что и реальные письма.

Endpoint поддерживает два режима проверки эталонного кадра:

```text
calibration_mode=strict
calibration_mode=scale_reference
```

### strict

Режим по умолчанию. Эталон не должен касаться ни одной стороны кадра:

```bash
make layout-calibrate FILE=/path/reference-c5.jpg FORMAT=C5
```

Любой `frame_contact_sides` приводит к `422 reference_partial_frame`.

### scale_reference

Режим для фиксированного сортировщика, где письмо штатно лежит у нижней границы изображения при неизменных camera/FOV/perspective.

Разрешён только:

```text
frame_contact_sides = ["bottom"]
```

Контакт с `top`, `left` или `right`, в том числе одновременно с `bottom`, остаётся блокирующим и возвращает 422.

Примеры для текущего production-положения письма:

```bash
make layout-calibrate \
  FILE=/path/reference-c5.jpg \
  FORMAT=C5 \
  CALIBRATION_MODE=scale_reference

make layout-calibrate \
  FILE=/path/reference-dl.tiff \
  FORMAT=DL \
  CALIBRATION_MODE=scale_reference
```

`scale_reference` допустим только если нижняя физическая сторона письма действительно присутствует в кадре и просто совпадает с нижней границей изображения. Если письмо реально обрезано, такую фотографию использовать как эталон нельзя.

Запись сохраняет диагностические поля:

```json
{
  "calibration_mode": "scale_reference",
  "reference_frame_contact_sides": ["bottom"],
  "reference_bottom_anchored": true
}
```

Для такой записи подразумевается та же физическая установка камеры и тот же FOV, что и при production-съёмке. Ручной resize, perspective correction или изменение масштаба эталонного изображения недопустимы.

Каждая команда обновляет только свою запись. C5 не удаляет DL и наоборот.

Просмотр набора:

```bash
make layout-calibrations
```

Команда дополнительно показывает `calibration_mode`, `reference_bottom_anchored`, `px_per_mm_x` и `px_per_mm_y`, чтобы C5/DL можно было сравнить напрямую.

## Совместимость старых записей

Старые одноформатные v1-файлы и ранние v2-записи, где ещё отсутствуют:

```text
reference_width_px
reference_height_px
px_per_mm_x
px_per_mm_y
```

остаются читаемыми. ToolOCR восстанавливает исходный эталонный quad через inverse homography и вычисляет pixel-scale автоматически.

У старых записей также могут отсутствовать `calibration_mode` и `reference_bottom_anchored`; это не мешает их загрузке.

Поэтому массовая повторная калибровка только ради нового формата JSON не требуется.

## Consensus нескольких эталонов

В режиме `format_mode=auto` формат production-письма заранее неизвестен, поэтому каждый сохранённый эталон независимо измеряет production quad через свой `px/mm`. Затем ToolOCR берёт медианный consensus.

Пример:

```json
{
  "metric_format": {
    "calibration": {
      "status": "loaded",
      "count": 2,
      "reference_formats": ["C5", "DL"],
      "metric_mode": "quad_pixel_scale"
    },
    "consensus": {
      "consistent": true,
      "metric_mode": "quad_pixel_scale",
      "width_spread_mm": 1.6,
      "height_spread_mm": 1.2,
      "measurement": {
        "width_mm": 229.4,
        "height_mm": 161.8
      }
    }
  }
}
```

Если эталоны расходятся более допустимого порога, `consensus.consistent=false` и метрическое решение не используется. Это может указывать на:

- реальное изменение масштаба/zoom камеры;
- upstream resize изображения;
- ошибочно указанный формат эталона;
- плохое выделение внешнего quad;
- заметную нелинейную дисторсию в разных частях кадра.

В `format_mode=session/fixed` используется только калибровка `expected_format`, поэтому калибровка другого формата не участвует в consensus.

## Что теперь допустимо

Допустимо изменение:

```text
3720x2888
3736x2888
3744x2888
3752x2888
```

если меняется только количество чёрного поля, а размер самого письма в пикселях остаётся тем же.

Недопустимо незаметное масштабирование самого содержимого, например когда одно и то же C5 в одном режиме занимает 2300 px по ширине, а в другом — 1800 px. Такой режим требует отдельной camera/profile calibration либо нормализации upstream.

## Partial frame

Если одна физическая сторона действительно обрезана кадром, соответствующее измерение остаётся `lower_bound`. Например при контакте только с нижней границей полная ширина остаётся сильным метрическим признаком.

Это поведение production analysis не следует смешивать с `scale_reference`: при создании эталона мы разрешаем `bottom` только в заранее известной конфигурации сортировщика и предполагаем, что физическая нижняя сторона письма не потеряна.

## Диагностика

```bash
make layout-smoke FILE=/path/letter.jpg | jq '{
  frame_normalization,
  detector,
  metric_format,
  format_status,
  orientation,
  timing
}'
```

Для анализа согласованности эталонов:

```bash
make layout-smoke FILE=/path/letter.jpg \
  | jq '.metric_format | {status, format, calibration, consensus, measurement, candidates}'
```

Если калибровка отсутствует или consensus некорректен, ToolOCR использует visual profile scoring как fallback.

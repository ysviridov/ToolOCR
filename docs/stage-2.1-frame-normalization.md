# Stage 2.1 — нормализация кадра на чёрном фоне

## Задача

В production письма фотографируются на чёрной ленте сортировщика. Полный JPEG может иметь немного различающуюся ширину, например для C5 встречаются `3720x2888`, `3736x2888`, `3744x2888`, `3752x2888`, хотя масштаб самого письма и рабочая плоскость камеры не меняются.

Поэтому полный размер JPEG нельзя использовать как жёсткий метрический инвариант.

## Новый pipeline

```text
source JPEG
    │
    ▼
light foreground segmentation
    │
    ▼
largest envelope component
    │
    ├── foreground bbox
    ├── bottom anchor
    └── crop + safety margin
    │
    ▼
normalized crop
    │
    ▼
external quad detector
    │
    ▼
quad crop -> source coordinates
    │
    ├── metric format по source quad
    └── perspective rectification из original JPEG
```

Crop используется только как рабочая область CV. Координаты четырёхугольника перед метрическим измерением возвращаются в систему исходного изображения, поэтому информация о масштабе не теряется.

## Почему tight crop не используется как физическая система координат

Если независимо растянуть C5 и C4 до одинакового размера, информация об абсолютном физическом размере исчезнет. Поэтому ToolOCR не измеряет формат в нормализованных координатах crop.

Вместо этого калибровка хранит масштаб самого эталонного четырёхугольника:

```text
reference_width_px / reference_width_mm  -> px_per_mm_x
reference_height_px / reference_height_mm -> px_per_mm_y
```

Например изменение полного JPEG с `3720x2888` на `3752x2888` не меняет размер quad письма в пикселях и, следовательно, не меняет метрическую оценку.

Homography `normalized image -> mm` сохранена в JSON для обратной совместимости и перспективной геометрии, но физический формат Stage 2.1 определяется через `metric_mode=quad_pixel_scale`.

## Совместимость старых калибровок

Повторная калибровка всех существующих эталонов не обязательна. Если в старой записи отсутствуют:

```text
reference_width_px
reference_height_px
px_per_mm_x
px_per_mm_y
```

ToolOCR восстанавливает исходный quad через inverse homography и вычисляет pixel-scale автоматически.

Новые `make layout-calibrate` записывают эти поля явно.

## API

`POST /v1/layout/analyze` теперь содержит блок:

```json
{
  "frame_normalization": {
    "status": "cropped",
    "source": {
      "width_px": 3744,
      "height_px": 2888
    },
    "crop": {
      "x": 120,
      "y": 580,
      "width_px": 3380,
      "height_px": 2308,
      "area_ratio": 0.72
    },
    "foreground_bbox": {
      "x": 160,
      "y": 620,
      "width_px": 3300,
      "height_px": 2268,
      "area_ratio": 0.69
    },
    "bottom_anchored": true
  }
}
```

Числа иллюстративные.

`status`:

- `cropped` — чёрный фон существенно удалён;
- `unchanged` — foreground найден, но crop почти равен исходному кадру;
- `foreground_not_found` — безопасный fallback на исходный кадр.

## Timing

Добавлена метрика:

```json
"normalization_ms": 12.5
```

Detector после этого работает уже на уменьшенной рабочей области. На кадрах с большим количеством чёрного фона это частично компенсирует стоимость самой сегментации.

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

Для debug-режима `include_debug_images=true` дополнительно возвращается `normalized_jpeg_base64`.

## Bottom anchor

Если светлая компонента письма заканчивается у нижней границы исходного кадра, выставляется:

```json
"bottom_anchored": true
```

Это соответствует текущей механике сортировщика, где отправление прижимается к нижнему краю рабочей области. Этот признак сохраняется для дальнейшего восстановления геометрии и ROI, но сам по себе не означает физический формат.

## Ограничение pixel-scale

Метод предполагает, что upstream не изменяет масштаб самого изображения письма между кадрами. Изменение только количества чёрного поля допустимо. Если разные режимы камеры дополнительно resize-ят содержимое, `px/mm` разных эталонов начнут расходиться; `metric_format.consensus.consistent=false` должен выявить такой случай.

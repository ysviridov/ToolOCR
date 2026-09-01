# Stage 2.2 — training adapter для postcode crop

## Проблема

`postcode_stencil` detector нормирует допустимые размеры верхних плашек относительно полного canonical-frame письма. Tight crop индексного блока нельзя подавать как полный кадр: одна плашка занимает слишком большую долю ширины/высоты и отбрасывается до structural matching.

На реальном crop 840×264 px верхние плашки имеют размер примерно 75–77×20–21 px. Если считать crop полным кадром, ширина плашки составляет около 9%, тогда как production detector ожидает не более 5.5% полной ширины.

Production detector и его thresholds для решения этой задачи не изменяются.

## `postcode_crop_virtual_canonical_v1`

Training-only adapter помещает неизменённый upright postcode crop в нижнюю левую часть виртуального landscape canonical-frame:

```text
virtual width  = crop_width × 2.20
virtual height = virtual_width / 1.42
bottom margin  = около 1% virtual height
```

Search rectangle остаётся ровно исходным crop. Белое окружение нужно только для восстановления системы относительных координат, в которой работает production detector.

После этого применяется неизменённый pipeline:

```text
postcode_stencil detector
→ confirmed 7 upper bars / start marker
→ six digit-cell geometry
→ stencil_dot_suppression_v1
→ 96×128 digit canvas
```

Исходный crop не масштабируется, не поворачивается и не сохраняется в изменённом виде.

## Mixed exporter

`scripts/export_postcode_training_dataset_mixed.py` сначала пытается обработать полное письмо. При неудаче ищется `<source_stem>_crop.*`, и fallback выполняется через `postcode_crop_virtual_canonical_v1`.

`summary.json` использует schema `toolocr.postcode-digit-dataset.v4` и фиксирует:

```text
postcode_crops.adapter = postcode_crop_virtual_canonical_v1
successful_full_envelope
successful_postcode_crop_fallback
```

Split train/val/test строится после обоих путей по успешно извлечённым исходным письмам.

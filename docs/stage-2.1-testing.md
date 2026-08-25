# Stage 2.1 — проверка геометрии на реальных фотографиях

Этот этап проверяет только геометрию полного письма: внешний четырёхугольник, perspective rectification и подбор кандидатов профилей ГОСТ Р 51506-99. OCR на этих endpoint не запускается.

## Обновление ветки

```bash
cd /opt/ToolOCR
git fetch origin
git switch stage-2.1-layout
git pull

docker compose up -d --build ocr
```

Проверка сервиса:

```bash
make ocr-health
```

Swagger:

```text
http://<host>:8090/docs
```

## 1. Каталог профилей

```bash
make layout-profiles | jq
```

На текущем инкременте опубликовано 16 профилей внутренних отправлений:

- C6, DL, C5: исполнение I/II, без окна и с окном;
- C4, B4: исполнение I/II, без окна.

Профили соответствуют физическим форматам ГОСТ и вариантам оформления из обязательных приложений А/Б. На этом шаге координаты адресных ROI ещё не применяются: сначала требуется стабилизировать поиск полного конверта и perspective rectification.

## 2. Анализ фотографии

```bash
make layout-smoke FILE=/path/photo.jpg | jq
```

или напрямую:

```bash
curl -sS -X POST \
  'http://localhost:8090/v1/layout/analyze' \
  -F 'file=@/path/photo.jpg' | jq
```

Ключевые поля ответа:

```json
{
  "layout_status": "detected",
  "detector": {
    "method": "foreground_otsu",
    "confidence": 0.91,
    "area_ratio": 0.57,
    "rectangularity": 0.96,
    "angle_score": 0.93,
    "quad_order": ["TL", "TR", "BR", "BL"],
    "quad": []
  },
  "rectified": {
    "width_px": 2310,
    "height_px": 1632,
    "landscape": true
  },
  "orientation": {
    "status": "pending_profile_scoring",
    "hypotheses_deg": [0, 180]
  },
  "format_status": "ambiguous_by_ratio",
  "format_candidates": [],
  "profile_candidates": [],
  "timing": {
    "decode_ms": 0,
    "detect_ms": 0,
    "rectify_ms": 0,
    "profile_ms": 0,
    "total_ms": 0
  }
}
```

Числа в примере условные.

### detector.method

Detector использует два независимых канала.

`foreground_otsu` — основной режим для камеры сортировщика, когда белый/светлый конверт лежит на тёмной транспортной ленте. Сначала строится яркостная маска листа, поэтому внутренние марки, штемпели, строки и штрихкоды не должны перехватывать роль внешнего контура.

`contour_approx` — второй канал на основе Canny. Он нужен для кадров, где письмо и фон хуже разделяются по яркости, но внешние края выражены достаточно хорошо.

Если точный выпуклый четырёхугольник не найден, допускается резервный `minAreaRect`. В `method` это видно как `foreground_otsu_min_area_rect` либо `contour_approx_min_area_rect`. Confidence такого результата специально ограничен; такие случаи желательно проверять визуально.

### format_status

`resolved_by_ratio` возможен прежде всего для DL, поскольку его отношение сторон около 2:1.

Для C6/C5/C4/B4 нормальным результатом является `ambiguous_by_ratio`: эти форматы имеют очень близкое отношение сторон. Окончательный выбор будет выполнен на следующем шаге по стандартным якорям ГОСТ.

### orientation

На сортировщике письмо может быть снято вверх ногами. Поэтому после rectification ориентация пока не фиксируется: обе гипотезы `0°` и `180°` считаются допустимыми. Это не ошибка. Выбор ориентации будет добавлен вместе с profile scoring по расположению марки, адресных зон и кодового штампа.

## 3. Сохранение rectified JPEG

```bash
make layout-rectify FILE=/path/photo.jpg
```

По умолчанию результат:

```text
/tmp/toolocr-rectified.jpg
```

Другой путь:

```bash
OUT=/tmp/photo-001-rectified.jpg make layout-rectify FILE=/path/photo.jpg
```

Проверять нужно:

1. в кадр попал весь конверт, а не внутренний прямоугольник/этикетка;
2. углы изображения соответствуют углам письма;
3. перспектива устранена;
4. длинная сторона стала горизонтальной;
5. содержимое не должно обязательно оказаться текстом вверх — поворот 180° будет решаться отдельным этапом;
6. края письма не должны заметно обрезаться.

## 4. Debug overlay

Для проблемного случая можно запросить две JPEG-картинки прямо внутри JSON:

```bash
curl -sS -X POST \
  'http://localhost:8090/v1/layout/analyze?include_debug_images=true' \
  -F 'file=@/path/photo.jpg' \
  -o /tmp/layout.json

jq -r '.debug_images.overlay_jpeg_base64' /tmp/layout.json \
  | base64 -d > /tmp/overlay.jpg

jq -r '.debug_images.rectified_jpeg_base64' /tmp/layout.json \
  | base64 -d > /tmp/rectified.jpg
```

На `overlay.jpg` точки подписаны в порядке `TL/TR/BR/BL`.

## 5. Layout reject

Если внешний четырёхугольник полного письма не найден, endpoint возвращает HTTP 422:

```json
{
  "detail": {
    "layout_status": "reject",
    "reason": "envelope_quad_not_found"
  }
}
```

`make layout-smoke` использует `curl --fail-with-body`, поэтому при HTTP 4xx/5xx тело диагностического ответа теперь также выводится в терминал.

Для production layout reject правильнее, чем продолжать OCR по случайному crop.

Для диагностического теста порог минимальной площади можно временно изменить:

```bash
curl -sS -X POST \
  'http://localhost:8090/v1/layout/analyze?min_area_ratio=0.10' \
  -F 'file=@/path/photo.jpg' | jq
```

В production значение следует зафиксировать после тестов на реальной камере сортировщика.

## 6. Что собрать по результатам теста

Для настройки detector полезно сохранить по проблемным фотографиям:

- исходное изображение;
- JSON `/v1/layout/analyze`;
- `overlay.jpg`;
- `rectified.jpg`;
- краткое замечание: `не найден контур`, `выбрана этикетка`, `обрезан край`, `искажена перспектива` и т. п.

После стабилизации этого слоя следующий инкремент — scoring профилей ГОСТ и автоматический выбор `формат + исполнение + окно + ориентация 0/180`, после чего можно проецировать адресные ROI в пиксели.

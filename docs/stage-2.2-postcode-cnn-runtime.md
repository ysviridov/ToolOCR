# Stage 2.2 — ONNX runtime распознавания почтового индекса

## Назначение

После validation/test CNN-модель `postcode_digit_v1.onnx` подключается как основной recognizer шести digit-cell почтового индекса.

Не меняются:

- canonical orientation;
- postcode stencil detector;
- геометрия шести digit-cell;
- `stencil_dot_suppression_v1`;
- размер входного canvas `96×128`.

Runtime получает точно тот же grayscale canvas, на котором обучалась модель, и нормализует его как `ink=(255-gray)/255`.

## Установка модели

Бинарные модели не хранятся в Git. После обучения:

```bash
cd /opt/ToolOCR
mkdir -p models
cp .toolocr-training/postcode-v1/model/postcode_digit_v1.onnx \
  models/postcode_digit_v1.onnx
```

`docker compose` монтирует каталог:

```text
./models -> /app/models:ro
```

Модель по умолчанию читается из:

```text
/app/models/postcode_digit_v1.onnx
```

## Настройки

```text
POSTCODE_RECOGNIZER_ENGINE=onnx
POSTCODE_ONNX_MODEL=/app/models/postcode_digit_v1.onnx
POSTCODE_ONNX_FALLBACK_TESSERACT=1
```

`POSTCODE_RECOGNIZER_ENGINE`:

- `onnx` — ONNX основной recognizer;
- `tesseract` — принудительный baseline;
- `auto` — ONNX при наличии model, иначе Tesseract.

При `POSTCODE_ONNX_FALLBACK_TESSERACT=1` ошибка загрузки модели или отдельного ONNX inference не делает индекс недоступным: runtime использует Tesseract fallback. В metadata это видно по `engine`.

Для строгого теста ONNX без fallback:

```text
POSTCODE_ONNX_FALLBACK_TESSERACT=0
```

## Runtime pipeline

```text
canonical image
→ postcode stencil
→ 6 digit cells
→ stencil_dot_suppression_v1
→ 96×128 grayscale canvas
→ ink=(255-gray)/255
→ cv2.dnn.readNetFromONNX()
→ logits 0..9
→ softmax
→ digit + probability + top3
→ 6-digit postcode
```

Текущая модель экспортирована с batch=1. Runtime кэширует один `cv2.dnn.Net` и выполняет шесть последовательных `forward()`. Повторная загрузка model для каждой цифры не выполняется.

Доступ к `setInput()/forward()` защищён lock, поскольку OpenCV `Net` хранит input внутри объекта.

## Metadata

`GET /v1/test-ui/images/{id}/roi/meta` возвращает для индекса:

```json
{
  "status": "recognized",
  "postcode": "129344",
  "confidence": 0.9981,
  "min_digit_confidence": 0.9932,
  "geometric_mean_confidence": 0.9980,
  "structurally_valid": true,
  "engine": "onnx_postcode_digit_v1+stencil_dot_suppression_v1",
  "model_path": "/app/models/postcode_digit_v1.onnx",
  "digits": [
    {
      "index": 1,
      "digit": "1",
      "confidence": 0.9991,
      "engine": "onnx",
      "top3": [
        {"digit": "1", "probability": 0.9991},
        {"digit": "7", "probability": 0.0005},
        {"digit": "4", "probability": 0.0002}
      ]
    }
  ]
}
```

`confidence` — средняя probability шести выбранных цифр. Дополнительно доступны minimum и geometric mean. До накопления production-корпуса эти значения не используются как hard gate.

Индекс, начинающийся с `0`, сохраняется в raw recognition, но получает `structurally_valid=false`.

## Tesseract fallback

Tesseract не удаляется. Возможные значения общего `engine`:

```text
onnx_postcode_digit_v1+stencil_dot_suppression_v1
tesseract_single_digit+stencil_dot_suppression_v1
tesseract_fallback_from_onnx+stencil_dot_suppression_v1
onnx_postcode_digit_v1+stencil_dot_suppression_v1+tesseract_fallback
```

Последний вариант означает, что часть цифр прошла ONNX, а для отдельной ошибки inference использован Tesseract.

## Проверка unit-tests

```bash
cd /opt/ToolOCR
git pull

docker compose run --rm --no-deps \
  -v "$PWD:/src:ro" \
  -e PYTHONPATH=/src \
  ocr sh -lc \
  'pip install -q pytest && pytest -q -p no:cacheprovider \
    /src/tests/test_postcode_runtime.py \
    /src/tests/test_postcode_recognizer.py \
    /src/tests/test_postcode_crop_training_adapter.py \
    /src/tests/test_postcode_digit_cells.py'
```

## Запуск OCR service

После копирования model:

```bash
cd /opt/ToolOCR
docker compose up -d --build ocr
```

Проверить, что модель видна внутри контейнера:

```bash
docker compose exec ocr sh -lc \
  'ls -lh /app/models/postcode_digit_v1.onnx'
```

После этого открыть Test UI и проверить ROI для писем с известным индексом. В `/roi/meta` основной критерий подключения ONNX:

```text
engine = onnx_postcode_digit_v1+stencil_dot_suppression_v1
```

а у каждой цифры должны присутствовать `confidence`, `top3` и `engine=onnx`.

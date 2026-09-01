# Runtime-модели ToolOCR

Каталог предназначен для локальных ONNX-моделей production/runtime. Бинарные модели не хранятся в Git.

Для Stage 2.2 ожидается файл:

```text
models/postcode_digit_v1.onnx
```

После обучения его можно скопировать из training-каталога:

```bash
mkdir -p models
cp .toolocr-training/postcode-v1/model/postcode_digit_v1.onnx models/
```

`docker compose` монтирует `./models` в OCR-контейнер как `/app/models:ro`.

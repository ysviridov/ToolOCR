# Stage 2.2 — Postcode challenge-set и A/B ONNX

## Назначение

Challenge-set нужен для проверки обобщающей способности postcode CNN на письмах, которые не входят в train/val/test. Challenge нельзя использовать для обучения, выбора split или ручной коррекции training manifest: иначе он перестаёт быть независимой проверкой.

Фиксированный набор `challenge-v1` хранится в:

```text
config/postcode-challenges/challenge-v1.csv
```

Формат минимальный:

```csv
filename,postcode
example.jpg,123456
```

`postcode` должен содержать ровно шесть цифр и не может начинаться с `0`.

## Что делает инструмент

`scripts/evaluate_postcode_challenge.py` работает с файлами, уже загруженными в Test UI volume. Для каждого `filename` он:

1. находит внутренний Test UI UUID;
2. запускает обычный layout pipeline с `AUTO` format;
3. строит canonical image;
4. запускает production postcode stencil detector;
5. получает frozen six-cell geometry;
6. вызывает тот же `_normalize_digit_crop_with_debug()`, который использует runtime;
7. сохраняет шесть grayscale canvas `96×128` после `stencil_dot_suppression_v1`;
8. прогоняет одну или несколько ONNX-моделей по этим же canvas;
9. формирует per-digit, per-file и aggregate отчёты.

Таким образом A/B моделей не зависит от повторной детекции: v1 и v2 получают идентичные входные canvas.

## Запуск текущей модели

Из `/opt/ToolOCR`:

```bash
mkdir -p .toolocr-training

docker compose run --rm --no-deps -T \
  -v "$PWD:/src:ro" \
  -v "$PWD/.toolocr-training:/work" \
  -e PYTHONPATH=/src \
  ocr python /src/scripts/evaluate_postcode_challenge.py \
    --challenge /src/config/postcode-challenges/challenge-v1.csv \
    --output /work/postcode-challenges/challenge-v1 \
    --model current=/app/models/postcode_digit_v1.onnx
```

Compose service `ocr` уже подключает persistent Test UI volume и `./models:/app/models:ro`, поэтому отдельные mount для исходных писем и runtime модели не нужны.

## A/B v1 против v2

Кандидатную модель следует хранить вне runtime `./models`, например в `.toolocr-training/postcode-v2/model/`. Каталог `.toolocr-training` в команде выше доступен как `/work`.

```bash
docker compose run --rm --no-deps -T \
  -v "$PWD:/src:ro" \
  -v "$PWD/.toolocr-training:/work" \
  -e PYTHONPATH=/src \
  ocr python /src/scripts/evaluate_postcode_challenge.py \
    --challenge /src/config/postcode-challenges/challenge-v1.csv \
    --output /work/postcode-challenges/challenge-v1-ab \
    --model v1=/app/models/postcode_digit_v1.onnx \
    --model v2=/work/postcode-v2/model/postcode_digit_v2.onnx
```

Первый `--model` считается baseline при построении `model_deltas.csv`.

## Артефакты

В `--output` создаются:

```text
canvases/
  <file_id>__<filename-stem>/
    digit_1_truth_X.png
    ...
    digit_6_truth_X.png
preprocess.jsonl
digits.csv
files.csv
model_deltas.csv
summary.json
```

`digits.csv` содержит `truth`, `predicted`, `correct`, `confidence`, `top3` и путь к точному runtime canvas. Это основной файл для анализа confusion `5/8/9/2`.

`files.csv` содержит итоговый шестизначный `predicted`, exact correctness, число правильных цифр, mean/min confidence.

`model_deltas.csv` показывает, исправил или сломал candidate каждый challenge-файл относительно baseline: `exact_delta=+1` — исправление, `-1` — регрессия.

`summary.json` содержит aggregate exact-postcode accuracy и digit accuracy по каждой модели, а также extraction failures.

## Правило использования

`challenge-v1` не добавляется в training manifest. Если требуется hard-negative training, в train добавляются другие новые письма с аналогичными вариантами написания цифр. Эти четыре challenge-файла остаются неизменной контрольной группой.

Модель-кандидат нельзя принимать только потому, что она исправила challenge-v1. Она должна одновременно не ухудшать исходные val/test метрики и подтверждаться на новых независимых письмах.

# Stage 2.2 — обучение классификатора цифр почтового индекса

## Цель

`stencil_dot_suppression_v1` формирует стабильные 96×128 binary-canvas отдельных цифр. Tesseract остаётся baseline; следующий recognizer — компактный 10-class CNN (`0..9`).

Обучение выполняется отдельно от production OCR. После обучения модель экспортируется в ONNX и проверяется через `cv2.dnn.readNetFromONNX()`.

Production OCR не переключается на CNN автоматически: сначала оцениваются validation и полностью независимый test holdout.

## Ground truth

Минимальный CSV:

```csv
filename,postcode
letter001.jpg,124460
letter002.jpg,119160
```

Обязательны только `filename` и `postcode`. Дополнительные поля разрешены, но training exporter их не использует как hard-filter. Поэтому в одном корпусе можно обучать рукописные и печатные цифры, если они находятся в том же трафаретном индексном блоке.

`postcode` валидируется по `^[1-9][0-9]{5}$`. Loader поддерживает UTF-8/UTF-8 BOM/CP1251 и `,`/`;`.

Локальные training-данные хранятся под `.toolocr-training/`; каталог исключён из Git.

## Организация корпуса

Рекомендуемая структура:

```text
.toolocr-training/postcode-v1/
  ground_truth.csv
  images/
    letter001.jpg
    letter002.jpg
    ...
  dataset/
  model/
```

Exporter ищет изображения рекурсивно. В CSV можно указывать относительный путь (`group-a/letter001.jpg`) либо уникальное basename. Если одинаковый basename встречается в нескольких подкаталогах, строка считается неоднозначной и попадёт в `failures`.

## Режимы входа

### `full-envelope`

Используется полное изображение письма:

```text
full image
→ layout / rectification
→ orientation / canonical
→ postcode stencil
→ 6 digit-cell
→ stencil_dot_suppression_v1
→ 96×128
```

По умолчанию формат определяется в AUTO. Для однородного корпуса можно задать `--expected-format DL|C5|C4`.

Если orientation остаётся ambiguous, письмо не используется и причина сохраняется в `summary.json`.

### `postcode-crop`

Используется заранее подготовленный crop всего индексного блока. Crop должен быть уже в нормальной ориентации:

```text
индекс читается слева направо
верхние stencil-bars находятся сверху
первая цифра слева, шестая справа
```

Не нужно вручную вырезать шесть цифр. ToolOCR сам применяет существующий postcode stencil detector, строит 6 digit-cell и запускает тот же `stencil_dot_suppression_v1`.

Для сохранения production `row_y_norm` rescue-инварианта upright crop помещается в нижнюю треть нейтрального training-canvas; содержимое search-zone и thresholds detector не меняются.

## 1. Экспорт digit-canvas из полных писем

```bash
cd /opt/ToolOCR
git pull

mkdir -p .toolocr-training/postcode-v1/dataset

docker compose run --rm --no-deps \
  -v "$PWD:/src" \
  -e PYTHONPATH=/src \
  ocr python /src/scripts/export_postcode_training_dataset.py \
    --ground-truth /src/.toolocr-training/postcode-v1/ground_truth.csv \
    --images-dir /src/.toolocr-training/postcode-v1/images \
    --output-dir /src/.toolocr-training/postcode-v1/dataset \
    --input-mode full-envelope \
    --expected-format auto
```

Для однородного C4-корпуса можно заменить `auto` на `C4`.

## 2. Экспорт из upright postcode-crops

Если подготовлены отдельные crop индексных блоков с теми же именами файлов:

```bash
docker compose run --rm --no-deps \
  -v "$PWD:/src" \
  -e PYTHONPATH=/src \
  ocr python /src/scripts/export_postcode_training_dataset.py \
    --ground-truth /src/.toolocr-training/postcode-v1/ground_truth.csv \
    --images-dir /src/.toolocr-training/postcode-v1/postcode-crops \
    --output-dir /src/.toolocr-training/postcode-v1/dataset \
    --input-mode postcode-crop
```

Результат:

```text
dataset/
  manifest.csv
  summary.json
  samples/
    <filename>__d1__y1.png
    ...
```

Exporter сохраняет письмо только если удалось получить все шесть digit-canvas.

## Train / validation / test

Split выполняется строго по исходным `filename`.

По умолчанию:

```text
train 70%
val   15%
test  15%
```

Все шесть цифр одного письма всегда находятся только в одном split. Randomized split дополнительно стремится сохранить распределение цифр и покрытие классов в holdout-наборах.

`validation` используется для выбора best checkpoint и early stopping. `test` не участвует в выборе модели и оценивается один раз после восстановления best validation checkpoint.

## 3. Проверка exporter

```bash
cd /opt/ToolOCR

docker compose run --rm --no-deps \
  -v "$PWD:/src:ro" \
  -e PYTHONPATH=/src \
  ocr sh -lc \
  'pip install -q pytest && pytest -q \
    /src/tests/test_postcode_training_dataset.py \
    /src/tests/test_postcode_recognizer.py \
    /src/tests/test_postcode_digit_cells.py'
```

## 4. Обучение CNN

Training-зависимости не входят в production image. PyTorch CPU устанавливается только в одноразовом training-container.

```bash
cd /opt/ToolOCR

docker compose run --rm --no-deps \
  -v "$PWD:/src" \
  -e PYTHONPATH=/src \
  ocr sh -lc '
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu &&
    pip install -q -r /src/ocr/requirements-training.txt &&
    python /src/scripts/train_postcode_cnn.py \
      --manifest /src/.toolocr-training/postcode-v1/dataset/manifest.csv \
      --output-dir /src/.toolocr-training/postcode-v1/model \
      --model-name postcode_digit_v1 \
      --cpu
  '
```

По умолчанию выполняется до 80 эпох с early stopping. Train получает мягкие rotation/translation/scale, небольшую вариацию толщины штриха и редкие остаточные stencil-dots. Validation и test идут без аугментации.

Из-за дисбаланса классов применяется ограниченный `sqrt-balanced` class weighting.

## 5. Метрики и артефакты

Результат:

```text
model/
  postcode_digit_v1.pt
  postcode_digit_v1.onnx
  metrics.json
  history.csv
  confusion_val_best.csv
  confusion_test.csv
```

Основные метрики:

```text
digit_accuracy
exact_postcode_accuracy
```

`exact_postcode_accuracy` считается только когда для исходного письма присутствуют все шесть цифр.

Best checkpoint выбирается только по validation: сначала `exact_postcode_accuracy`, затем `digit_accuracy`, затем loss. После выбора выполняется независимая оценка test и результат пишется в `final_test` внутри `metrics.json`.

После ONNX export скрипт сравнивает PyTorch/OpenCV DNN на validation sample. Если argmax отличается, run завершается ошибкой.

## Перед подключением в production

Нужно проверить:

```text
summary.json: достаточно successful_files, непустые train/val/test
metrics.json: best_validation и final_test
confusion_test.csv: систематические путаницы цифр
ONNX validation: same_argmax=true
```

Только после этого ONNX подключается вместо Tesseract. ROI, stencil detector, digit-cell geometry и `stencil_dot_suppression_v1` при этом не меняются.

# Stage 2.2 — обучение классификатора цифр почтового индекса

## Цель

Текущий `stencil_dot_suppression_v1` формирует стабильные 96×128 binary-canvas отдельных цифр. Tesseract используется только как baseline и плохо распознаёт специальное рукописное начертание цифр в индексном штампе.

Следующий recognizer — компактный 10-class CNN (`0..9`). Обучение выполняется отдельно от production OCR. После обучения модель экспортируется в ONNX и проверяется загрузкой через `cv2.dnn.readNetFromONNX()`.

Production OCR не переключается на CNN автоматически: сначала оцениваются `digit_accuracy` и `exact_postcode_accuracy` на holdout-письмах.

## Ground truth

Для обучения достаточно полей:

```text
filename
format
postcode
postcode_source
```

Адресные поля в training pipeline не читаются и в репозиторий не сохраняются.

Loader поддерживает UTF-8 и Windows-1251, разделитель `;` или `,`, а также нормализует кириллическую `С4` в латинскую `C4`.

В обучение допускаются только строки:

```text
format == C4
postcode_source == stencil
postcode matches ^[1-9][0-9]{5}$
```

Строки `printed` исключаются.

Локальный ground truth и training artifacts хранятся под `.toolocr-training/`, который добавлен в `.gitignore`.

## 1. Экспорт digit-canvas

Положить ground truth локально:

```bash
cd /opt/ToolOCR
mkdir -p .toolocr-training
cp /path/to/C4-ground_truth.csv .toolocr-training/C4-ground_truth.csv
```

Обновить OCR-контейнер:

```bash
git pull
docker compose up -d --build ocr
```

Экспортировать датасет из test UI Docker volume:

```bash
docker compose run --rm --no-deps \
  -v "$PWD:/src" \
  -e PYTHONPATH=/src \
  ocr python /src/scripts/export_postcode_training_dataset.py \
    --ground-truth /src/.toolocr-training/C4-ground_truth.csv \
    --test-data-dir /app/test-data \
    --output-dir /src/.toolocr-training/c4-dataset
```

Exporter:

1. сопоставляет `filename` с sidecar metadata в `toolocr_testdata`;
2. выполняет FIXED C4 layout + canonicalization;
3. находит postcode stencil и 6 digit-cell;
4. применяет тот же `stencil_dot_suppression_v1`, что используется runtime;
5. сохраняет финальные 96×128 canvas;
6. формирует train/val split строго по исходным письмам;
7. пишет `manifest.csv` и `summary.json`.

Результат:

```text
.toolocr-training/c4-dataset/
  manifest.csv
  summary.json
  samples/
    <filename>__d1__y1.png
    ...
```

Если layout/orientation/digit geometry конкретного письма не позволяют получить все 6 цифр, письмо целиком не используется и причина фиксируется в `summary.json`.

## 2. Обучение CNN

Training-зависимости не входят в production Docker image. Для одноразового training container устанавливается CPU PyTorch.

```bash
cd /opt/ToolOCR

docker compose run --rm --no-deps \
  -v "$PWD:/src" \
  -e PYTHONPATH=/src \
  ocr sh -lc '
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu &&
    pip install -q -r /src/ocr/requirements-training.txt &&
    python /src/scripts/train_postcode_cnn.py \
      --manifest /src/.toolocr-training/c4-dataset/manifest.csv \
      --output-dir /src/.toolocr-training/c4-model \
      --cpu
  '
```

По умолчанию выполняется до 80 эпох с early stopping. Для train применяются мягкие аугментации: небольшие rotation/translation/scale, вариация толщины штриха и редкие остаточные stencil dots.

Из-за дисбаланса классов используется `sqrt-balanced` weighting, ограниченный сверху, чтобы редкие классы не доминировали.

## 3. Метрики и артефакты

Результат:

```text
.toolocr-training/c4-model/
  postcode_digit_c4.pt
  postcode_digit_c4.onnx
  metrics.json
  history.csv
  confusion_best.csv
```

Основные метрики:

```text
digit_accuracy
exact_postcode_accuracy
```

`exact_postcode_accuracy` считается по шести цифрам одного исходного письма и является главной метрикой для принятия модели.

Train/validation leakage запрещён: все 6 цифр одного письма всегда находятся только в одном split.

После ONNX export скрипт загружает модель через OpenCV DNN и сравнивает argmax PyTorch/OpenCV на validation sample. При несовпадении training run завершается ошибкой.

## Ограничения первого C4 корпуса

Первый C4 ground truth содержит небольшой объём реальных примеров и заметный дисбаланс цифр. Поэтому первая модель — baseline для проверки архитектуры. Расширение ground truth новыми C4/C5/DL письмами должно выполняться до фиксации production-модели.

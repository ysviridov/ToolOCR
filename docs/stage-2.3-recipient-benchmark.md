# Stage 2.3.1 — benchmark RECIPIENT ROI

## Назначение

Stage 2.3.1 фиксирует измеримый baseline текущего `recipient_address` detector до изменения его алгоритма.

RECIPIENT — адресный блок получателя в нижней правой части canonical-изображения письма. В дальнейшем он будет источником для OCR/LLM распознавания адреса, распознавания дубликата индекса назначения и cross-check с трафаретным индексом.

Текущий detector в `ocr/app/roi.py` использует широкую GOST-guided search zone и общий bbox подходящих foreground connected-components. На этом этапе алгоритм detector **не меняется**.

## Почему ground truth отделён от detector

Bootstrap может использовать текущий зелёный RECIPIENT bbox только как стартовую подсказку для ручной разметки. Каждая bootstrap-строка получает:

```text
ground_truth_status=needs_review
```

Такие строки evaluator игнорирует. В метрики попадают только строки:

```text
ground_truth_status=verified
```

Это не позволяет получить круговую оценку detector по его собственным bbox.

## Формат CSV

```csv
filename,format,ground_truth_status,recipient_status,x,y,width,height,postcode_box_status,postcode_x,postcode_y,postcode_width,postcode_height,notes
```

Координаты задаются в canonical coordinate space и нормализованы в диапазон `0..1`.

`format`:

```text
DL | C5 | C4
```

`ground_truth_status`:

```text
needs_review — bootstrap/незавершённая ручная разметка, не считать
verified     — ручная разметка проверена, считать в benchmark
exclude      — сознательно исключить изображение из benchmark
```

`recipient_status`:

```text
present — RECIPIENT существует; verified требует x/y/width/height
absent  — RECIPIENT отсутствует; bbox должен быть пустым
```

`postcode_box_status` относится к отдельной квадратной/прямоугольной рамке с рукописным дубликатом индекса назначения:

```text
present | absent | unknown
```

При `present` verified-строка должна содержать `postcode_x/postcode_y/postcode_width/postcode_height`.

Stage 2.3.1 сохраняет эту разметку для будущего detector рамки, но текущий `recipient_address` detector её отдельно не выделяет, поэтому baseline не вычисляет postcode-box detection rate.

## Bootstrap из Test UI

### Визуальный редактор (рекомендуемый способ)

В Test UI нажмите **«Разметка RECIPIENT»**, либо откройте `/test-ui/recipient`.

1. Выберите папку, например `C4-Рукописные`, и формат C4; нажмите «Открыть папку».
2. На canonical-изображении мышью обведите адресный блок. Перетаскивание внутри рамки перемещает её; перетаскивание угла меняет размер. Новый прямоугольник рисуется с пустого места. Масштабирование и прокрутка позволяют работать с мелким текстом.
3. При необходимости выберите «Рамка индекса» и обведите дубликат индекса. Для отсутствующих блоков используйте соответствующие статусы.
4. «Взять рамку детектора» переносит текущую подсказку только в редактируемый черновик. Проверьте и исправьте её вручную.
5. «Сохранить черновик» сохраняет `needs_review`; «Проверено → далее» сохраняет `verified` и открывает следующее письмо. «Исключить» сохраняет `exclude`.
6. «Скачать разметку CSV» выгружает сохранённые записи открытой папки/формата в формате evaluator. Вручную редактировать координаты в CSV не требуется.

Разметка хранится в `TEST_UI_STORAGE_DIR/recipient-annotations/` в существующем volume `toolocr_testdata`, отдельно для каждого file ID и формата. Пересборка контейнера её не удаляет. Bootstrap CSV для работы редактора не нужен; импорт ранее созданного CSV в этой версии не предусмотрен.

Редактор использует production canonicalization с FIXED format. При неразрешённой ориентации/отсутствующем quad показывает ошибку вместо разметки в неверной системе координат. Хеш canonical pixels проверяется при сохранении и экспорте verified записей. Если canonical изменился, письмо надо открыть и разметить заново. Экспорт проверяет уникальность имён во всей библиотеке, поскольку evaluator ищет изображения по имени.

Скачанный CSV перенесите на сервер в `.toolocr-training/recipient-benchmarks/recipient-v1.csv` и запустите evaluator по инструкции ниже. Экспорт проверяет canonical для каждого verified письма, поэтому на большой коллекции может занять время. Редактор не запускает evaluator автоматически.

### CLI bootstrap (необязательный альтернативный способ)

Bootstrap предназначен для однородной Test UI folder одного формата.

Пример для C5:

```bash
cd /opt/ToolOCR
mkdir -p .toolocr-training/recipient-benchmarks

docker compose run --rm --no-deps -T \
  -v "$PWD:/src:ro" \
  -v "$PWD/.toolocr-training:/work" \
  -e PYTHONPATH=/src \
  ocr python /src/scripts/bootstrap_recipient_benchmark.py \
    --folder "C5" \
    --format C5 \
    --output /work/recipient-benchmarks/recipient-v1.csv \
    --overlays /work/recipient-benchmarks/bootstrap-overlays
```

Скрипт:

1. находит изображения Test UI folder по имени;
2. запускает тот же production layout/orientation wiring;
3. использует `FIXED` expected format, чтобы benchmark RECIPIENT не зависел от AUTO format detection;
4. строит canonical image;
5. запускает текущий `detect_simple_mail_rois()`;
6. переносит текущий RECIPIENT bbox в CSV как normalized seed;
7. сохраняет overlay `CURRENT RECIPIENT - NOT GROUND TRUTH`;
8. ставит каждой строке `needs_review`.

Seed bbox необходимо проверить вручную. После проверки:

- поправить координаты RECIPIENT, если нужно;
- отметить `recipient_status`;
- при возможности разметить destination postcode box;
- поставить `ground_truth_status=verified`;
- сложные/неоднозначные случаи можно временно оставить `needs_review` или поставить `exclude` с причиной в `notes`.

## Что считать границей RECIPIENT

GT bbox должен включать полезное содержимое блока назначения целиком:

- печатную подпись `Кому` и заполненное поле;
- печатную подпись `Куда` и адрес назначения;
- строку/подпись индекса места назначения;
- рукописный дубликат индекса и его рамку, если они пространственно являются частью адресного блока.

Небольшой внешний margin допустим. Нельзя обрезать полезную строку только ради более высокого IoU.

Не следует расширять GT bbox до соседнего независимого barcode, штемпеля, логотипа или другого постороннего блока.

## Запуск evaluator

```bash
cd /opt/ToolOCR

docker compose run --rm --no-deps -T \
  -v "$PWD:/src:ro" \
  -v "$PWD/.toolocr-training:/work" \
  -e PYTHONPATH=/src \
  ocr python /src/scripts/evaluate_recipient_benchmark.py \
    --ground-truth /work/recipient-benchmarks/recipient-v1.csv \
    --output /work/recipient-benchmarks/recipient-v1-results
```

Evaluator обрабатывает только `verified`.

## Метрики

Для каждого positive GT:

```text
IoU
content_recall
undercrop_fraction = 1 - content_recall
overcrop_fraction
```

`content_recall` — доля площади GT RECIPIENT, покрытая bbox detector. Для будущего OCR эта метрика важнее симметричного IoU: обрезка полезного адресного текста опаснее небольшого внешнего margin.

`overcrop_fraction` — доля площади predicted bbox, лежащая вне GT.

Диагностический `outcome` по умолчанию:

```text
good        content_recall >= 0.95 и overcrop_fraction <= 0.35
too_small   потеря полезного GT content
too_large   слишком много посторонней площади
poor_fit    одновременно undercrop и overcrop
wrong_block IoU < 0.10 и content_recall < 0.25
miss        RECIPIENT не найден
```

Для negative GT доступны:

```text
correct_negative
false_positive
```

Пороги outcome — только удобная диагностическая классификация. Основные непрерывные метрики сохраняются независимо от неё.

## Артефакты

```text
recipient-v1-results/
├── files.csv
├── failures.csv
├── summary.json
├── overlays/
└── crops/
```

Overlay содержит:

```text
GT RECIPIENT          — ручной ground truth
CURRENT DETECTOR      — текущий recipient_address detector
GT DEST POSTCODE BOX  — ручная рамка дубликата индекса, если размечена
```

В `crops/` сохраняются GT и predicted crop для быстрого визуального сравнения.

## Критерий следующего шага

После накопления достаточного числа verified писем сначала фиксируется baseline текущего detector. Только затем начинается Stage 2.3.2: line/component grouping и candidate scoring.

Изменения detector принимаются только по сравнению с тем же неизменным verified benchmark, без подгонки GT под новый алгоритм.

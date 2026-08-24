# ToolOCR — Stage 1 / 1.1 / 2

`ToolOCR` использует PostgreSQL как версионируемое хранилище адресного справочника РФ и как первый слой валидации результатов OCR. Источник данных — ежемесячный полный снимок `ADDRESS_YYYYMMDDHHMMSS.zip`, предоставляемый заказчиком.

В Stage 1 реализованы схема БД, импорт и безопасное ежемесячное обновление. В Stage 1.1 добавлены фактические проверки адресов, OCR-нормализация, fuzzy-поиск и SQL/HTTP API `find_address_candidates()`. В Stage 2 добавлен отдельный сервис приёма изображений, предобработки и базового OCR.

## Архитектура текущего этапа

```text
ADDRESS_*.zip
     │
     ▼
 importer
     │
     ▼
PostgreSQL / toolocr
     │
     ├── versioned datasets
     ├── current_* views
     ├── pg_trgm fuzzy matching
     └── find_address_candidates()
                  │
                  ▼
             FastAPI
        /v1/address/candidates
```

## Почему используются версии наборов данных вместо TRUNCATE/COPY

Новый ежемесячный снимок загружается рядом с текущим активным набором данных. Во время загрузки и проверки производственные запросы продолжают работать со старой версией. Переключение выполняется только после успешной валидации нового набора данных и происходит одной короткой транзакцией. Предыдущая версия остаётся доступной для отката.

Код приложения обращается к активному набору через `toolocr.current_*` либо через `toolocr.find_address_candidates()` и не должен знать конкретный `dataset_id`.

## Запуск Stage 1

```bash
cp .env.example .env
# обязательно изменить POSTGRES_PASSWORD

docker compose up -d db
```

## Первый импорт

```bash
make import FILE=/absolute/path/ADDRESS_20260115223000.zip
```

Во время импорта выполняются:

1. PostgreSQL advisory lock, предотвращающий одновременный запуск двух импортов.
2. Защита от повторной загрузки одного снимка по SHA-256. Неудавшийся импорт разрешено повторить.
3. Проверка точных имён файлов и заголовков FS/DT/PC/MC/CT/SR/AR.
4. Потоковый `COPY` непосредственно из ZIP без предварительной распаковки всего архива.
5. Контроль минимально допустимого количества строк.
6. Проверка ссылочной целостности между таблицами.
7. `ANALYZE` после загрузки.
8. Атомарная активация только после успешной валидации.

## Ежемесячное обновление

Поместить новый архив в `data/incoming/` и выполнить:

```bash
./scripts/update_latest.sh
```

или указать архив явно:

```bash
make update FILE=/absolute/path/ADDRESS_20260215223000.zip
```

Если импорт или проверка завершатся ошибкой, текущий активный набор данных останется без изменений.

## Состояние наборов и откат

```bash
make datasets
make activate ID=1
KEEP=2 make prune
```

`prune` никогда не удаляет активный набор данных.

---

# Stage 1.1

## 1. Применение миграции к уже работающей БД

Если volume PostgreSQL был создан на Stage 1, init-скрипты повторно автоматически не запускаются. Поэтому после обновления файлов проекта миграцию Stage 1.1 нужно применить один раз:

```bash
make migrate-stage11
```

Миграция идемпотентна. Она создаёт/обновляет:

- `toolocr.ocr_norm_text(text)`;
- `toolocr.norm_house(text)`;
- `toolocr.house_leading_number(text)`;
- `toolocr.house_match_score(...)`;
- `toolocr.find_address_candidates(...)`;
- `toolocr.schema_migration` — журнал применённых миграций схемы.

При создании нового PostgreSQL volume файл `db/init/002_stage_1_1.sql` выполняется автоматически после `001_schema.sql`.

## 2. Фактическая проверка импортированной БД

После миграции выполнить:

```bash
make check-stage11
```

Скрипт `db/checks/stage_1_1_smoke.sql` проверяет:

- наличие активного snapshot;
- фактические объёмы активных таблиц;
- реальный адрес из предоставленного снимка: `142100 / Подольск / Кирова / дом 4`;
- fuzzy-ошибку `КНРОВА` вместо `КИРОВА`;
- смешение латиницы и кириллицы в OCR-строке;
- автоматические assertions для exact/fuzzy результата;
- планы выполнения сужения по индексу и fuzzy-поиска через `EXPLAIN (ANALYZE, BUFFERS)`.

В исходной выгрузке адресу `142100 / Подольск / Кирова / дом 4` соответствует, в частности, `IdAddress=467310`.

## 3. SQL-функция find_address_candidates()

Сигнатура:

```sql
toolocr.find_address_candidates(
    p_postal_code text DEFAULT NULL,
    p_city        text DEFAULT NULL,
    p_street      text DEFAULT NULL,
    p_house       text DEFAULT NULL,
    p_limit       integer DEFAULT 20
)
```

Пример точного поиска:

```sql
SELECT *
FROM toolocr.find_address_candidates(
    '142100',
    'Подольск',
    'Кирова',
    '4',
    10
);
```

Пример с типичной ошибкой OCR:

```sql
SELECT id_address, postal_code, matched_city_name, street_name,
       city_score, street_score, house_score, score
FROM toolocr.find_address_candidates(
    '142100',
    'Подольск',
    'КНРОВА',
    '4',
    10
);
```

`КНРОВА` должна дать кандидатом `КИРОВА`, поскольку сравнение названий выполняется через `pg_trgm`.

### Нормализация OCR

Перед fuzzy-сопоставлением запрос:

- переводится в верхний регистр;
- `Ё` приводится к `Е`;
- лишняя пунктуация и пробелы нормализуются;
- визуально похожие латинские буквы `A B C E H K M O P T X Y` приводятся к соответствующим кириллическим символам.

Например смешанная OCR-строка `KИРОВА` приводится к кириллическому варианту перед поиском.

### Формирование score

На Stage 1.1 используются веса:

| Признак | Вес |
|---|---:|
| почтовый индекс | 0.35 |
| населённый пункт | 0.25 |
| улица | 0.30 |
| дом | 0.10 |

Вес учитывается только если соответствующее поле было передано.

Индекс на этом этапе является сильным точным фильтром: если он передан, после удаления посторонних символов должно остаться ровно 6 цифр. Город и улица являются fuzzy-признаками. Дом влияет на рейтинг кандидата, но не удаляет строку из результатов жёстко — это сделано специально для последующего объединения нескольких OCR-гипотез.

Для дома:

- `1.0` — точное совпадение с границей диапазона;
- `0.95` — числовая часть попадает внутрь диапазона;
- `0.0` — диапазон не подтверждает дом.

## 4. HTTP API

API — тонкий слой над SQL-функцией. Запуск:

```bash
make api-up
```

Проверка:

```bash
curl http://localhost:8080/health
```

Swagger/OpenAPI:

```text
http://localhost:8080/docs
```

### GET

```bash
curl --get 'http://localhost:8080/v1/address/candidates' \
  --data-urlencode 'postal_code=142100' \
  --data-urlencode 'city=Подольск' \
  --data-urlencode 'street=КНРОВА' \
  --data-urlencode 'house=4' \
  --data-urlencode 'limit=10'
```

### POST

```bash
curl -X POST 'http://localhost:8080/v1/address/candidates' \
  -H 'Content-Type: application/json' \
  -d '{
    "postal_code": "142100",
    "city": "Подольск",
    "street": "КНРОВА",
    "house": "4",
    "limit": 10
  }'
```

Формат ответа:

```json
{
  "query": {
    "postal_code": "142100",
    "city": "Подольск",
    "street": "КНРОВА",
    "house": "4",
    "limit": 10
  },
  "count": 3,
  "timing": {
    "db_ms": 21.4,
    "total_ms": 23.7
  },
  "candidates": [
    {
      "id_address": 467310,
      "postal_code": "142100",
      "matched_city_name": "ПОДОЛЬСК ГОРОД",
      "street_name": "КИРОВА",
      "from_house_number": "4",
      "to_house_number": "4",
      "city_score": 1.0,
      "house_score": 1.0,
      "score": 0.87
    }
  ]
}
```

В примере показан только первый кандидат из ответа. Численные значения округлены для иллюстрации структуры. Фактические `street_score` и `score` определяются текущим снимком и `pg_trgm`.

### Время выполнения запроса

API версии `1.1.1` возвращает объект `timing`:

```json
"timing": {
  "db_ms": 21.4,
  "total_ms": 23.7
}
```

- `db_ms` — время от отправки SQL через `cursor.execute()` до получения всех строк `fetchall()`. Открытие соединения с PostgreSQL сюда не входит.
- `total_ms` — время обработки внутри endpoint: подключение к PostgreSQL, SQL-запрос, получение строк и подготовка объекта ответа. Финальная JSON-сериализация FastAPI и передача данных по сети клиенту сюда не входят.
- Для полного времени, наблюдаемого клиентом, используйте `curl -w '%{time_total}'`.

После изменения кода API достаточно пересобрать только сервис API:

```bash
docker compose up -d --build api
```

Базу данных перезапускать и повторно импортировать адресный snapshot не требуется.

Быстрая проверка API:

```bash
make api-smoke
# если порт изменён:
make api-smoke TOOLOCR_API_PORT=18080
```

Логи:

```bash
make api-logs
```

Остановка API без остановки PostgreSQL:

```bash
make api-down
```

## 5. Таблицы и представления

- `toolocr.dataset`, `toolocr.runtime_state` — реестр снимков и указатель активной версии.
- `toolocr.federal_subject` — FS, субъекты РФ.
- `toolocr.district` — DT, районы.
- `toolocr.postal_code` — PC, почтовые индексы.
- `toolocr.main_city` — MC, основные населённые пункты/объекты.
- `toolocr.city` — CT, подчинённые населённые пункты/объекты.
- `toolocr.street` — SR, улицы.
- `toolocr.address_range` — AR, индекс и идентификаторы адресных сущностей, диапазоны домов/строений.
- `toolocr.current_*` — представления активного снимка.

Расширения `pg_trgm` и `fuzzystrmatch` включены в Stage 1. `name_norm` хранится как generated column и индексируется GIN для fuzzy-поиска.

## 6. Доступ к БД вручную

```bash
make db-shell
```

Полезные запросы:

```sql
SELECT * FROM toolocr.active_dataset;
SELECT count(*) FROM toolocr.current_address_range;

SELECT s.id_street,
       s.street_name,
       similarity(s.name_norm, toolocr.ocr_norm_text('КНРОВА')) AS score
FROM toolocr.current_street s
WHERE s.name_norm % toolocr.ocr_norm_text('КНРОВА')
ORDER BY score DESC
LIMIT 20;
```

## 7. Производственная эксплуатация

- PostgreSQL должен иметь отдельное резервное копирование независимо от snapshot rollback.
- Ежемесячный архив следует публиковать в `data/incoming/` атомарно: сначала `.part`, после полной передачи — rename в `.zip`.
- `KEEP=2 make prune` выполнять только после проверки новой версии в production.
- Изменения схемы после инициализации volume оформлять идемпотентными миграциями и применять явно.
- HTTP API сейчас создаёт короткое подключение к PostgreSQL на запрос. Пул подключений целесообразно включить на этапе подключения параллельных OCR workers.

## 8. Ограничения Stage 1.1

Текущая функция — первый валидатор кандидатов, а не окончательный адресный решатель.

Пока намеренно не реализованы:

- несколько альтернативных OCR-вариантов индекса одновременно;
- исправление ошибочной цифры в индексе;
- отдельные веса OCR confidence от распознавателя;
- полноценный разбор `дом/корпус/строение` из одной строки;
- ранжирование по субъекту/району;
- кэширование популярных запросов;
- connection pool для API.

Следующий слой сможет принимать `top-N` гипотезы OCR и объединять их с `score` адресной БД.

---

# Stage 2 — приём изображения, предобработка и базовое OCR

На Stage 2 добавлен отдельный сервис `ocr`. Он не меняет адресную БД и не зависит от PostgreSQL: его задача — превратить входное изображение в стабильный OCR-контракт, который затем будет использоваться адресным парсером и валидатором.

Текущий baseline-движок — **Tesseract 5** с языками `rus+eng`. Он выбран как простой воспроизводимый open-source baseline. Внешний HTTP-контракт отделён от конкретного OCR-движка, поэтому позднее можно добавить PaddleOCR/ONNX как второй движок без изменения клиента.

## Архитектура Stage 2

```text
JPEG / PNG / BMP / TIFF
          │
          ▼
     toolocr-ocr :8090
          │
          ├── проверка размера
          ├── decode OpenCV
          ├── deskew
          ├── gray / Otsu / adaptive
          └── Tesseract rus+eng
                    │
                    ▼
              OCR JSON
          ┌─────────┼──────────┐
          ▼         ▼          ▼
        text      lines       words
                  bbox        bbox
                  score       score
```

На Stage 2 сервис **не сохраняет изображения на диск и не пишет OCR-результаты в БД**. Он является stateless worker. Это упрощает масштабирование и исключает накопление пользовательских изображений в контейнере.

## Запуск

После обновления файлов проекта существующие контейнеры БД и адресного API трогать не требуется:

```bash
docker compose up -d --build ocr
```

или:

```bash
make ocr-up
```

Проверка:

```bash
make ocr-health
```

Swagger/OpenAPI:

```text
http://localhost:8090/docs
```

Если внешний порт изменён в `.env`, например:

```text
TOOLOCR_OCR_PORT=18090
```

для make-команд можно передать его явно:

```bash
make ocr-health TOOLOCR_OCR_PORT=18090
```

## Конфигурация

Параметры `.env`:

```text
TOOLOCR_OCR_PORT=8090
OCR_MAX_UPLOAD_MB=20
OCR_MAX_PIXELS=40000000
OCR_DEFAULT_LANG=rus+eng
OCR_DEFAULT_PSM=11
```

`OCR_MAX_UPLOAD_MB` ограничивает размер загружаемого файла. `OCR_MAX_PIXELS` ограничивает уже декодированное изображение и защищает сервис от изображений с чрезмерным разрешением.

По умолчанию используется Tesseract PSM 11 (`Sparse text`), что удобнее для конвертов и изображений, где текст расположен отдельными областями. Через API допускаются PSM `3`, `6`, `11`, `12`.

## OCR endpoint

```text
POST /v1/ocr/recognize
```

Изображение передаётся как `multipart/form-data` в поле `file`.

Пример:

```bash
curl -sS -X POST \
  'http://localhost:8090/v1/ocr/recognize?preprocess=auto&deskew=true&psm=11&include_alternatives=true' \
  -F 'file=@/path/envelope.jpg' | jq
```

Быстрая make-команда:

```bash
make ocr-smoke FILE=/path/envelope.jpg
```

## Режимы предобработки

Параметр `preprocess`:

| Значение | Поведение |
|---|---|
| `none` | исходное цветное изображение |
| `gray` | перевод в оттенки серого |
| `otsu` | глобальная бинаризация Otsu |
| `adaptive` | адаптивная Gaussian-бинаризация |
| `auto` | OCR выполняется для `gray` и `otsu`, выбирается вариант с лучшим confidence |

`auto` выполняет два OCR-прохода и поэтому медленнее. Его задача на этом этапе — проверить идею нескольких гипотез. Для высоконагруженного production-пайплайна позднее будет введён ранний выход: если первый проход уже достаточно уверен, второй не запускается.

`deskew=true` включает осторожную коррекцию небольшого наклона. Если оценённый угол меньше `0.2°` либо больше `15°`, автоматическое вращение не выполняется.

## Формат ответа

Сокращённый пример:

```json
{
  "engine": "tesseract",
  "engine_version": "5.x",
  "language": "rus+eng",
  "psm": 11,
  "selected_preprocess": "otsu",
  "confidence": 0.8734,
  "text": "119501\nМОСКВА\nУЛ ВЕЕРНАЯ Д 30",
  "image": {
    "filename": "envelope.jpg",
    "content_type": "image/jpeg",
    "width": 2480,
    "height": 1754,
    "channels": 3,
    "bytes_received": 481233,
    "deskew_angle": -1.125
  },
  "timing": {
    "decode_ms": 7.841,
    "preprocess_ms": 15.238,
    "ocr_ms": 241.621,
    "total_ms": 266.102
  },
  "lines": [
    {
      "text": "УЛ ВЕЕРНАЯ Д 30",
      "confidence": 0.8912,
      "bbox": {"x": 422, "y": 804, "width": 911, "height": 96},
      "block": 1,
      "paragraph": 1,
      "line": 3
    }
  ],
  "words": [
    {
      "text": "ВЕЕРНАЯ",
      "confidence": 0.9134,
      "bbox": {"x": 531, "y": 804, "width": 391, "height": 92},
      "block": 1,
      "paragraph": 1,
      "line": 3
    }
  ],
  "alternatives": null
}
```

Числа в примере иллюстративные. Confidence нормализован в диапазон `0..1`.

Если передан `include_alternatives=true`, ответ дополнительно содержит результаты каждого выполненного варианта предобработки:

```json
"alternatives": [
  {
    "preprocess": "gray",
    "confidence": 0.81,
    "text": "...",
    "ocr_ms": 102.3
  },
  {
    "preprocess": "otsu",
    "confidence": 0.87,
    "text": "...",
    "ocr_ms": 109.7
  }
]
```

## Метрики времени

`timing` измеряется внутри OCR-сервиса:

- `decode_ms` — декодирование файла OpenCV;
- `preprocess_ms` — deskew и подготовка одного/нескольких вариантов;
- `ocr_ms` — суммарное время всех запусков OCR;
- `total_ms` — обработка запроса внутри endpoint до формирования ответа.

Полное время с передачей файла и JSON по сети по-прежнему измеряется на клиенте через `curl -w '%{time_total}'`.

## Healthcheck

```bash
curl -sS http://localhost:8090/health | jq
```

Сервис проверяет наличие бинарника Tesseract и языковых моделей, необходимых для `OCR_DEFAULT_LANG`. Ожидаемый ответ содержит установленную версию Tesseract и список языков.

## Ограничения текущего baseline

Stage 2 предназначен прежде всего для отладки контракта и pipeline. Tesseract не рассматривается как окончательный движок для рукописных российских адресов. Также пока отсутствуют:

- детектор адресного блока на полном конверте;
- отдельный recognizer шестизначного индекса;
- PaddleOCR/ONNX и ансамбль OCR-движков;
- распознавание штрихкодов/DataMatrix;
- семантический parser `индекс / регион / город / улица / дом`;
- автоматическая передача OCR-гипотез в `find_address_candidates()`.

Следующий подэтап после проверки Stage 2 на реальных изображениях — выделение адресного блока и построение **OCR → address parser → address candidates**, где адресная БД Stage 1.1 начнёт автоматически корректировать OCR-ошибки.

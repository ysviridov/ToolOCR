\set ON_ERROR_STOP on
\pset pager off
\timing on

\echo '=== ToolOCR Stage 1.1: активный набор ==='
SELECT dataset_id, source_filename, status, imported_at, activated_at
FROM toolocr.active_dataset;

\echo '=== Применённые миграции ==='
SELECT version, description, applied_at
FROM toolocr.schema_migration
ORDER BY version;

\echo '=== Контроль объёма активного снимка ==='
SELECT
    (SELECT count(*) FROM toolocr.current_federal_subject) AS subjects,
    (SELECT count(*) FROM toolocr.current_district) AS districts,
    (SELECT count(*) FROM toolocr.current_postal_code) AS postcodes,
    (SELECT count(*) FROM toolocr.current_main_city) AS main_cities,
    (SELECT count(*) FROM toolocr.current_city) AS cities,
    (SELECT count(*) FROM toolocr.current_street) AS streets,
    (SELECT count(*) FROM toolocr.current_address_range) AS address_ranges;

\echo '=== Реальный адрес из снимка: 142100, Подольск, Кирова, дом 4 ==='
SELECT id_address, postal_code, matched_city_name, street_name, street_qualifier,
       from_house_number, to_house_number,
       city_score, street_score, house_score, score
FROM toolocr.find_address_candidates('142100', 'Подольск', 'Кирова', '4', 10);

\echo '=== Fuzzy: намеренная OCR-ошибка КНРОВА вместо КИРОВА ==='
SELECT id_address, postal_code, matched_city_name, street_name,
       city_score, street_score, house_score, score
FROM toolocr.find_address_candidates('142100', 'Подольск', 'КНРОВА', '4', 10);

\echo '=== Проверка смешанной латиницы: KИРОВА/ПOДOЛЬСК ==='
SELECT id_address, postal_code, matched_city_name, street_name,
       city_score, street_score, house_score, score
FROM toolocr.find_address_candidates('142100', 'ПOДOЛЬСК', 'KИРОВА', '4', 10);


\echo '=== Автоматические проверки exact/fuzzy ==='
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM toolocr.find_address_candidates('142100', 'Подольск', 'Кирова', '4', 20)
        WHERE postal_code = '142100'
          AND street_name = 'КИРОВА'
          AND toolocr.ocr_norm_text(matched_city_name) LIKE 'ПОДОЛЬСК%'
          AND house_match = true
    ) THEN
        RAISE EXCEPTION 'Exact-проверка не пройдена: ожидается 142100 / ПОДОЛЬСК / КИРОВА / дом 4';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM toolocr.find_address_candidates('142100', 'Подольск', 'КНРОВА', '4', 20)
        WHERE postal_code = '142100'
          AND street_name = 'КИРОВА'
          AND toolocr.ocr_norm_text(matched_city_name) LIKE 'ПОДОЛЬСК%'
          AND house_match = true
    ) THEN
        RAISE EXCEPTION 'Fuzzy-проверка не пройдена: КНРОВА должна находить 142100 / ПОДОЛЬСК / КИРОВА / дом 4';
    END IF;
END;
$$;

\echo '=== EXPLAIN: точное сужение address_range по индексу ==='
EXPLAIN (ANALYZE, BUFFERS, VERBOSE OFF)
SELECT ar.id_address
FROM toolocr.current_address_range ar
WHERE ar.postal_code = '142100'
LIMIT 100;

\echo '=== EXPLAIN: fuzzy-поиск улицы через pg_trgm ==='
EXPLAIN (ANALYZE, BUFFERS, VERBOSE OFF)
SELECT s.id_street, s.street_name
FROM toolocr.current_street s
WHERE s.name_norm % toolocr.ocr_norm_text('КНРОВА')
ORDER BY similarity(s.name_norm, toolocr.ocr_norm_text('КНРОВА')) DESC
LIMIT 20;

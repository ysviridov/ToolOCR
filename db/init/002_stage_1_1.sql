\set ON_ERROR_STOP on

BEGIN;

CREATE TABLE IF NOT EXISTS toolocr.schema_migration (
    version      text PRIMARY KEY,
    description  text NOT NULL,
    applied_at   timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE toolocr.schema_migration IS
'Журнал применённых миграций схемы ToolOCR.';

INSERT INTO toolocr.schema_migration(version, description)
VALUES ('1.0', 'Базовая схема адресной БД и импорт версионируемых snapshots')
ON CONFLICT (version) DO NOTHING;

-- Stage 1.1: поиск кандидатов адреса поверх активного снимка.
-- Файл является одновременно init-скриптом для новой БД и идемпотентной
-- миграцией для уже существующего volume (make migrate-stage11).

CREATE OR REPLACE FUNCTION toolocr.ocr_norm_text(value text)
RETURNS text
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT translate(
        toolocr.norm_text(value),
        'ABCEHKMOPTXY',
        'АВСЕНКМОРТХУ'
    );
$$;

COMMENT ON FUNCTION toolocr.ocr_norm_text(text) IS
'Нормализация OCR-строки: базовая нормализация плюс замена визуально похожих латинских букв на кириллицу.';

CREATE OR REPLACE FUNCTION toolocr.norm_house(value text)
RETURNS text
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT regexp_replace(
        toolocr.ocr_norm_text(value),
        '[^0-9А-ЯA-Z]+',
        '',
        'g'
    );
$$;

COMMENT ON FUNCTION toolocr.norm_house(text) IS
'Нормализует номер дома/строения для точного сравнения и выделения числовой части.';

CREATE OR REPLACE FUNCTION toolocr.house_leading_number(value text)
RETURNS bigint
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT NULLIF(substring(toolocr.norm_house(value) FROM '^[0-9]+'), '')::bigint;
$$;

COMMENT ON FUNCTION toolocr.house_leading_number(text) IS
'Возвращает ведущую числовую часть номера дома либо NULL.';

CREATE OR REPLACE FUNCTION toolocr.house_match_score(
    query_house text,
    range_from text,
    range_to text
)
RETURNS real
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
WITH p AS (
    SELECT
        toolocr.norm_house(query_house) AS q,
        toolocr.norm_house(range_from) AS f,
        toolocr.norm_house(range_to) AS t,
        toolocr.house_leading_number(query_house) AS qn,
        toolocr.house_leading_number(range_from) AS fn,
        toolocr.house_leading_number(range_to) AS tn
)
SELECT CASE
    WHEN q = '' THEN NULL::real
    WHEN q = f OR q = t THEN 1.0::real
    WHEN qn IS NOT NULL AND fn IS NOT NULL AND tn IS NOT NULL
         AND qn BETWEEN LEAST(fn, tn) AND GREATEST(fn, tn)
      THEN 0.95::real
    ELSE 0.0::real
END
FROM p;
$$;

COMMENT ON FUNCTION toolocr.house_match_score(text, text, text) IS
'Оценивает попадание дома в адресный диапазон: 1.0 точное совпадение, 0.95 попадание числовой части в диапазон, 0.0 несовпадение.';

CREATE OR REPLACE FUNCTION toolocr.find_address_candidates(
    p_postal_code text DEFAULT NULL,
    p_city        text DEFAULT NULL,
    p_street      text DEFAULT NULL,
    p_house       text DEFAULT NULL,
    p_limit       integer DEFAULT 20
)
RETURNS TABLE (
    dataset_id            bigint,
    id_address            bigint,
    postal_code           varchar(6),
    id_subject            integer,
    subject_name          text,
    id_district           integer,
    district_name         text,
    id_main_city          integer,
    main_city_name        text,
    id_city               integer,
    city_name             text,
    matched_city_name     text,
    id_street             integer,
    street_name           text,
    street_qualifier      text,
    from_house_number     text,
    to_house_number       text,
    from_building_number  text,
    to_building_number    text,
    postcode_score        real,
    city_score            real,
    street_score          real,
    house_score           real,
    house_match           boolean,
    score                 real
)
LANGUAGE plpgsql
STABLE
PARALLEL SAFE
AS $$
DECLARE
    v_dataset_id bigint;
    v_postcode   text := regexp_replace(coalesce(p_postal_code, ''), '[^0-9]', '', 'g');
    v_city       text := toolocr.ocr_norm_text(p_city);
    v_street     text := toolocr.ocr_norm_text(p_street);
    v_house      text := toolocr.norm_house(p_house);
    v_limit      integer := LEAST(GREATEST(coalesce(p_limit, 20), 1), 100);
BEGIN
    SELECT rs.active_dataset_id
      INTO v_dataset_id
      FROM toolocr.runtime_state rs
     WHERE rs.singleton = true;

    IF v_dataset_id IS NULL THEN
        RAISE EXCEPTION USING
            ERRCODE = 'object_not_in_prerequisite_state',
            MESSAGE = 'В ToolOCR нет активного набора адресных данных';
    END IF;

    IF coalesce(trim(p_postal_code), '') <> '' AND length(v_postcode) <> 6 THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'Почтовый индекс должен содержать ровно 6 цифр';
    END IF;

    IF v_postcode = '' AND v_city = '' AND v_street = '' THEN
        RAISE EXCEPTION USING
            ERRCODE = '22023',
            MESSAGE = 'Для поиска необходимо передать хотя бы индекс, населённый пункт или улицу';
    END IF;

    RETURN QUERY
    WITH main_city_candidates AS MATERIALIZED (
        SELECT
            mc.id_main_city,
            GREATEST(
                similarity(mc.name_norm, v_city),
                similarity(toolocr.ocr_norm_text(mc.lex_key1), v_city)
            )::real AS sim
        FROM toolocr.main_city mc
        WHERE mc.dataset_id = v_dataset_id
          AND v_city <> ''
          AND mc.name_norm % v_city
    ),
    city_candidates AS MATERIALIZED (
        SELECT
            c.id_city,
            c.id_main_city,
            GREATEST(
                similarity(c.name_norm, v_city),
                similarity(toolocr.ocr_norm_text(c.lex_key1), v_city)
            )::real AS sim
        FROM toolocr.city c
        WHERE c.dataset_id = v_dataset_id
          AND v_city <> ''
          AND c.name_norm % v_city
    ),
    street_candidates AS MATERIALIZED (
        SELECT
            s.id_street,
            GREATEST(
                similarity(s.name_norm, v_street),
                similarity(toolocr.ocr_norm_text(s.lex_key1), v_street)
            )::real AS sim
        FROM toolocr.street s
        WHERE s.dataset_id = v_dataset_id
          AND v_street <> ''
          AND s.name_norm % v_street
    ),
    base AS (
        SELECT
            ar.dataset_id,
            ar.id_address,
            ar.postal_code,
            ar.id_subject,
            fs.subject_name,
            ar.id_district,
            dt.district_name,
            ar.id_main_city,
            mc.main_city_name,
            ar.id_city,
            ct.city_name,
            coalesce(ct.city_name, mc.main_city_name, ar.post_office_name) AS matched_city_name,
            ar.id_street,
            st.street_name,
            st.qualifier AS street_qualifier,
            ar.from_house_number,
            ar.to_house_number,
            ar.from_building_number,
            ar.to_building_number,
            CASE WHEN v_postcode = '' THEN NULL::real ELSE 1.0::real END AS postcode_score,
            CASE
                WHEN v_city = '' THEN NULL::real
                ELSE GREATEST(coalesce(mcc.sim, 0.0::real), coalesce(cc.sim, 0.0::real))
            END AS city_score,
            CASE WHEN v_street = '' THEN NULL::real ELSE sc.sim END AS street_score,
            CASE
                WHEN v_house = '' THEN NULL::real
                ELSE toolocr.house_match_score(v_house, ar.from_house_number, ar.to_house_number)
            END AS house_score
        FROM toolocr.address_range ar
        LEFT JOIN toolocr.federal_subject fs
          ON fs.dataset_id = ar.dataset_id
         AND fs.id_subject = ar.id_subject
        LEFT JOIN toolocr.district dt
          ON dt.dataset_id = ar.dataset_id
         AND dt.id_district = ar.id_district
        LEFT JOIN toolocr.main_city mc
          ON mc.dataset_id = ar.dataset_id
         AND mc.id_main_city = ar.id_main_city
        LEFT JOIN toolocr.city ct
          ON ct.dataset_id = ar.dataset_id
         AND ct.id_city = ar.id_city
        LEFT JOIN toolocr.street st
          ON st.dataset_id = ar.dataset_id
         AND st.id_street = ar.id_street
        LEFT JOIN main_city_candidates mcc
          ON mcc.id_main_city = ar.id_main_city
        LEFT JOIN city_candidates cc
          ON cc.id_city = ar.id_city
        LEFT JOIN street_candidates sc
          ON sc.id_street = ar.id_street
        WHERE ar.dataset_id = v_dataset_id
          AND (v_postcode = '' OR ar.postal_code = v_postcode)
          AND (v_city = '' OR mcc.id_main_city IS NOT NULL OR cc.id_city IS NOT NULL)
          AND (v_street = '' OR sc.id_street IS NOT NULL)
    ),
    scored AS (
        SELECT
            b.*,
            (
                (
                    CASE WHEN v_postcode <> '' THEN 0.35 * coalesce(b.postcode_score, 0) ELSE 0 END +
                    CASE WHEN v_city     <> '' THEN 0.25 * coalesce(b.city_score, 0)     ELSE 0 END +
                    CASE WHEN v_street   <> '' THEN 0.30 * coalesce(b.street_score, 0)   ELSE 0 END +
                    CASE WHEN v_house    <> '' THEN 0.10 * coalesce(b.house_score, 0)    ELSE 0 END
                ) /
                NULLIF(
                    CASE WHEN v_postcode <> '' THEN 0.35 ELSE 0 END +
                    CASE WHEN v_city     <> '' THEN 0.25 ELSE 0 END +
                    CASE WHEN v_street   <> '' THEN 0.30 ELSE 0 END +
                    CASE WHEN v_house    <> '' THEN 0.10 ELSE 0 END,
                    0
                )
            )::real AS final_score
        FROM base b
    )
    SELECT
        s.dataset_id,
        s.id_address,
        s.postal_code,
        s.id_subject,
        s.subject_name,
        s.id_district,
        s.district_name,
        s.id_main_city,
        s.main_city_name,
        s.id_city,
        s.city_name,
        s.matched_city_name,
        s.id_street,
        s.street_name,
        s.street_qualifier,
        s.from_house_number,
        s.to_house_number,
        s.from_building_number,
        s.to_building_number,
        s.postcode_score,
        s.city_score,
        s.street_score,
        s.house_score,
        CASE WHEN s.house_score IS NULL THEN NULL ELSE s.house_score >= 0.95 END AS house_match,
        s.final_score
    FROM scored s
    ORDER BY
        s.final_score DESC,
        coalesce(s.house_score, 0) DESC,
        coalesce(s.street_score, 0) DESC,
        coalesce(s.city_score, 0) DESC,
        s.id_address
    LIMIT v_limit;
END;
$$;

COMMENT ON FUNCTION toolocr.find_address_candidates(text, text, text, text, integer) IS
'Ищет кандидатов адреса в активном снимке. Индекс сравнивается точно, город и улица — fuzzy через pg_trgm, дом влияет на итоговый score.';


INSERT INTO toolocr.schema_migration(version, description)
VALUES ('1.1', 'Fuzzy-поиск адресных кандидатов и SQL/API-контракт')
ON CONFLICT (version) DO UPDATE
SET description = EXCLUDED.description;

COMMIT;

\set ON_ERROR_STOP on

CREATE SCHEMA IF NOT EXISTS toolocr;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;

CREATE OR REPLACE FUNCTION toolocr.norm_text(value text)
RETURNS text
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT trim(regexp_replace(
        upper(translate(coalesce(value, ''), 'Ё', 'Е')),
        '[^0-9A-ZА-Я-]+', ' ', 'g'
    ));
$$;

CREATE TABLE IF NOT EXISTS toolocr.dataset (
    dataset_id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_filename  text NOT NULL,
    source_timestamp timestamptz,
    source_sha256    char(64) NOT NULL UNIQUE,
    status           text NOT NULL DEFAULT 'loading'
                     CHECK (status IN ('loading', 'ready', 'active', 'retired', 'failed')),
    imported_at      timestamptz NOT NULL DEFAULT now(),
    activated_at     timestamptz,
    validation_json  jsonb,
    row_counts_json  jsonb,
    error_text       text
);

CREATE TABLE IF NOT EXISTS toolocr.runtime_state (
    singleton         boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    active_dataset_id bigint REFERENCES toolocr.dataset(dataset_id),
    updated_at        timestamptz NOT NULL DEFAULT now()
);
INSERT INTO toolocr.runtime_state(singleton)
VALUES (true)
ON CONFLICT (singleton) DO NOTHING;

CREATE TABLE IF NOT EXISTS toolocr.federal_subject (
    dataset_id   bigint NOT NULL REFERENCES toolocr.dataset(dataset_id) ON DELETE CASCADE,
    id_subject   integer NOT NULL,
    subject_name text NOT NULL,
    lex_key1     text,
    lex_key2     text,
    name_norm    text GENERATED ALWAYS AS (toolocr.norm_text(subject_name)) STORED,
    PRIMARY KEY (dataset_id, id_subject)
);

CREATE TABLE IF NOT EXISTS toolocr.district (
    dataset_id    bigint NOT NULL REFERENCES toolocr.dataset(dataset_id) ON DELETE CASCADE,
    id_district   integer NOT NULL,
    id_subject    integer,
    district_name text NOT NULL,
    lex_key1      text,
    lex_key2      text,
    name_norm     text GENERATED ALWAYS AS (toolocr.norm_text(district_name)) STORED,
    PRIMARY KEY (dataset_id, id_district)
);

CREATE TABLE IF NOT EXISTS toolocr.postal_code (
    dataset_id           bigint NOT NULL REFERENCES toolocr.dataset(dataset_id) ON DELETE CASCADE,
    postal_code          varchar(6) NOT NULL,
    three_digit_flag     smallint,
    federal_subject_flag smallint,
    district_flag        smallint,
    main_city_flag       smallint,
    city_flag            smallint,
    street_flag          smallint,
    PRIMARY KEY (dataset_id, postal_code)
);

CREATE TABLE IF NOT EXISTS toolocr.main_city (
    dataset_id     bigint NOT NULL REFERENCES toolocr.dataset(dataset_id) ON DELETE CASCADE,
    id_main_city   integer NOT NULL,
    id_subject     integer,
    id_district    integer,
    main_city_name text NOT NULL,
    lex_key1       text,
    lex_key2       text,
    name_norm      text GENERATED ALWAYS AS (toolocr.norm_text(main_city_name)) STORED,
    PRIMARY KEY (dataset_id, id_main_city)
);

CREATE TABLE IF NOT EXISTS toolocr.city (
    dataset_id bigint NOT NULL REFERENCES toolocr.dataset(dataset_id) ON DELETE CASCADE,
    id_city    integer NOT NULL,
    id_main_city integer,
    city_name  text NOT NULL,
    lex_key1   text,
    lex_key2   text,
    name_norm  text GENERATED ALWAYS AS (toolocr.norm_text(city_name)) STORED,
    PRIMARY KEY (dataset_id, id_city)
);

CREATE TABLE IF NOT EXISTS toolocr.street (
    dataset_id bigint NOT NULL REFERENCES toolocr.dataset(dataset_id) ON DELETE CASCADE,
    id_street  integer NOT NULL,
    street_name text NOT NULL,
    qualifier  text,
    lex_key1   text,
    lex_key2   text,
    name_norm  text GENERATED ALWAYS AS (toolocr.norm_text(street_name)) STORED,
    PRIMARY KEY (dataset_id, id_street)
);

CREATE TABLE IF NOT EXISTS toolocr.address_range (
    dataset_id            bigint NOT NULL REFERENCES toolocr.dataset(dataset_id) ON DELETE CASCADE,
    id_address            bigint NOT NULL,
    postal_code           varchar(6),
    post_office_name      text,
    id_subject            integer,
    id_district           integer,
    id_main_city          integer,
    id_city               integer,
    id_street             integer,
    from_house_number     text,
    to_house_number       text,
    from_building_number  text,
    to_building_number    text,
    even_odd_indicator    varchar(8),
    PRIMARY KEY (dataset_id, id_address)
);

-- Exact/narrowing indexes used by the OCR validator.
CREATE INDEX IF NOT EXISTS ix_district_dataset_subject
    ON toolocr.district(dataset_id, id_subject);
CREATE INDEX IF NOT EXISTS ix_main_city_dataset_subject
    ON toolocr.main_city(dataset_id, id_subject);
CREATE INDEX IF NOT EXISTS ix_main_city_dataset_district
    ON toolocr.main_city(dataset_id, id_district);
CREATE INDEX IF NOT EXISTS ix_city_dataset_main_city
    ON toolocr.city(dataset_id, id_main_city);
CREATE INDEX IF NOT EXISTS ix_address_dataset_postcode
    ON toolocr.address_range(dataset_id, postal_code);
CREATE INDEX IF NOT EXISTS ix_address_dataset_main_city
    ON toolocr.address_range(dataset_id, id_main_city);
CREATE INDEX IF NOT EXISTS ix_address_dataset_city
    ON toolocr.address_range(dataset_id, id_city);
CREATE INDEX IF NOT EXISTS ix_address_dataset_street
    ON toolocr.address_range(dataset_id, id_street);
CREATE INDEX IF NOT EXISTS ix_address_candidate
    ON toolocr.address_range(dataset_id, postal_code, id_main_city, id_city, id_street);

-- Fuzzy lookup indexes. dataset_id remains a separate btree filter.
CREATE INDEX IF NOT EXISTS ix_subject_name_trgm
    ON toolocr.federal_subject USING gin(name_norm gin_trgm_ops);
CREATE INDEX IF NOT EXISTS ix_district_name_trgm
    ON toolocr.district USING gin(name_norm gin_trgm_ops);
CREATE INDEX IF NOT EXISTS ix_main_city_name_trgm
    ON toolocr.main_city USING gin(name_norm gin_trgm_ops);
CREATE INDEX IF NOT EXISTS ix_city_name_trgm
    ON toolocr.city USING gin(name_norm gin_trgm_ops);
CREATE INDEX IF NOT EXISTS ix_street_name_trgm
    ON toolocr.street USING gin(name_norm gin_trgm_ops);

-- Stable views: application code never has to know the current dataset_id.
CREATE OR REPLACE VIEW toolocr.current_federal_subject AS
SELECT t.* FROM toolocr.federal_subject t
JOIN toolocr.runtime_state r ON r.singleton AND r.active_dataset_id = t.dataset_id;

CREATE OR REPLACE VIEW toolocr.current_district AS
SELECT t.* FROM toolocr.district t
JOIN toolocr.runtime_state r ON r.singleton AND r.active_dataset_id = t.dataset_id;

CREATE OR REPLACE VIEW toolocr.current_postal_code AS
SELECT t.* FROM toolocr.postal_code t
JOIN toolocr.runtime_state r ON r.singleton AND r.active_dataset_id = t.dataset_id;

CREATE OR REPLACE VIEW toolocr.current_main_city AS
SELECT t.* FROM toolocr.main_city t
JOIN toolocr.runtime_state r ON r.singleton AND r.active_dataset_id = t.dataset_id;

CREATE OR REPLACE VIEW toolocr.current_city AS
SELECT t.* FROM toolocr.city t
JOIN toolocr.runtime_state r ON r.singleton AND r.active_dataset_id = t.dataset_id;

CREATE OR REPLACE VIEW toolocr.current_street AS
SELECT t.* FROM toolocr.street t
JOIN toolocr.runtime_state r ON r.singleton AND r.active_dataset_id = t.dataset_id;

CREATE OR REPLACE VIEW toolocr.current_address_range AS
SELECT t.* FROM toolocr.address_range t
JOIN toolocr.runtime_state r ON r.singleton AND r.active_dataset_id = t.dataset_id;

CREATE OR REPLACE VIEW toolocr.active_dataset AS
SELECT d.*
FROM toolocr.dataset d
JOIN toolocr.runtime_state r ON r.singleton AND r.active_dataset_id = d.dataset_id;

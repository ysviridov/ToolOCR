SHELL := /bin/bash
COMPOSE := docker compose
TOOLOCR_API_PORT ?= 8080
TOOLOCR_OCR_PORT ?= 8090

.PHONY: db-up db-down db-logs db-shell import update datasets activate prune reset \
        migrate-stage11 check-stage11 api-up api-down api-logs api-smoke \
        ocr-up ocr-down ocr-logs ocr-health ocr-smoke \
        layout-profiles layout-smoke layout-calibrate layout-calibrations \
        layout-rectify layout-rectify-raw test

db-up:
	$(COMPOSE) up -d db

db-down:
	$(COMPOSE) down

db-logs:
	$(COMPOSE) logs -f db

db-shell:
	$(COMPOSE) exec db sh -lc 'exec psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'

import:
	@test -n "$(FILE)" || (echo "Использование: make import FILE=/path/ADDRESS_YYYYMMDDHHMMSS.zip"; exit 2)
	$(COMPOSE) run --rm -v "$(abspath $(FILE)):/import/address.zip:ro" importer import /import/address.zip --activate

# Ежемесячный полный снимок обновляется безопасно: load -> validate -> atomic activate.
update: import

datasets:
	$(COMPOSE) run --rm importer list

activate:
	@test -n "$(ID)" || (echo "Использование: make activate ID=<dataset_id>"; exit 2)
	$(COMPOSE) run --rm importer activate $(ID)

prune:
	$(COMPOSE) run --rm importer prune --keep $${KEEP:-2}

# Stage 1.1. Файл идемпотентен и может быть применён к уже работающему volume.
migrate-stage11:
	$(COMPOSE) exec -T db sh -lc 'psql -v ON_ERROR_STOP=1 -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"' < db/init/002_stage_1_1.sql

check-stage11:
	$(COMPOSE) exec -T db sh -lc 'psql -v ON_ERROR_STOP=1 -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"' < db/checks/stage_1_1_smoke.sql

api-up:
	$(COMPOSE) up -d --build api

api-down:
	$(COMPOSE) stop api

api-logs:
	$(COMPOSE) logs -f api

api-smoke:
	@echo "Проверка /health"
	@curl --fail --silent --show-error "http://localhost:$(TOOLOCR_API_PORT)/health"; echo
	@echo "Проверка fuzzy-поиска: КНРОВА -> КИРОВА"
	@curl --fail --silent --show-error --get "http://localhost:$(TOOLOCR_API_PORT)/v1/address/candidates" \
		--data-urlencode "postal_code=142100" \
		--data-urlencode "city=Подольск" \
		--data-urlencode "street=КНРОВА" \
		--data-urlencode "house=4" \
		--data-urlencode "limit=3"; echo

# Stage 2: сервис базового OCR.
ocr-up:
	$(COMPOSE) up -d --build ocr

ocr-down:
	$(COMPOSE) stop ocr

ocr-logs:
	$(COMPOSE) logs -f ocr

ocr-health:
	@curl --fail --silent --show-error "http://localhost:$(TOOLOCR_OCR_PORT)/health"; echo

ocr-smoke:
	@test -n "$(FILE)" || (echo "Использование: make ocr-smoke FILE=/path/image.jpg"; exit 2)
	@curl --fail --silent --show-error -X POST \
		"http://localhost:$(TOOLOCR_OCR_PORT)/v1/ocr/recognize?preprocess=auto&deskew=true&psm=11&include_alternatives=true" \
		-F "file=@$(abspath $(FILE))"

# Stage 2.1: геометрия полного письма и ГОСТ-профили.
layout-profiles:
	@curl --fail-with-body --silent --show-error \
		"http://localhost:$(TOOLOCR_OCR_PORT)/v1/layout/profiles"

layout-smoke:
	@test -n "$(FILE)" || (echo "Использование: make layout-smoke FILE=/path/full-envelope.jpg"; exit 2)
	@curl --fail-with-body --silent --show-error -X POST \
		"http://localhost:$(TOOLOCR_OCR_PORT)/v1/layout/analyze" \
		-F "file=@$(abspath $(FILE))"

# Калибровка фиксированной камеры по полностью видимому эталону известного
# ГОСТ-формата. Каждый FORMAT обновляет только свою запись в общем JSON.
# Старый одноформатный v1-файл автоматически мигрирует в набор version=2.
layout-calibrate:
	@test -n "$(FILE)" || (echo "Использование: make layout-calibrate FILE=/path/reference.jpg FORMAT=C4 [OUT=config/camera-calibration.json]"; exit 2)
	@test -n "$(FORMAT)" || (echo "Укажите FORMAT=C6|DL|C5|C4|B4"; exit 2)
	@set -euo pipefail; \
	out="$${OUT:-config/camera-calibration.json}"; \
	mkdir -p "$$(dirname "$$out")"; \
	entry="$$out.entry.tmp"; \
	merged="$$out.tmp"; \
	trap 'rm -f "$$entry" "$$merged"' EXIT; \
	curl --fail-with-body --silent --show-error -X POST \
		"http://localhost:$(TOOLOCR_OCR_PORT)/v1/layout/calibration/estimate?known_format=$(FORMAT)" \
		-F "file=@$(abspath $(FILE))" \
		| jq -e '.calibration' > "$$entry"; \
	jq -e '.version == 1 and .homography_norm_to_mm and .reference_format and .standard' "$$entry" >/dev/null; \
	if [[ -s "$$out" ]]; then \
		jq --slurpfile e "$$entry" ' \
			($$e[0]) as $$new | \
			if (.version == 2 and (.calibrations | type) == "object") then \
				if .standard != $$new.standard then error("standard mismatch") \
				else .calibrations[$$new.reference_format] = $$new end \
			elif (has("homography_norm_to_mm") and has("reference_format")) then \
				. as $$old | \
				if $$old.standard != $$new.standard then error("standard mismatch") \
				else { \
					version: 2, \
					standard: $$new.standard, \
					calibrations: ({($$old.reference_format): $$old} + {($$new.reference_format): $$new}) \
				} end \
			else error("unsupported calibration file") end \
		' "$$out" > "$$merged"; \
	else \
		jq -n --slurpfile e "$$entry" ' \
			($$e[0]) as $$new | { \
				version: 2, \
				standard: $$new.standard, \
				calibrations: {($$new.reference_format): $$new} \
			} \
		' > "$$merged"; \
	fi; \
	jq -e '.version == 2 and (.calibrations | type) == "object" and (.calibrations | length) > 0' "$$merged" >/dev/null; \
	mv "$$merged" "$$out"; \
	rm -f "$$entry"; \
	trap - EXIT; \
	echo "Калибровка $(FORMAT) сохранена/обновлена: $$out"; \
	jq '{version, standard, count:(.calibrations|length), formats:(.calibrations|keys)}' "$$out"

layout-calibrations:
	@out="$${OUT:-config/camera-calibration.json}"; \
	test -s "$$out" || (echo "Файл калибровок не найден: $$out"; exit 2); \
	jq '{version, standard, count:(if .calibrations then (.calibrations|length) else 1 end), formats:(if .calibrations then (.calibrations|keys) else [.reference_format] end)}' "$$out"

# По умолчанию /rectify применяет автоматически определённый поворот 0/180.
layout-rectify:
	@test -n "$(FILE)" || (echo "Использование: make layout-rectify FILE=/path/full-envelope.jpg [OUT=/tmp/rectified.jpg]"; exit 2)
	@curl --fail-with-body --silent --show-error -X POST \
		"http://localhost:$(TOOLOCR_OCR_PORT)/v1/layout/rectify?canonical_orientation=true" \
		-F "file=@$(abspath $(FILE))" \
		-o "$${OUT:-/tmp/toolocr-rectified.jpg}"
	@echo "Сохранено: $${OUT:-/tmp/toolocr-rectified.jpg}"

# Диагностический вариант без автоматического поворота 0/180.
layout-rectify-raw:
	@test -n "$(FILE)" || (echo "Использование: make layout-rectify-raw FILE=/path/full-envelope.jpg [OUT=/tmp/rectified-raw.jpg]"; exit 2)
	@curl --fail-with-body --silent --show-error -X POST \
		"http://localhost:$(TOOLOCR_OCR_PORT)/v1/layout/rectify?canonical_orientation=false" \
		-F "file=@$(abspath $(FILE))" \
		-o "$${OUT:-/tmp/toolocr-rectified-raw.jpg}"
	@echo "Сохранено: $${OUT:-/tmp/toolocr-rectified-raw.jpg}"

test:
	python3 tests/test_archive_contract.py "$${ARCHIVE:?Укажите ARCHIVE=/path/ADDRESS_*.zip}"
	python3 -m compileall -q importer api ocr

# Разрушительная операция: удаляет volume PostgreSQL.
reset:
	$(COMPOSE) down -v

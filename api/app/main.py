from __future__ import annotations

import os
import time
from typing import Any

import psycopg
from fastapi import FastAPI, HTTPException, Query
from psycopg.rows import dict_row
from pydantic import BaseModel, Field, model_validator

DB_DSN = os.environ.get("TOOLOCR_DB_DSN", "")

app = FastAPI(
    title="ToolOCR Address API",
    version="1.1.1",
    description="Поиск кандидатов российского адреса по активному снимку адресной базы.",
)


class AddressCandidateRequest(BaseModel):
    postal_code: str | None = Field(default=None, pattern=r"^\d{6}$")
    city: str | None = Field(default=None, max_length=256)
    street: str | None = Field(default=None, max_length=256)
    house: str | None = Field(default=None, max_length=64)
    limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def validate_search_scope(self) -> "AddressCandidateRequest":
        if not any(
            value and value.strip()
            for value in (self.postal_code, self.city, self.street)
        ):
            raise ValueError("Нужно передать хотя бы postal_code, city или street")
        return self


class TimingResponse(BaseModel):
    db_ms: float
    total_ms: float


class CandidateResponse(BaseModel):
    query: AddressCandidateRequest
    count: int
    timing: TimingResponse
    candidates: list[dict[str, Any]]


def run_candidates(request: AddressCandidateRequest) -> tuple[list[dict[str, Any]], float]:
    sql = """
        SELECT *
        FROM toolocr.find_address_candidates(%s, %s, %s, %s, %s)
    """
    try:
        with psycopg.connect(DB_DSN, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                db_started = time.perf_counter()
                cur.execute(
                    sql,
                    (
                        request.postal_code,
                        request.city,
                        request.street,
                        request.house,
                        request.limit,
                    ),
                )
                rows = [dict(row) for row in cur.fetchall()]
                db_ms = (time.perf_counter() - db_started) * 1000.0
                return rows, db_ms
    except psycopg.Error as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ошибка обращения к адресной БД: {exc.diag.message_primary or str(exc)}",
        ) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        with psycopg.connect(DB_DSN, row_factory=dict_row) as conn:
            row = conn.execute(
                """
                SELECT dataset_id, source_filename, status, imported_at, activated_at,
                       to_regprocedure('toolocr.find_address_candidates(text,text,text,text,integer)') IS NOT NULL
                           AS stage11_ready
                FROM toolocr.active_dataset
                """
            ).fetchone()
            if not row:
                raise HTTPException(status_code=503, detail="Нет активного набора адресных данных")
            if not row["stage11_ready"]:
                raise HTTPException(status_code=503, detail="Миграция Stage 1.1 не применена")
            payload = dict(row)
            payload.pop("stage11_ready", None)
            return {"status": "ok", "active_dataset": payload}
    except HTTPException:
        raise
    except psycopg.Error as exc:
        raise HTTPException(
            status_code=503,
            detail=f"База данных недоступна: {exc.diag.message_primary or str(exc)}",
        ) from exc


@app.post("/v1/address/candidates", response_model=CandidateResponse)
def candidates_post(request: AddressCandidateRequest) -> CandidateResponse:
    total_started = time.perf_counter()
    rows, db_ms = run_candidates(request)
    total_ms = (time.perf_counter() - total_started) * 1000.0
    return CandidateResponse(
        query=request,
        count=len(rows),
        timing=TimingResponse(
            db_ms=round(db_ms, 3),
            total_ms=round(total_ms, 3),
        ),
        candidates=rows,
    )


@app.get("/v1/address/candidates", response_model=CandidateResponse)
def candidates_get(
    postal_code: str | None = Query(default=None, pattern=r"^\d{6}$"),
    city: str | None = Query(default=None, max_length=256),
    street: str | None = Query(default=None, max_length=256),
    house: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=20, ge=1, le=100),
) -> CandidateResponse:
    request = AddressCandidateRequest(
        postal_code=postal_code,
        city=city,
        street=street,
        house=house,
        limit=limit,
    )
    total_started = time.perf_counter()
    rows, db_ms = run_candidates(request)
    total_ms = (time.perf_counter() - total_started) * 1000.0
    return CandidateResponse(
        query=request,
        count=len(rows),
        timing=TimingResponse(
            db_ms=round(db_ms, 3),
            total_ms=round(total_ms, 3),
        ),
        candidates=rows,
    )

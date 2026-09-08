from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field, model_validator

from . import test_ui
from .format_modes import FormatMode
from .gost_r_51506_99 import EnvelopeFormat
from .roi import detect_simple_mail_rois
from .roi_test_ui import _analyze_saved_image, _canonical_from_analysis

router = APIRouter(tags=["test-ui"])
Format = Literal["DL", "C5", "C4"]


class Box(BaseModel):
    x: float = Field(ge=0, le=1, allow_inf_nan=False)
    y: float = Field(ge=0, le=1, allow_inf_nan=False)
    width: float = Field(gt=0, le=1, allow_inf_nan=False)
    height: float = Field(gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def bounds(self):
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("Прямоугольник выходит за границы изображения")
        return self


class Annotation(BaseModel):
    format: Format
    canonical_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ground_truth_status: Literal["needs_review", "verified", "exclude"]
    recipient_status: Literal["present", "absent"]
    recipient_bbox: Box | None = None
    postcode_box_status: Literal["present", "absent", "unknown"] = "unknown"
    postcode_bbox: Box | None = None
    notes: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def consistency(self):
        if self.recipient_status == "absent" and self.recipient_bbox:
            raise ValueError("Удалите рамку отсутствующего адресного блока")
        if self.postcode_box_status != "present" and self.postcode_bbox:
            raise ValueError("Удалите рамку индекса или укажите её наличие")
        if self.ground_truth_status == "verified":
            if self.recipient_status == "present" and not self.recipient_bbox:
                raise ValueError("Нарисуйте адресный блок перед подтверждением")
            if self.postcode_box_status == "present" and not self.postcode_bbox:
                raise ValueError("Нарисуйте рамку индекса перед подтверждением")
        return self


def _path(file_id: str, fmt: str) -> Path:
    return test_ui.TEST_STORAGE_DIR / "recipient-annotations" / f"{test_ui._validate_file_id(file_id)}-{fmt}.json"


def _read(file_id: str, fmt: str):
    path = _path(file_id, fmt)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


async def _canonical(file_id: str, fmt: str):
    analysis, image, _ = await _analyze_saved_image(
        file_id, format_mode=FormatMode.FIXED, expected_format=EnvelopeFormat(fmt)
    )
    canonical = _canonical_from_analysis(analysis, image).image
    digest = hashlib.sha256(str(canonical.shape).encode() + canonical.tobytes()).hexdigest()
    return canonical, digest


@router.get("/test-ui/recipient", response_class=HTMLResponse)
def page():
    return HTMLResponse(Path(__file__).with_name("recipient_annotation.html").read_text(encoding="utf-8"))


@router.get("/v1/test-ui/recipient/items")
def items(folder_id: str, format: Format):
    test_ui._require_folder(folder_id)
    result = []
    for meta in test_ui._iter_metadata():
        if meta.get("folder_id") == folder_id:
            saved = _read(meta["id"], format)
            result.append({"id": meta["id"], "name": meta["name"],
                           "status": saved["ground_truth_status"] if saved else "needs_review"})
    return {"items": sorted(result, key=lambda item: item["name"].casefold())}


@router.get("/v1/test-ui/recipient/images/{file_id}")
async def image(file_id: str, format: Format):
    canonical, digest = await _canonical(file_id, format)
    ok, encoded = cv2.imencode(".png", canonical)
    if not ok:
        raise HTTPException(500, "Не удалось подготовить изображение")
    saved = _read(file_id, format)
    stale = bool(saved and saved["canonical_hash"] != digest)
    if stale:
        # Old coordinates are unsafe after layout/orientation changes.
        saved = None
    roi = detect_simple_mail_rois(canonical, EnvelopeFormat(format))
    rect = next(r for r in roi.regions if r.kind == "recipient_address").detected_bbox
    h, w = canonical.shape[:2]
    seed = None if rect is None else dict(x=rect.x/w, y=rect.y/h, width=rect.width/w, height=rect.height/h)
    return {"image": "data:image/png;base64," + base64.b64encode(encoded).decode(),
            "canonical_hash": digest, "saved": saved, "stale": stale, "seed": seed}


@router.put("/v1/test-ui/recipient/images/{file_id}")
async def save(file_id: str, request: Annotation):
    # Recheck the actual canonical pixels, not client-supplied dimensions.
    _, digest = await _canonical(file_id, request.format)
    if digest != request.canonical_hash:
        raise HTTPException(409, "Изображение изменилось. Откройте его заново и проверьте разметку.")
    _write(_path(file_id, request.format), request.model_dump())
    return {"status": "saved"}


@router.get("/v1/test-ui/recipient/export")
async def export(folder_id: str, format: Format):
    test_ui._require_folder(folder_id)
    fields = ["filename", "format", "ground_truth_status", "recipient_status", "x", "y", "width", "height",
              "postcode_box_status", "postcode_x", "postcode_y", "postcode_width", "postcode_height", "notes"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    seen = set()
    metadata = list(test_ui._iter_metadata())
    for meta in metadata:
        if meta.get("folder_id") != folder_id:
            continue
        saved = _read(meta["id"], format)
        if not saved:
            continue
        key = meta["name"].casefold()
        if key in seen or sum(str(m["name"]).casefold() == key for m in metadata) > 1:
            raise HTTPException(409, "В библиотеке есть одинаковые имена файлов; evaluator требует уникальные имена")
        if saved["ground_truth_status"] == "verified":
            _, digest = await _canonical(meta["id"], format)
            if digest != saved["canonical_hash"]:
                raise HTTPException(409, f"Изменилась геометрия {meta['name']}; откройте письмо и проверьте разметку заново")
        seen.add(key)
        row = {k: saved[k] for k in ("format", "ground_truth_status", "recipient_status", "postcode_box_status", "notes")}
        row["filename"] = meta["name"]
        for source, prefix in (("recipient_bbox", ""), ("postcode_bbox", "postcode_")):
            row.update({prefix + key: value for key, value in (saved.get(source) or {}).items()})
        writer.writerow(row)
    return Response(stream.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="recipient-benchmark.csv"'})

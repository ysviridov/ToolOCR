from __future__ import annotations

import io
import json
import mimetypes
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from .frame_normalization import normalize_black_background
from .layout_api import MAX_PIXELS, MAX_UPLOAD_BYTES, analyze_layout


router = APIRouter(tags=["test-ui"])
TEST_STORAGE_DIR = Path(os.environ.get("TEST_UI_STORAGE_DIR", "/app/test-data"))
MAX_TEST_FILES_PER_RUN = int(os.environ.get("TEST_UI_MAX_FILES_PER_RUN", "100"))
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FILE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
HTML_PATH = Path(__file__).with_name("test_ui.html")
FOLDERS_PATH = TEST_STORAGE_DIR / "folders.json"
MAX_FOLDER_NAME = 80


class TestRunRequest(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=MAX_TEST_FILES_PER_RUN)


class DeleteRequest(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=500)


class MoveRequest(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=500)
    folder_id: str | None = None


class FolderRequest(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_FOLDER_NAME)


def _ensure_storage() -> None:
    TEST_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _validate_file_id(file_id: str) -> str:
    value = str(file_id).lower()
    if not FILE_ID_RE.fullmatch(value):
        raise HTTPException(status_code=400, detail="Некорректный идентификатор файла")
    return value


def _validate_folder_id(folder_id: str) -> str:
    value = str(folder_id).lower()
    if not FILE_ID_RE.fullmatch(value):
        raise HTTPException(status_code=400, detail="Некорректный идентификатор папки")
    return value


def _sanitize_folder_name(name: str) -> str:
    value = re.sub(r"[\x00-\x1f\x7f]", "", str(name)).strip()
    if not value:
        raise HTTPException(status_code=400, detail="Имя папки не может быть пустым")
    return value[:MAX_FOLDER_NAME]


def _metadata_path(file_id: str) -> Path:
    return TEST_STORAGE_DIR / f"{_validate_file_id(file_id)}.json"


def _load_metadata(file_id: str) -> dict[str, Any]:
    path = _metadata_path(file_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Тестовое изображение не найдено") from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail="Повреждены метаданные тестового изображения") from exc
    if payload.get("id") != file_id:
        raise HTTPException(status_code=500, detail="Несогласованные метаданные тестового изображения")
    return payload


def _image_path(meta: dict[str, Any]) -> Path:
    file_id = _validate_file_id(str(meta["id"]))
    suffix = str(meta.get("suffix", "")).lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=500, detail="Некорректное расширение сохранённого изображения")
    path = TEST_STORAGE_DIR / f"{file_id}{suffix}"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Тестовое изображение не найдено")
    return path


def _sanitize_display_name(name: str | None) -> str:
    value = Path(name or "image").name.replace("\x00", "").strip()
    return value[:240] or "image"


def _decode_image(raw: bytes) -> np.ndarray:
    encoded = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=415, detail="Файл не является поддерживаемым изображением")
    height, width = image.shape[:2]
    if height * width > MAX_PIXELS:
        raise HTTPException(status_code=413, detail=f"Изображение слишком большое: {width}x{height}")
    return image


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _load_folders() -> list[dict[str, Any]]:
    _ensure_storage()
    try:
        payload = json.loads(FOLDERS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail="Повреждён список тестовых папок") from exc

    folders = payload.get("folders") if isinstance(payload, dict) else None
    if not isinstance(folders, list):
        raise HTTPException(status_code=500, detail="Некорректный формат списка тестовых папок")

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in folders:
        if not isinstance(item, dict):
            continue
        folder_id = str(item.get("id", "")).lower()
        if not FILE_ID_RE.fullmatch(folder_id) or folder_id in seen:
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        seen.add(folder_id)
        result.append(
            {
                "id": folder_id,
                "name": name[:MAX_FOLDER_NAME],
                "created_at": item.get("created_at") or datetime.now(timezone.utc).isoformat(),
            }
        )
    return result


def _save_folders(folders: list[dict[str, Any]]) -> None:
    _ensure_storage()
    _write_json_atomic(FOLDERS_PATH, {"version": 1, "folders": folders})


def _folder_map() -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in _load_folders()}


def _require_folder(folder_id: str | None) -> str | None:
    if folder_id is None or str(folder_id).strip() == "":
        return None
    value = _validate_folder_id(folder_id)
    if value not in _folder_map():
        raise HTTPException(status_code=404, detail="Тестовая папка не найдена")
    return value


def _iter_metadata() -> list[dict[str, Any]]:
    _ensure_storage()
    items: list[dict[str, Any]] = []
    for path in TEST_STORAGE_DIR.glob("*.json"):
        if not FILE_ID_RE.fullmatch(path.stem.lower()):
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            image_path = _image_path(meta)
            if image_path.is_file():
                items.append(meta)
        except Exception:
            # Повреждённый sidecar не должен ломать всю тестовую библиотеку.
            continue
    return items


def _public_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": meta["id"],
        "name": meta["name"],
        "bytes": meta["bytes"],
        "width_px": meta["width_px"],
        "height_px": meta["height_px"],
        "uploaded_at": meta["uploaded_at"],
        "folder_id": meta.get("folder_id"),
    }


@router.get("/test-ui", response_class=HTMLResponse, include_in_schema=False)
def test_ui_page() -> HTMLResponse:
    try:
        html = HTML_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Не найден шаблон test-ui") from exc
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@router.get("/v1/test-ui/folders")
def list_test_folders() -> dict[str, Any]:
    folders = _load_folders()
    counts = {item["id"]: 0 for item in folders}
    ungrouped = 0
    for meta in _iter_metadata():
        folder_id = meta.get("folder_id")
        if folder_id in counts:
            counts[folder_id] += 1
        else:
            ungrouped += 1
    public = [
        {**item, "count": counts[item["id"]]}
        for item in sorted(folders, key=lambda value: value["name"].casefold())
    ]
    return {"count": len(public), "ungrouped_count": ungrouped, "folders": public}


@router.post("/v1/test-ui/folders")
def create_test_folder(request: FolderRequest) -> dict[str, Any]:
    folders = _load_folders()
    name = _sanitize_folder_name(request.name)
    if any(item["name"].casefold() == name.casefold() for item in folders):
        raise HTTPException(status_code=409, detail="Папка с таким именем уже существует")
    folder = {
        "id": uuid.uuid4().hex,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    folders.append(folder)
    _save_folders(folders)
    return {"folder": {**folder, "count": 0}}


@router.post("/v1/test-ui/folders/{folder_id}/rename")
def rename_test_folder(folder_id: str, request: FolderRequest) -> dict[str, Any]:
    value = _validate_folder_id(folder_id)
    folders = _load_folders()
    name = _sanitize_folder_name(request.name)
    found = False
    for item in folders:
        if item["id"] == value:
            found = True
            continue
        if item["name"].casefold() == name.casefold():
            raise HTTPException(status_code=409, detail="Папка с таким именем уже существует")
    if not found:
        raise HTTPException(status_code=404, detail="Тестовая папка не найдена")
    for item in folders:
        if item["id"] == value:
            item["name"] = name
            break
    _save_folders(folders)
    return {"folder": next(item for item in folders if item["id"] == value)}


@router.delete("/v1/test-ui/folders/{folder_id}")
def delete_test_folder(folder_id: str) -> dict[str, Any]:
    value = _validate_folder_id(folder_id)
    folders = _load_folders()
    if not any(item["id"] == value for item in folders):
        raise HTTPException(status_code=404, detail="Тестовая папка не найдена")

    moved = 0
    for meta in _iter_metadata():
        if meta.get("folder_id") == value:
            meta["folder_id"] = None
            _write_json_atomic(_metadata_path(meta["id"]), meta)
            moved += 1

    folders = [item for item in folders if item["id"] != value]
    _save_folders(folders)
    return {"deleted": value, "images_moved_to_ungrouped": moved}


@router.get("/v1/test-ui/images")
def list_test_images() -> dict[str, Any]:
    items = [_public_metadata(meta) for meta in _iter_metadata()]
    items.sort(key=lambda item: item["uploaded_at"], reverse=True)
    return {"count": len(items), "images": items}


@router.post("/v1/test-ui/images")
async def upload_test_images(
    files: list[UploadFile] = File(...),
    folder_id: str | None = Form(default=None),
) -> dict[str, Any]:
    _ensure_storage()
    target_folder_id = _require_folder(folder_id)
    if not files:
        raise HTTPException(status_code=400, detail="Не переданы изображения")

    uploaded: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for source in files:
        display_name = _sanitize_display_name(source.filename)
        suffix = Path(display_name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            errors.append({"name": display_name, "error": "Неподдерживаемое расширение файла"})
            continue
        raw = await source.read(MAX_UPLOAD_BYTES + 1)
        if not raw:
            errors.append({"name": display_name, "error": "Пустой файл"})
            continue
        if len(raw) > MAX_UPLOAD_BYTES:
            errors.append({"name": display_name, "error": f"Размер превышает лимит {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ"})
            continue
        try:
            image = _decode_image(raw)
        except HTTPException as exc:
            errors.append({"name": display_name, "error": str(exc.detail)})
            continue

        file_id = uuid.uuid4().hex
        image_path = TEST_STORAGE_DIR / f"{file_id}{suffix}"
        image_tmp = TEST_STORAGE_DIR / f".{file_id}.upload"
        image_tmp.write_bytes(raw)
        os.replace(image_tmp, image_path)

        height, width = image.shape[:2]
        meta = {
            "id": file_id,
            "name": display_name,
            "suffix": suffix,
            "bytes": len(raw),
            "width_px": int(width),
            "height_px": int(height),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "folder_id": target_folder_id,
        }
        _write_json_atomic(_metadata_path(file_id), meta)
        uploaded.append(_public_metadata(meta))

    return {"uploaded": uploaded, "errors": errors}


@router.post("/v1/test-ui/images/move")
def move_test_images(request: MoveRequest) -> dict[str, Any]:
    _ensure_storage()
    target_folder_id = _require_folder(request.folder_id)
    metas: list[dict[str, Any]] = []
    for raw_id in dict.fromkeys(request.ids):
        metas.append(_load_metadata(_validate_file_id(raw_id)))
    for meta in metas:
        meta["folder_id"] = target_folder_id
        _write_json_atomic(_metadata_path(meta["id"]), meta)
    return {"moved": [meta["id"] for meta in metas], "folder_id": target_folder_id}


@router.post("/v1/test-ui/images/delete")
def delete_test_images(request: DeleteRequest) -> dict[str, Any]:
    _ensure_storage()
    deleted: list[str] = []
    missing: list[str] = []
    for raw_id in dict.fromkeys(request.ids):
        file_id = _validate_file_id(raw_id)
        try:
            meta = _load_metadata(file_id)
        except HTTPException as exc:
            if exc.status_code == 404:
                missing.append(file_id)
                continue
            raise
        image_path = _image_path(meta)
        try:
            image_path.unlink(missing_ok=True)
            _metadata_path(file_id).unlink(missing_ok=True)
            deleted.append(file_id)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Не удалось удалить {meta['name']}: {exc}") from exc
    return {"deleted": deleted, "missing": missing}


@router.get("/v1/test-ui/images/{file_id}/original", response_class=Response)
def original_image(file_id: str) -> Response:
    """Возвращает исходный файл из постоянной тестовой библиотеки без преобразований."""

    meta = _load_metadata(_validate_file_id(file_id))
    path = _image_path(meta)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Не удалось прочитать исходное изображение") from exc
    media_type = mimetypes.guess_type(meta.get("name") or path.name)[0] or "application/octet-stream"
    return Response(
        content=raw,
        media_type=media_type,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
            "Content-Disposition": f'inline; filename="{path.name}"',
        },
    )


@router.get("/v1/test-ui/images/{file_id}/normalized", response_class=Response)
def normalized_crop(file_id: str) -> Response:
    meta = _load_metadata(_validate_file_id(file_id))
    raw = _image_path(meta).read_bytes()
    image = _decode_image(raw)
    normalization = normalize_black_background(image)
    ok, encoded = cv2.imencode(".jpg", normalization.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise HTTPException(status_code=500, detail="Не удалось закодировать normalized crop")
    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
            "X-ToolOCR-Normalization-Status": normalization.status,
        },
    )


@router.post("/v1/test-ui/run")
async def run_test(request: TestRunRequest) -> dict[str, Any]:
    _ensure_storage()
    results: list[dict[str, Any]] = []
    folders = _folder_map()
    for raw_id in dict.fromkeys(request.ids):
        file_id = _validate_file_id(raw_id)
        try:
            meta = _load_metadata(file_id)
            raw = _image_path(meta).read_bytes()
            upload = UploadFile(file=io.BytesIO(raw), filename=meta["name"])
            try:
                analysis = await analyze_layout(
                    file=upload,
                    include_debug_images=False,
                    min_area_ratio=0.15,
                    scoring_top_n=8,
                )
            finally:
                await upload.close()

            format_value = analysis.get("format")
            format_status = analysis.get("format_status")
            orientation = analysis.get("orientation") or {}
            normalization = analysis.get("frame_normalization") or {}
            timing = analysis.get("timing") or {}
            folder_id = meta.get("folder_id")
            results.append(
                {
                    "id": file_id,
                    "name": meta["name"],
                    "folder_id": folder_id,
                    "folder_name": folders.get(folder_id, {}).get("name") if folder_id else None,
                    "ok": True,
                    "layout_status": analysis.get("layout_status"),
                    "format": format_value,
                    "format_status": format_status,
                    "orientation_status": orientation.get("status"),
                    "orientation_deg": orientation.get("value_deg"),
                    "normalization_status": normalization.get("status"),
                    "total_ms": timing.get("total_ms"),
                    "debug": analysis,
                }
            )
        except HTTPException as exc:
            name = None
            folder_id = None
            try:
                meta = _load_metadata(file_id)
                name = meta.get("name")
                folder_id = meta.get("folder_id")
            except HTTPException:
                pass
            detail = exc.detail
            results.append(
                {
                    "id": file_id,
                    "name": name or file_id,
                    "folder_id": folder_id,
                    "folder_name": folders.get(folder_id, {}).get("name") if folder_id else None,
                    "ok": False,
                    "layout_status": "error",
                    "format": None,
                    "format_status": None,
                    "orientation_status": None,
                    "orientation_deg": None,
                    "normalization_status": None,
                    "total_ms": None,
                    "error": detail,
                    "debug": {"status_code": exc.status_code, "detail": detail},
                }
            )
        except Exception as exc:
            results.append(
                {
                    "id": file_id,
                    "name": file_id,
                    "folder_id": None,
                    "folder_name": None,
                    "ok": False,
                    "layout_status": "error",
                    "format": None,
                    "format_status": None,
                    "orientation_status": None,
                    "orientation_deg": None,
                    "normalization_status": None,
                    "total_ms": None,
                    "error": f"Внутренняя ошибка: {type(exc).__name__}: {exc}",
                    "debug": {"error": type(exc).__name__, "message": str(exc)},
                }
            )
    return {"count": len(results), "results": results}

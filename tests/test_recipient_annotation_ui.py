import asyncio

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from ocr.app import recipient_annotation_ui as ui
from ocr.app.recipient_benchmark import load_ground_truth


def payload(**changes):
    data = dict(format="C4", canonical_hash="a" * 64, ground_truth_status="verified",
                recipient_status="present", recipient_bbox=dict(x=.5, y=.4, width=.4, height=.4))
    return ui.Annotation(**(data | changes))


@pytest.mark.parametrize("changes", [
    {"recipient_bbox": None},
    {"recipient_status": "absent"},
    {"postcode_box_status": "present"},
    {"recipient_bbox": dict(x=.9, y=.2, width=.2, height=.2)},
    {"recipient_bbox": dict(x=float("nan"), y=.2, width=.2, height=.2)},
])
def test_invalid_ground_truth_rejected(changes):
    with pytest.raises(ValidationError):
        payload(**changes)


def test_draft_can_have_no_box():
    assert payload(ground_truth_status="needs_review", recipient_bbox=None).recipient_bbox is None


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(ui.test_ui, "TEST_STORAGE_DIR", tmp_path)
    monkeypatch.setattr(ui.test_ui, "_require_folder", lambda value: value)
    meta = {"id": "1" * 32, "folder_id": "2" * 32, "name": "письмо.jpg"}
    monkeypatch.setattr(ui.test_ui, "_iter_metadata", lambda: [meta])

    async def canonical(file_id, fmt):
        return np.zeros((10, 20, 3), dtype=np.uint8), "a" * 64

    monkeypatch.setattr(ui, "_canonical", canonical)
    return tmp_path, meta


def test_save_and_export_roundtrip(storage):
    root, meta = storage
    asyncio.run(ui.save(meta["id"], payload(notes="Ручная проверка")))
    response = asyncio.run(ui.export(meta["folder_id"], "C4"))
    csv_path = root / "export.csv"
    csv_path.write_bytes(response.body)
    rows = load_ground_truth(csv_path)
    assert len(rows) == 1
    assert rows[0].filename == meta["name"]
    assert rows[0].ground_truth_status == "verified"
    assert rows[0].recipient_bbox.x == .5
    assert rows[0].notes == "Ручная проверка"
    assert ui.items(meta["folder_id"], "C4")["items"][0]["status"] == "verified"
    assert not list((root / "recipient-annotations").glob("*.tmp"))


def test_save_refuses_changed_canonical(storage):
    _, meta = storage
    with pytest.raises(HTTPException) as error:
        asyncio.run(ui.save(meta["id"], payload(canonical_hash="b" * 64)))
    assert error.value.status_code == 409
    assert ui._read(meta["id"], "C4") is None


def test_export_refuses_stale_verified_annotation(storage):
    _, meta = storage
    ui._write(ui._path(meta["id"], "C4"), payload(canonical_hash="b" * 64).model_dump())
    with pytest.raises(HTTPException) as error:
        asyncio.run(ui.export(meta["folder_id"], "C4"))
    assert error.value.status_code == 409


def test_export_refuses_ambiguous_library_names(storage, monkeypatch):
    _, meta = storage
    asyncio.run(ui.save(meta["id"], payload()))
    monkeypatch.setattr(ui.test_ui, "_iter_metadata", lambda: [meta, meta | {"id": "3" * 32, "folder_id": "4" * 32}])
    with pytest.raises(HTTPException) as error:
        asyncio.run(ui.export(meta["folder_id"], "C4"))
    assert error.value.status_code == 409


def test_annotation_path_rejects_traversal():
    with pytest.raises(HTTPException):
        ui._path("../escape", "C4")

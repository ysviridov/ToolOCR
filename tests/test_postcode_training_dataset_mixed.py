import asyncio
from pathlib import Path

import numpy as np

from scripts.export_postcode_training_dataset import _index_images
from scripts.export_postcode_training_dataset_mixed import (
    _crop_candidate_names,
    _extract_with_fallback,
    _resolve_postcode_crop,
)


def test_crop_candidate_prefers_source_extension_and_crop_suffix():
    candidates = _crop_candidate_names("group/a.jpg")
    assert candidates[0] == "group/a_crop.jpg"
    assert "group/a_crop.png" in candidates


def test_resolve_postcode_crop_by_unique_basename(tmp_path: Path):
    crop = tmp_path / "0B0EC6000015_0011_20260825001257_00350_1_crop.jpg"
    crop.write_bytes(b"x")

    by_relative, unique_basename, duplicates = _index_images(tmp_path)
    resolved = _resolve_postcode_crop(
        "0B0EC6000015_0011_20260825001257_00350_1.jpg",
        by_relative=by_relative,
        unique_basename=unique_basename,
    )

    assert duplicates == []
    assert resolved == crop


def test_full_envelope_success_does_not_use_crop(tmp_path: Path, monkeypatch):
    full = tmp_path / "a.jpg"
    crop = tmp_path / "a_crop.jpg"
    full.write_bytes(b"full")
    crop.write_bytes(b"crop")
    calls = []

    async def fake_extract(path, *, input_mode, expected_format):
        calls.append((path.name, input_mode))
        canvas = np.full((128, 96), 255, dtype=np.uint8)
        return canvas, [(1, canvas, {"status": "applied"})] * 6, {}

    monkeypatch.setattr(
        "scripts.export_postcode_training_dataset_mixed._extract_one",
        fake_extract,
    )

    cells, mode, reason = asyncio.run(
        _extract_with_fallback(
            filename="a.jpg",
            full_image_path=full,
            crop_path=crop,
            expected_format=None,
        )
    )

    assert cells is not None
    assert mode == "full-envelope"
    assert reason is None
    assert calls == [("a.jpg", "full-envelope")]


def test_full_envelope_failure_uses_crop(tmp_path: Path, monkeypatch):
    full = tmp_path / "a.jpg"
    crop = tmp_path / "a_crop.jpg"
    full.write_bytes(b"full")
    crop.write_bytes(b"crop")
    calls = []

    async def fake_extract(path, *, input_mode, expected_format):
        calls.append((path.name, input_mode))
        if input_mode == "full-envelope":
            raise RuntimeError("orientation_unresolved")
        canvas = np.full((128, 96), 255, dtype=np.uint8)
        return canvas, [(1, canvas, {"status": "applied"})] * 6, {}

    monkeypatch.setattr(
        "scripts.export_postcode_training_dataset_mixed._extract_one",
        fake_extract,
    )

    cells, mode, reason = asyncio.run(
        _extract_with_fallback(
            filename="a.jpg",
            full_image_path=full,
            crop_path=crop,
            expected_format=None,
        )
    )

    assert cells is not None
    assert mode == "postcode-crop-fallback"
    assert reason is None
    assert calls == [
        ("a.jpg", "full-envelope"),
        ("a_crop.jpg", "postcode-crop"),
    ]


def test_missing_full_image_can_use_crop(tmp_path: Path, monkeypatch):
    crop = tmp_path / "a_crop.jpg"
    crop.write_bytes(b"crop")

    async def fake_extract(path, *, input_mode, expected_format):
        assert path == crop
        assert input_mode == "postcode-crop"
        canvas = np.full((128, 96), 255, dtype=np.uint8)
        return canvas, [(1, canvas, {"status": "applied"})] * 6, {}

    monkeypatch.setattr(
        "scripts.export_postcode_training_dataset_mixed._extract_one",
        fake_extract,
    )

    cells, mode, reason = asyncio.run(
        _extract_with_fallback(
            filename="a.jpg",
            full_image_path=None,
            crop_path=crop,
            expected_format=None,
        )
    )

    assert cells is not None
    assert mode == "postcode-crop-fallback"
    assert reason is None

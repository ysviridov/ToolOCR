import asyncio
from pathlib import Path

import cv2
import numpy as np

from ocr.app.roi import PixelRect
from scripts.export_postcode_training_dataset import (
    _choose_dataset_splits,
    _extract_one,
    _index_images,
    _read_ground_truth,
    _resolve_image_path,
    _validate_training_rows,
)


def test_ground_truth_minimal_cp1251_semicolon(tmp_path: Path):
    content = (
        "filename;postcode\r\n"
        "a.jpg;123456\r\n"
        "b.jpg;654321\r\n"
    )
    path = tmp_path / "gt.csv"
    path.write_bytes(content.encode("cp1251"))

    rows, encoding, delimiter = _read_ground_truth(path)
    valid, skipped = _validate_training_rows(rows)

    assert encoding == "cp1251"
    assert delimiter == ";"
    assert [item["filename"] for item in valid] == ["a.jpg", "b.jpg"]
    assert [item["postcode"] for item in valid] == ["123456", "654321"]
    assert skipped == []


def test_optional_legacy_fields_do_not_filter_printed_digits(tmp_path: Path):
    path = tmp_path / "gt.csv"
    path.write_text(
        "filename,postcode,format,postcode_source\n"
        "a.jpg,123456,C4,stencil\n"
        "b.jpg,654321,C5,printed\n",
        encoding="utf-8",
    )

    rows, _, _ = _read_ground_truth(path)
    valid, skipped = _validate_training_rows(rows)

    assert {item["filename"] for item in valid} == {"a.jpg", "b.jpg"}
    assert skipped == []


def test_ground_truth_rejects_invalid_postcode_and_casefold_duplicates(tmp_path: Path):
    path = tmp_path / "gt.csv"
    path.write_text(
        "filename,postcode\n"
        "a.jpg,012345\n"
        "B.jpg,123456\n"
        "b.JPG,654321\n",
        encoding="utf-8",
    )

    rows, _, delimiter = _read_ground_truth(path)
    valid, skipped = _validate_training_rows(rows)

    assert delimiter == ","
    assert [item["filename"] for item in valid] == ["B.jpg"]
    assert skipped[0]["reason"] == "invalid_postcode=012345"
    assert skipped[1]["reason"] == "duplicate_filename"


def test_direct_image_index_supports_relative_path_and_unique_basename(tmp_path: Path):
    (tmp_path / "group-a").mkdir()
    (tmp_path / "group-b").mkdir()
    first = tmp_path / "group-a" / "a.jpg"
    second = tmp_path / "group-b" / "b.png"
    duplicate_1 = tmp_path / "group-a" / "same.jpg"
    duplicate_2 = tmp_path / "group-b" / "same.jpg"
    for path in (first, second, duplicate_1, duplicate_2):
        path.write_bytes(b"x")

    by_relative, unique_basename, duplicate_basenames = _index_images(tmp_path)

    assert _resolve_image_path(
        "group-a/a.jpg",
        by_relative=by_relative,
        unique_basename=unique_basename,
    ) == first
    assert _resolve_image_path(
        "b.png",
        by_relative=by_relative,
        unique_basename=unique_basename,
    ) == second
    assert _resolve_image_path(
        "same.jpg",
        by_relative=by_relative,
        unique_basename=unique_basename,
    ) is None
    assert duplicate_basenames == ["same.jpg"]


def test_train_val_test_split_is_by_source_file():
    rows = [
        {"filename": "a.jpg", "postcode": "123456"},
        {"filename": "b.jpg", "postcode": "789012"},
        {"filename": "c.jpg", "postcode": "345678"},
        {"filename": "d.jpg", "postcode": "901234"},
        {"filename": "e.jpg", "postcode": "567890"},
        {"filename": "f.jpg", "postcode": "112233"},
        {"filename": "g.jpg", "postcode": "445566"},
        {"filename": "h.jpg", "postcode": "778899"},
        {"filename": "i.jpg", "postcode": "102938"},
        {"filename": "j.jpg", "postcode": "475869"},
    ]

    split = _choose_dataset_splits(
        rows,
        val_fraction=0.20,
        test_fraction=0.20,
        seed=20260831,
    )

    train = {name for name, value in split.items() if value == "train"}
    val = {name for name, value in split.items() if value == "val"}
    test = {name for name, value in split.items() if value == "test"}
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2
    assert not train.intersection(val)
    assert not train.intersection(test)
    assert not val.intersection(test)
    assert train | val | test == {row["filename"] for row in rows}


def test_postcode_crop_wrapper_produces_six_canvases(tmp_path: Path, monkeypatch):
    crop = np.full((150, 320, 3), 235, dtype=np.uint8)
    for center_x in (60, 86, 112, 138, 164, 190):
        cv2.line(crop, (center_x, 34), (center_x, 70), (25, 25, 25), 4)
        cv2.line(crop, (center_x, 70), (center_x + 8, 76), (25, 25, 25), 4)

    path = tmp_path / "postcode.png"
    assert cv2.imwrite(str(path), crop)

    def fake_detector(image, search):
        assert search.y == 300
        assert image.shape[:2] == (450, 320)
        return (
            PixelRect(21, 310, 190, 70),
            7,
            0.05,
            0.95,
            {
                "confirmation_mode": "strict_start_marker",
                "bar_width_px": 20.0,
                "bar_step_px": 26.0,
                "row_y_norm": 320.0 / 450.0,
            },
        )

    monkeypatch.setattr(
        "scripts.export_postcode_training_dataset._postcode_stencil_bbox",
        fake_detector,
    )

    _, cells, debug = asyncio.run(
        _extract_one(
            path,
            input_mode="postcode-crop",
            expected_format=None,
        )
    )

    assert debug["input_mode"] == "postcode-crop"
    assert len(cells) == 6
    assert [index for index, _, _ in cells] == [1, 2, 3, 4, 5, 6]
    assert all(canvas.shape == (128, 96) for _, canvas, _ in cells)

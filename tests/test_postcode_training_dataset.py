from pathlib import Path

from scripts.export_postcode_training_dataset import (
    _choose_validation_files,
    _read_ground_truth,
    _validate_training_rows,
)


def test_ground_truth_cp1251_semicolon_and_cyrillic_c4(tmp_path: Path):
    content = (
        "filename;format;postcode;postcode_source;recipient_address_raw\r\n"
        "a.jpg;С4;123456;stencil;Москва\r\n"
        "b.jpg;C4;654321;printed;Москва\r\n"
        "c.jpg;C5;111111;stencil;Москва\r\n"
    )
    path = tmp_path / "gt.csv"
    path.write_bytes(content.encode("cp1251"))

    rows, encoding, delimiter = _read_ground_truth(path)
    valid, skipped = _validate_training_rows(rows)

    assert encoding == "cp1251"
    assert delimiter == ";"
    assert len(valid) == 1
    assert valid[0]["filename"] == "a.jpg"
    assert valid[0]["format"] == "C4"
    assert valid[0]["postcode"] == "123456"
    assert {item["reason"] for item in skipped} == {"postcode_source=printed", "format=C5"}


def test_ground_truth_rejects_invalid_postcode_and_duplicates(tmp_path: Path):
    path = tmp_path / "gt.csv"
    path.write_text(
        "filename,format,postcode,postcode_source\n"
        "a.jpg,C4,012345,stencil\n"
        "b.jpg,C4,123456,stencil\n"
        "b.jpg,C4,654321,stencil\n",
        encoding="utf-8",
    )

    rows, _, delimiter = _read_ground_truth(path)
    valid, skipped = _validate_training_rows(rows)

    assert delimiter == ","
    assert [item["filename"] for item in valid] == ["b.jpg"]
    assert skipped[0]["reason"] == "invalid_postcode=012345"
    assert skipped[1]["reason"] == "duplicate_filename"


def test_validation_split_is_by_source_file_and_covers_all_digits():
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

    val_files = _choose_validation_files(rows, fraction=0.30, seed=20260828)
    train_files = {row["filename"] for row in rows}.difference(val_files)

    assert val_files
    assert train_files
    assert not val_files.intersection(train_files)
    assert len(val_files) == 3

    val_digits = set(
        "".join(row["postcode"] for row in rows if row["filename"] in val_files)
    )
    train_digits = set(
        "".join(row["postcode"] for row in rows if row["filename"] in train_files)
    )
    assert val_digits == set("0123456789")
    assert train_digits == set("0123456789")

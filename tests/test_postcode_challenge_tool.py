import csv
from pathlib import Path

import pytest

from scripts.evaluate_postcode_challenge import (
    _model_deltas,
    _parse_model_specs,
    _read_challenge,
    _safe_stem,
)


def test_challenge_csv_requires_unique_valid_six_digit_postcodes(tmp_path: Path):
    path = tmp_path / "challenge.csv"
    path.write_text(
        "filename,postcode\n"
        "a.jpg,123290\n"
        "b.tiff,105005\n",
        encoding="utf-8",
    )

    rows = _read_challenge(path)

    assert [(item.filename, item.postcode) for item in rows] == [
        ("a.jpg", "123290"),
        ("b.tiff", "105005"),
    ]


def test_challenge_csv_rejects_leading_zero_and_duplicate_filename(tmp_path: Path):
    bad_postcode = tmp_path / "bad_postcode.csv"
    bad_postcode.write_text(
        "filename,postcode\n"
        "a.jpg,012345\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="не начинаться с 0"):
        _read_challenge(bad_postcode)

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "filename,postcode\n"
        "A.jpg,123456\n"
        "a.JPG,654321\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate filename"):
        _read_challenge(duplicate)


def test_model_specs_support_ab_and_reject_duplicate_labels():
    specs = _parse_model_specs(
        [
            "v1=/app/models/postcode_digit_v1.onnx",
            "v2=/work/postcode-v2/model.onnx",
        ]
    )
    assert [item.label for item in specs] == ["v1", "v2"]
    assert str(specs[1].path) == "/work/postcode-v2/model.onnx"

    with pytest.raises(ValueError, match="Повторный label"):
        _parse_model_specs(["v1=/a.onnx", "V1=/b.onnx"])


def test_model_delta_marks_candidate_recovery_and_regression():
    rows = [
        {
            "model": "v1",
            "filename": "a.jpg",
            "truth": "123290",
            "predicted": "123280",
            "exact_correct": 0,
        },
        {
            "model": "v2",
            "filename": "a.jpg",
            "truth": "123290",
            "predicted": "123290",
            "exact_correct": 1,
        },
        {
            "model": "v1",
            "filename": "b.jpg",
            "truth": "105005",
            "predicted": "105005",
            "exact_correct": 1,
        },
        {
            "model": "v2",
            "filename": "b.jpg",
            "truth": "105005",
            "predicted": "109009",
            "exact_correct": 0,
        },
    ]
    specs = _parse_model_specs(["v1=/a.onnx", "v2=/b.onnx"])

    deltas = _model_deltas(rows, specs)

    assert [item["exact_delta"] for item in deltas] == [1, -1]
    assert deltas[0]["candidate_predicted"] == "123290"


def test_safe_stem_never_contains_path_separators():
    value = _safe_stem("../../Письмо 01?.jpg")
    assert "/" not in value
    assert "\\" not in value
    assert value

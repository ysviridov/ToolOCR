from pathlib import Path

import pytest

from ocr.app.recipient_benchmark import NormalizedRect, bbox_metrics, classify_fit, load_ground_truth


HEADER = (
    "filename,format,ground_truth_status,recipient_status,x,y,width,height,"
    "postcode_box_status,postcode_x,postcode_y,postcode_width,postcode_height,notes\n"
)


def test_bbox_metrics_identical_boxes_are_perfect():
    rect = NormalizedRect(0.5, 0.4, 0.3, 0.2)
    metrics = bbox_metrics(rect, rect)

    assert metrics.iou == pytest.approx(1.0)
    assert metrics.content_recall == pytest.approx(1.0)
    assert metrics.overcrop_fraction == pytest.approx(0.0)
    assert metrics.undercrop_fraction == pytest.approx(0.0)
    assert classify_fit(metrics) == "good"


def test_bbox_metrics_detect_undercrop():
    truth = NormalizedRect(0.4, 0.4, 0.4, 0.3)
    predicted = NormalizedRect(0.4, 0.4, 0.2, 0.3)
    metrics = bbox_metrics(truth, predicted)

    assert metrics.content_recall == pytest.approx(0.5)
    assert metrics.overcrop_fraction == pytest.approx(0.0)
    assert classify_fit(metrics) == "too_small"


def test_bbox_metrics_detect_overcrop():
    truth = NormalizedRect(0.5, 0.5, 0.2, 0.2)
    predicted = NormalizedRect(0.3, 0.3, 0.6, 0.6)
    metrics = bbox_metrics(truth, predicted)

    assert metrics.content_recall == pytest.approx(1.0)
    assert metrics.overcrop_fraction > 0.8
    assert classify_fit(metrics) == "too_large"


def test_load_ground_truth_accepts_verified_recipient_and_postcode_box(tmp_path: Path):
    path = tmp_path / "gt.csv"
    path.write_text(
        HEADER
        + "letter.jpg,C5,verified,present,0.50,0.35,0.40,0.40,present,0.78,0.65,0.10,0.08,ok\n",
        encoding="utf-8",
    )

    rows = load_ground_truth(path)

    assert len(rows) == 1
    row = rows[0]
    assert row.filename == "letter.jpg"
    assert row.recipient_bbox is not None
    assert row.postcode_box_bbox is not None
    assert row.postcode_box_status == "present"


def test_load_ground_truth_rejects_verified_present_without_bbox(tmp_path: Path):
    path = tmp_path / "gt.csv"
    path.write_text(
        HEADER
        + "letter.jpg,DL,verified,present,,,,,unknown,,,,,missing bbox\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires bbox|требует bbox"):
        load_ground_truth(path)


def test_needs_review_may_keep_incomplete_seed_bbox(tmp_path: Path):
    path = tmp_path / "gt.csv"
    path.write_text(
        HEADER
        + "letter.jpg,C4,needs_review,present,,,,,unknown,,,,,bootstrap miss\n",
        encoding="utf-8",
    )

    rows = load_ground_truth(path)

    assert rows[0].ground_truth_status == "needs_review"
    assert rows[0].recipient_bbox is None

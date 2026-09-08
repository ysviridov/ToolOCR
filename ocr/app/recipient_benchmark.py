from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


GROUND_TRUTH_STATUSES = {"needs_review", "verified", "exclude"}
RECIPIENT_STATUSES = {"present", "absent"}
POSTCODE_BOX_STATUSES = {"present", "absent", "unknown"}
SUPPORTED_FORMATS = {"DL", "C5", "C4"}


@dataclass(frozen=True, slots=True)
class NormalizedRect:
    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class RecipientGroundTruth:
    filename: str
    format: str
    ground_truth_status: str
    recipient_status: str
    recipient_bbox: NormalizedRect | None
    postcode_box_status: str
    postcode_box_bbox: NormalizedRect | None
    notes: str


@dataclass(frozen=True, slots=True)
class BboxMetrics:
    intersection_area: float
    union_area: float
    iou: float
    content_recall: float
    overcrop_fraction: float
    undercrop_fraction: float


def _parse_float(raw: str | None, *, field: str, line: int) -> float | None:
    value = str(raw or "").strip()
    if value == "":
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"CSV:{line}: {field} должен быть числом 0..1") from exc
    if not math.isfinite(result):
        raise ValueError(f"CSV:{line}: {field} должен быть конечным числом")
    return result


def _read_rect(raw: dict[str, str], prefix: str, *, line: int) -> NormalizedRect | None:
    fields = [f"{prefix}x", f"{prefix}y", f"{prefix}width", f"{prefix}height"]
    values = [_parse_float(raw.get(name), field=name, line=line) for name in fields]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"CSV:{line}: bbox {prefix or 'recipient_'} должен содержать все x,y,width,height")
    x, y, width, height = (float(value) for value in values)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"CSV:{line}: bbox должен иметь x,y>=0 и width,height>0")
    if x > 1 or y > 1 or x + width > 1.000001 or y + height > 1.000001:
        raise ValueError(f"CSV:{line}: bbox должен находиться в normalized диапазоне 0..1")
    return NormalizedRect(x=x, y=y, width=width, height=height)


def load_ground_truth(path: Path) -> list[RecipientGroundTruth]:
    if not path.is_file():
        raise FileNotFoundError(f"RECIPIENT benchmark CSV не найден: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "filename",
            "format",
            "ground_truth_status",
            "recipient_status",
            "x",
            "y",
            "width",
            "height",
            "postcode_box_status",
            "postcode_x",
            "postcode_y",
            "postcode_width",
            "postcode_height",
            "notes",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("RECIPIENT benchmark CSV: отсутствуют поля " + ", ".join(sorted(missing)))

        rows: list[RecipientGroundTruth] = []
        seen: set[str] = set()
        for line, raw in enumerate(reader, start=2):
            filename = str(raw.get("filename") or "").strip()
            if not filename:
                raise ValueError(f"CSV:{line}: пустой filename")
            key = filename.casefold()
            if key in seen:
                raise ValueError(f"CSV:{line}: duplicate filename={filename}")
            seen.add(key)

            format_value = str(raw.get("format") or "").strip().upper()
            if format_value not in SUPPORTED_FORMATS:
                raise ValueError(f"CSV:{line}: format должен быть DL/C5/C4")

            gt_status = str(raw.get("ground_truth_status") or "").strip().lower()
            if gt_status not in GROUND_TRUTH_STATUSES:
                raise ValueError(f"CSV:{line}: ground_truth_status должен быть needs_review/verified/exclude")

            recipient_status = str(raw.get("recipient_status") or "").strip().lower()
            if recipient_status not in RECIPIENT_STATUSES:
                raise ValueError(f"CSV:{line}: recipient_status должен быть present/absent")
            recipient_bbox = _read_rect(raw, "", line=line)

            postcode_status = str(raw.get("postcode_box_status") or "unknown").strip().lower()
            if postcode_status not in POSTCODE_BOX_STATUSES:
                raise ValueError(f"CSV:{line}: postcode_box_status должен быть present/absent/unknown")
            postcode_bbox = _read_rect(raw, "postcode_", line=line)

            if gt_status == "verified":
                if recipient_status == "present" and recipient_bbox is None:
                    raise ValueError(f"CSV:{line}: verified recipient_status=present требует bbox")
                if recipient_status == "absent" and recipient_bbox is not None:
                    raise ValueError(f"CSV:{line}: recipient_status=absent не должен иметь bbox")
                if postcode_status == "present" and postcode_bbox is None:
                    raise ValueError(f"CSV:{line}: postcode_box_status=present требует postcode bbox")
                if postcode_status != "present" and postcode_bbox is not None:
                    raise ValueError(f"CSV:{line}: postcode bbox разрешён только при postcode_box_status=present")

            rows.append(
                RecipientGroundTruth(
                    filename=filename,
                    format=format_value,
                    ground_truth_status=gt_status,
                    recipient_status=recipient_status,
                    recipient_bbox=recipient_bbox,
                    postcode_box_status=postcode_status,
                    postcode_box_bbox=postcode_bbox,
                    notes=str(raw.get("notes") or "").strip(),
                )
            )
    return rows


def bbox_metrics(truth: NormalizedRect, predicted: NormalizedRect) -> BboxMetrics:
    x1 = max(truth.x, predicted.x)
    y1 = max(truth.y, predicted.y)
    x2 = min(truth.x2, predicted.x2)
    y2 = min(truth.y2, predicted.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = truth.area + predicted.area - intersection
    iou = intersection / union if union > 0 else 0.0
    recall = intersection / truth.area if truth.area > 0 else 0.0
    overcrop = max(0.0, predicted.area - intersection) / predicted.area if predicted.area > 0 else 0.0
    return BboxMetrics(
        intersection_area=intersection,
        union_area=union,
        iou=iou,
        content_recall=recall,
        overcrop_fraction=overcrop,
        undercrop_fraction=max(0.0, 1.0 - recall),
    )


def classify_fit(
    metrics: BboxMetrics,
    *,
    recall_good: float = 0.95,
    overcrop_good: float = 0.35,
    wrong_block_iou: float = 0.10,
) -> str:
    if metrics.iou < wrong_block_iou and metrics.content_recall < 0.25:
        return "wrong_block"
    too_small = metrics.content_recall < recall_good
    too_large = metrics.overcrop_fraction > overcrop_good
    if too_small and too_large:
        return "poor_fit"
    if too_small:
        return "too_small"
    if too_large:
        return "too_large"
    return "good"

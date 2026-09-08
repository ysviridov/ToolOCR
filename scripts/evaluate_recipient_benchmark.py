from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import statistics
from pathlib import Path

import cv2

# application устанавливает те же layout/orientation hooks, что production OCR service.
from ocr.app import application as _application  # noqa: F401
from ocr.app.format_modes import FormatMode
from ocr.app.gost_r_51506_99 import EnvelopeFormat
from ocr.app.recipient_benchmark import NormalizedRect, bbox_metrics, classify_fit, load_ground_truth
from ocr.app.roi import detect_simple_mail_rois
from ocr.app.roi_test_ui import _analyze_saved_image, _canonical_from_analysis
from ocr.app.test_ui import _iter_metadata


_DEFAULT_GT = "/work/recipient-benchmarks/recipient-v1.csv"
_DEFAULT_OUTPUT = "/work/recipient-benchmarks/recipient-v1-results"


def _safe_stem(filename: str) -> str:
    value = re.sub(r"[^A-Za-z0-9А-Яа-яЁё._-]+", "_", Path(filename).stem).strip("._-")
    return value[:120] or "image"


def _find_file_id(filename: str) -> str:
    matches = [
        item for item in _iter_metadata()
        if str(item.get("name") or "").casefold() == filename.casefold()
    ]
    if not matches:
        raise FileNotFoundError(f"Файл не найден в Test UI volume: {filename}")
    if len(matches) > 1:
        ids = ",".join(str(item.get("id")) for item in matches)
        raise RuntimeError(f"Несколько Test UI файлов с именем {filename}; ids={ids}")
    return str(matches[0]["id"])


def _normalized(rect, width: int, height: int) -> NormalizedRect:
    return NormalizedRect(
        x=rect.x / width,
        y=rect.y / height,
        width=rect.width / width,
        height=rect.height / height,
    )


def _pixel_rect(rect: NormalizedRect, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(round(rect.x * width))))
    y1 = max(0, min(height - 1, int(round(rect.y * height))))
    x2 = max(x1 + 1, min(width, int(round(rect.x2 * width))))
    y2 = max(y1 + 1, min(height, int(round(rect.y2 * height))))
    return x1, y1, x2, y2


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _fmt_rect(rect: NormalizedRect | None) -> dict[str, float | None]:
    if rect is None:
        return {"x": None, "y": None, "width": None, "height": None}
    return {
        "x": round(rect.x, 6),
        "y": round(rect.y, 6),
        "width": round(rect.width, 6),
        "height": round(rect.height, 6),
    }


async def _run(args: argparse.Namespace) -> int:
    gt_rows = load_ground_truth(Path(args.ground_truth))
    verified = [row for row in gt_rows if row.ground_truth_status == "verified"]
    needs_review = sum(row.ground_truth_status == "needs_review" for row in gt_rows)
    excluded = sum(row.ground_truth_status == "exclude" for row in gt_rows)
    if not verified:
        raise ValueError(
            "Нет verified строк. Bootstrap-строки needs_review намеренно не участвуют в метриках."
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    overlay_dir = output / "overlays"
    crop_dir = output / "crops"
    overlay_dir.mkdir(exist_ok=True)
    crop_dir.mkdir(exist_ok=True)

    results: list[dict] = []
    failures: list[dict] = []

    for truth in verified:
        try:
            file_id = _find_file_id(truth.filename)
            envelope_format = EnvelopeFormat(truth.format)
            analysis, image, _ = await _analyze_saved_image(
                file_id,
                format_mode=FormatMode.FIXED,
                expected_format=envelope_format,
            )
            canonical = _canonical_from_analysis(analysis, image)
            roi = detect_simple_mail_rois(canonical.image, envelope_format)
            recipient = next(region for region in roi.regions if region.kind == "recipient_address")
            height, width = canonical.image.shape[:2]
            predicted = (
                _normalized(recipient.detected_bbox, width, height)
                if recipient.detected_bbox is not None
                else None
            )
            predicted_present = predicted is not None and recipient.status == "detected"

            row: dict[str, object] = {
                "filename": truth.filename,
                "file_id": file_id,
                "format": truth.format,
                "recipient_truth": truth.recipient_status,
                "recipient_predicted": "present" if predicted_present else "absent",
                "detector_status": recipient.status,
                "detector_confidence": recipient.confidence,
                "component_count": recipient.component_count,
                "ink_density": recipient.ink_density,
                "canonical_width_px": width,
                "canonical_height_px": height,
                "outcome": "",
                "iou": "",
                "content_recall": "",
                "overcrop_fraction": "",
                "undercrop_fraction": "",
                "gt_x": "",
                "gt_y": "",
                "gt_width": "",
                "gt_height": "",
                "pred_x": "",
                "pred_y": "",
                "pred_width": "",
                "pred_height": "",
                "postcode_box_status": truth.postcode_box_status,
                "postcode_box_evaluated": 0,
                "notes": truth.notes,
            }

            if truth.recipient_bbox is not None:
                gt_rect = _fmt_rect(truth.recipient_bbox)
                row.update({f"gt_{key}": value for key, value in gt_rect.items()})
            if predicted is not None:
                pred_rect = _fmt_rect(predicted)
                row.update({f"pred_{key}": value for key, value in pred_rect.items()})

            if truth.recipient_status == "absent":
                row["outcome"] = "false_positive" if predicted_present else "correct_negative"
            elif not predicted_present or predicted is None:
                row["outcome"] = "miss"
            else:
                assert truth.recipient_bbox is not None
                metrics = bbox_metrics(truth.recipient_bbox, predicted)
                row["iou"] = round(metrics.iou, 6)
                row["content_recall"] = round(metrics.content_recall, 6)
                row["overcrop_fraction"] = round(metrics.overcrop_fraction, 6)
                row["undercrop_fraction"] = round(metrics.undercrop_fraction, 6)
                row["outcome"] = classify_fit(
                    metrics,
                    recall_good=args.recall_good,
                    overcrop_good=args.overcrop_good,
                    wrong_block_iou=args.wrong_block_iou,
                )

            overlay = canonical.image.copy()
            if truth.recipient_bbox is not None:
                x1, y1, x2, y2 = _pixel_rect(truth.recipient_bbox, width, height)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 190, 60), 4)
                cv2.putText(
                    overlay, "GT RECIPIENT", (x1, max(28, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 150, 40), 2, cv2.LINE_AA,
                )
                gt_crop = canonical.image[y1:y2, x1:x2]
                cv2.imwrite(str(crop_dir / f"{file_id}__{_safe_stem(truth.filename)}__gt.jpg"), gt_crop)
            if predicted is not None:
                x1, y1, x2, y2 = _pixel_rect(predicted, width, height)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 140, 255), 3)
                cv2.putText(
                    overlay, "CURRENT DETECTOR", (x1, min(height - 8, y2 + 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 105, 220), 2, cv2.LINE_AA,
                )
                pred_crop = canonical.image[y1:y2, x1:x2]
                cv2.imwrite(str(crop_dir / f"{file_id}__{_safe_stem(truth.filename)}__pred.jpg"), pred_crop)

            if truth.postcode_box_bbox is not None:
                x1, y1, x2, y2 = _pixel_rect(truth.postcode_box_bbox, width, height)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 120, 40), 3)
                cv2.putText(
                    overlay, "GT DEST POSTCODE BOX", (x1, max(28, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 80, 20), 2, cv2.LINE_AA,
                )

            overlay_path = overlay_dir / f"{file_id}__{_safe_stem(truth.filename)}__{row['outcome']}.jpg"
            cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
            results.append(row)
            print(f"OK {truth.filename}: {row['outcome']}")
        except Exception as exc:
            failures.append(
                {
                    "filename": truth.filename,
                    "format": truth.format,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"FAIL {truth.filename}: {type(exc).__name__}: {exc}")

    fieldnames = [
        "filename", "file_id", "format", "recipient_truth", "recipient_predicted",
        "detector_status", "detector_confidence", "component_count", "ink_density",
        "canonical_width_px", "canonical_height_px", "outcome", "iou", "content_recall",
        "overcrop_fraction", "undercrop_fraction", "gt_x", "gt_y", "gt_width", "gt_height",
        "pred_x", "pred_y", "pred_width", "pred_height", "postcode_box_status",
        "postcode_box_evaluated", "notes",
    ]
    _write_csv(output / "files.csv", results, fieldnames)
    _write_csv(output / "failures.csv", failures, ["filename", "format", "error"])

    positive = [row for row in results if row["recipient_truth"] == "present"]
    negative = [row for row in results if row["recipient_truth"] == "absent"]
    detected_positive = [row for row in positive if row["recipient_predicted"] == "present"]
    fitted = [row for row in positive if row["iou"] != ""]
    ious = [float(row["iou"]) for row in fitted]
    recalls = [float(row["content_recall"]) for row in fitted]
    overcrops = [float(row["overcrop_fraction"]) for row in fitted]
    good = sum(row["outcome"] == "good" for row in positive)
    false_positive = sum(row["outcome"] == "false_positive" for row in negative)

    outcomes: dict[str, int] = {}
    for row in results:
        key = str(row["outcome"])
        outcomes[key] = outcomes.get(key, 0) + 1

    summary = {
        "schema": "toolocr.recipient-benchmark.v1",
        "stage": "2.3.1",
        "ground_truth": str(Path(args.ground_truth)),
        "rows_total": len(gt_rows),
        "rows_verified": len(verified),
        "rows_needs_review": needs_review,
        "rows_excluded": excluded,
        "rows_evaluated": len(results),
        "failures": failures,
        "thresholds_for_diagnostic_labels": {
            "recall_good": args.recall_good,
            "overcrop_good": args.overcrop_good,
            "wrong_block_iou": args.wrong_block_iou,
        },
        "recipient": {
            "positive_total": len(positive),
            "negative_total": len(negative),
            "detected_positive": len(detected_positive),
            "detection_rate": len(detected_positive) / len(positive) if positive else None,
            "good_total": good,
            "good_rate": good / len(positive) if positive else None,
            "false_positive_total": false_positive,
            "false_positive_rate": false_positive / len(negative) if negative else None,
            "mean_iou": _mean(ious),
            "median_iou": _median(ious),
            "mean_content_recall": _mean(recalls),
            "median_content_recall": _median(recalls),
            "mean_overcrop_fraction": _mean(overcrops),
            "median_overcrop_fraction": _median(overcrops),
            "outcomes": outcomes,
        },
        "destination_postcode_box": {
            "ground_truth_present": sum(row.postcode_box_status == "present" for row in verified),
            "detector_available": False,
            "note": "Stage 2.3.1 сохраняет GT рамки, но текущий RECIPIENT detector ещё не выделяет destination postcode box отдельно.",
        },
        "artifacts": {
            "files_csv": str(output / "files.csv"),
            "failures_csv": str(output / "failures.csv"),
            "overlays": str(overlay_dir),
            "crops": str(crop_dir),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    recipient_summary = summary["recipient"]
    print("\nSUMMARY")
    print(f"verified: {len(verified)}  evaluated: {len(results)}  failures: {len(failures)}")
    print(f"positive: {len(positive)}  detected: {len(detected_positive)}")
    if recipient_summary["detection_rate"] is not None:
        print(f"detection_rate: {recipient_summary['detection_rate']:.2%}")
    if recipient_summary["mean_iou"] is not None:
        print(f"mean_iou: {recipient_summary['mean_iou']:.4f}")
        print(f"mean_content_recall: {recipient_summary['mean_content_recall']:.4f}")
        print(f"mean_overcrop_fraction: {recipient_summary['mean_overcrop_fraction']:.4f}")
    print("outcomes:", outcomes)
    print(f"output: {output}")
    return 2 if failures else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Оценивает текущий RECIPIENT ROI detector по вручную verified normalized bbox."
    )
    parser.add_argument("--ground-truth", default=_DEFAULT_GT)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    parser.add_argument("--recall-good", type=float, default=0.95)
    parser.add_argument("--overcrop-good", type=float, default=0.35)
    parser.add_argument("--wrong-block-iou", type=float, default=0.10)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        for name in ("recall_good", "overcrop_good", "wrong_block_iou"):
            value = float(getattr(args, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"--{name.replace('_', '-')} должен быть 0..1")
        return asyncio.run(_run(args))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

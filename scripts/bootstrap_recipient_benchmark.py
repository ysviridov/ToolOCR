from __future__ import annotations

import argparse
import asyncio
import csv
import re
from pathlib import Path

import cv2

from ocr.app.format_modes import FormatMode
from ocr.app.gost_r_51506_99 import EnvelopeFormat
from ocr.app.roi import detect_simple_mail_rois
from ocr.app.roi_test_ui import _analyze_saved_image, _canonical_from_analysis
from ocr.app.test_ui import _iter_metadata, _load_folders


HEADER = [
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
]


def _safe_stem(filename: str) -> str:
    value = re.sub(r"[^A-Za-z0-9А-Яа-яЁё._-]+", "_", Path(filename).stem).strip("._-")
    return value[:120] or "image"


def _norm_rect(rect, width: int, height: int) -> tuple[str, str, str, str]:
    return (
        f"{rect.x / width:.6f}",
        f"{rect.y / height:.6f}",
        f"{rect.width / width:.6f}",
        f"{rect.height / height:.6f}",
    )


def _folder_id_by_name(name: str) -> str:
    matches = [item for item in _load_folders() if str(item.get("name") or "").casefold() == name.casefold()]
    if not matches:
        raise ValueError(f"Test UI folder не найден: {name}")
    if len(matches) > 1:
        raise ValueError(f"Найдено несколько Test UI folders с именем: {name}")
    return str(matches[0]["id"])


async def _run(args: argparse.Namespace) -> int:
    envelope_format = EnvelopeFormat(args.format)
    folder_id = _folder_id_by_name(args.folder)
    items = [item for item in _iter_metadata() if item.get("folder_id") == folder_id]
    items.sort(key=lambda item: str(item.get("name") or "").casefold())
    if not items:
        raise RuntimeError(f"В папке {args.folder!r} нет изображений")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay_dir = Path(args.overlays)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    failures: list[str] = []

    for meta in items:
        file_id = str(meta["id"])
        filename = str(meta["name"])
        try:
            analysis, image, _ = await _analyze_saved_image(
                file_id,
                format_mode=FormatMode.FIXED,
                expected_format=envelope_format,
            )
            canonical = _canonical_from_analysis(analysis, image)
            roi = detect_simple_mail_rois(canonical.image, envelope_format)
            recipient = next(region for region in roi.regions if region.kind == "recipient_address")
            height, width = canonical.image.shape[:2]

            if recipient.detected_bbox is not None:
                x, y, w, h = _norm_rect(recipient.detected_bbox, width, height)
                note = "bootstrap=current_detector_bbox; обязательно проверить вручную"
            else:
                x = y = w = h = ""
                note = "bootstrap=detector_miss; вручную разметить RECIPIENT bbox"

            rows.append(
                {
                    "filename": filename,
                    "format": envelope_format.value,
                    "ground_truth_status": "needs_review",
                    "recipient_status": "present",
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "postcode_box_status": "unknown",
                    "postcode_x": "",
                    "postcode_y": "",
                    "postcode_width": "",
                    "postcode_height": "",
                    "notes": note,
                }
            )

            overlay = canonical.image.copy()
            if recipient.detected_bbox is not None:
                rect = recipient.detected_bbox
                cv2.rectangle(overlay, (rect.x, rect.y), (rect.x2, rect.y2), (0, 180, 255), 4)
                cv2.putText(
                    overlay,
                    "CURRENT RECIPIENT - NOT GROUND TRUTH",
                    (max(8, rect.x), max(28, rect.y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 100, 220),
                    2,
                    cv2.LINE_AA,
                )
            overlay_path = overlay_dir / f"{file_id}__{_safe_stem(filename)}.jpg"
            if not cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"Не удалось сохранить overlay {overlay_path}")
            print(f"BOOTSTRAP OK   {filename}")
        except Exception as exc:
            failures.append(f"{filename}: {type(exc).__name__}: {exc}")
            print(f"BOOTSTRAP FAIL {filename}: {type(exc).__name__}: {exc}")

    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSUMMARY")
    print(f"folder: {args.folder}")
    print(f"format: {envelope_format.value}")
    print(f"rows: {len(rows)}/{len(items)}")
    print(f"csv: {output}")
    print(f"overlays: {overlay_dir}")
    if failures:
        print(f"failures: {len(failures)}")
        for item in failures:
            print(f"  {item}")
    print("Все строки имеют ground_truth_status=needs_review и НЕ участвуют в метриках до ручной проверки.")
    return 2 if failures else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Создаёт seed CSV для ручной разметки RECIPIENT benchmark из Test UI folder."
    )
    parser.add_argument("--folder", required=True, help="Точное имя Test UI folder")
    parser.add_argument("--format", required=True, choices=["DL", "C5", "C4"])
    parser.add_argument("--output", default="/work/recipient-benchmarks/recipient-v1.csv")
    parser.add_argument("--overlays", default="/work/recipient-benchmarks/bootstrap-overlays")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return asyncio.run(_run(args))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import UploadFile

from ocr.app.format_modes import FormatMode
from ocr.app.gost_r_51506_99 import EnvelopeFormat
from ocr.app.layout_api import analyze_layout
from ocr.app.postcode_digit_cells import derive_postcode_digit_geometry
from ocr.app.postcode_recognizer import _normalize_digit_crop_with_debug
from ocr.app.roi import detect_simple_mail_rois
from ocr.app.roi_test_ui import _canonical_from_analysis
from ocr.app.test_ui import _decode_image


POSTCODE_RE = re.compile(r"^[1-9][0-9]{5}$")
SUPPORTED_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251")


def _read_ground_truth(path: Path) -> tuple[list[dict[str, str]], str, str]:
    raw = path.read_bytes()
    text = None
    encoding = None
    for candidate in SUPPORTED_ENCODINGS:
        try:
            text = raw.decode(candidate)
            encoding = candidate
            break
        except UnicodeDecodeError:
            continue
    if text is None or encoding is None:
        raise ValueError("CSV не удалось декодировать как UTF-8/CP1251")

    first_line = text.splitlines()[0] if text.splitlines() else ""
    delimiter = ";" if first_line.count(";") >= first_line.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    required = {"filename", "format", "postcode", "postcode_source"}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise ValueError("В ground truth отсутствуют поля: " + ", ".join(sorted(missing)))

    rows: list[dict[str, str]] = []
    for index, row in enumerate(reader, start=2):
        normalized = {str(key): str(value or "").strip() for key, value in row.items()}
        normalized["_line"] = str(index)
        rows.append(normalized)
    return rows, encoding, delimiter


def _normalize_format(value: str) -> str:
    # Кириллическая С визуально совпадает с латинской C и часто попадает в CSV.
    return value.strip().upper().replace("С", "C")


def _validate_training_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    valid: list[dict[str, str]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in rows:
        filename = row.get("filename", "")
        postcode = row.get("postcode", "")
        source = row.get("postcode_source", "").casefold()
        fmt = _normalize_format(row.get("format", ""))
        line = int(row.get("_line", "0") or 0)

        if source != "stencil":
            skipped.append({"line": line, "filename": filename, "reason": f"postcode_source={source or 'empty'}"})
            continue
        if fmt != "C4":
            skipped.append({"line": line, "filename": filename, "reason": f"format={fmt or 'empty'}"})
            continue
        if not POSTCODE_RE.fullmatch(postcode):
            skipped.append({"line": line, "filename": filename, "reason": f"invalid_postcode={postcode}"})
            continue
        if not filename:
            skipped.append({"line": line, "filename": filename, "reason": "empty_filename"})
            continue
        if filename in seen:
            skipped.append({"line": line, "filename": filename, "reason": "duplicate_filename"})
            continue
        seen.add(filename)
        valid.append({**row, "format": "C4", "postcode_source": "stencil"})

    return valid, skipped


def _index_test_data(test_data_dir: Path) -> tuple[dict[str, Path], list[str]]:
    by_name: dict[str, Path] = {}
    duplicate_names: list[str] = []
    for sidecar in sorted(test_data_dir.glob("*.json")):
        if len(sidecar.stem) != 32:
            continue
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = str(meta.get("name") or "").strip()
        suffix = str(meta.get("suffix") or "").lower()
        file_id = str(meta.get("id") or "").lower()
        if not name or len(file_id) != 32 or not suffix:
            continue
        image_path = test_data_dir / f"{file_id}{suffix}"
        if not image_path.is_file():
            continue
        if name in by_name:
            duplicate_names.append(name)
            continue
        by_name[name] = image_path
    return by_name, sorted(set(duplicate_names))


async def _extract_one(image_path: Path) -> tuple[np.ndarray, list[tuple[int, np.ndarray, dict[str, Any]]], dict[str, Any]]:
    raw = image_path.read_bytes()
    image = _decode_image(raw)
    upload = UploadFile(file=io.BytesIO(raw), filename=image_path.name)
    try:
        analysis = await analyze_layout(
            file=upload,
            include_debug_images=False,
            min_area_ratio=0.15,
            scoring_top_n=8,
            format_mode=FormatMode.FIXED,
            expected_format=EnvelopeFormat.C4,
        )
    finally:
        await upload.close()

    canonical = _canonical_from_analysis(analysis, image)
    roi = detect_simple_mail_rois(canonical.image, EnvelopeFormat.C4)
    geometry = derive_postcode_digit_geometry(
        roi,
        image_width=int(canonical.image.shape[1]),
        image_height=int(canonical.image.shape[0]),
    )
    if geometry.status != "ready" or len(geometry.cells) != 6:
        raise RuntimeError(f"digit_geometry:{geometry.status}:{geometry.reason}")

    cells: list[tuple[int, np.ndarray, dict[str, Any]]] = []
    for cell in geometry.cells:
        canvas, debug = _normalize_digit_crop_with_debug(canonical.image, cell)
        if canvas is None:
            raise RuntimeError(f"digit_{cell.index}_preprocess:{debug.get('status')}")
        cells.append((cell.index, canvas, debug))
    return canonical.image, cells, analysis


def _digit_counts(rows: list[dict[str, str]]) -> Counter[str]:
    return Counter("".join(row["postcode"] for row in rows))


def _choose_validation_files(rows: list[dict[str, str]], *, fraction: float, seed: int) -> set[str]:
    filenames = [row["filename"] for row in rows]
    if len(filenames) < 3:
        return set(filenames[-1:])

    val_size = max(1, min(len(filenames) - 1, round(len(filenames) * fraction)))
    total_counts = _digit_counts(rows)
    total_digits = sum(total_counts.values())
    target = {digit: total_counts[digit] / total_digits for digit in "0123456789"}
    row_by_name = {row["filename"]: row for row in rows}
    rng = random.Random(seed)

    best: tuple[float, set[str]] | None = None
    iterations = max(3000, len(filenames) * 100)
    for _ in range(iterations):
        candidate = set(rng.sample(filenames, val_size))
        val_rows = [row_by_name[name] for name in candidate]
        train_rows = [row for row in rows if row["filename"] not in candidate]
        val_counts = _digit_counts(val_rows)
        train_counts = _digit_counts(train_rows)
        val_total = max(1, sum(val_counts.values()))

        score = 0.0
        for digit in "0123456789":
            if val_counts[digit] == 0:
                score += 8.0
            if train_counts[digit] == 0:
                score += 20.0
            score += abs(val_counts[digit] / val_total - target[digit])
        if best is None or score < best[0]:
            best = (score, candidate)
    assert best is not None
    return best[1]


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "filename",
        "digit_index",
        "label",
        "split",
        "sample_path",
        "preprocess_status",
        "suppressed_components",
        "restored_components",
        "suppressed_ink_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


async def export_dataset(args: argparse.Namespace) -> int:
    ground_truth_path = Path(args.ground_truth).resolve()
    test_data_dir = Path(args.test_data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    rows, encoding, delimiter = _read_ground_truth(ground_truth_path)
    valid_rows, skipped_rows = _validate_training_rows(rows)
    image_index, duplicate_test_names = _index_test_data(test_data_dir)
    if not valid_rows:
        raise RuntimeError("После валидации не осталось stencil/C4 строк")

    val_files = _choose_validation_files(valid_rows, fraction=args.val_fraction, seed=args.seed)
    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    used_source_rows: list[dict[str, str]] = []

    for current, row in enumerate(valid_rows, start=1):
        filename = row["filename"]
        postcode = row["postcode"]
        image_path = image_index.get(filename)
        print(f"[{current:02d}/{len(valid_rows):02d}] {filename}")
        if image_path is None:
            failures.append({"filename": filename, "postcode": postcode, "reason": "image_not_found_in_test_data"})
            continue
        try:
            _, cells, _ = await _extract_one(image_path)
        except Exception as exc:
            failures.append({"filename": filename, "postcode": postcode, "reason": f"{type(exc).__name__}: {exc}"})
            continue

        split = "val" if filename in val_files else "train"
        stem = Path(filename).stem
        sample_rows: list[dict[str, Any]] = []
        sample_paths: list[Path] = []
        complete = True
        for digit_index, canvas, debug in cells:
            label = postcode[digit_index - 1]
            relative = Path("samples") / f"{stem}__d{digit_index}__y{label}.png"
            target = output_dir / relative
            if not cv2.imwrite(str(target), canvas):
                complete = False
                failures.append({"filename": filename, "postcode": postcode, "reason": f"cannot_write_digit_{digit_index}"})
                break
            sample_paths.append(target)
            sample_rows.append(
                {
                    "filename": filename,
                    "digit_index": digit_index,
                    "label": label,
                    "split": split,
                    "sample_path": relative.as_posix(),
                    "preprocess_status": debug.get("status"),
                    "suppressed_components": debug.get("suppressed_components"),
                    "restored_components": debug.get("restored_components"),
                    "suppressed_ink_ratio": debug.get("suppressed_ink_ratio"),
                }
            )
        if not complete or len(sample_rows) != 6:
            for target in sample_paths:
                target.unlink(missing_ok=True)
            continue
        manifest.extend(sample_rows)
        used_source_rows.append(row)

    if not manifest:
        raise RuntimeError("Не удалось экспортировать ни одной digit-cell")

    _write_manifest(output_dir / "manifest.csv", manifest)
    successful_files = sorted({row["filename"] for row in manifest})
    train_labels = Counter(row["label"] for row in manifest if row["split"] == "train")
    val_labels = Counter(row["label"] for row in manifest if row["split"] == "val")
    summary = {
        "schema": "toolocr.postcode-digit-dataset.v1",
        "ground_truth": {
            "path": str(ground_truth_path),
            "encoding": encoding,
            "delimiter": delimiter,
            "rows_total": len(rows),
            "rows_valid_stencil_c4": len(valid_rows),
            "rows_skipped": skipped_rows,
        },
        "test_data": {
            "path": str(test_data_dir),
            "indexed_images": len(image_index),
            "duplicate_display_names": duplicate_test_names,
        },
        "dataset": {
            "successful_files": len(successful_files),
            "digit_samples": len(manifest),
            "train_files": len({row["filename"] for row in manifest if row["split"] == "train"}),
            "val_files": len({row["filename"] for row in manifest if row["split"] == "val"}),
            "train_digit_distribution": {digit: train_labels[digit] for digit in "0123456789"},
            "val_digit_distribution": {digit: val_labels[digit] for digit in "0123456789"},
            "seed": args.seed,
            "val_fraction": args.val_fraction,
        },
        "failures": failures,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary["dataset"], ensure_ascii=False, indent=2))
    if failures:
        print(f"WARN: failures={len(failures)}; детали: {output_dir / 'summary.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Экспорт C4 stencil digit-canvas из ToolOCR test-data")
    parser.add_argument("--ground-truth", required=True, help="CSV ground truth (UTF-8 или CP1251; ; или ,)")
    parser.add_argument("--test-data-dir", default="/app/test-data", help="Каталог Docker volume test UI")
    parser.add_argument("--output-dir", default="/work/postcode-c4-dataset", help="Каталог выходного датасета")
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not (0.05 <= args.val_fraction <= 0.50):
        raise SystemExit("--val-fraction должен быть в диапазоне 0.05..0.50")
    return asyncio.run(export_dataset(args))


if __name__ == "__main__":
    raise SystemExit(main())

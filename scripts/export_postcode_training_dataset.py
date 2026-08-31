from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import random
import re
import shutil
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
from ocr.app.roi import (
    PixelRect,
    RoiDetection,
    RoiRegion,
    _postcode_stencil_bbox,
    detect_simple_mail_rois,
)
from ocr.app.roi_test_ui import _canonical_from_analysis
from ocr.app.test_ui import _decode_image


POSTCODE_RE = re.compile(r"^[1-9][0-9]{5}$")
SUPPORTED_ENCODINGS = ("utf-8-sig", "utf-8", "cp1251")
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_FULL_ENVELOPE_FORMATS = {EnvelopeFormat.DL, EnvelopeFormat.C5, EnvelopeFormat.C4}


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

    lines = text.splitlines()
    first_line = lines[0] if lines else ""
    delimiter = ";" if first_line.count(";") >= first_line.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    required = {"filename", "postcode"}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise ValueError("В ground truth отсутствуют поля: " + ", ".join(sorted(missing)))

    rows: list[dict[str, str]] = []
    for index, row in enumerate(reader, start=2):
        normalized = {str(key): str(value or "").strip() for key, value in row.items()}
        normalized["_line"] = str(index)
        rows.append(normalized)
    return rows, encoding, delimiter


def _validate_training_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    valid: list[dict[str, str]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in rows:
        filename = row.get("filename", "").strip()
        postcode = row.get("postcode", "").strip()
        line = int(row.get("_line", "0") or 0)

        if not filename:
            skipped.append({"line": line, "filename": filename, "reason": "empty_filename"})
            continue
        if not POSTCODE_RE.fullmatch(postcode):
            skipped.append(
                {"line": line, "filename": filename, "reason": f"invalid_postcode={postcode}"}
            )
            continue

        duplicate_key = filename.casefold()
        if duplicate_key in seen:
            skipped.append({"line": line, "filename": filename, "reason": "duplicate_filename"})
            continue
        seen.add(duplicate_key)
        valid.append({**row, "filename": filename, "postcode": postcode})

    return valid, skipped


def _index_images(images_dir: Path) -> tuple[dict[str, Path], dict[str, Path], list[str]]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Каталог изображений не найден: {images_dir}")

    by_relative: dict[str, Path] = {}
    basename_candidates: dict[str, list[Path]] = {}
    for path in sorted(images_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(images_dir).as_posix()
        by_relative[relative.casefold()] = path
        basename_candidates.setdefault(path.name.casefold(), []).append(path)

    unique_basename = {
        key: paths[0]
        for key, paths in basename_candidates.items()
        if len(paths) == 1
    }
    duplicate_basenames = sorted(
        paths[0].name for paths in basename_candidates.values() if len(paths) > 1
    )
    return by_relative, unique_basename, duplicate_basenames


def _resolve_image_path(
    filename: str,
    *,
    by_relative: dict[str, Path],
    unique_basename: dict[str, Path],
) -> Path | None:
    normalized = filename.replace("\\", "/").lstrip("./").casefold()
    direct = by_relative.get(normalized)
    if direct is not None:
        return direct
    return unique_basename.get(Path(filename).name.casefold())


def _digit_counts(rows: list[dict[str, str]]) -> Counter[str]:
    return Counter("".join(row["postcode"] for row in rows))


def _split_distribution_score(
    split_rows: list[dict[str, str]],
    *,
    target: dict[str, float],
    missing_penalty: float,
) -> float:
    counts = _digit_counts(split_rows)
    total = max(1, sum(counts.values()))
    score = 0.0
    for digit in "0123456789":
        if counts[digit] == 0:
            score += missing_penalty
        score += abs(counts[digit] / total - target[digit])
    return score


def _choose_dataset_splits(
    rows: list[dict[str, str]],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    filenames = [row["filename"] for row in rows]
    count = len(filenames)
    if count < 3:
        raise ValueError("Для train/val/test нужны минимум 3 письма")

    val_size = max(1, round(count * val_fraction))
    test_size = max(1, round(count * test_fraction))
    if val_size + test_size >= count:
        excess = val_size + test_size - (count - 1)
        while excess > 0 and (val_size > 1 or test_size > 1):
            if val_size >= test_size and val_size > 1:
                val_size -= 1
            elif test_size > 1:
                test_size -= 1
            excess -= 1
    if val_size + test_size >= count:
        raise ValueError("Недостаточно писем для непустых train/val/test")

    total_counts = _digit_counts(rows)
    total_digits = max(1, sum(total_counts.values()))
    target = {digit: total_counts[digit] / total_digits for digit in "0123456789"}
    row_by_name = {row["filename"]: row for row in rows}

    rng = random.Random(seed)
    best: tuple[float, set[str], set[str]] | None = None
    iterations = max(5000, count * 120)

    for _ in range(iterations):
        test_files = set(rng.sample(filenames, test_size))
        remaining = [name for name in filenames if name not in test_files]
        val_files = set(rng.sample(remaining, val_size))
        train_files = set(filenames).difference(test_files, val_files)

        train_rows = [row_by_name[name] for name in train_files]
        val_rows = [row_by_name[name] for name in val_files]
        test_rows = [row_by_name[name] for name in test_files]

        score = (
            _split_distribution_score(train_rows, target=target, missing_penalty=40.0)
            + 1.5 * _split_distribution_score(val_rows, target=target, missing_penalty=10.0)
            + 1.5 * _split_distribution_score(test_rows, target=target, missing_penalty=10.0)
        )
        if best is None or score < best[0]:
            best = (score, val_files, test_files)

    assert best is not None
    _, val_files, test_files = best
    return {
        filename: (
            "test" if filename in test_files else "val" if filename in val_files else "train"
        )
        for filename in filenames
    }


def _decode_path(path: Path) -> np.ndarray:
    return _decode_image(path.read_bytes())


async def _extract_full_envelope(
    image_path: Path,
    *,
    expected_format: EnvelopeFormat | None,
) -> tuple[np.ndarray, RoiDetection, dict[str, Any]]:
    raw = image_path.read_bytes()
    image = _decode_image(raw)
    upload = UploadFile(file=io.BytesIO(raw), filename=image_path.name)
    format_mode = FormatMode.FIXED if expected_format is not None else FormatMode.AUTO
    try:
        analysis = await analyze_layout(
            file=upload,
            include_debug_images=False,
            min_area_ratio=0.15,
            scoring_top_n=8,
            format_mode=format_mode,
            expected_format=expected_format,
        )
    finally:
        await upload.close()

    canonical = _canonical_from_analysis(analysis, image)
    format_value = expected_format.value if expected_format is not None else analysis.get("format")
    if not format_value:
        raise RuntimeError("format_unresolved")
    try:
        envelope_format = EnvelopeFormat(str(format_value))
    except ValueError as exc:
        raise RuntimeError(f"unsupported_format:{format_value}") from exc
    if envelope_format not in SUPPORTED_FULL_ENVELOPE_FORMATS:
        raise RuntimeError(f"roi_format_unsupported:{envelope_format.value}")

    roi = detect_simple_mail_rois(canonical.image, envelope_format)
    return canonical.image, roi, analysis


def _extract_postcode_crop(
    image_path: Path,
) -> tuple[np.ndarray, RoiDetection, dict[str, Any]]:
    crop = _decode_path(image_path)
    crop_height, crop_width = crop.shape[:2]

    # Crop уже должен быть upright: индекс читается слева направо, верхние
    # stencil-bars находятся сверху. Нейтральный top-pad сохраняет production
    # row_y_norm rescue-инвариант, не меняя содержимое search-zone и thresholds.
    top_pad = max(64, crop_height * 2)
    canvas = np.full((top_pad + crop_height, crop_width, 3), 255, dtype=np.uint8)
    canvas[top_pad : top_pad + crop_height, :crop_width] = crop
    search = PixelRect(0, top_pad, crop_width, crop_height)

    bbox, bar_count, density, confidence, features = _postcode_stencil_bbox(canvas, search)
    if bbox is None:
        reason = features.get("rejection_reason") if isinstance(features, dict) else None
        raise RuntimeError(f"postcode_not_detected:{reason or 'unknown'}")

    region = RoiRegion(
        kind="recipient_postcode",
        status="stencil_detected",
        confidence=confidence,
        search_bbox=search,
        detected_bbox=bbox,
        bbox=bbox,
        component_count=bar_count,
        ink_density=density,
        detector="postcode_stencil",
        features=features,
    )
    roi = RoiDetection(
        status="detected",
        format=EnvelopeFormat.C4,
        coordinate_space="training_postcode_crop",
        mail_class="simple",
        regions=(region,),
        source_reference="ToolOCR postcode stencil detector; training crop",
    )
    debug = {
        "input_mode": "postcode-crop",
        "source_width_px": crop_width,
        "source_height_px": crop_height,
        "training_canvas_top_pad_px": top_pad,
        "stencil_features": features,
    }
    return canvas, roi, debug


async def _extract_one(
    image_path: Path,
    *,
    input_mode: str,
    expected_format: EnvelopeFormat | None,
) -> tuple[np.ndarray, list[tuple[int, np.ndarray, dict[str, Any]]], dict[str, Any]]:
    if input_mode == "full-envelope":
        working_image, roi, debug = await _extract_full_envelope(
            image_path,
            expected_format=expected_format,
        )
    elif input_mode == "postcode-crop":
        working_image, roi, debug = _extract_postcode_crop(image_path)
    else:
        raise ValueError(f"Неизвестный input_mode: {input_mode}")

    geometry = derive_postcode_digit_geometry(
        roi,
        image_width=int(working_image.shape[1]),
        image_height=int(working_image.shape[0]),
    )
    if geometry.status != "ready" or len(geometry.cells) != 6:
        raise RuntimeError(f"digit_geometry:{geometry.status}:{geometry.reason}")

    cells: list[tuple[int, np.ndarray, dict[str, Any]]] = []
    for cell in geometry.cells:
        canvas, preprocess = _normalize_digit_crop_with_debug(working_image, cell)
        if canvas is None:
            raise RuntimeError(f"digit_{cell.index}_preprocess:{preprocess.get('status')}")
        cells.append((cell.index, canvas, preprocess))
    return working_image, cells, debug


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "filename",
        "digit_index",
        "label",
        "split",
        "sample_path",
        "input_mode",
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


def _split_stats(manifest: list[dict[str, Any]], split: str) -> dict[str, Any]:
    rows = [row for row in manifest if row["split"] == split]
    labels = Counter(str(row["label"]) for row in rows)
    return {
        "files": len({row["filename"] for row in rows}),
        "digit_samples": len(rows),
        "digit_distribution": {digit: labels[digit] for digit in "0123456789"},
    }


async def export_dataset(args: argparse.Namespace) -> int:
    ground_truth_path = Path(args.ground_truth).resolve()
    images_dir = Path(args.images_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    samples_dir = output_dir / "samples"

    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    rows, encoding, delimiter = _read_ground_truth(ground_truth_path)
    valid_rows, skipped_rows = _validate_training_rows(rows)
    if not valid_rows:
        raise RuntimeError("После валидации ground truth не осталось строк")

    by_relative, unique_basename, duplicate_basenames = _index_images(images_dir)
    split_by_filename = _choose_dataset_splits(
        valid_rows,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    expected_format = None if args.expected_format == "auto" else EnvelopeFormat(args.expected_format)

    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for current, row in enumerate(valid_rows, start=1):
        filename = row["filename"]
        postcode = row["postcode"]
        image_path = _resolve_image_path(
            filename,
            by_relative=by_relative,
            unique_basename=unique_basename,
        )
        print(f"[{current:03d}/{len(valid_rows):03d}] {filename}")
        if image_path is None:
            failures.append(
                {"filename": filename, "postcode": postcode, "reason": "image_not_found_or_ambiguous"}
            )
            continue

        try:
            _, cells, _ = await _extract_one(
                image_path,
                input_mode=args.input_mode,
                expected_format=expected_format,
            )
        except Exception as exc:
            failures.append(
                {
                    "filename": filename,
                    "postcode": postcode,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        split = split_by_filename[filename]
        stem = Path(filename).stem
        sample_rows: list[dict[str, Any]] = []
        sample_paths: list[Path] = []
        complete = True

        for digit_index, canvas, preprocess in cells:
            label = postcode[digit_index - 1]
            relative = Path("samples") / f"{stem}__d{digit_index}__y{label}.png"
            target = output_dir / relative
            if not cv2.imwrite(str(target), canvas):
                complete = False
                failures.append(
                    {"filename": filename, "postcode": postcode, "reason": f"cannot_write_digit_{digit_index}"}
                )
                break
            sample_paths.append(target)
            sample_rows.append(
                {
                    "filename": filename,
                    "digit_index": digit_index,
                    "label": label,
                    "split": split,
                    "sample_path": relative.as_posix(),
                    "input_mode": args.input_mode,
                    "preprocess_status": preprocess.get("status"),
                    "suppressed_components": preprocess.get("suppressed_components"),
                    "restored_components": preprocess.get("restored_components"),
                    "suppressed_ink_ratio": preprocess.get("suppressed_ink_ratio"),
                }
            )

        if not complete or len(sample_rows) != 6:
            for target in sample_paths:
                target.unlink(missing_ok=True)
            continue
        manifest.extend(sample_rows)

    if not manifest:
        raise RuntimeError("Не удалось экспортировать ни одной digit-cell")

    _write_manifest(output_dir / "manifest.csv", manifest)
    successful_files = {row["filename"] for row in manifest}
    summary = {
        "schema": "toolocr.postcode-digit-dataset.v2",
        "ground_truth": {
            "path": str(ground_truth_path),
            "encoding": encoding,
            "delimiter": delimiter,
            "rows_total": len(rows),
            "rows_valid": len(valid_rows),
            "rows_skipped": skipped_rows,
        },
        "images": {
            "path": str(images_dir),
            "indexed_images": len(by_relative),
            "duplicate_basenames": duplicate_basenames,
        },
        "dataset": {
            "input_mode": args.input_mode,
            "expected_format": args.expected_format,
            "successful_files": len(successful_files),
            "digit_samples": len(manifest),
            "train": _split_stats(manifest, "train"),
            "val": _split_stats(manifest, "val"),
            "test": _split_stats(manifest, "test"),
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
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
    parser = argparse.ArgumentParser(
        description="Экспорт 96x128 digit-canvas из full-envelope или upright postcode-crop"
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="CSV с минимум filename,postcode (UTF-8/CP1251; ; или ,)",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Каталог исходных изображений; поиск рекурсивный",
    )
    parser.add_argument("--output-dir", required=True, help="Каталог выходного датасета")
    parser.add_argument(
        "--input-mode",
        choices=("full-envelope", "postcode-crop"),
        default="full-envelope",
        help="full-envelope использует layout/orientation; postcode-crop ожидает upright crop индексного блока",
    )
    parser.add_argument(
        "--expected-format",
        choices=("auto", "DL", "C5", "C4"),
        default="auto",
        help="Только для full-envelope: AUTO или жёстко ожидаемый формат",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not (0.05 <= args.val_fraction <= 0.40):
        raise SystemExit("--val-fraction должен быть в диапазоне 0.05..0.40")
    if not (0.05 <= args.test_fraction <= 0.40):
        raise SystemExit("--test-fraction должен быть в диапазоне 0.05..0.40")
    if args.val_fraction + args.test_fraction >= 0.80:
        raise SystemExit("Сумма val/test fractions должна быть < 0.80")
    if args.input_mode == "postcode-crop" and args.expected_format != "auto":
        print("INFO: --expected-format игнорируется в режиме postcode-crop")
    return asyncio.run(export_dataset(args))


if __name__ == "__main__":
    raise SystemExit(main())

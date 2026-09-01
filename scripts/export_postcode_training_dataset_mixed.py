from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import cv2

from ocr.app.gost_r_51506_99 import EnvelopeFormat
from scripts.export_postcode_training_dataset import (
    SUPPORTED_IMAGE_SUFFIXES,
    _choose_dataset_splits,
    _extract_one,
    _index_images,
    _read_ground_truth,
    _resolve_image_path,
    _split_stats,
    _validate_training_rows,
    _write_manifest,
)
from scripts.postcode_crop_training_adapter import extract_postcode_crop_cells


def _crop_candidate_names(filename: str) -> list[str]:
    """Возвращает допустимые имена upright postcode-crop для source filename.

    Для `a.jpg` основное соглашение — `a_crop.jpg`. Дополнительно допускается
    другое поддерживаемое расширение crop, чтобы пересохранение JPEG/PNG не
    требовало менять ground truth.
    """

    normalized = filename.replace("\\", "/").lstrip("./")
    source = Path(normalized)
    base = source.with_suffix("")
    preferred_suffix = source.suffix.lower()

    suffixes: list[str] = []
    if preferred_suffix in SUPPORTED_IMAGE_SUFFIXES:
        suffixes.append(preferred_suffix)
    suffixes.extend(sorted(SUPPORTED_IMAGE_SUFFIXES.difference(suffixes)))

    return [f"{base.as_posix()}_crop{suffix}" for suffix in suffixes]


def _resolve_postcode_crop(
    filename: str,
    *,
    by_relative: dict[str, Path],
    unique_basename: dict[str, Path],
) -> Path | None:
    for candidate in _crop_candidate_names(filename):
        path = _resolve_image_path(
            candidate,
            by_relative=by_relative,
            unique_basename=unique_basename,
        )
        if path is not None:
            return path
    return None


async def _extract_with_fallback(
    *,
    filename: str,
    full_image_path: Path | None,
    crop_path: Path | None,
    expected_format: EnvelopeFormat | None,
) -> tuple[list[tuple[int, Any, dict[str, Any]]] | None, str | None, str | None]:
    primary_reason: str | None = None

    if full_image_path is not None:
        try:
            _, cells, _ = await _extract_one(
                full_image_path,
                input_mode="full-envelope",
                expected_format=expected_format,
            )
            return cells, "full-envelope", None
        except Exception as exc:
            primary_reason = f"{type(exc).__name__}: {exc}"
    else:
        primary_reason = "image_not_found_or_ambiguous"

    if crop_path is not None:
        try:
            cells, _ = extract_postcode_crop_cells(crop_path)
            return cells, "postcode-crop-fallback", None
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            return (
                None,
                None,
                f"full-envelope={primary_reason}; postcode-crop={fallback_reason}",
            )

    return None, None, f"full-envelope={primary_reason}; postcode-crop=not_found"


async def export_mixed_dataset(args: argparse.Namespace) -> int:
    ground_truth_path = Path(args.ground_truth).resolve()
    images_dir = Path(args.images_dir).resolve()
    crops_dir = Path(args.postcode_crops_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    samples_dir = output_dir / "samples"

    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    rows, encoding, delimiter = _read_ground_truth(ground_truth_path)
    valid_rows, skipped_rows = _validate_training_rows(rows)
    if not valid_rows:
        raise RuntimeError("После валидации ground truth не осталось строк")

    image_by_relative, image_unique_basename, image_duplicate_basenames = _index_images(images_dir)
    crop_by_relative, crop_unique_basename, crop_duplicate_basenames = _index_images(crops_dir)
    expected_format = None if args.expected_format == "auto" else EnvelopeFormat(args.expected_format)

    manifest: list[dict[str, Any]] = []
    successful_rows: list[dict[str, str]] = []
    failures: list[dict[str, Any]] = []
    full_successes = 0
    fallback_successes = 0

    for current, row in enumerate(valid_rows, start=1):
        filename = row["filename"]
        postcode = row["postcode"]
        full_image_path = _resolve_image_path(
            filename,
            by_relative=image_by_relative,
            unique_basename=image_unique_basename,
        )
        crop_path = _resolve_postcode_crop(
            filename,
            by_relative=crop_by_relative,
            unique_basename=crop_unique_basename,
        )

        print(f"[{current:03d}/{len(valid_rows):03d}] {filename}")
        cells, actual_input_mode, failure_reason = await _extract_with_fallback(
            filename=filename,
            full_image_path=full_image_path,
            crop_path=crop_path,
            expected_format=expected_format,
        )
        if cells is None or actual_input_mode is None:
            failures.append(
                {
                    "filename": filename,
                    "postcode": postcode,
                    "crop": None if crop_path is None else crop_path.name,
                    "reason": failure_reason or "unknown_failure",
                }
            )
            continue

        if actual_input_mode == "full-envelope":
            full_successes += 1
        else:
            fallback_successes += 1

        source_id = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:16]
        sample_rows: list[dict[str, Any]] = []
        sample_paths: list[Path] = []
        complete = True

        for digit_index, canvas, preprocess in cells:
            label = postcode[digit_index - 1]
            relative = Path("samples") / f"{source_id}__d{digit_index}__y{label}.png"
            target = output_dir / relative
            if not cv2.imwrite(str(target), canvas):
                complete = False
                failures.append(
                    {
                        "filename": filename,
                        "postcode": postcode,
                        "crop": None if crop_path is None else crop_path.name,
                        "reason": f"cannot_write_digit_{digit_index}",
                    }
                )
                break

            sample_paths.append(target)
            sample_rows.append(
                {
                    "filename": filename,
                    "digit_index": digit_index,
                    "label": label,
                    "split": "",
                    "sample_path": relative.as_posix(),
                    "input_mode": actual_input_mode,
                    "preprocess_status": preprocess.get("status"),
                    "suppressed_components": preprocess.get("suppressed_components"),
                    "restored_components": preprocess.get("restored_components"),
                    "suppressed_ink_ratio": preprocess.get("suppressed_ink_ratio"),
                }
            )

        if not complete or len(sample_rows) != 6:
            for target in sample_paths:
                target.unlink(missing_ok=True)
            if actual_input_mode == "full-envelope":
                full_successes -= 1
            else:
                fallback_successes -= 1
            continue

        manifest.extend(sample_rows)
        successful_rows.append(row)

    if not manifest:
        raise RuntimeError("Не удалось экспортировать ни одной digit-cell")
    if len(successful_rows) < 3:
        raise RuntimeError(
            "После извлечения осталось меньше 3 писем; невозможно построить train/val/test"
        )

    split_by_filename = _choose_dataset_splits(
        successful_rows,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    for row in manifest:
        row["split"] = split_by_filename[str(row["filename"])]

    _write_manifest(output_dir / "manifest.csv", manifest)
    summary = {
        "schema": "toolocr.postcode-digit-dataset.v4",
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
            "indexed_images": len(image_by_relative),
            "duplicate_basenames": image_duplicate_basenames,
        },
        "postcode_crops": {
            "path": str(crops_dir),
            "indexed_images": len(crop_by_relative),
            "duplicate_basenames": crop_duplicate_basenames,
            "naming": "<source_stem>_crop.<supported_ext>",
            "adapter": "postcode_crop_virtual_canonical_v2",
        },
        "dataset": {
            "input_mode": "full-envelope+postcode-crop-fallback",
            "expected_format": args.expected_format,
            "successful_files": len(successful_rows),
            "successful_full_envelope": full_successes,
            "successful_postcode_crop_fallback": fallback_successes,
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
        description=(
            "Mixed exporter: full-envelope, затем fallback на upright "
            "<source_stem>_crop.*"
        )
    )
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--postcode-crops-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-format",
        choices=("auto", "DL", "C5", "C4"),
        default="auto",
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
    return asyncio.run(export_mixed_dataset(args))


if __name__ == "__main__":
    raise SystemExit(main())

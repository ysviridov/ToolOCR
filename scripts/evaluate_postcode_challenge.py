from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


_DEFAULT_MODEL = "/app/models/postcode_digit_v1.onnx"
_DEFAULT_CHALLENGE = "/src/config/postcode-challenges/challenge-v1.csv"
_DEFAULT_OUTPUT = "/work/postcode-challenges/challenge-v1"
_POSTCODE_RE = re.compile(r"^[1-9][0-9]{5}$")
_MODEL_LABEL_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True, slots=True)
class ChallengeRow:
    filename: str
    postcode: str


@dataclass(frozen=True, slots=True)
class ModelSpec:
    label: str
    path: Path


@dataclass(frozen=True, slots=True)
class ExportedFile:
    filename: str
    file_id: str
    postcode: str
    format_value: str
    canvas_paths: tuple[Path, ...]


def _read_challenge(path: Path) -> list[ChallengeRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Challenge CSV не найден: {path}")

    rows: list[ChallengeRow] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"filename", "postcode"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "Challenge CSV: отсутствуют поля " + ", ".join(sorted(missing))
            )
        for line, raw in enumerate(reader, start=2):
            filename = str(raw.get("filename") or "").strip()
            postcode = str(raw.get("postcode") or "").strip()
            if not filename:
                raise ValueError(f"Challenge CSV:{line}: пустой filename")
            if not _POSTCODE_RE.fullmatch(postcode):
                raise ValueError(
                    f"Challenge CSV:{line}: postcode должен быть шестизначным и не начинаться с 0"
                )
            key = filename.casefold()
            if key in seen:
                raise ValueError(f"Challenge CSV:{line}: duplicate filename={filename}")
            seen.add(key)
            rows.append(ChallengeRow(filename=filename, postcode=postcode))

    if not rows:
        raise ValueError("Challenge CSV пуст")
    return rows


def _parse_model_specs(values: list[str] | None) -> list[ModelSpec]:
    raw_values = list(values or []) or [f"current={_DEFAULT_MODEL}"]
    result: list[ModelSpec] = []
    seen: set[str] = set()
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(
                f"Некорректный --model {raw!r}; ожидается label=/path/model.onnx"
            )
        label, path_value = raw.split("=", 1)
        label = label.strip()
        path_value = path_value.strip()
        if not label or not _MODEL_LABEL_RE.fullmatch(label):
            raise ValueError(
                f"Некорректный label модели {label!r}; разрешены A-Z a-z 0-9 . _ -"
            )
        if label.casefold() in seen:
            raise ValueError(f"Повторный label модели: {label}")
        if not path_value:
            raise ValueError(f"Пустой path для модели {label}")
        seen.add(label.casefold())
        result.append(ModelSpec(label=label, path=Path(path_value)))
    return result


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return value[:100] or "image"


def _find_test_ui_file_id(filename: str) -> str:
    from ocr.app.test_ui import _iter_metadata

    matches = [
        item
        for item in _iter_metadata()
        if str(item.get("name") or "").casefold() == filename.casefold()
    ]
    if not matches:
        raise FileNotFoundError(f"Файл не найден в Test UI volume: {filename}")
    if len(matches) > 1:
        ids = ",".join(str(item.get("id")) for item in matches)
        raise RuntimeError(
            f"В Test UI несколько файлов с именем {filename}; ids={ids}. "
            "Challenge требует однозначного filename."
        )
    return str(matches[0]["id"])


async def _export_runtime_canvases(
    row: ChallengeRow,
    *,
    output_dir: Path,
) -> tuple[ExportedFile, list[dict[str, Any]]]:
    # application устанавливает те же preprocessing/wiring hooks, с которыми
    # работает production OCR service. Импорт намеренно отложен до runtime,
    # чтобы unit-тесты парсера не поднимали FastAPI приложение.
    from ocr.app import application as _application  # noqa: F401
    from ocr.app.format_modes import FormatMode
    from ocr.app.gost_r_51506_99 import EnvelopeFormat
    from ocr.app.postcode_digit_cells import derive_postcode_digit_geometry
    from ocr.app.postcode_recognizer import _normalize_digit_crop_with_debug
    from ocr.app.roi import detect_simple_mail_rois
    from ocr.app.roi_test_ui import _analyze_saved_image, _canonical_from_analysis

    file_id = _find_test_ui_file_id(row.filename)
    analysis, image, _ = await _analyze_saved_image(
        file_id,
        format_mode=FormatMode.AUTO,
        expected_format=None,
    )
    canonical = _canonical_from_analysis(analysis, image)

    format_value = str(analysis.get("format") or "")
    if not format_value:
        raise RuntimeError("format_unresolved")
    try:
        envelope_format = EnvelopeFormat(format_value)
    except ValueError as exc:
        raise RuntimeError(f"unsupported_format={format_value}") from exc

    roi = detect_simple_mail_rois(canonical.image, envelope_format)
    geometry = derive_postcode_digit_geometry(
        roi,
        image_width=int(canonical.image.shape[1]),
        image_height=int(canonical.image.shape[0]),
    )
    if geometry.status != "ready" or len(geometry.cells) != 6:
        raise RuntimeError(
            f"digit_geometry_unavailable: status={geometry.status} reason={geometry.reason}"
        )

    file_dir = output_dir / "canvases" / f"{file_id}__{_safe_stem(row.filename)}"
    file_dir.mkdir(parents=True, exist_ok=True)

    canvas_paths: list[Path] = []
    preprocess_rows: list[dict[str, Any]] = []
    for cell in geometry.cells:
        canvas, preprocess = _normalize_digit_crop_with_debug(canonical.image, cell)
        if canvas is None:
            raise RuntimeError(
                f"digit_canvas_unavailable: index={cell.index} preprocess={preprocess}"
            )
        truth = row.postcode[cell.index - 1]
        canvas_path = file_dir / f"digit_{cell.index}_truth_{truth}.png"
        if not cv2.imwrite(str(canvas_path), canvas):
            raise RuntimeError(f"Не удалось сохранить canvas: {canvas_path}")
        canvas_paths.append(canvas_path)
        preprocess_rows.append(
            {
                "filename": row.filename,
                "file_id": file_id,
                "postcode": row.postcode,
                "digit_index": cell.index,
                "truth": truth,
                "format": format_value,
                "canvas": str(canvas_path),
                "preprocess": preprocess,
            }
        )

    return (
        ExportedFile(
            filename=row.filename,
            file_id=file_id,
            postcode=row.postcode,
            format_value=format_value,
            canvas_paths=tuple(canvas_paths),
        ),
        preprocess_rows,
    )


def _predict_models(
    exported: list[ExportedFile],
    model_specs: list[ModelSpec],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from ocr.app.postcode_onnx import predict_digit_onnx

    digit_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    model_summary: dict[str, Any] = {}

    for spec in model_specs:
        if not spec.path.is_file():
            raise FileNotFoundError(f"ONNX model не найден: {spec.path}")
        try:
            net = cv2.dnn.readNetFromONNX(str(spec.path))
        except Exception as exc:
            raise RuntimeError(
                f"Не удалось загрузить ONNX {spec.label}={spec.path}: {type(exc).__name__}: {exc}"
            ) from exc

        exact_correct = 0
        digit_correct = 0
        digit_total = 0

        for item in exported:
            predicted_digits: list[str] = []
            confidences: list[float] = []
            file_digit_correct = 0

            for digit_index, canvas_path in enumerate(item.canvas_paths, start=1):
                canvas = cv2.imread(str(canvas_path), cv2.IMREAD_GRAYSCALE)
                if canvas is None:
                    raise RuntimeError(f"Не удалось прочитать canvas: {canvas_path}")
                prediction = predict_digit_onnx(canvas, net=net)
                truth = item.postcode[digit_index - 1]
                correct = prediction.digit == truth
                digit_total += 1
                if correct:
                    digit_correct += 1
                    file_digit_correct += 1
                predicted_digits.append(prediction.digit)
                confidences.append(float(prediction.confidence))
                digit_rows.append(
                    {
                        "model": spec.label,
                        "model_path": str(spec.path),
                        "filename": item.filename,
                        "file_id": item.file_id,
                        "format": item.format_value,
                        "digit_index": digit_index,
                        "truth": truth,
                        "predicted": prediction.digit,
                        "correct": int(correct),
                        "confidence": prediction.confidence,
                        "top3": json.dumps(
                            [
                                {"digit": digit, "probability": probability}
                                for digit, probability in prediction.top3
                            ],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        "canvas": str(canvas_path),
                    }
                )

            predicted = "".join(predicted_digits)
            exact = predicted == item.postcode
            if exact:
                exact_correct += 1
            file_rows.append(
                {
                    "model": spec.label,
                    "model_path": str(spec.path),
                    "filename": item.filename,
                    "file_id": item.file_id,
                    "format": item.format_value,
                    "truth": item.postcode,
                    "predicted": predicted,
                    "exact_correct": int(exact),
                    "digit_correct": file_digit_correct,
                    "digit_total": 6,
                    "mean_confidence": round(sum(confidences) / 6.0, 6),
                    "min_confidence": round(min(confidences), 6),
                }
            )

        exact_total = len(exported)
        model_summary[spec.label] = {
            "model_path": str(spec.path),
            "exact_postcode_correct": exact_correct,
            "exact_postcode_total": exact_total,
            "exact_postcode_accuracy": (
                exact_correct / exact_total if exact_total else None
            ),
            "digit_correct": digit_correct,
            "digit_total": digit_total,
            "digit_accuracy": digit_correct / digit_total if digit_total else None,
        }

    return digit_rows, file_rows, model_summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_preprocess(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for item in rows:
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
            stream.write("\n")


def _model_deltas(
    file_rows: list[dict[str, Any]],
    model_specs: list[ModelSpec],
) -> list[dict[str, Any]]:
    if len(model_specs) < 2:
        return []
    by_file: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in file_rows:
        by_file[str(row["filename"])][str(row["model"])] = row

    baseline = model_specs[0].label
    result: list[dict[str, Any]] = []
    for filename in sorted(by_file):
        base = by_file[filename].get(baseline)
        if not base:
            continue
        for spec in model_specs[1:]:
            other = by_file[filename].get(spec.label)
            if not other:
                continue
            result.append(
                {
                    "filename": filename,
                    "truth": base["truth"],
                    "baseline_model": baseline,
                    "baseline_predicted": base["predicted"],
                    "baseline_exact": base["exact_correct"],
                    "candidate_model": spec.label,
                    "candidate_predicted": other["predicted"],
                    "candidate_exact": other["exact_correct"],
                    "exact_delta": int(other["exact_correct"]) - int(base["exact_correct"]),
                }
            )
    return result


async def _run(args: argparse.Namespace) -> int:
    challenge_path = Path(args.challenge)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    challenge = _read_challenge(challenge_path)
    models = _parse_model_specs(args.model)

    exported: list[ExportedFile] = []
    preprocess_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for row in challenge:
        try:
            item, debug_rows = await _export_runtime_canvases(
                row,
                output_dir=output_dir,
            )
            exported.append(item)
            preprocess_rows.extend(debug_rows)
            print(f"CANVAS OK  {row.filename}  {row.postcode}")
        except Exception as exc:
            failures.append(
                {
                    "filename": row.filename,
                    "postcode": row.postcode,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(
                f"CANVAS FAIL {row.filename}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    _write_preprocess(output_dir / "preprocess.jsonl", preprocess_rows)

    digit_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    model_summary: dict[str, Any] = {}
    if exported:
        digit_rows, file_rows, model_summary = _predict_models(exported, models)

    _write_csv(
        output_dir / "digits.csv",
        digit_rows,
        [
            "model",
            "model_path",
            "filename",
            "file_id",
            "format",
            "digit_index",
            "truth",
            "predicted",
            "correct",
            "confidence",
            "top3",
            "canvas",
        ],
    )
    _write_csv(
        output_dir / "files.csv",
        file_rows,
        [
            "model",
            "model_path",
            "filename",
            "file_id",
            "format",
            "truth",
            "predicted",
            "exact_correct",
            "digit_correct",
            "digit_total",
            "mean_confidence",
            "min_confidence",
        ],
    )

    deltas = _model_deltas(file_rows, models)
    _write_csv(
        output_dir / "model_deltas.csv",
        deltas,
        [
            "filename",
            "truth",
            "baseline_model",
            "baseline_predicted",
            "baseline_exact",
            "candidate_model",
            "candidate_predicted",
            "candidate_exact",
            "exact_delta",
        ],
    )

    summary = {
        "challenge": str(challenge_path),
        "preprocess": "stencil_dot_suppression_v1",
        "canvas_shape": [128, 96],
        "files_requested": len(challenge),
        "files_exported": len(exported),
        "failures": failures,
        "models": model_summary,
        "artifacts": {
            "digits_csv": str(output_dir / "digits.csv"),
            "files_csv": str(output_dir / "files.csv"),
            "model_deltas_csv": str(output_dir / "model_deltas.csv"),
            "preprocess_jsonl": str(output_dir / "preprocess.jsonl"),
            "canvases_dir": str(output_dir / "canvases"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print(f"files: {len(exported)}/{len(challenge)}")
    for label, metrics in model_summary.items():
        exact = metrics["exact_postcode_accuracy"]
        digit = metrics["digit_accuracy"]
        print(
            f"{label}: exact={metrics['exact_postcode_correct']}/{metrics['exact_postcode_total']} "
            f"({exact:.2%}) digit={metrics['digit_correct']}/{metrics['digit_total']} "
            f"({digit:.2%})"
        )
    print(f"output: {output_dir}")

    return 2 if failures else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Экспортирует production-equivalent postcode 96x128 canvases из Test UI "
            "и сравнивает одну или несколько ONNX-моделей на фиксированном challenge-set."
        )
    )
    parser.add_argument(
        "--challenge",
        default=_DEFAULT_CHALLENGE,
        help=f"CSV filename,postcode (default: {_DEFAULT_CHALLENGE})",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help=f"Каталог результатов (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model",
        action="append",
        help=(
            "Модель в формате label=/path/model.onnx. Можно повторять. "
            f"Без параметра используется current={_DEFAULT_MODEL}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

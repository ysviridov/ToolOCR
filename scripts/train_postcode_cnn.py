from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_WIDTH = 96
IMAGE_HEIGHT = 128
NUM_CLASSES = 10
ALLOWED_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class Sample:
    filename: str
    digit_index: int
    label: int
    split: str
    path: Path


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_manifest(path: Path) -> list[Sample]:
    root = path.parent
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"filename", "digit_index", "label", "split", "sample_path"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("manifest.csv: отсутствуют поля " + ", ".join(sorted(missing)))
        for line, row in enumerate(reader, start=2):
            try:
                digit_index = int(row["digit_index"])
                label = int(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"manifest.csv:{line}: invalid digit_index/label") from exc
            split = str(row["split"] or "").strip().lower()
            if not (1 <= digit_index <= 6):
                raise ValueError(f"manifest.csv:{line}: digit_index должен быть 1..6")
            if not (0 <= label <= 9):
                raise ValueError(f"manifest.csv:{line}: label должен быть 0..9")
            if split not in ALLOWED_SPLITS:
                raise ValueError(f"manifest.csv:{line}: split должен быть train/val/test")
            sample_path = (root / str(row["sample_path"])).resolve()
            if not sample_path.is_file():
                raise FileNotFoundError(f"manifest.csv:{line}: sample не найден: {sample_path}")
            samples.append(
                Sample(
                    filename=str(row["filename"]),
                    digit_index=digit_index,
                    label=label,
                    split=split,
                    path=sample_path,
                )
            )
    if not samples:
        raise ValueError("manifest.csv пуст")
    return samples


def _augment(binary_canvas: np.ndarray, rng: random.Random) -> np.ndarray:
    image = binary_canvas.copy()
    center = (IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0)
    angle = rng.uniform(-7.0, 7.0)
    scale = rng.uniform(0.92, 1.08)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += rng.uniform(-4.0, 4.0)
    matrix[1, 2] += rng.uniform(-5.0, 5.0)
    image = cv2.warpAffine(
        image,
        matrix,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    morph = rng.random()
    if morph < 0.18:
        image = cv2.erode(image, np.ones((2, 2), dtype=np.uint8), iterations=1)
    elif morph < 0.36:
        image = cv2.dilate(image, np.ones((2, 2), dtype=np.uint8), iterations=1)

    if rng.random() < 0.35:
        for _ in range(rng.randint(1, 7)):
            x = rng.randint(5, IMAGE_WIDTH - 6)
            y = rng.randint(5, IMAGE_HEIGHT - 6)
            radius = 1 if rng.random() < 0.85 else 2
            cv2.circle(image, (x, y), radius, 0, -1)
    return image


class DigitDataset(Dataset):
    def __init__(self, samples: list[Sample], *, augment: bool, seed: int) -> None:
        self.samples = samples
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Не удалось прочитать {sample.path}")
        if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            rng = random.Random(self.seed + self.epoch * 1_000_003 + index * 7_919)
            image = _augment(image, rng)

        tensor = torch.from_numpy((255.0 - image.astype(np.float32)) / 255.0).unsqueeze(0)
        return tensor, sample.label, sample.filename, sample.digit_index


class PostcodeDigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 4 * 3, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(96, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _class_weights(samples: list[Sample], device: torch.device) -> torch.Tensor:
    counts = Counter(sample.label for sample in samples)
    total = max(1, len(samples))
    weights = []
    for digit in range(NUM_CLASSES):
        count = max(1, counts[digit])
        value = math.sqrt(total / (NUM_CLASSES * count))
        weights.append(min(3.0, max(0.55, value)))
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    grouped: dict[str, dict[int, tuple[int, int]]] = defaultdict(dict)

    for images, labels, filenames, digit_indexes in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = torch.argmax(logits, dim=1)

        batch = labels.shape[0]
        total_loss += float(loss.item()) * batch
        total += batch
        correct += int((predictions == labels).sum().item())

        labels_cpu = labels.cpu().tolist()
        predictions_cpu = predictions.cpu().tolist()
        digit_indexes_cpu = digit_indexes.tolist() if hasattr(digit_indexes, "tolist") else list(digit_indexes)
        for filename, digit_index, truth, pred in zip(
            filenames,
            digit_indexes_cpu,
            labels_cpu,
            predictions_cpu,
            strict=False,
        ):
            confusion[int(truth), int(pred)] += 1
            grouped[str(filename)][int(digit_index)] = (int(truth), int(pred))

    exact_total = 0
    exact_correct = 0
    for digits in grouped.values():
        if set(digits) != {1, 2, 3, 4, 5, 6}:
            continue
        exact_total += 1
        if all(digits[index][0] == digits[index][1] for index in range(1, 7)):
            exact_correct += 1

    return {
        "loss": total_loss / max(1, total),
        "digit_accuracy": correct / max(1, total),
        "exact_postcode_accuracy": exact_correct / max(1, exact_total),
        "exact_postcode_correct": exact_correct,
        "exact_postcode_total": exact_total,
        "confusion": confusion,
    }


def _write_confusion(path: Path, confusion: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true\\pred", *range(NUM_CLASSES)])
        for digit in range(NUM_CLASSES):
            writer.writerow([digit, *[int(value) for value in confusion[digit]]])


def _export_onnx(model: nn.Module, output_path: Path) -> None:
    model_cpu = model.to("cpu").eval()
    dummy = torch.zeros((1, 1, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def _validate_onnx_with_opencv(
    model: nn.Module,
    onnx_path: Path,
    sample_path: Path,
) -> dict[str, Any]:
    image = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Не удалось прочитать validation sample: {sample_path}")
    if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    tensor_np = ((255.0 - image.astype(np.float32)) / 255.0)[None, None, :, :]

    model_cpu = model.to("cpu").eval()
    with torch.no_grad():
        torch_logits = model_cpu(torch.from_numpy(tensor_np)).numpy()

    net = cv2.dnn.readNetFromONNX(str(onnx_path))
    net.setInput(tensor_np)
    opencv_logits = net.forward()
    max_abs_diff = float(np.max(np.abs(torch_logits - opencv_logits)))
    same_argmax = int(np.argmax(torch_logits, axis=1)[0]) == int(np.argmax(opencv_logits, axis=1)[0])
    return {
        "opencv_load_ok": True,
        "same_argmax": bool(same_argmax),
        "max_abs_logit_diff": max_abs_diff,
    }


def _public_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {key: value for key, value in metrics.items() if key != "confusion"}


def train(args: argparse.Namespace) -> int:
    _set_seed(args.seed)
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = _load_manifest(manifest_path)
    train_samples = [sample for sample in all_samples if sample.split == "train"]
    val_samples = [sample for sample in all_samples if sample.split == "val"]
    test_samples = [sample for sample in all_samples if sample.split == "test"]
    if not train_samples or not val_samples:
        raise RuntimeError("Нужны непустые train и val split")

    split_files = {
        "train": {sample.filename for sample in train_samples},
        "val": {sample.filename for sample in val_samples},
        "test": {sample.filename for sample in test_samples},
    }
    if split_files["train"].intersection(split_files["val"]):
        raise RuntimeError("Leakage: письма одновременно в train и val")
    if split_files["train"].intersection(split_files["test"]):
        raise RuntimeError("Leakage: письма одновременно в train и test")
    if split_files["val"].intersection(split_files["test"]):
        raise RuntimeError("Leakage: письма одновременно в val и test")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}")
    print(f"train: files={len(split_files['train'])} digits={len(train_samples)}")
    print(f"val:   files={len(split_files['val'])} digits={len(val_samples)}")
    print(f"test:  files={len(split_files['test'])} digits={len(test_samples)}")
    print("train labels:", dict(sorted(Counter(sample.label for sample in train_samples).items())))
    print("val labels:  ", dict(sorted(Counter(sample.label for sample in val_samples).items())))
    print("test labels: ", dict(sorted(Counter(sample.label for sample in test_samples).items())))

    train_dataset = DigitDataset(train_samples, augment=True, seed=args.seed)
    val_dataset = DigitDataset(val_samples, augment=False, seed=args.seed)
    test_dataset = DigitDataset(test_samples, augment=False, seed=args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch_size * 2),
        shuffle=False,
        num_workers=0,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=max(1, args.batch_size * 2),
            shuffle=False,
            num_workers=0,
        )
        if test_samples
        else None
    )

    model = PostcodeDigitCNN().to(device)
    weights = _class_weights(train_samples, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-5,
    )

    best_score: tuple[float, float, float] | None = None
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        running_total = 0
        running_correct = 0

        for images, labels, _, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch = labels.shape[0]
            running_loss += float(loss.item()) * batch
            running_total += batch
            running_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())

        train_loss = running_loss / max(1, running_total)
        train_accuracy = running_correct / max(1, running_total)
        val_metrics = _evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])
        lr = float(optimizer.param_groups[0]["lr"])

        epoch_row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "train_digit_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_digit_accuracy": val_metrics["digit_accuracy"],
            "val_exact_postcode_accuracy": val_metrics["exact_postcode_accuracy"],
        }
        history.append(epoch_row)
        print(
            f"epoch={epoch:03d} lr={lr:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['digit_accuracy']:.3f} "
            f"val_postcode={val_metrics['exact_postcode_accuracy']:.3f}"
        )

        score = (
            float(val_metrics["exact_postcode_accuracy"]),
            float(val_metrics["digit_accuracy"]),
            -float(val_metrics["loss"]),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = _public_metrics(val_metrics)
            _write_confusion(output_dir / "confusion_val_best.csv", val_metrics["confusion"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
            print(f"early_stop: no improvement {epochs_without_improvement} epochs")
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Не удалось выбрать best checkpoint")

    model.load_state_dict(best_state)
    test_metrics = (
        _evaluate(model, test_loader, criterion, device)
        if test_loader is not None
        else None
    )
    if test_metrics is not None:
        _write_confusion(output_dir / "confusion_test.csv", test_metrics["confusion"])

    model_name = args.model_name.strip()
    if not model_name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in model_name):
        raise RuntimeError("--model-name может содержать только A-Z, a-z, 0-9, _ и -")

    checkpoint_path = output_dir / f"{model_name}.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "architecture": "PostcodeDigitCNN.v1",
            "input_width": IMAGE_WIDTH,
            "input_height": IMAGE_HEIGHT,
            "classes": list(range(NUM_CLASSES)),
            "seed": args.seed,
            "best_epoch": best_epoch,
        },
        checkpoint_path,
    )

    onnx_path = output_dir / f"{model_name}.onnx"
    _export_onnx(model, onnx_path)
    onnx_validation = _validate_onnx_with_opencv(model, onnx_path, val_samples[0].path)
    if not onnx_validation["same_argmax"]:
        raise RuntimeError(f"OpenCV ONNX validation mismatch: {onnx_validation}")

    metadata = {
        "schema": "toolocr.postcode-digit-model.v2",
        "architecture": "PostcodeDigitCNN.v1",
        "input": {
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "channels": 1,
            "normalization": "ink=(255-gray)/255",
            "preprocess": "stencil_dot_suppression_v1",
        },
        "classes": list(range(NUM_CLASSES)),
        "training": {
            "manifest": str(manifest_path),
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "train_files": len(split_files["train"]),
            "val_files": len(split_files["val"]),
            "test_files": len(split_files["test"]),
            "train_digits": len(train_samples),
            "val_digits": len(val_samples),
            "test_digits": len(test_samples),
            "train_distribution": dict(sorted(Counter(sample.label for sample in train_samples).items())),
            "val_distribution": dict(sorted(Counter(sample.label for sample in val_samples).items())),
            "test_distribution": dict(sorted(Counter(sample.label for sample in test_samples).items())),
            "class_weights": [round(float(value), 6) for value in weights.detach().cpu().tolist()],
        },
        "best_validation": best_metrics,
        "final_test": _public_metrics(test_metrics),
        "onnx_validation": onnx_validation,
        "artifacts": {
            "pytorch": checkpoint_path.name,
            "onnx": onnx_path.name,
            "validation_confusion": "confusion_val_best.csv",
            "test_confusion": "confusion_test.csv" if test_metrics is not None else None,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    print("\nBEST VALIDATION")
    print(json.dumps(metadata["best_validation"], ensure_ascii=False, indent=2))
    print("\nFINAL TEST")
    print(json.dumps(metadata["final_test"], ensure_ascii=False, indent=2))
    print("ONNX", json.dumps(onnx_validation, ensure_ascii=False))
    print(f"model={onnx_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Обучение CNN для 6-значного stencil postcode")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="postcode_digit_v1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.min_epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise SystemExit("epochs/min-epochs/patience/batch-size должны быть > 0")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())

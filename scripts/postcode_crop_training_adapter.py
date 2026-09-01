from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ocr.app.gost_r_51506_99 import EnvelopeFormat
from ocr.app.postcode_digit_cells import derive_postcode_digit_geometry
from ocr.app.postcode_recognizer import _normalize_digit_crop_with_debug
from ocr.app.roi import PixelRect, RoiDetection, RoiRegion, _postcode_stencil_bbox
from ocr.app.test_ui import _decode_image


# Postcode detector нормирует размеры плашек относительно полного canonical-frame.
# Tight crop нельзя подавать как полный кадр: плашки становятся искусственно
# слишком крупными. Training-only adapter помещает исходный upright crop в
# виртуальный landscape-frame, не меняя сам crop и production thresholds.
_TRAINING_CANVAS_WIDTH_SCALE = 2.20
_TRAINING_CANVAS_ASPECT_RATIO = 1.42
_TRAINING_CANVAS_BOTTOM_MARGIN_RATIO = 0.01


def build_training_postcode_canvas(
    crop: np.ndarray,
) -> tuple[np.ndarray, PixelRect, dict[str, Any]]:
    if crop is None or crop.size == 0:
        raise ValueError("Пустой postcode crop")

    crop_height, crop_width = crop.shape[:2]
    canvas_width = max(crop_width + 1, int(round(crop_width * _TRAINING_CANVAS_WIDTH_SCALE)))
    canvas_height = max(
        crop_height + 8,
        int(round(canvas_width / _TRAINING_CANVAS_ASPECT_RATIO)),
    )
    bottom_margin = max(
        4,
        int(round(canvas_height * _TRAINING_CANVAS_BOTTOM_MARGIN_RATIO)),
    )
    origin_x = 0
    origin_y = max(0, canvas_height - bottom_margin - crop_height)

    if crop.ndim == 2:
        canvas = np.full((canvas_height, canvas_width), 255, dtype=crop.dtype)
        canvas[origin_y : origin_y + crop_height, :crop_width] = crop
    else:
        channels = crop.shape[2]
        canvas = np.full((canvas_height, canvas_width, channels), 255, dtype=crop.dtype)
        canvas[origin_y : origin_y + crop_height, :crop_width, :] = crop

    search = PixelRect(origin_x, origin_y, crop_width, crop_height)
    debug = {
        "adapter": "postcode_crop_virtual_canonical_v1",
        "source_width_px": crop_width,
        "source_height_px": crop_height,
        "canvas_width_px": canvas_width,
        "canvas_height_px": canvas_height,
        "crop_origin_x_px": origin_x,
        "crop_origin_y_px": origin_y,
        "bottom_margin_px": bottom_margin,
        "width_scale": _TRAINING_CANVAS_WIDTH_SCALE,
        "canvas_aspect_ratio": _TRAINING_CANVAS_ASPECT_RATIO,
    }
    return canvas, search, debug


def extract_postcode_crop_cells(
    image_path: Path,
) -> tuple[list[tuple[int, np.ndarray, dict[str, Any]]], dict[str, Any]]:
    crop = _decode_image(image_path.read_bytes())
    working_image, search, debug = build_training_postcode_canvas(crop)

    bbox, bar_count, density, confidence, features = _postcode_stencil_bbox(
        working_image,
        search,
    )
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
        coordinate_space="training_postcode_crop_virtual_canonical",
        mail_class="simple",
        regions=(region,),
        source_reference="ToolOCR postcode stencil detector; training crop virtual canonical adapter",
    )

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

    debug["stencil_features"] = features
    debug["digit_cell_count"] = len(cells)
    return cells, debug

from __future__ import annotations

import io

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query, UploadFile
from fastapi.responses import Response

from .format_modes import FormatMode
from .gost_r_51506_99 import EnvelopeFormat
from .layout import rectify_envelope
from .layout_api import analyze_layout
from .roi import canonicalize_rectified, detect_simple_mail_rois, draw_roi_overlay, roi_detection_to_dict
from .test_ui import _decode_image, _image_path, _load_metadata, _validate_file_id


router = APIRouter(tags=["test-ui"])


async def _analyze_saved_image(
    file_id: str,
    *,
    format_mode: FormatMode,
    expected_format: EnvelopeFormat | None,
) -> tuple[dict, np.ndarray, str]:
    meta = _load_metadata(_validate_file_id(file_id))
    raw = _image_path(meta).read_bytes()
    image = _decode_image(raw)

    upload = UploadFile(file=io.BytesIO(raw), filename=meta["name"])
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

    return analysis, image, str(meta["name"])


def _canonical_from_analysis(analysis: dict, image: np.ndarray):
    detector = analysis.get("detector") or {}
    quad = detector.get("quad") or []
    if len(quad) != 4:
        raise HTTPException(
            status_code=422,
            detail={"reason": "quad_unavailable", "message": "Не найден quad для canonicalization"},
        )

    points = np.array([[float(item["x"]), float(item["y"])] for item in quad], dtype=np.float32)
    rectified = rectify_envelope(image, points)
    orientation = analysis.get("orientation") or {}
    canonical = canonicalize_rectified(
        rectified.image,
        orientation_status=orientation.get("status"),
        orientation_deg=orientation.get("value_deg"),
    )
    if not canonical.reliable:
        raise HTTPException(
            status_code=422,
            detail={
                "reason": "orientation_unresolved",
                "orientation_status": orientation.get("status"),
                "orientation_deg": orientation.get("value_deg"),
                "message": "Canonical/ROI preview не строится при неоднозначной ориентации",
            },
        )
    return canonical


def _encode_jpeg(image: np.ndarray, *, quality: int = 93) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Не удалось закодировать preview JPEG")
    return encoded.tobytes()


def _common_headers(analysis: dict, *, view: str, rotation: int) -> dict[str, str]:
    return {
        "Cache-Control": "no-store, max-age=0",
        "Pragma": "no-cache",
        "X-ToolOCR-Preview": view,
        "X-ToolOCR-Format": str(analysis.get("format") or ""),
        "X-ToolOCR-Orientation": str((analysis.get("orientation") or {}).get("value_deg") or 0),
        "X-ToolOCR-Canonical-Rotation": str(rotation),
    }


@router.get("/v1/test-ui/images/{file_id}/canonical", response_class=Response)
async def canonical_preview(
    file_id: str,
    format_mode: FormatMode = Query(default=FormatMode.AUTO),
    expected_format: EnvelopeFormat | None = Query(default=None),
) -> Response:
    analysis, image, _ = await _analyze_saved_image(
        file_id,
        format_mode=format_mode,
        expected_format=expected_format,
    )
    canonical = _canonical_from_analysis(analysis, image)
    return Response(
        content=_encode_jpeg(canonical.image),
        media_type="image/jpeg",
        headers=_common_headers(
            analysis,
            view="canonical",
            rotation=canonical.rotation_applied_deg,
        ),
    )


@router.get("/v1/test-ui/images/{file_id}/roi", response_class=Response)
async def roi_preview(
    file_id: str,
    format_mode: FormatMode = Query(default=FormatMode.AUTO),
    expected_format: EnvelopeFormat | None = Query(default=None),
) -> Response:
    analysis, image, _ = await _analyze_saved_image(
        file_id,
        format_mode=format_mode,
        expected_format=expected_format,
    )
    canonical = _canonical_from_analysis(analysis, image)

    format_value = analysis.get("format")
    if not format_value:
        raise HTTPException(
            status_code=422,
            detail={"reason": "format_unresolved", "message": "Формат письма не определён"},
        )
    try:
        envelope_format = EnvelopeFormat(format_value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Некорректный формат письма") from exc

    roi = detect_simple_mail_rois(canonical.image, envelope_format)
    if roi.status == "unsupported_format":
        raise HTTPException(
            status_code=422,
            detail={
                "reason": "roi_format_unsupported",
                "format": envelope_format.value,
                "supported_formats": [EnvelopeFormat.DL.value, EnvelopeFormat.C5.value],
                "message": "На первом этапе ROI простых писем поддерживает DL и C5",
            },
        )

    overlay = draw_roi_overlay(canonical.image, roi)
    headers = _common_headers(
        analysis,
        view="roi",
        rotation=canonical.rotation_applied_deg,
    )
    headers.update(
        {
            "X-ToolOCR-ROI-Status": roi.status,
            "X-ToolOCR-ROI-Coordinate-Space": roi.coordinate_space,
        }
    )
    return Response(
        content=_encode_jpeg(overlay),
        media_type="image/jpeg",
        headers=headers,
    )


@router.get("/v1/test-ui/images/{file_id}/roi/meta")
async def roi_metadata(
    file_id: str,
    format_mode: FormatMode = Query(default=FormatMode.AUTO),
    expected_format: EnvelopeFormat | None = Query(default=None),
) -> dict:
    analysis, image, name = await _analyze_saved_image(
        file_id,
        format_mode=format_mode,
        expected_format=expected_format,
    )
    canonical = _canonical_from_analysis(analysis, image)
    format_value = analysis.get("format")
    if not format_value:
        raise HTTPException(status_code=422, detail={"reason": "format_unresolved"})
    envelope_format = EnvelopeFormat(format_value)
    roi = detect_simple_mail_rois(canonical.image, envelope_format)
    return {
        "stage": "2.2",
        "filename": name,
        "canonical": {
            "status": canonical.status,
            "source_orientation_deg": canonical.source_orientation_deg,
            "rotation_applied_deg": canonical.rotation_applied_deg,
            "width_px": int(canonical.image.shape[1]),
            "height_px": int(canonical.image.shape[0]),
        },
        "roi": roi_detection_to_dict(roi),
    }

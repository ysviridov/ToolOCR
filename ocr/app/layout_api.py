from __future__ import annotations

import base64
import os
import time
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile

from .gost_r_51506_99 import ENVELOPE_SPECS, GOST_ID, candidate_formats_by_aspect_ratio
from .layout import EnvelopeNotFoundError, detect_envelope_quad, draw_detection_overlay, rectify_envelope
from .profile_scoring import ProfileHypothesis, score_gost_profiles
from .profiles import DOMESTIC_PROFILES, PROFILE_BY_ID, profiles_for_format

router = APIRouter(prefix="/v1/layout", tags=["layout"])
MAX_UPLOAD_MB = int(os.environ.get("OCR_MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_PIXELS = int(os.environ.get("OCR_MAX_PIXELS", "40000000"))


def _round_ms(value: float) -> float:
    return round(value, 3)


async def _read_and_decode(file: UploadFile) -> tuple[bytes, np.ndarray]:
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Максимальный размер файла: {MAX_UPLOAD_MB} МБ")
    if not raw:
        raise HTTPException(status_code=400, detail="Передан пустой файл")
    encoded = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=415, detail="Не удалось декодировать изображение")
    height, width = image.shape[:2]
    if height * width > MAX_PIXELS:
        raise HTTPException(status_code=413, detail=f"Изображение слишком большое: {width}x{height}")
    return raw, image


def _encode_debug_jpeg(image: np.ndarray, *, max_side: int = 2200) -> str:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / float(max(width, height)))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise HTTPException(status_code=500, detail="Не удалось закодировать debug JPEG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _profile_to_dict(profile: Any, ratio_error: float | None = None) -> dict[str, Any]:
    result = {
        "profile_id": profile.profile_id,
        "format": profile.format.value,
        "layout": profile.layout.value,
        "window": profile.window,
        "figure": profile.figure,
        "width_mm": profile.width_mm,
        "height_mm": profile.height_mm,
    }
    if ratio_error is not None:
        result["ratio_error"] = ratio_error
    return result


def _hypothesis_to_dict(hypothesis: ProfileHypothesis) -> dict[str, Any]:
    components = hypothesis.components
    return {
        "profile_id": hypothesis.profile_id,
        "format": hypothesis.format,
        "layout": hypothesis.layout,
        "window": hypothesis.window,
        "orientation_deg": hypothesis.orientation_deg,
        "score": hypothesis.score,
        "components": {
            "aspect": components.aspect,
            "postage": components.postage,
            "code_stamp": components.code_stamp,
            "layout": components.layout,
            "window": components.window,
            "orientation_signal": components.orientation_signal,
        },
    }


@router.get("/profiles")
def list_profiles() -> dict[str, Any]:
    return {
        "standard": GOST_ID,
        "scope": "domestic",
        "count": len(DOMESTIC_PROFILES),
        "profiles": [_profile_to_dict(profile) for profile in DOMESTIC_PROFILES],
    }


@router.post("/analyze")
async def analyze_layout(
    file: UploadFile = File(..., description="Полная фотография лицевой стороны письма"),
    include_debug_images: bool = Query(default=False),
    min_area_ratio: float = Query(default=0.15, ge=0.05, le=0.90),
    scoring_top_n: int = Query(default=8, ge=1, le=32),
) -> dict[str, Any]:
    total_started = time.perf_counter()

    decode_started = time.perf_counter()
    raw, image = await _read_and_decode(file)
    decode_ms = (time.perf_counter() - decode_started) * 1000.0

    detect_started = time.perf_counter()
    try:
        detection = detect_envelope_quad(image, min_area_ratio=min_area_ratio)
    except EnvelopeNotFoundError as exc:
        raise HTTPException(
            status_code=422,
            detail={"layout_status": "reject", "reason": "envelope_quad_not_found", "message": str(exc)},
        ) from exc
    detect_ms = (time.perf_counter() - detect_started) * 1000.0

    rectify_started = time.perf_counter()
    rectified = rectify_envelope(image, detection.points)
    rectify_ms = (time.perf_counter() - rectify_started) * 1000.0

    candidate_started = time.perf_counter()
    format_candidates = candidate_formats_by_aspect_ratio(
        rectified.width_px,
        rectified.height_px,
        max_relative_error=0.08,
    )
    profile_candidates: list[dict[str, Any]] = []
    for candidate in format_candidates:
        for profile in profiles_for_format(candidate.format):
            profile_candidates.append(_profile_to_dict(profile, candidate.ratio_error))
    candidate_ms = (time.perf_counter() - candidate_started) * 1000.0

    scoring_started = time.perf_counter()
    scoring = score_gost_profiles(
        rectified.image,
        DOMESTIC_PROFILES,
        frame_contact_sides=detection.frame_contact_sides,
        profile_min_margin=0.085 if detection.frame_contact_sides else 0.055,
    )
    scoring_ms = (time.perf_counter() - scoring_started) * 1000.0

    selected_profile = (
        PROFILE_BY_ID.get(scoring.profile.profile_id)
        if scoring.profile.profile_id is not None
        else None
    )

    if selected_profile is not None:
        format_status = (
            "resolved_by_profile_scoring_partial_frame"
            if detection.frame_contact_sides
            else "resolved_by_profile_scoring"
        )
    elif detection.frame_contact_sides:
        format_status = "unreliable_partial_frame"
    elif not format_candidates:
        format_status = "unknown"
    elif len(format_candidates) == 1:
        format_status = "resolved_by_ratio"
    else:
        format_status = "ambiguous_by_ratio"

    debug_images = None
    if include_debug_images:
        overlay = draw_detection_overlay(image, detection)
        canonical = rectified.image
        if scoring.orientation.status == "resolved" and scoring.orientation.value_deg == 180:
            canonical = cv2.rotate(canonical, cv2.ROTATE_180)
        debug_images = {
            "overlay_jpeg_base64": _encode_debug_jpeg(overlay),
            "rectified_jpeg_base64": _encode_debug_jpeg(rectified.image),
            "canonical_jpeg_base64": _encode_debug_jpeg(canonical),
        }

    quad = [
        {"x": round(float(point[0]), 2), "y": round(float(point[1]), 2)}
        for point in detection.points
    ]
    candidates_json = []
    for candidate in format_candidates:
        spec = ENVELOPE_SPECS[candidate.format]
        candidates_json.append(
            {
                "format": candidate.format.value,
                "width_mm": spec.width_mm,
                "height_mm": spec.height_mm,
                "aspect_ratio": round(spec.aspect_ratio, 6),
                "ratio_error": candidate.ratio_error,
            }
        )

    total_ms = (time.perf_counter() - total_started) * 1000.0
    return {
        "stage": "2.1",
        "standard": GOST_ID,
        "layout_status": "detected" if not detection.frame_contact_sides else "partial_frame",
        "input": {
            "filename": file.filename,
            "content_type": file.content_type,
            "bytes_received": len(raw),
            "width_px": int(image.shape[1]),
            "height_px": int(image.shape[0]),
        },
        "detector": {
            "method": detection.method,
            "confidence": detection.confidence,
            "raw_confidence": detection.raw_confidence,
            "frame_status": detection.frame_status,
            "frame_contact_sides": list(detection.frame_contact_sides),
            "area_ratio": detection.area_ratio,
            "rectangularity": detection.rectangularity,
            "angle_score": detection.angle_score,
            "quad_order": ["TL", "TR", "BR", "BL"],
            "quad": quad,
        },
        "rectified": {
            "width_px": rectified.width_px,
            "height_px": rectified.height_px,
            "landscape": rectified.width_px >= rectified.height_px,
        },
        "orientation": {
            "status": scoring.orientation.status,
            "value_deg": scoring.orientation.value_deg,
            "confidence": scoring.orientation.confidence,
            "margin": scoring.orientation.margin,
            "scores": [
                {"orientation_deg": degree, "score": score}
                for degree, score in scoring.orientation.scores
            ],
        },
        "format_status": format_status,
        "format_candidates": candidates_json,
        "profile_scope": "domestic",
        "profile_candidates": profile_candidates,
        "profile_scoring": {
            "status": scoring.profile.status,
            "profile_id": scoring.profile.profile_id,
            "confidence": scoring.profile.confidence,
            "margin": scoring.profile.margin,
            "selected": _profile_to_dict(selected_profile) if selected_profile is not None else None,
            "top_hypotheses": [
                _hypothesis_to_dict(item)
                for item in scoring.hypotheses[:scoring_top_n]
            ],
        },
        "timing": {
            "decode_ms": _round_ms(decode_ms),
            "detect_ms": _round_ms(detect_ms),
            "rectify_ms": _round_ms(rectify_ms),
            "candidate_ms": _round_ms(candidate_ms),
            "profile_scoring_ms": _round_ms(scoring_ms),
            "profile_ms": _round_ms(candidate_ms + scoring_ms),
            "total_ms": _round_ms(total_ms),
        },
        "debug_images": debug_images,
    }


@router.post("/rectify", response_class=Response)
async def rectify_image(
    file: UploadFile = File(..., description="Полная фотография лицевой стороны письма"),
    min_area_ratio: float = Query(default=0.15, ge=0.05, le=0.90),
    canonical_orientation: bool = Query(
        default=True,
        description="Если ориентация 0/180 определена, вернуть изображение текстом вверх",
    ),
) -> Response:
    _, image = await _read_and_decode(file)
    try:
        detection = detect_envelope_quad(image, min_area_ratio=min_area_ratio)
    except EnvelopeNotFoundError as exc:
        raise HTTPException(
            status_code=422,
            detail={"layout_status": "reject", "reason": "envelope_quad_not_found", "message": str(exc)},
        ) from exc

    rectified = rectify_envelope(image, detection.points)
    scoring = score_gost_profiles(
        rectified.image,
        DOMESTIC_PROFILES,
        frame_contact_sides=detection.frame_contact_sides,
        profile_min_margin=0.085 if detection.frame_contact_sides else 0.055,
    )

    output = rectified.image
    if canonical_orientation and scoring.orientation.status == "resolved" and scoring.orientation.value_deg == 180:
        output = cv2.rotate(output, cv2.ROTATE_180)

    ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise HTTPException(status_code=500, detail="Не удалось закодировать rectified JPEG")

    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-ToolOCR-Stage": "2.1",
            "X-ToolOCR-Detector": detection.method,
            "X-ToolOCR-Quad-Confidence": f"{detection.confidence:.4f}",
            "X-ToolOCR-Frame-Status": detection.frame_status,
            "X-ToolOCR-Frame-Contact-Sides": ",".join(detection.frame_contact_sides),
            "X-ToolOCR-Orientation-Status": scoring.orientation.status,
            "X-ToolOCR-Orientation": (
                str(scoring.orientation.value_deg)
                if scoring.orientation.value_deg is not None
                else "ambiguous"
            ),
            "X-ToolOCR-Profile": scoring.profile.profile_id or "ambiguous",
        },
    )

from __future__ import annotations

import base64
import os
import time
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile

from .camera_calibration import (
    CameraCalibrationError,
    CalibrationConsensus,
    build_plane_calibration,
    calibration_to_dict,
    load_plane_calibrations,
    match_format_by_metric,
    measure_quad_mm_consensus,
)
from .format_modes import (
    FormatMode,
    decide_format,
    expected_aspect_error,
    require_expected_format,
)
from .frame_normalization import (
    detection_to_source,
    normalization_to_dict,
    normalize_black_background,
)
from .gost_r_51506_99 import (
    ENVELOPE_SPECS,
    GOST_ID,
    EnvelopeFormat,
    candidate_formats_by_aspect_ratio,
)
from .layout import EnvelopeNotFoundError, detect_envelope_quad, draw_detection_overlay, rectify_envelope
from .profile_scoring import ProfileHypothesis, score_gost_profiles
from .profiles import DOMESTIC_PROFILES, PROFILE_BY_ID, profiles_for_format

router = APIRouter(prefix="/v1/layout", tags=["layout"])
MAX_UPLOAD_MB = int(os.environ.get("OCR_MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_PIXELS = int(os.environ.get("OCR_MAX_PIXELS", "40000000"))
CAMERA_CALIBRATION_PATH = os.environ.get(
    "LAYOUT_CAMERA_CALIBRATION",
    "/app/config/camera-calibration.json",
)
FORMAT_RATIO_TOLERANCE = 0.08


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
            "barcode_layout": components.barcode_layout,
            "address_layout": components.address_layout,
            "text_direction": components.text_direction,
            "content_orientation": components.content_orientation,
            "orientation_signal": components.orientation_signal,
        },
    }


def _orientation_evidence_to_dict(evidence: Any) -> dict[str, Any]:
    return {
        "orientation_deg": evidence.orientation_deg,
        "postage": evidence.postage,
        "code_stamp": evidence.code_stamp,
        "barcode_layout": evidence.barcode_layout,
        "address_layout": evidence.address_layout,
        "text_direction": evidence.text_direction,
        "content_orientation": evidence.content_orientation,
        "base_score": evidence.base_score,
        "contrast": {
            "postage_delta": evidence.postage_delta,
            "code_stamp_delta": evidence.code_stamp_delta,
            "barcode_delta": evidence.barcode_delta,
            "address_delta": evidence.address_delta,
            "text_delta": evidence.text_delta,
            "bonus": evidence.contrast_bonus,
        },
        "agreement": {
            "channels": evidence.agreement_channels,
            "bonus": evidence.agreement_bonus,
        },
        "score": evidence.score,
    }


def _measurement_to_dict(measurement: Any) -> dict[str, Any]:
    return {
        "width_mm": measurement.width_mm,
        "height_mm": measurement.height_mm,
        "top_width_mm": measurement.top_width_mm,
        "bottom_width_mm": measurement.bottom_width_mm,
        "left_height_mm": measurement.left_height_mm,
        "right_height_mm": measurement.right_height_mm,
        "width_exact": measurement.width_exact,
        "height_exact": measurement.height_exact,
    }


def _validate_format_contract(
    format_mode: FormatMode,
    expected_format: EnvelopeFormat | None,
) -> EnvelopeFormat | None:
    try:
        return require_expected_format(format_mode, expected_format)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "reason": "expected_format_required",
                "format_mode": format_mode.value,
                "message": str(exc),
            },
        ) from exc


def _load_calibration_status(
    reference_format: EnvelopeFormat | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if not CAMERA_CALIBRATION_PATH:
        return (), {"status": "disabled", "path": None}
    try:
        all_calibrations = load_plane_calibrations(CAMERA_CALIBRATION_PATH)
    except CameraCalibrationError as exc:
        return (), {
            "status": "unavailable",
            "path": CAMERA_CALIBRATION_PATH,
            "reason": str(exc),
        }

    available_formats = [item.reference_format for item in all_calibrations]
    if reference_format is None:
        calibrations = all_calibrations
    else:
        calibrations = tuple(
            item for item in all_calibrations
            if item.reference_format == reference_format.value
        )
        if not calibrations:
            return (), {
                "status": "reference_missing",
                "path": CAMERA_CALIBRATION_PATH,
                "requested_reference_format": reference_format.value,
                "available_reference_formats": available_formats,
                "reason": f"Нет калибровки для expected_format={reference_format.value}",
            }

    return calibrations, {
        "status": "loaded",
        "path": CAMERA_CALIBRATION_PATH,
        "count": len(calibrations),
        "reference_formats": [item.reference_format for item in calibrations],
        "available_reference_formats": available_formats,
        "requested_reference_format": reference_format.value if reference_format is not None else None,
        "metric_mode": "quad_pixel_scale",
        "entries": [
            {
                "reference_format": item.reference_format,
                "image_width_px": item.image_width_px,
                "image_height_px": item.image_height_px,
                "image_aspect_ratio": round(item.image_aspect_ratio, 9),
            }
            for item in calibrations
        ],
    }


def _consensus_to_dict(consensus: CalibrationConsensus | None) -> dict[str, Any] | None:
    if consensus is None:
        return None
    return {
        "consistent": consensus.consistent,
        "metric_mode": "quad_pixel_scale",
        "reference_formats": list(consensus.reference_formats),
        "width_spread_mm": consensus.width_spread_mm,
        "height_spread_mm": consensus.height_spread_mm,
        "measurement": _measurement_to_dict(consensus.measurement),
        "per_reference": [
            {
                "reference_format": item.reference_format,
                "measurement": _measurement_to_dict(item.measurement),
            }
            for item in consensus.per_reference
        ],
    }


def _metric_decision_to_dict(
    decision: Any | None,
    calibration_status: dict[str, Any],
    consensus: CalibrationConsensus | None,
) -> dict[str, Any]:
    if decision is None:
        return {
            "status": "unavailable",
            "format": None,
            "confidence": 0.0,
            "margin": 0.0,
            "calibration": calibration_status,
            "consensus": _consensus_to_dict(consensus),
            "measurement": _measurement_to_dict(consensus.measurement) if consensus is not None else None,
            "candidates": [],
        }

    measurement = decision.measurement
    return {
        "status": decision.status,
        "format": decision.format.value if decision.format is not None else None,
        "confidence": decision.confidence,
        "margin": decision.margin,
        "calibration": calibration_status,
        "consensus": _consensus_to_dict(consensus),
        "measurement": _measurement_to_dict(measurement),
        "candidates": [
            {
                "format": candidate.format.value,
                "score": candidate.score,
                "width_error_mm": candidate.width_error_mm,
                "height_error_mm": candidate.height_error_mm,
                "width_mode": candidate.width_mode,
                "height_mode": candidate.height_mode,
            }
            for candidate in decision.candidates
        ],
    }


def _normalize_and_detect(image: np.ndarray, *, min_area_ratio: float) -> tuple[Any, Any]:
    normalization = normalize_black_background(image)
    detection_local = detect_envelope_quad(
        normalization.image,
        min_area_ratio=min_area_ratio,
    )
    detection = detection_to_source(detection_local, normalization)
    return normalization, detection


def _resolve_metric(
    image: np.ndarray,
    detection: Any,
    *,
    reference_format: EnvelopeFormat | None = None,
) -> tuple[Any | None, CalibrationConsensus | None, dict[str, Any]]:
    calibrations, status = _load_calibration_status(reference_format)
    if not calibrations:
        return None, None, status

    try:
        consensus = measure_quad_mm_consensus(
            calibrations,
            detection.points,
            image_width_px=int(image.shape[1]),
            image_height_px=int(image.shape[0]),
            frame_contact_sides=detection.frame_contact_sides,
        )
    except CameraCalibrationError as exc:
        return None, None, {
            **status,
            "status": "invalid_for_frame",
            "reason": str(exc),
        }

    if not consensus.consistent:
        return None, consensus, {
            **status,
            "status": "inconsistent",
            "reason": (
                "Pixel-scale калибровки расходятся более допустимого порога: "
                f"width_spread={consensus.width_spread_mm} мм, "
                f"height_spread={consensus.height_spread_mm} мм"
            ),
        }

    return match_format_by_metric(consensus.measurement), consensus, status


def _profile_candidates_for_mode(
    *,
    format_mode: FormatMode,
    expected_format: EnvelopeFormat | None,
    format_candidates: tuple[Any, ...],
    rectified_width_px: int,
    rectified_height_px: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if format_mode is FormatMode.AUTO:
        for candidate in format_candidates:
            for profile in profiles_for_format(candidate.format):
                result.append(_profile_to_dict(profile, candidate.ratio_error))
        return result

    assert expected_format is not None
    ratio_error = expected_aspect_error(
        rectified_width_px,
        rectified_height_px,
        expected_format,
    )
    for profile in profiles_for_format(expected_format):
        result.append(_profile_to_dict(profile, round(ratio_error, 6)))
    return result


@router.get("/profiles")
def list_profiles() -> dict[str, Any]:
    return {
        "standard": GOST_ID,
        "scope": "domestic",
        "count": len(DOMESTIC_PROFILES),
        "profiles": [_profile_to_dict(profile) for profile in DOMESTIC_PROFILES],
    }


@router.post("/calibration/estimate")
async def estimate_camera_calibration(
    file: UploadFile = File(..., description="Полная фотография эталонного конверта"),
    known_format: EnvelopeFormat = Query(..., description="Физический формат эталона по ГОСТ"),
    min_area_ratio: float = Query(default=0.15, ge=0.05, le=0.90),
) -> dict[str, Any]:
    """Строит запись калибровки по эталону, предварительно убрав черный фон."""

    raw, image = await _read_and_decode(file)
    try:
        normalization, detection = _normalize_and_detect(image, min_area_ratio=min_area_ratio)
    except EnvelopeNotFoundError as exc:
        raise HTTPException(
            status_code=422,
            detail={"calibration_status": "reject", "reason": "envelope_quad_not_found", "message": str(exc)},
        ) from exc

    if detection.frame_contact_sides:
        raise HTTPException(
            status_code=422,
            detail={
                "calibration_status": "reject",
                "reason": "reference_partial_frame",
                "frame_contact_sides": list(detection.frame_contact_sides),
                "message": "Калибровочный эталон должен целиком попадать в кадр",
            },
        )

    try:
        calibration = build_plane_calibration(
            detection.points,
            image_width_px=int(image.shape[1]),
            image_height_px=int(image.shape[0]),
            spec=ENVELOPE_SPECS[known_format],
            standard=GOST_ID,
        )
    except CameraCalibrationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"calibration_status": "reject", "reason": "calibration_geometry", "message": str(exc)},
        ) from exc

    return {
        "stage": "2.1",
        "standard": GOST_ID,
        "calibration_status": "estimated",
        "input": {
            "filename": file.filename,
            "bytes_received": len(raw),
            "width_px": int(image.shape[1]),
            "height_px": int(image.shape[0]),
        },
        "frame_normalization": normalization_to_dict(normalization),
        "known_format": known_format.value,
        "detector": {
            "method": detection.method,
            "confidence": detection.confidence,
            "frame_status": detection.frame_status,
        },
        "calibration": calibration_to_dict(calibration),
    }


@router.post("/analyze")
async def analyze_layout(
    file: UploadFile = File(..., description="Полная фотография лицевой стороны письма"),
    include_debug_images: bool = Query(default=False),
    min_area_ratio: float = Query(default=0.15, ge=0.05, le=0.90),
    scoring_top_n: int = Query(default=8, ge=1, le=32),
    format_mode: FormatMode = Query(default=FormatMode.AUTO),
    expected_format: EnvelopeFormat | None = Query(default=None),
) -> dict[str, Any]:
    total_started = time.perf_counter()
    expected = _validate_format_contract(format_mode, expected_format)

    decode_started = time.perf_counter()
    raw, image = await _read_and_decode(file)
    decode_ms = (time.perf_counter() - decode_started) * 1000.0

    normalization_started = time.perf_counter()
    normalization = normalize_black_background(image)
    normalization_ms = (time.perf_counter() - normalization_started) * 1000.0

    detect_started = time.perf_counter()
    try:
        detection_local = detect_envelope_quad(
            normalization.image,
            min_area_ratio=min_area_ratio,
        )
        detection = detection_to_source(detection_local, normalization)
    except EnvelopeNotFoundError as exc:
        raise HTTPException(
            status_code=422,
            detail={"layout_status": "reject", "reason": "envelope_quad_not_found", "message": str(exc)},
        ) from exc
    detect_ms = (time.perf_counter() - detect_started) * 1000.0

    metric_started = time.perf_counter()
    metric_reference = expected if format_mode is not FormatMode.AUTO else None
    metric_decision, metric_consensus, calibration_status = _resolve_metric(
        image,
        detection,
        reference_format=metric_reference,
    )
    metric_ms = (time.perf_counter() - metric_started) * 1000.0

    rectify_started = time.perf_counter()
    rectified = rectify_envelope(image, detection.points)
    rectify_ms = (time.perf_counter() - rectify_started) * 1000.0

    candidate_started = time.perf_counter()
    format_candidates = tuple(candidate_formats_by_aspect_ratio(
        rectified.width_px,
        rectified.height_px,
        max_relative_error=FORMAT_RATIO_TOLERANCE,
    ))
    profile_candidates = _profile_candidates_for_mode(
        format_mode=format_mode,
        expected_format=expected,
        format_candidates=format_candidates,
        rectified_width_px=rectified.width_px,
        rectified_height_px=rectified.height_px,
    )
    candidate_ms = (time.perf_counter() - candidate_started) * 1000.0

    metric_observed_format = (
        metric_decision.format
        if metric_decision is not None and metric_decision.status == "resolved"
        else None
    )
    if format_mode is FormatMode.AUTO:
        scoring_profiles = (
            profiles_for_format(metric_observed_format)
            if metric_observed_format is not None
            else DOMESTIC_PROFILES
        )
    else:
        assert expected is not None
        scoring_profiles = profiles_for_format(expected)

    scoring_started = time.perf_counter()
    scoring = score_gost_profiles(
        rectified.image,
        scoring_profiles,
        frame_contact_sides=detection.frame_contact_sides,
        profile_min_margin=(
            0.055
            if format_mode is not FormatMode.AUTO or metric_observed_format is not None
            else (0.085 if detection.frame_contact_sides else 0.055)
        ),
    )
    scoring_ms = (time.perf_counter() - scoring_started) * 1000.0

    selected_profile = (
        PROFILE_BY_ID.get(scoring.profile.profile_id)
        if scoring.profile.profile_id is not None
        else None
    )
    format_decision = decide_format(
        format_mode=format_mode,
        expected_format=expected,
        metric_decision=metric_decision,
        selected_profile=selected_profile,
        format_candidates=format_candidates,
        rectified_width_px=rectified.width_px,
        rectified_height_px=rectified.height_px,
        partial_frame=bool(detection.frame_contact_sides),
        ratio_tolerance=FORMAT_RATIO_TOLERANCE,
    )

    debug_images = None
    if include_debug_images:
        overlay = draw_detection_overlay(image, detection)
        canonical = rectified.image
        if scoring.orientation.status == "resolved" and scoring.orientation.value_deg == 180:
            canonical = cv2.rotate(canonical, cv2.ROTATE_180)
        debug_images = {
            "normalized_jpeg_base64": _encode_debug_jpeg(normalization.image),
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
        "frame_normalization": normalization_to_dict(normalization),
        "detector": {
            "method": detection.method,
            "confidence": detection.confidence,
            "raw_confidence": detection.raw_confidence,
            "frame_status": detection.frame_status,
            "frame_contact_sides": list(detection.frame_contact_sides),
            "area_ratio": detection.area_ratio,
            "area_ratio_scope": "normalized_crop",
            "rectangularity": detection.rectangularity,
            "angle_score": detection.angle_score,
            "quad_order": ["TL", "TR", "BR", "BL"],
            "quad": quad,
        },
        "format_mode": format_mode.value,
        "expected_format": expected.value if expected is not None else None,
        "format_validation": format_decision.validation,
        "metric_format": _metric_decision_to_dict(
            metric_decision,
            calibration_status,
            metric_consensus,
        ),
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
            "evidence": [
                _orientation_evidence_to_dict(item)
                for item in scoring.orientation_evidence
            ],
        },
        "format_status": format_decision.status,
        "format": format_decision.format.value if format_decision.format is not None else None,
        "format_candidates": candidates_json,
        "profile_scope": (
            "domestic" if format_mode is FormatMode.AUTO else f"expected_format:{expected.value}"
        ),
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
            "normalization_ms": _round_ms(normalization_ms),
            "detect_ms": _round_ms(detect_ms),
            "metric_ms": _round_ms(metric_ms),
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
    format_mode: FormatMode = Query(default=FormatMode.AUTO),
    expected_format: EnvelopeFormat | None = Query(default=None),
) -> Response:
    expected = _validate_format_contract(format_mode, expected_format)
    _, image = await _read_and_decode(file)
    try:
        normalization, detection = _normalize_and_detect(image, min_area_ratio=min_area_ratio)
    except EnvelopeNotFoundError as exc:
        raise HTTPException(
            status_code=422,
            detail={"layout_status": "reject", "reason": "envelope_quad_not_found", "message": str(exc)},
        ) from exc

    metric_reference = expected if format_mode is not FormatMode.AUTO else None
    metric_decision, _, _ = _resolve_metric(
        image,
        detection,
        reference_format=metric_reference,
    )
    metric_observed_format = (
        metric_decision.format
        if metric_decision is not None and metric_decision.status == "resolved"
        else None
    )

    rectified = rectify_envelope(image, detection.points)
    if format_mode is FormatMode.AUTO:
        scoring_profiles = (
            profiles_for_format(metric_observed_format)
            if metric_observed_format is not None
            else DOMESTIC_PROFILES
        )
    else:
        assert expected is not None
        scoring_profiles = profiles_for_format(expected)

    scoring = score_gost_profiles(
        rectified.image,
        scoring_profiles,
        frame_contact_sides=detection.frame_contact_sides,
    )
    selected_profile = (
        PROFILE_BY_ID.get(scoring.profile.profile_id)
        if scoring.profile.profile_id is not None
        else None
    )
    format_candidates = tuple(candidate_formats_by_aspect_ratio(
        rectified.width_px,
        rectified.height_px,
        max_relative_error=FORMAT_RATIO_TOLERANCE,
    ))
    format_decision = decide_format(
        format_mode=format_mode,
        expected_format=expected,
        metric_decision=metric_decision,
        selected_profile=selected_profile,
        format_candidates=format_candidates,
        rectified_width_px=rectified.width_px,
        rectified_height_px=rectified.height_px,
        partial_frame=bool(detection.frame_contact_sides),
        ratio_tolerance=FORMAT_RATIO_TOLERANCE,
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
            "X-ToolOCR-Frame-Normalization": normalization.status,
            "X-ToolOCR-Detector": detection.method,
            "X-ToolOCR-Quad-Confidence": f"{detection.confidence:.4f}",
            "X-ToolOCR-Frame-Status": detection.frame_status,
            "X-ToolOCR-Frame-Contact-Sides": ",".join(detection.frame_contact_sides),
            "X-ToolOCR-Format-Mode": format_mode.value,
            "X-ToolOCR-Expected-Format": expected.value if expected is not None else "",
            "X-ToolOCR-Format-Status": format_decision.status,
            "X-ToolOCR-Format": (
                format_decision.format.value if format_decision.format is not None else "ambiguous"
            ),
            "X-ToolOCR-Orientation-Status": scoring.orientation.status,
            "X-ToolOCR-Orientation": (
                str(scoring.orientation.value_deg)
                if scoring.orientation.value_deg is not None
                else "ambiguous"
            ),
            "X-ToolOCR-Profile": scoring.profile.profile_id or "ambiguous",
        },
    )

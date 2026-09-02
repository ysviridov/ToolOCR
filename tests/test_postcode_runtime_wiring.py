from ocr.app import application, postcode_runtime, test_ui_postcode_runtime


def test_roi_runtime_uses_onnx_primary_recognizer():
    assert (
        application._roi_test_ui.recognize_postcode_digits
        is postcode_runtime.recognize_postcode_digits
    )
    assert (
        application._roi_test_ui.postcode_recognition_to_dict
        is postcode_runtime.postcode_recognition_to_dict
    )
    assert (
        application._roi_test_ui.postcode_digit_overlay_labels
        is postcode_runtime.postcode_digit_overlay_labels
    )
    assert (
        application._roi_test_ui.draw_postcode_recognition_summary
        is postcode_runtime.draw_postcode_recognition_summary
    )


def test_batch_debug_uses_same_onnx_runtime_as_roi_meta():
    legacy = test_ui_postcode_runtime._legacy_test_ui
    assert legacy.recognize_postcode_digits is postcode_runtime.recognize_postcode_digits
    assert legacy.postcode_recognition_to_dict is postcode_runtime.postcode_recognition_to_dict
    assert 'data-toolocr-postcode-meta="v1"' in legacy._PREVIEW_SCRIPT
    assert "/roi/meta" in legacy._PREVIEW_SCRIPT


def test_current_batch_router_exposes_expected_endpoint():
    # FastAPI 0.141.x сохраняет include_router() в app.routes как
    # _IncludedRouter и не flatten-ит дочерние APIRoute. Поэтому контракт
    # проверяем непосредственно на исходном APIRouter.
    matches = [
        route
        for route in test_ui_postcode_runtime.router.routes
        if getattr(route, "path", None) == "/v1/test-ui/run"
        and "POST" in (getattr(route, "methods", set()) or set())
    ]
    assert matches
    assert matches[0].endpoint is test_ui_postcode_runtime.run_test_with_current_postcode_runtime


def test_runtime_debug_exposes_engine_top_level_and_roi_meta_url(monkeypatch):
    monkeypatch.setenv("POSTCODE_RECOGNIZER_ENGINE", "onnx")
    monkeypatch.setenv("POSTCODE_ONNX_MODEL", "/app/models/postcode_digit_v1.onnx")
    result = {
        "id": "a" * 32,
        "debug": {
            "test_ui_postcode_ocr": {
                "status": "recognized",
                "text": "125009",
                "postcode": "125009",
                "confidence": 0.99,
                "min_digit_confidence": 0.98,
                "digits": [
                    {
                        "index": index,
                        "digit": str(index % 10),
                        "confidence": 0.99,
                        "engine": "onnx",
                        "top3": [],
                    }
                    for index in range(1, 7)
                ],
            }
        },
    }

    test_ui_postcode_runtime._enrich_runtime_debug(result)

    ocr = result["debug"]["test_ui_postcode_ocr"]
    assert ocr["engine"] == "onnx_postcode_digit_v1+stencil_dot_suppression_v1"
    assert ocr["model_path"] == "/app/models/postcode_digit_v1.onnx"
    assert ocr["geometric_mean_confidence"] == 0.99
    assert ocr["roi_meta_url"] == f"/v1/test-ui/images/{'a' * 32}/roi/meta"
    assert result["debug"]["test_ui_roi_meta"]["source"] == "live_endpoint"

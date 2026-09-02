from ocr.app import application, test_ui_diagnostics, test_ui_library_preview


def test_diagnostic_layer_is_installed_into_test_ui():
    assert application._test_ui_diagnostics is test_ui_diagnostics
    assert 'data-toolocr-debug-diagnostics="v1"' in test_ui_library_preview._ROI_STATUS_STYLE
    assert 'data-toolocr-debug-diagnostics="v1"' in test_ui_library_preview._PREVIEW_SCRIPT


def test_debug_button_opens_human_readable_diagnostics_before_raw_json():
    script = test_ui_library_preview._PREVIEW_SCRIPT
    assert "window.showDebug = id =>" in script
    assert "renderDiagnostics(item)" in script
    assert "Диагностика —" in script
    assert "Показать JSON" in script
    assert "Скачать JSON" in script
    assert "debugJsonDialog" in script


def test_diagnostics_have_documented_tooltips_for_core_pipeline():
    script = test_ui_library_preview._PREVIEW_SCRIPT
    for expected in (
        "stage-2.1-frame-normalization.md",
        "stage-2.1-format-modes.md",
        "stage-2.1-orientation-content.md",
        "stage-2.1-profile-scoring.md",
        "stage-2.2-postcode-cnn-runtime.md",
        "stage-2.2-test-ui-diagnostics.md",
    ):
        assert expected in script
    assert "docFor(path, key)" in script
    assert "diag-info" in script


def test_heavy_debug_payloads_are_collapsed_not_rendered_inline():
    script = test_ui_library_preview._PREVIEW_SCRIPT
    assert "diag-nested" in script
    assert "diag-array-item" in script
    assert "key.endsWith('_jpeg_base64')" in script
    assert "binary/base64 скрыт" in script


def test_single_file_json_export_keeps_original_debug_payload():
    script = test_ui_library_preview._PREVIEW_SCRIPT
    assert "toolocr.test-ui.debug-item.v1" in script
    assert "debug:item.debug" in script
    assert "toolocr-debug-${safeFilename(item.name)}.json" in script

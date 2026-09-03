from ocr.app import application, test_ui_diagnostics_layout_v3, test_ui_library_preview


def test_layout_v3_layer_is_installed_after_existing_diagnostics():
    assert application._test_ui_diagnostics_layout_v3 is test_ui_diagnostics_layout_v3
    assert 'data-toolocr-debug-diagnostics-layout="v3"' in test_ui_library_preview._ROI_STATUS_STYLE
    assert 'data-toolocr-debug-diagnostics-layout="v3"' in test_ui_library_preview._PREVIEW_SCRIPT


def test_debug_dialog_uses_available_viewport_and_supports_manual_resize():
    style = test_ui_library_preview._ROI_STATUS_STYLE
    assert 'width:calc(100vw - 24px)' in style
    assert 'max-width:none' in style
    assert 'resize:both' in style
    assert 'max-height:calc(100vh - 24px)' in style


def test_complex_arrays_get_specialized_human_readable_views():
    script = test_ui_library_preview._PREVIEW_SCRIPT
    for expected in (
        "renderFormatCandidates",
        "renderOrientationScores",
        "renderOrientationEvidence",
        "renderProfileHypotheses",
        "renderProfileSelected",
        "renderQuad",
        "orientation.evidence",
        "profile_scoring.top_hypotheses",
        "format_candidates",
        "detector.quad",
    ):
        assert expected in script


def test_layout_v3_preserves_existing_show_debug_pipeline():
    script = test_ui_library_preview._PREVIEW_SCRIPT
    assert "const previousShowDebug = window.showDebug" in script
    assert "previousShowDebug(id)" in script
    assert "refineComplexSections(id)" in script

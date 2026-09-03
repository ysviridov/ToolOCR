from ocr.app import (
    application,
    test_ui_diagnostics_refinements,
    test_ui_library_preview,
)


def test_diagnostic_refinements_are_installed_after_base_layer():
    assert application._test_ui_diagnostics_refinements is test_ui_diagnostics_refinements
    assert 'data-toolocr-debug-diagnostics-refinements="v2"' in test_ui_library_preview._ROI_STATUS_STYLE
    assert 'data-toolocr-debug-diagnostics-refinements="v2"' in test_ui_library_preview._PREVIEW_SCRIPT


def test_postcode_digits_use_flat_cards_instead_of_deep_top3_tables():
    script = test_ui_diagnostics_refinements._REFINEMENT_SCRIPT
    style = test_ui_diagnostics_refinements._REFINEMENT_STYLE
    assert "renderDigitCards(digits)" in script
    assert "renderTop3(digit?.top3)" in script
    assert "cells[1].colSpan = 2" in script
    assert "diag-digit-grid" in style
    assert "diag-top3-mini" in style


def test_nested_tables_do_not_inherit_fixed_column_widths():
    style = test_ui_diagnostics_refinements._REFINEMENT_STYLE
    assert ".diag-nested-body table.diag-table" in style
    assert "table-layout:auto" in style
    assert "min-width:640px" in style


def test_generic_payload_phrase_is_replaced_by_human_fallback():
    script = test_ui_diagnostics_refinements._REFINEMENT_SCRIPT
    assert "humanFallbackDescription" in script
    assert "Диагностический признак" in script
    assert "из диагностического payload ToolOCR" not in script


def test_postcode_runtime_fields_have_specific_human_descriptions():
    script = test_ui_diagnostics_refinements._REFINEMENT_SCRIPT
    for path in (
        "test_ui_postcode_ocr.reason",
        "test_ui_postcode_ocr.engine",
        "test_ui_postcode_ocr.model_path",
        "test_ui_postcode_ocr.digits[].top3",
        "test_ui_postcode_ocr.digits[].preprocess",
    ):
        assert path in script
    assert "Высокое значение не гарантирует отсутствие ошибки" in script

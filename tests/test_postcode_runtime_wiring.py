from ocr.app import application, postcode_runtime


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

from ocr.app.calibration_modes import CalibrationMode, validate_calibration_frame


def test_strict_rejects_bottom_contact():
    result = validate_calibration_frame(CalibrationMode.STRICT, ("bottom",))

    assert result.accepted is False
    assert result.contact_sides == ("bottom",)
    assert result.allowed_contact_sides == ()
    assert result.reason


def test_scale_reference_accepts_bottom_contact():
    result = validate_calibration_frame(CalibrationMode.SCALE_REFERENCE, ("bottom",))

    assert result.accepted is True
    assert result.contact_sides == ("bottom",)
    assert result.allowed_contact_sides == ("bottom",)
    assert result.reason is None


def test_scale_reference_accepts_fully_visible_reference():
    result = validate_calibration_frame(CalibrationMode.SCALE_REFERENCE, ())

    assert result.accepted is True
    assert result.contact_sides == ()


def test_scale_reference_rejects_side_or_top_contact():
    for sides in (("left",), ("right",), ("top",), ("bottom", "left")):
        result = validate_calibration_frame(CalibrationMode.SCALE_REFERENCE, sides)
        assert result.accepted is False
        assert result.reason

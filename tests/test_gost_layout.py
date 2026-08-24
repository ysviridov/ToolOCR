from ocr.app.gost_r_51506_99 import (
    ENVELOPE_SPECS,
    EnvelopeFormat,
    RectMM,
    candidate_formats_by_aspect_ratio,
    mm_to_normalized,
)


def test_dl_is_distinguishable_by_aspect_ratio():
    candidates = candidate_formats_by_aspect_ratio(2200, 1100)
    assert [item.format for item in candidates] == [EnvelopeFormat.DL]


def test_sqrt2_family_must_remain_ambiguous_by_ratio():
    candidates = candidate_formats_by_aspect_ratio(1620, 1140)
    formats = {item.format for item in candidates}

    assert EnvelopeFormat.C6 in formats
    assert EnvelopeFormat.C5 in formats
    assert EnvelopeFormat.C4 in formats
    assert EnvelopeFormat.B4 in formats
    assert EnvelopeFormat.DL not in formats


def test_mm_to_normalized_uses_gost_physical_size():
    spec = ENVELOPE_SPECS[EnvelopeFormat.C6]
    rect = RectMM(x=0.0, y=0.0, width=81.0, height=57.0)

    assert mm_to_normalized(rect, spec) == (0.0, 0.0, 0.5, 0.5)

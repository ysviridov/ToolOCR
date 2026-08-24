from ocr.app.gost_r_51506_99 import (
    ENVELOPE_SPECS,
    EnvelopeFormat,
    RectMM,
    RectNormalized,
    candidate_formats_by_aspect_ratio,
    mm_to_normalized,
    rotate_normalized_rect_180,
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


def test_aspect_ratio_candidates_are_180_degree_invariant():
    normal = candidate_formats_by_aspect_ratio(2200, 1100)
    rotated_180 = candidate_formats_by_aspect_ratio(2200, 1100)
    assert normal == rotated_180


def test_mm_to_normalized_uses_gost_physical_size():
    spec = ENVELOPE_SPECS[EnvelopeFormat.C6]
    rect = RectMM(x=0.0, y=0.0, width=81.0, height=57.0)

    assert mm_to_normalized(rect, spec) == (0.0, 0.0, 0.5, 0.5)


def test_rotate_normalized_rect_180_moves_top_right_to_bottom_left():
    top_right = RectNormalized(x=0.75, y=0.0, width=0.25, height=0.20)
    rotated = rotate_normalized_rect_180(top_right)

    assert rotated == RectNormalized(x=0.0, y=0.80, width=0.25, height=0.20)


def test_rotate_normalized_rect_180_is_involution():
    original = RectNormalized(x=0.11, y=0.23, width=0.31, height=0.17)
    rotated = rotate_normalized_rect_180(original)
    restored = rotate_normalized_rect_180(rotated)

    assert restored == original

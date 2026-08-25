import numpy as np

from ocr.app.frame_normalization import (
    detection_to_source,
    normalize_black_background,
)
from ocr.app.layout import QuadDetection


def _scene(width: int, height: int = 1000):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # Светлое письмо на черной ленте; нижняя сторона совпадает с краем кадра.
    x0 = 180
    y0 = 280
    x1 = min(width - 160, x0 + 1100)
    y1 = height
    image[y0:y1, x0:x1] = 235
    return image, (x0, y0, x1, y1)


def test_black_background_is_cropped_and_bottom_anchor_is_preserved():
    image, (x0, y0, x1, y1) = _scene(1500)
    normalized = normalize_black_background(image)

    assert normalized.status == "cropped"
    assert normalized.bottom_anchored is True
    assert normalized.crop_width_px < image.shape[1]
    assert normalized.crop_height_px < image.shape[0]
    assert normalized.crop_x <= x0
    assert normalized.crop_y <= y0
    assert normalized.crop_x + normalized.crop_width_px >= x1
    assert normalized.crop_y + normalized.crop_height_px == y1


def test_different_black_frame_widths_do_not_change_foreground_width():
    widths = []
    for source_width in (1450, 1470, 1490, 1510):
        image, _ = _scene(source_width)
        normalized = normalize_black_background(image)
        widths.append(normalized.foreground_width_px)

    assert max(widths) - min(widths) <= 2


def test_detection_points_are_translated_back_to_source_coordinates():
    image, _ = _scene(1500)
    normalized = normalize_black_background(image)
    local = np.array(
        [
            [20.0, 30.0],
            [500.0, 30.0],
            [500.0, 400.0],
            [20.0, 400.0],
        ],
        dtype=np.float32,
    )
    detection = QuadDetection(
        points=local,
        method="foreground_otsu",
        confidence=0.9,
        raw_confidence=0.9,
        area_ratio=0.7,
        rectangularity=0.99,
        angle_score=0.99,
        frame_contact_sides=(),
    )

    source = detection_to_source(detection, normalized)

    assert np.allclose(
        source.points,
        local + np.array([normalized.crop_x, normalized.crop_y], dtype=np.float32),
    )
    assert source.method.startswith("normalized:")

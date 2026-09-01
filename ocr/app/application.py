from __future__ import annotations

from .main import app
from .orientation_photometric import (
    install_layout_debug_metadata,
    install_photometric_orientation_preprocessing,
)

# Устанавливаем photometric preprocessing до импорта layout/test-ui роутеров,
# чтобы все вызовы score_gost_profiles использовали один и тот же pipeline.
install_photometric_orientation_preprocessing()

from . import layout_api as _layout_api

install_layout_debug_metadata(_layout_api)

from .layout_api import router as layout_router
from . import postcode_runtime as _postcode_runtime
from . import roi_test_ui as _roi_test_ui
from .test_ui import router as test_ui_router
from .test_ui_library_preview import router as test_ui_library_preview_router

# Stage 2.2 runtime wiring. ROI/Test UI исторически импортировал функции
# postcode_recognizer напрямую. Оставляем preprocessing/preview без изменений,
# но переключаем runtime recognition на ONNX-primary adapter. Это также
# сохраняет Tesseract как настраиваемый fallback без дублирования роутов.
_roi_test_ui.recognize_postcode_digits = _postcode_runtime.recognize_postcode_digits
_roi_test_ui.postcode_recognition_to_dict = _postcode_runtime.postcode_recognition_to_dict
_roi_test_ui.postcode_digit_overlay_labels = _postcode_runtime.postcode_digit_overlay_labels
_roi_test_ui.draw_postcode_recognition_summary = _postcode_runtime.draw_postcode_recognition_summary
roi_test_ui_router = _roi_test_ui.router


app.include_router(layout_router)
# Расширенный /test-ui подключаем раньше базового route: FastAPI использует
# первый совпавший маршрут. Остальные API-маршруты test_ui_router остаются без изменений.
app.include_router(test_ui_library_preview_router)
app.include_router(test_ui_router)
app.include_router(roi_test_ui_router)

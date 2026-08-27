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
from .roi_test_ui import router as roi_test_ui_router
from .test_ui import router as test_ui_router
from .test_ui_library_preview import router as test_ui_library_preview_router


app.include_router(layout_router)
# Расширенный /test-ui подключаем раньше базового route: FastAPI использует
# первый совпавший маршрут. Остальные API-маршруты test_ui_router остаются без изменений.
app.include_router(test_ui_library_preview_router)
app.include_router(test_ui_router)
app.include_router(roi_test_ui_router)

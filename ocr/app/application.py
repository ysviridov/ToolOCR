from __future__ import annotations

from .layout_api import router as layout_router
from .main import app
from .roi_test_ui import router as roi_test_ui_router
from .test_ui import router as test_ui_router


app.include_router(layout_router)
app.include_router(test_ui_router)
app.include_router(roi_test_ui_router)

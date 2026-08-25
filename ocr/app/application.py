from __future__ import annotations

from .layout_api import router as layout_router
from .main import app
from .test_ui import router as test_ui_router


app.include_router(layout_router)
app.include_router(test_ui_router)

from __future__ import annotations

from .layout_api import router as layout_router
from .main import app


app.include_router(layout_router)

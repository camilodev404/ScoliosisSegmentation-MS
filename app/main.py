from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="API para ejecutar inferencia del pipeline de segmentacion thoracolumbar.",
    )
    app.include_router(router, prefix=settings.api_prefix)
    return app


app = create_app()


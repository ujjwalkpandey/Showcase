from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import api_router
from app.core.config import settings
from app.adapters.database import engine
from app import models

models.Portfolio.metadata.create_all(bind=engine)

def get_application() -> FastAPI:

    _app = FastAPI(
        title=settings.PROJECT_NAME,
        version="1.0.0",
        description="AI: Transforming resumes into portfolios."
    )

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _app.include_router(api_router, prefix=settings.API_V1_STR)

    return _app

app = get_application()

@app.get("/", tags=["Health Check"])
async def health_check():
    return {
        "status": "online",
        "message": "Welcome AI. Your portfolio is one upload away."
    }
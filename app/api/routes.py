from fastapi import APIRouter
from app.api.v1 import resume, portfolio

api_router = APIRouter()

api_router.include_router(resume.router, prefix="/v1/resume", tags=["Resume Processing"])
api_router.include_router(portfolio.router, prefix="/v1/portfolio", tags=["Portfolio Management"])
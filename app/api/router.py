from fastapi import APIRouter

from app.api.routes.health import router as health_router
from app.api.routes.batches import router as batches_router

api_router = APIRouter()
api_router.include_router(health_router)

api_router.include_router(batches_router, tags=["batches"])

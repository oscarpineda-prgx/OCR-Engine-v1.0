import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings
from app.db import models  # noqa: F401
from app.db.session import Base, SessionLocal, engine
from app.services.vendor_master.loader import bootstrap_vendor_master_if_empty

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(application: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        bootstrap_vendor_master_if_empty(db)
    except Exception as exc:  # pragma: no cover
        logger.warning("Vendor master bootstrap skipped: %s", exc)
    finally:
        db.close()
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.include_router(api_router, prefix=settings.api_prefix)

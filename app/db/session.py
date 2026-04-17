from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()
DATABASE_URL = f"sqlite:///{settings.db_path.as_posix()}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

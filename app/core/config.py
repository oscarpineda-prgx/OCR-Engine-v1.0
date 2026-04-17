import logging
import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "OCR Local MVP"
    app_version: str = "0.2.0"
    api_prefix: str = "/api/v1"

    base_dir: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = base_dir / "data"
    db_path: Path = data_dir / "ocr_local.db"
    tesseract_cmd: str = ""
    max_upload_size_mb: int = 50
    max_workers: int = 4
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
        force=True,
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(settings.log_level)
    return settings

import logging
import os
import shutil
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_DEFAULT_WINDOWS_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_COMMON_WINDOWS_TESSERACT_PATHS = (
    Path(_DEFAULT_WINDOWS_TESSERACT_CMD),
    Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
)


def _normalize_candidate(candidate: str | None) -> str | None:
    raw = (candidate or "").strip().strip('"').strip("'")
    if not raw:
        return None

    path_candidate = Path(raw).expanduser()
    if path_candidate.is_file():
        return str(path_candidate.resolve())

    resolved = shutil.which(raw)
    if resolved:
        return resolved

    return None


@lru_cache(maxsize=1)
def resolve_tesseract_cmd() -> str | None:
    settings = get_settings()
    candidates = [
        settings.tesseract_cmd,
        os.environ.get("TESSERACT_CMD"),
        _DEFAULT_WINDOWS_TESSERACT_CMD,
        shutil.which("tesseract"),
        *[str(path) for path in _COMMON_WINDOWS_TESSERACT_PATHS],
    ]

    for candidate in candidates:
        resolved = _normalize_candidate(candidate)
        if resolved:
            return resolved

    return None


def ensure_tesseract_on_path() -> str | None:
    resolved = resolve_tesseract_cmd()
    if not resolved:
        return None

    parent = str(Path(resolved).resolve().parent)
    current_path = os.environ.get("PATH", "")
    path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
    normalized_entries = {entry.lower() for entry in path_entries}

    if parent.lower() not in normalized_entries:
        os.environ["PATH"] = f"{parent}{os.pathsep}{current_path}" if current_path else parent

    return resolved


def configure_tesseract_cmd() -> str | None:
    resolved = ensure_tesseract_on_path()
    if not resolved:
        return None

    try:
        import pytesseract
    except ImportError:
        logger.debug("pytesseract package is not installed")
        return resolved

    if getattr(pytesseract.pytesseract, "tesseract_cmd", None) != resolved:
        pytesseract.pytesseract.tesseract_cmd = resolved

    return resolved


def is_tesseract_available() -> bool:
    return resolve_tesseract_cmd() is not None

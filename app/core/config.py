import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RF_MODEL_PATH = MODELS_DIR / "air_quality_rf.pkl"
SVM_MODEL_PATH = MODELS_DIR / "air_quality_svm.pkl"
IF_MODEL_PATH = MODELS_DIR / "air_quality_if.pkl"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost.pkl"

# Backward-compatible alias currently pointing to the Random Forest model.
MODEL_PATH = RF_MODEL_PATH

# Railway uses dynamic PORT environment variable
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8000"))

# Load .env once so all service-level settings are available at startup.
# .env is optional in production (Railway uses environment variables directly)
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=False)


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


CHATBOT_USE_OLLAMA = _env_bool("CHATBOT_USE_OLLAMA", default=False)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_TIMEOUT_SECONDS = _env_int("OLLAMA_TIMEOUT_SECONDS", 20)
OLLAMA_TEMPERATURE = _env_float("OLLAMA_TEMPERATURE", 0.8)
OLLAMA_TOP_P = _env_float("OLLAMA_TOP_P", 0.9)
OLLAMA_REPEAT_PENALTY = _env_float("OLLAMA_REPEAT_PENALTY", 1.12)

CHATBOT_ENABLE_WEB_SEARCH = _env_bool("CHATBOT_ENABLE_WEB_SEARCH", default=True)
CHATBOT_WEB_SEARCH_TIMEOUT_SECONDS = _env_int("CHATBOT_WEB_SEARCH_TIMEOUT_SECONDS", 10)
CHATBOT_WEB_SEARCH_MAX_SNIPPETS = _env_int("CHATBOT_WEB_SEARCH_MAX_SNIPPETS", 2)

CHATBOT_ENABLE_LEARNING = _env_bool("CHATBOT_ENABLE_LEARNING", default=True)
# Local fallback store used whenever the Supabase "chatbot_notes" table is unavailable.
CHATBOT_NOTES_LOCAL_PATH = DATA_DIR / "chatbot_notes.json"


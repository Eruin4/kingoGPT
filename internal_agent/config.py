from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = REPO_ROOT / "state"

DEFAULT_TOKEN_CACHE = STATE_DIR / "kingogpt_token_cache.json"
DEFAULT_TOKEN_CONFIG = STATE_DIR / "kingogpt_config.json"
DEFAULT_PROFILE_DIR = STATE_DIR / "kingogpt_chrome_profile"

DEFAULT_SESSION_KEY = "internal_agent"
DEFAULT_MAX_STEPS = 8
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_TOKEN_REFRESH_TIMEOUT_SECONDS = 300
DEFAULT_OBSERVATION_LIMIT = 8000


"""
SongGeneration Studio - Configuration
Directories, constants, and shared state initialization.
"""

import json
import sys
from pathlib import Path
from typing import Dict
from datetime import datetime

# ============================================================================
# Directory Configuration
# ============================================================================

ROOT_DIR = Path(__file__).parent
BASE_DIR = ROOT_DIR

# --- CONFIG: Default to Base (Legacy) for Mac Stability ---
DEFAULT_MODEL = "songgeneration_base"

OUTPUT_DIR = BASE_DIR / "output"
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "web" / "static"
QUEUE_FILE = BASE_DIR / "queue.json"
VERIFIED_MODELS_FILE = BASE_DIR / "verified_models.json"
TIMING_FILE = BASE_DIR / "timing_history.json"

OUTPUT_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SERVER_PORT = 42100
MODEL_SERVER_URL = f"http://127.0.0.1:{MODEL_SERVER_PORT}"
USE_MODEL_SERVER = True

MAX_TIMING_RECORDS = 1000

# Verified Models Cache
verified_models_cache: Dict[str, dict] = {}

def load_verified_models() -> dict:
    try:
        if VERIFIED_MODELS_FILE.exists():
            with open(VERIFIED_MODELS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_verified_models(cache: dict):
    try:
        with open(VERIFIED_MODELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

def mark_model_verified(model_id: str, model_pt_size: int):
    global verified_models_cache
    verified_models_cache[model_id] = {
        "verified": True,
        "model_pt_size": model_pt_size,
        "verified_at": datetime.now().isoformat()
    }
    save_verified_models(verified_models_cache)

def is_model_verified(model_id: str) -> bool:
    return model_id in verified_models_cache and verified_models_cache[model_id].get("verified")

def get_verified_model_size(model_id: str) -> int:
    if model_id in verified_models_cache:
        return verified_models_cache[model_id].get("model_pt_size", 0)
    return 0

verified_models_cache = load_verified_models()

# Queue Storage
def load_queue() -> list:
    try:
        if QUEUE_FILE.exists():
            with open(QUEUE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[QUEUE] Error loading queue: {e}")
    return []

def save_queue(queue: list):
    try:
        with open(QUEUE_FILE, 'w', encoding='utf-8') as f:
            json.dump(queue, f, indent=2)
    except Exception as e:
        print(f"[QUEUE] Error saving queue: {e}")

def log_startup_info():
    print(f"[CONFIG] Base dir: {BASE_DIR}")
    print(f"[CONFIG] Output dir: {OUTPUT_DIR}")
    print(f"[CONFIG] Python: {sys.executable}")

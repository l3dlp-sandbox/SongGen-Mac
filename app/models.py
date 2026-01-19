"""
SongGeneration Studio - Model Registry & Download Manager
Model definitions, status checking, and download management.
Updated for Mac Stability (Direct Python Download).
"""

import sys
import os
import shutil
import threading
import time
import subprocess
from typing import Dict, Optional, List
from pathlib import Path

# --- Auto-install dependency if missing ---
try:
    import huggingface_hub
    from huggingface_hub import snapshot_download
except ImportError:
    print("[loader] Installing huggingface_hub...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    import huggingface_hub
    from huggingface_hub import snapshot_download
# ------------------------------------------

from config import (
    BASE_DIR, DEFAULT_MODEL, mark_model_verified, is_model_verified,
    get_verified_model_size
)
from gpu import gpu_info

# ============================================================================
# Model Registry (NO PRIORITY SYSTEM)
# ============================================================================

MODEL_REGISTRY: Dict[str, dict] = {
    # The Stable One
    "songgeneration_base": {
        "name": "SongGeneration - Base (Fastest)",
        "description": "Legacy model. Best for Mac RAM usage (<16GB).",
        "vram_required": 10,
        "hf_repo": "lglg666/SongGeneration-base",
        "size_gb": 11.3,
        # Direct path to your installed folder
        "path": "ckpt/model_1rvq" 
    },
    
    # The Heavy One (If you want to try 80GB RAM again)
    "songgeneration_base_new": {
        "name": "SongGeneration - Base New",
        "description": "Newer architecture. High RAM usage on Mac.",
        "vram_required": 10,
        "hf_repo": "lglg666/SongGeneration-base-new",
        "size_gb": 11.3,
        "path": "songgeneration_base_new"
    },
    
    "songgeneration_large": {
        "name": "SongGeneration - Large",
        "description": "Best quality. Requires 32GB+ RAM.",
        "vram_required": 22,
        "hf_repo": "lglg666/SongGeneration-large",
        "size_gb": 20.5,
        "path": "songgeneration_large"
    },
    
    "songgeneration_base_full": {
        "name": "SongGeneration - Base Full",
        "description": "Longer duration context.",
        "vram_required": 12,
        "hf_repo": "lglg666/SongGeneration-base-full",
        "size_gb": 11.3,
        "path": "songgeneration_base_full",
    },
}

# ============================================================================
# Download State Tracking
# ============================================================================

download_states: Dict[str, dict] = {}
download_threads: Dict[str, threading.Thread] = {}
download_processes: Dict[str, subprocess.Popen] = {}
download_cancel_flags: Dict[str, threading.Event] = {}
expected_file_sizes_cache: Dict[str, dict] = {}
download_start_lock = threading.Lock()

# ============================================================================
# HuggingFace Helpers
# ============================================================================

def get_repo_file_sizes_from_hf(repo_id: str) -> dict:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
        file_sizes = {}
        total_bytes = 0
        for sibling in repo_info.siblings:
            if hasattr(sibling, 'size') and sibling.size:
                filename = sibling.rfilename
                size_bytes = sibling.size
                file_sizes[filename] = size_bytes
                total_bytes += size_bytes
        file_sizes['__total__'] = total_bytes
        return file_sizes
    except Exception as e:
        print(f"[DOWNLOAD] Could not get file sizes from HF API: {e}")
        return {}

def get_expected_file_sizes(model_id: str) -> dict:
    global expected_file_sizes_cache
    if model_id in expected_file_sizes_cache:
        return expected_file_sizes_cache[model_id]
    if model_id not in MODEL_REGISTRY:
        return {}
    hf_repo = MODEL_REGISTRY[model_id]["hf_repo"]
    file_sizes = get_repo_file_sizes_from_hf(hf_repo)
    if file_sizes:
        expected_file_sizes_cache[model_id] = file_sizes
    return file_sizes

def get_directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    try:
        for f in path.rglob('*'):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except (OSError, IOError):
                    pass
    except Exception:
        pass
    return total

# ============================================================================
# Model Status Functions
# ============================================================================

def get_model_status(model_id: str) -> str:
    if model_id in download_states and download_states[model_id].get("status") == "downloading":
        return "downloading"

    # Direct Path Lookup
    # If the registry says it's in 'ckpt/model_1rvq', we check exactly that.
    if model_id not in MODEL_REGISTRY:
        return "not_downloaded"
        
    path_config = MODEL_REGISTRY[model_id]["path"]
    folder_path = BASE_DIR / path_config
    
    if not folder_path.exists():
        return "not_downloaded"

    # Check for various model file names
    possible_files = ["model.pt", "model.safetensors", "model_2.safetensors", "model_2_fixed.safetensors"]
    
    for fname in possible_files:
        if (folder_path / fname).exists():
            return "ready"

    return "not_downloaded"

def get_model_status_quick(model_id: str) -> str:
    return get_model_status(model_id)

def is_model_ready_quick(model_id: str) -> bool:
    return get_model_status_quick(model_id) == "ready"

def get_download_progress(model_id: str) -> dict:
    if model_id not in download_states:
        return {"status": "not_started", "progress": 0}
    return download_states[model_id]

# --- MODIFIED: NO PRIORITY SORTING ---
def get_recommended_model(refresh: bool = False) -> Optional[str]:
    """Return default model if valid, otherwise any valid model."""
    from gpu import gpu_info, refresh_gpu_info
    current_gpu_info = refresh_gpu_info() if refresh else gpu_info

    # 1. Prefer the Config Default
    if DEFAULT_MODEL in MODEL_REGISTRY:
        return DEFAULT_MODEL
        
    # 2. Fallback to Base
    return "songgeneration_base"

# --- MODIFIED: NO PRIORITY SORTING ---
def get_best_ready_model(refresh: bool = False) -> Optional[str]:
    """Return default model if ready, otherwise first ready model."""
    
    # 1. Check if the Config Default is ready
    if is_model_ready_quick(DEFAULT_MODEL):
        return DEFAULT_MODEL
        
    # 2. Return any ready model
    for model_id in MODEL_REGISTRY.keys():
        if is_model_ready_quick(model_id):
            return model_id
            
    return None

def get_available_models_sync() -> List[dict]:
    models = []
    for model_id, info in MODEL_REGISTRY.items():
        status = get_model_status_quick(model_id)
        if status == "ready":
            models.append({"id": model_id, "name": info["name"], "status": status})
    return models

# ============================================================================
# Download Management
# ============================================================================

def run_model_download(model_id: str, notify_callback=None):
    global download_states, download_cancel_flags
    if model_id not in MODEL_REGISTRY:
        download_states[model_id] = {"status": "error", "error": "Unknown model"}
        return
    model_info = MODEL_REGISTRY[model_id]
    hf_repo = model_info["hf_repo"]
    local_dir = BASE_DIR / model_info["path"]
    
    print(f"[DOWNLOAD] Starting download of {model_id} from {hf_repo} to {local_dir}")
    download_states[model_id] = {
        "status": "downloading",
        "progress": 0,
        "message": "Downloading...",
        "downloaded_gb": 0,
        "total_gb": model_info["size_gb"],
    }
    if notify_callback:
        notify_callback()
    try:
        snapshot_download(
            repo_id=hf_repo,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=8
        )
        final_size = get_directory_size(local_dir)
        final_gb = final_size / (1024 * 1024 * 1024)
        download_states[model_id] = {
            "status": "completed",
            "progress": 100,
            "downloaded_gb": round(final_gb, 2),
            "total_gb": round(final_gb, 2),
        }
        if notify_callback:
            notify_callback()
    except Exception as e:
        print(f"[DOWNLOAD] Error downloading {model_id}: {e}")
        download_states[model_id] = {"status": "error", "error": str(e)}
        if notify_callback:
            notify_callback()
    finally:
        download_threads.pop(model_id, None)

def start_model_download(model_id: str, notify_callback=None) -> dict:
    if model_id not in MODEL_REGISTRY:
        return {"error": "Unknown model"}
    with download_start_lock:
        current_status = get_model_status(model_id)
        if current_status == "ready":
            return {"error": "Model already downloaded"}
        if current_status == "downloading":
            return {"error": "Model is already downloading"}
        download_states[model_id] = {
            "status": "downloading",
            "progress": 0,
            "message": "Initializing..."
        }
        thread = threading.Thread(
            target=run_model_download,
            args=(model_id, notify_callback),
            daemon=True
        )
        download_threads[model_id] = thread
        thread.start()
    return {"status": "started", "model_id": model_id}

def cancel_model_download(model_id: str) -> dict:
    if model_id in download_states:
        download_states[model_id] = {"status": "cancelled", "progress": 0}
    download_threads.pop(model_id, None)
    return {"status": "cancelled", "model_id": model_id}

def delete_model(model_id: str) -> dict:
    if model_id not in MODEL_REGISTRY:
        return {"error": "Unknown model"}
    path_name = MODEL_REGISTRY[model_id]["path"]
    folder_path = BASE_DIR / path_name
    try:
        if folder_path.exists():
            shutil.rmtree(folder_path)
        print(f"[MODEL] Deleted model {model_id} at {folder_path}")
        return {"status": "deleted", "model_id": model_id}
    except Exception as e:
        return {"error": f"Failed to delete: {e}"}

def cleanup_download_states():
    download_states.clear()
    download_threads.clear()
    print(f"[CONFIG] Cleared stale download states")

def get_model_path(model_name):
    if model_name not in MODEL_REGISTRY:
        return str(BASE_DIR / "songgeneration_base")
    return str(BASE_DIR / MODEL_REGISTRY[model_name]["path"])

def get_model_list():
    return list(MODEL_REGISTRY.keys())

def get_model_info(model_name):
    return MODEL_REGISTRY.get(model_name, {})

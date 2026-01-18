"""
SongGeneration Studio - Model Registry & Download Manager
Model definitions, status checking, and download management.
Updated for Mac Stability (Subprocess API Download + Cancel Support).
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
except ImportError:
    print("[loader] Installing huggingface_hub...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    import huggingface_hub
# ------------------------------------------

from config import (
    BASE_DIR, DEFAULT_MODEL, mark_model_verified, is_model_verified,
    get_verified_model_size
)
from gpu import gpu_info

# ============================================================================
# Model Registry
# ============================================================================

MODEL_REGISTRY: Dict[str, dict] = {
    "songgeneration_base": {
        "name": "SongGeneration - Base (Fastest)",
        "description": "Legacy model. Best for Mac RAM usage (<16GB).",
        "vram_required": 10,
        "hf_repo": "lglg666/SongGeneration-base",
        "size_gb": 11.3,
        "path": "ckpt/model_1rvq" 
    },
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
download_start_lock = threading.Lock()

# ============================================================================
# Helpers
# ============================================================================

def get_directory_size(path: Path) -> int:
    """Calculate total size of a directory in bytes."""
    if not path.exists():
        return 0
    total = 0
    try:
        # Fast walk
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except: pass
    except: pass
    return total

def get_model_status(model_id: str) -> str:
    if model_id in download_states:
        s = download_states[model_id].get("status")
        if s == "downloading": return "downloading"

    if model_id not in MODEL_REGISTRY: return "not_downloaded"
    path_config = MODEL_REGISTRY[model_id]["path"]
    folder_path = BASE_DIR / path_config
    
    if not folder_path.exists(): return "not_downloaded"

    # Check for completion (simple file check)
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

def get_recommended_model(refresh: bool = False) -> Optional[str]:
    if DEFAULT_MODEL in MODEL_REGISTRY: return DEFAULT_MODEL
    return "songgeneration_base"

def get_best_ready_model(refresh: bool = False) -> Optional[str]:
    if is_model_ready_quick(DEFAULT_MODEL): return DEFAULT_MODEL
    for model_id in MODEL_REGISTRY.keys():
        if is_model_ready_quick(model_id): return model_id
    return None

def get_available_models_sync() -> List[dict]:
    models = []
    for model_id, info in MODEL_REGISTRY.items():
        status = get_model_status_quick(model_id)
        if status == "ready":
            models.append({"id": model_id, "name": info["name"], "status": status})
    return models

# ============================================================================
# Download Management (Subprocess API Implementation)
# ============================================================================

def run_model_download_monitor(model_id: str, notify_callback=None):
    """Monitors the download subprocess and updates progress."""
    global download_states, download_processes
    
    if model_id not in MODEL_REGISTRY:
        download_states[model_id] = {"status": "error", "error": "Unknown model"}
        return

    model_info = MODEL_REGISTRY[model_id]
    hf_repo = model_info["hf_repo"]
    local_dir = BASE_DIR / model_info["path"]
    target_gb = model_info["size_gb"]
    
    print(f"[DOWNLOAD] Starting download of {model_id} from {hf_repo}")
    
    # 1. Start the Download Process
    # Instead of 'huggingface-cli', we run a small python script that imports the library directly.
    # This avoids "No module named huggingface_hub.cli" errors.
    safe_dir = str(local_dir).replace("\\", "\\\\").replace("'", "\\'")
    
    download_script = (
        f"from huggingface_hub import snapshot_download; "
        f"print('Starting internal download...'); "
        f"snapshot_download("
        f"repo_id='{hf_repo}', "
        f"local_dir='{safe_dir}', "
        f"local_dir_use_symlinks=False, "
        f"resume_download=True, "
        f"max_workers=8"
        f")"
    )

    cmd = [sys.executable, "-c", download_script]
    
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        download_processes[model_id] = proc
        
        # 2. Monitor Loop
        download_states[model_id] = {"status": "downloading", "progress": 0, "message": "Starting..."}
        if notify_callback: notify_callback()
        
        while proc.poll() is None:
            # Calculate size
            current_bytes = get_directory_size(local_dir)
            current_gb = current_bytes / (1024 * 1024 * 1024)
            progress = min(99, int((current_gb / target_gb) * 100)) if target_gb > 0 else 0
            
            download_states[model_id].update({
                "progress": progress,
                "downloaded_gb": round(current_gb, 2),
                "total_gb": target_gb,
                "message": f"Downloading... {round(current_gb, 1)}GB / {target_gb}GB"
            })
            if notify_callback: notify_callback()
            time.sleep(1.0)
            
        # 3. Check Result
        if proc.returncode == 0:
            download_states[model_id] = {
                "status": "completed",
                "progress": 100,
                "message": "Download Complete",
                "downloaded_gb": target_gb,
                "total_gb": target_gb
            }
        else:
            # Check if it was cancelled intentionally
            if download_states[model_id].get("status") == "cancelled":
                print(f"[DOWNLOAD] Download cancelled by user: {model_id}")
            else:
                stderr = proc.stderr.read().decode()
                print(f"[DOWNLOAD] Process failed with code {proc.returncode}: {stderr}")
                download_states[model_id] = {"status": "error", "error": "Download failed (network or disk)"}
                
    except Exception as e:
        print(f"[DOWNLOAD] Exception in monitor: {e}")
        download_states[model_id] = {"status": "error", "error": str(e)}
        
    finally:
        if notify_callback: notify_callback()
        download_processes.pop(model_id, None)
        download_threads.pop(model_id, None)


def start_model_download(model_id: str, notify_callback=None) -> dict:
    if model_id not in MODEL_REGISTRY: return {"error": "Unknown model"}
    
    with download_start_lock:
        if model_id in download_processes:
            return {"error": "Already downloading"}
            
        thread = threading.Thread(
            target=run_model_download_monitor,
            args=(model_id, notify_callback),
            daemon=True
        )
        download_threads[model_id] = thread
        thread.start()
        
    return {"status": "started", "model_id": model_id}


def cancel_model_download(model_id: str) -> dict:
    """Kill the download process immediately."""
    if model_id in download_states:
        download_states[model_id] = {"status": "cancelled", "progress": 0, "message": "Cancelled"}
    
    # KILL THE PROCESS
    if model_id in download_processes:
        proc = download_processes[model_id]
        print(f"[DOWNLOAD] Killing download process for {model_id}...")
        try:
            proc.kill() # Force kill
            proc.wait(timeout=2) # Wait for it to die
        except:
            pass
        download_processes.pop(model_id, None)
        
    return {"status": "cancelled", "model_id": model_id}


def delete_model(model_id: str) -> dict:
    if model_id not in MODEL_REGISTRY: return {"error": "Unknown model"}
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
    download_processes.clear()

def get_model_path(model_name):
    if model_name not in MODEL_REGISTRY:
        return str(BASE_DIR / "songgeneration_base")
    return str(BASE_DIR / MODEL_REGISTRY[model_name]["path"])

def get_model_list(): return list(MODEL_REGISTRY.keys())
def get_model_info(model_name): return MODEL_REGISTRY.get(model_name, {})

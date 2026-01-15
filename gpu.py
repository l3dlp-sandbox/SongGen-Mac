"""
SongGeneration Studio - GPU Detection
GPU and VRAM detection utilities.
Updated for Mac M1 Max (64GB) Support.
"""

import subprocess
import platform
import os
from typing import Optional
from pathlib import Path

# Try to import torch to check for MPS
try:
    import torch
except ImportError:
    torch = None

# ============================================================================
# GPU/VRAM Detection
# ============================================================================

def get_gpu_info() -> dict:
    """Detect GPU (NVIDIA or Apple Silicon) and available VRAM."""
    
    # 1. Check for Apple Silicon (MPS)
    if torch and torch.backends.mps.is_available():
        # You have 64GB Unified Memory.
        # macOS typically allows ~75% of RAM for Metal (GPU) tasks.
        # 64GB * 0.75 = 48GB available for the AI.
        return {
            'available': True,
            'gpu': {
                'name': 'Apple M1 Max (MPS)',
                'total_mb': 65536, # 64GB Total
                'free_mb': 49152,  # 48GB Free (Safe Metal limit)
                'used_mb': 16384,  # Reserved/System
                'total_gb': 64.0,
                'free_gb': 48.0,
            },
            'recommended_mode': 'full',
            'can_run_full': True, # 48GB > 24GB required for large models
            'can_run_low': True,
        }

    # 2. Check for NVIDIA (Original Logic)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpus.append({
                        'name': parts[0],
                        'total_mb': int(parts[1]),
                        'free_mb': int(parts[2]),
                        'used_mb': int(parts[3]),
                        'total_gb': round(int(parts[1]) / 1024, 1),
                        'free_gb': round(int(parts[2]) / 1024, 1),
                    })
            if gpus:
                gpu = gpus[0]  # Primary GPU
                if gpu['free_gb'] >= 24:
                    recommended = 'full'
                else:
                    recommended = 'low'
                return {
                    'available': True,
                    'gpu': gpu,
                    'recommended_mode': recommended,
                    'can_run_full': gpu['free_gb'] >= 24,
                    'can_run_low': gpu['free_gb'] >= 10,
                }
    except Exception as e:
        print(f"[GPU] NVIDIA detection failed (normal for Mac): {e}")

    # 3. Fallback to CPU
    return {
        'available': False,
        'gpu': None,
        'recommended_mode': 'low',
        'can_run_full': False,
        'can_run_low': False,
    }

# ============================================================================
# Audio Duration Helper
# ============================================================================

def get_audio_duration(audio_path: Path) -> Optional[float]:
    """Get audio duration in seconds using ffprobe."""
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        print(f"[AUDIO] Failed to get duration for {audio_path}: {e}")
    return None

# Global GPU info - initialized on import
gpu_info = get_gpu_info()

def log_gpu_info():
    """Log GPU detection results"""
    if gpu_info['available']:
        print(f"[GPU] Detected: {gpu_info['gpu']['name']}")
        print(f"[GPU] VRAM: {gpu_info['gpu']['free_gb']}GB free / {gpu_info['gpu']['total_gb']}GB total")
        print(f"[GPU] Recommended mode: {gpu_info['recommended_mode']}")
    else:
        print("[GPU] No NVIDIA GPU or Apple MPS detected.")

def refresh_gpu_info() -> dict:
    """Refresh GPU info and return updated data"""
    global gpu_info
    gpu_info = get_gpu_info()
    return gpu_info

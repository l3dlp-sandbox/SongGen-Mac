"""
SongGeneration Studio - Model Registry & Download Manager
Model definitions, status checking, and download management.
Updated for Mac Stability (Subprocess API Download).
"""

import os
import sys
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import uuid
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Response, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse # Required for progress bars
import uvicorn

from config import (BASE_DIR, DEFAULT_MODEL, OUTPUT_DIR, UPLOADS_DIR, STATIC_DIR, load_queue, save_queue, log_startup_info)
from gpu import gpu_info, refresh_gpu_info, log_gpu_info
from schemas import Section, SongRequest, UpdateGenerationRequest
from timing import get_timing_stats
from models import (MODEL_REGISTRY, get_model_status, get_model_status_quick, get_download_progress, get_recommended_model, get_best_ready_model, get_available_models_sync, start_model_download, cancel_model_download, delete_model, cleanup_download_states, is_model_ready_quick)
from model_server import (is_model_server_running_async, start_model_server, stop_model_server, get_model_server_status_async, load_model_on_server_async, unload_model_on_server)
from sse import (notify_queue_update, notify_generation_update as sse_notify_gen, notify_library_update as sse_notify_lib, notify_models_update, notify_models_update_sync, event_generator)
from generation import (generations, generation_lock, is_generation_active, get_active_generation_id, restore_library, run_generation)

log_gpu_info(); log_startup_info(); cleanup_download_states(); restore_library()

def notify_gen(gen_id, gen_data): sse_notify_gen(gen_id, gen_data)
def notify_lib(gens=None): sse_notify_lib(gens or generations)
async def notify_models(): await notify_models_update(get_all_models)

async def get_all_models():
    server_status = await get_model_server_status_async()
    models = []
    for model_id, info in MODEL_REGISTRY.items():
        status = get_model_status_quick(model_id)
        warmth = "cold"
        if status == "ready":
            if server_status.get("loading"): warmth = "loading"
            elif server_status.get("loaded") and server_status.get("model_id") == model_id:
                is_generating = any(gen.get("status") in ("processing", "pending") and gen.get("model") == model_id for gen in generations.values())
                warmth = "generating" if is_generating else "loaded"
            else: warmth = "not_loaded"
        models.append({"id": model_id, "name": info["name"], "status": status, "warmth": warmth, "size_gb": info["size_gb"]})
    return models

async def process_queue_item():
    with generation_lock:
        if is_generation_active(): return
        queue = load_queue()
        if not queue: return
        item = queue.pop(0)
        save_queue(queue)
        notify_queue_update()

        gen_id = item.get("id") or str(uuid.uuid4())[:8]
        sections = [Section(type=s.get('type', 'verse'), lyrics=s.get('lyrics')) for s in item.get('sections', [])]
        
        try:
            request = SongRequest(
                title=item.get('title', 'Untitled'),
                sections=sections,
                gender=item.get('gender', 'female'),
                genre=item.get('genre', ''),
                emotion=item.get('emotion', ''),
                timbre=item.get('timbre', ''),
                instruments=item.get('instruments', ''),
                custom_style=item.get('custom_style'),
                bpm=item.get('bpm', 120),
                model=item.get('model', DEFAULT_MODEL),
                output_mode=item.get('output_mode', 'mixed'),
                reference_audio_id=item.get('reference_audio_id'),
                cfg_coef=item.get('cfg_coef', 1.5),
                temperature=item.get('temperature', 0.8),
                top_k=item.get('top_k', 50),
                top_p=item.get('top_p', 0.0),
                extend_stride=item.get('extend_stride', 5),
                duration=item.get('duration', 240)
            )
        except: return

        reference_path = None
        if request.reference_audio_id:
            ref_files = list(UPLOADS_DIR.glob(f"{request.reference_audio_id}_*"))
            if ref_files: reference_path = str(ref_files[0])

        generations[gen_id] = {
            "id": gen_id, "title": request.title, "model": request.model,
            "status": "pending", "progress": 0, "message": "Starting...",
            "created_at": datetime.now().isoformat(),
        }

    try: await run_generation(gen_id, request, reference_path, notify_gen, notify_lib, notify_models)
    except Exception as e:
        generations[gen_id]["status"] = "failed"; generations[gen_id]["message"] = str(e)

async def background_queue_processor():
    while True:
        try:
            if not is_generation_active():
                if load_queue(): await process_queue_item()
            await asyncio.sleep(2.0)
        except: await asyncio.sleep(5.0)

@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(background_queue_processor())
    yield
    task.cancel()

app = FastAPI(title="SongGeneration", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root(): return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

@app.get("/api/health")
async def health_check(): return {"status": "ok"}

@app.get("/api/models")
async def list_models():
    all_models = await get_all_models()
    ready_models = [m for m in all_models if m["status"] == "ready"]
    return {"models": all_models, "ready_models": ready_models, "default": DEFAULT_MODEL, "has_ready_model": len(ready_models) > 0}

# --- FIX: ADDED MISSING MODEL MANAGEMENT ROUTES ---
@app.post("/api/models/{model_id}/download")
async def download_model_route(model_id: str):
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(404, "Model not found")
    start_model_download(model_id)
    return {"status": "started", "model_id": model_id}

@app.post("/api/models/{model_id}/cancel")
async def cancel_download_route(model_id: str):
    cancel_model_download(model_id)
    return {"status": "cancelled", "model_id": model_id}

@app.delete("/api/models/{model_id}")
async def delete_model_route(model_id: str):
    delete_model(model_id)
    return {"status": "deleted", "model_id": model_id}
# --------------------------------------------------

@app.post("/api/generate")
async def generate_song(request: SongRequest, background_tasks: BackgroundTasks):
    with generation_lock:
        if get_active_generation_id(): raise HTTPException(409, "Busy")
        gen_id = str(uuid.uuid4())[:8]
        generations[gen_id] = {
            "id": gen_id, "title": request.title, "model": request.model,
            "status": "pending", "progress": 0, "message": "Queued...",
            "created_at": datetime.now().isoformat(),
        }
    
    reference_path = None
    if request.reference_audio_id:
        ref_files = list(UPLOADS_DIR.glob(f"{request.reference_audio_id}_*"))
        if ref_files: reference_path = str(ref_files[0])

    background_tasks.add_task(run_generation, gen_id, request, reference_path, notify_gen, notify_lib, notify_models)
    return {"generation_id": gen_id}

@app.get("/api/generations")
async def list_generations(): return list(generations.values())

@app.get("/api/audio/{gen_id}/{track_idx}")
async def get_audio_track(gen_id: str, track_idx: int):
    if gen_id not in generations: raise HTTPException(404)
    return FileResponse(generations[gen_id]["output_files"][track_idx])

@app.get("/api/queue")
async def get_queue(): return load_queue()

@app.post("/api/queue")
async def add_to_queue(payload: dict):
    queue = load_queue()
    queue.append({"id": str(uuid.uuid4())[:8], **payload})
    save_queue(queue)
    notify_queue_update()
    return {"status": "added"}

@app.delete("/api/queue/{item_id}")
async def remove_from_queue(item_id: str):
    queue = [i for i in load_queue() if i.get("id") != item_id]
    save_queue(queue); notify_queue_update()
    return {"status": "removed"}

# --- FIX: RESTORED REAL-TIME EVENTS (Progress Bar) ---
@app.get("/api/events")
async def sse_endpoint(request: Request):
    return EventSourceResponse(event_generator(request))
# -----------------------------------------------------

if STATIC_DIR.exists(): app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    

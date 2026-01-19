"""
SongGeneration Studio - Generation Logic
"""
import os
import re
import json
import asyncio
import threading
from pathlib import Path
from datetime import datetime

from config import BASE_DIR, DEFAULT_MODEL, OUTPUT_DIR, UPLOADS_DIR, USE_MODEL_SERVER

generations = {}
generation_lock = threading.Lock()
model_server_busy = False

LYRICS_FILTER_REGEX = re.compile(r"[^\w\s\[\]\-\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u00c0-\u017f]")
VOCAL_SECTION_TYPES = {"verse", "chorus", "bridge", "prechorus"}

def clean_lyrics_line(line): return LYRICS_FILTER_REGEX.sub("", line).strip()
def is_generation_active(): return model_server_busy or any(g.get("status") in ("pending", "processing") for g in generations.values())
def get_active_generation_id():
    for gen_id, gen in generations.items():
        if gen.get("status") in ("pending", "processing"): return gen_id
    return None

def build_lyrics_string(sections):
    parts = []
    for section in sections:
        tag = f"[{section.type}]"
        if section.lyrics and section.type.split('-')[0].lower() in VOCAL_SECTION_TYPES:
            lines = [clean_lyrics_line(l) for l in section.lyrics.strip().split('\n') if clean_lyrics_line(l)]
            parts.append(f"{tag} {'.'.join(lines)}" if lines else tag)
        else: parts.append(tag)
    return " ; ".join(parts)

def build_description(request):
    parts = []
    if request.gender: parts.append(request.gender)
    if request.genre: parts.append(request.genre)
    if request.emotion: parts.append(request.emotion)
    if request.bpm: parts.append(f"the bpm is {request.bpm}")
    return ", ".join(parts) + "." if parts else ""

def restore_library():
    if not OUTPUT_DIR.exists(): return
    for subdir in OUTPUT_DIR.iterdir():
        if not subdir.is_dir(): continue
        meta_path = subdir / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f: meta = json.load(f)
                files = list((subdir/"audios").glob("*.flac"))
                generations[subdir.name] = {
                    "id": subdir.name, "status": "completed", "progress": 100,
                    "title": meta.get("title", "Untitled"), "model": meta.get("model", "unknown"),
                    "output_files": [str(f) for f in files], "metadata": meta
                }
            except: pass

async def run_generation(gen_id, request, reference_path, notify_gen, notify_lib, notify_models):
    global generations, model_server_busy
    from model_server import is_model_server_running_async, start_model_server, get_model_server_status_async, load_model_on_server_async, generate_via_server_async

    try:
        generations[gen_id].update({"status": "processing", "started_at": datetime.now().isoformat(), "message": "Initializing...", "progress": 0})
        await notify_models()

        model_id = request.model or DEFAULT_MODEL
        input_file = UPLOADS_DIR / f"{gen_id}_input.jsonl"
        output_subdir = OUTPUT_DIR / gen_id
        output_subdir.mkdir(exist_ok=True)

        input_data = {"idx": gen_id, "gt_lyric": build_lyrics_string(request.sections)}
        if reference_path: input_data["prompt_audio_path"] = reference_path
        else:
            input_data["auto_prompt_audio_type"] = "Auto"
            input_data["descriptions"] = build_description(request)

        input_data.update({
            "cfg_coef": request.cfg_coef, "temperature": request.temperature,
            "top_k": request.top_k, "top_p": request.top_p, "extend_stride": request.extend_stride,
            
            # --- FIX: Write duration to file ---
            "duration": request.duration or 240
        })

        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(input_data, f, ensure_ascii=False); f.write('\n')

        generations[gen_id]["message"] = "Loading Model..."
        notify_gen(gen_id, generations[gen_id])

        if USE_MODEL_SERVER:
            if not await is_model_server_running_async(): await start_model_server()
            status = await get_model_server_status_async()
            if not status.get("loaded") or status.get("model_id") != model_id:
                await load_model_on_server_async(model_id)
                for i in range(600):
                    await asyncio.sleep(1)
                    s = await get_model_server_status_async()
                    if s.get("loaded"): break

            generations[gen_id].update({"message": "Generating...", "progress": 35})
            notify_gen(gen_id, generations[gen_id])
            
            model_server_busy = True
            try: result = await generate_via_server_async(str(input_file), str(output_subdir), request.output_mode)
            finally: model_server_busy = False

            if "error" in result: raise Exception(result['error'])

            output_files = list((output_subdir / "audios").glob("*.flac"))
            generations[gen_id].update({
                "status": "completed", "progress": 100, "message": "Done",
                "output_files": [str(f) for f in output_files]
            })
            
            notify_gen(gen_id, generations[gen_id])
            notify_lib(generations)

    except Exception as e:
        generations[gen_id].update({"status": "failed", "message": str(e)})
        notify_gen(gen_id, generations[gen_id])

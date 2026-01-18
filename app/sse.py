import asyncio
import json
from typing import Dict, Any, List
from fastapi import Request

# Global list of queues (one per connected browser tab)
_event_queues: List[asyncio.Queue] = []

async def event_generator(request: Request):
    """
    Generator function that maintains the connection with the browser
    and sends updates (progress bars, status changes) in real-time.
    """
    queue = asyncio.Queue()
    _event_queues.append(queue)
    
    try:
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break
                
            try:
                # Wait for data (timeout allows sending keep-alive pings)
                data = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield data
            except asyncio.TimeoutError:
                # Send empty comment to keep connection alive
                yield {"comment": "ping"}
                
    except Exception as e:
        print(f"[SSE] Error in event stream: {e}", flush=True)
    finally:
        if queue in _event_queues:
            _event_queues.remove(queue)

def _broadcast(event_name: str, data: Any):
    """Internal helper to push data to all active clients."""
    message = {
        "event": event_name,
        "data": json.dumps(data)
    }
    # Send to all connected queues
    for q in _event_queues:
        try:
            q.put_nowait(message)
        except:
            pass # Queue full or closed

# --- Public API Functions (Imported by main.py) ---

def notify_queue_update():
    """Tell UI to refresh the queue list."""
    _broadcast("queue_update", {})

def notify_generation_update(gen_id: str, data: Dict[str, Any]):
    """Send progress bar updates for a specific generation."""
    payload = data.copy()
    payload["id"] = gen_id
    _broadcast("generation_update", payload)

def notify_library_update(generations: Dict[str, Any]):
    """Send the full updated list of generations."""
    _broadcast("library_update", list(generations.values()))

async def notify_models_update(get_models_func):
    """
    Notify UI that model statuses changed (e.g., downloading finished).
    Takes an async function to fetch fresh data.
    """
    data = await get_models_func()
    _broadcast("models_update", data)

def notify_models_update_sync(data):
    """Sync version of model update notification."""
    _broadcast("models_update", data)

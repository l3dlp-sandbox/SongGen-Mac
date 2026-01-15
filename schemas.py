from typing import Optional, List
from pydantic import BaseModel

class Section(BaseModel):
    type: str
    lyrics: Optional[str] = None

class SongRequest(BaseModel):
    title: str = "Untitled"
    sections: List[Section]
    gender: str = "female"
    timbre: str = ""
    genre: str = ""
    emotion: str = ""
    instruments: str = ""
    custom_style: Optional[str] = None
    bpm: int = 120
    output_mode: str = "mixed"
    auto_prompt_type: Optional[str] = None
    reference_audio_id: Optional[str] = None
    model: str = "songgeneration_base"
    memory_mode: str = "auto"
    cfg_coef: float = 1.5
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.0
    extend_stride: int = 5
    
    # --- ADDED ---
    duration: Optional[int] = None

class UpdateGenerationRequest(BaseModel):
    title: Optional[str] = None

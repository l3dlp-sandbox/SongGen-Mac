import os
import gc
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from omegaconf import OmegaConf

from codeclm.trainer.codec_song_pl import CodecLM_PL
from codeclm.models import CodecLM
from separator import Separator

# ============================================================================
# 🧠 MANUAL ATTENTION PATCH (The "Safe Mode")
# Uses standard math instead of fused kernels to prevent driver memory leaks
# ============================================================================
def apply_manual_attention_patch():
    try:
        from transformers.models.llama import modeling_llama
        
        print("[PATCH] Applying Manual Attention (Standard Math) for Mac Stability...", flush=True)

        def manual_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
            bsz, q_len, _ = hidden_states.size()

            # 1. Projection
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # 2. RoPE
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            # 3. Update Cache
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # 4. GQA Handling
            key_states = modeling_llama.repeat_kv(key_states, self.num_key_value_groups)
            value_states = modeling_llama.repeat_kv(value_states, self.num_key_value_groups)

            # 5. MANUAL ATTENTION MATH (Standard Q @ K / V)
            # Standard implementation O(N^2) but stable on MPS drivers
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            # Upcast to fp32 for softmax stability
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                attn_output = attn_output.transpose(1, 2).contiguous()
            
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value

        # Apply to both standard and Flash classes
        modeling_llama.LlamaAttention.forward = manual_forward
        if hasattr(modeling_llama, "LlamaFlashAttention2"):
            modeling_llama.LlamaFlashAttention2.forward = manual_forward
        if hasattr(modeling_llama, "LlamaSdpaAttention"):
            modeling_llama.LlamaSdpaAttention.forward = manual_forward
            
        print("[PATCH] Manual Attention Patch Applied Successfully!", flush=True)

    except ImportError:
        print("[PATCH] Failed to patch LlamaAttention", flush=True)
    except Exception as e:
        print(f"[PATCH] Error applying Manual patch: {e}", flush=True)

apply_manual_attention_patch()
# ============================================================================


class LeVoInference(torch.nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()

        # Optimizations
        torch.backends.cudnn.enabled = False 
        
        # --- FIX: Safe Resolver Registration (Prevents crashes on model switch) ---
        try:
            OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        except ValueError:
            pass # Already registered

        try:
            OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
        except ValueError:
            pass

        try:
            OmegaConf.register_new_resolver("get_fname", lambda: 'default')
        except ValueError:
            pass

        try:
            OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))
        except ValueError:
            pass
        # ------------------------------------------------------------------------

        cfg_path = os.path.join(ckpt_path, 'config.yaml')
        pt_path = os.path.join(ckpt_path, 'model.pt')

        self.cfg = OmegaConf.load(cfg_path)
        self.cfg.mode = 'inference'
        self.max_duration = self.cfg.max_dur

        # Load Full Model Once (Faster for high RAM devices)
        print("[INFERENCE] Loading Full Model into Memory...", flush=True)
        model_light = CodecLM_PL(self.cfg, pt_path)

        # Move to MPS immediately in FP16
        model_light = model_light.eval().cuda().to(torch.float16)
        model_light.audiolm.cfg = self.cfg

        self.model_lm = model_light.audiolm
        self.model_audio_tokenizer = model_light.audio_tokenizer
        self.model_seperate_tokenizer = model_light.seperate_tokenizer

        self.model = CodecLM(name = "tmp",
            lm = self.model_lm,
            audiotokenizer = self.model_audio_tokenizer,
            max_duration = self.max_duration,
            seperate_tokenizer = self.model_seperate_tokenizer,
        )
        self.separator = Separator()

        self.default_params = dict(
            cfg_coef = 1.5,
            temperature = 1.0,
            top_k = 50,
            top_p = 0.0,
            record_tokens = True,
            record_window = 50,
            extend_stride = 5,
            duration = self.max_duration,
        )

        self.model.set_generation_params(**self.default_params)

    def forward(self, lyric: str, description: str = None, prompt_audio_path: os.PathLike = None, genre: str = None, auto_prompt_path: os.PathLike = None, gen_type: str = "mixed", params = dict()):
        params = {**self.default_params, **params}
        self.model.set_generation_params(**params)

        if prompt_audio_path is not None and os.path.exists(prompt_audio_path):
            pmt_wav, vocal_wav, bgm_wav = self.separator.run(prompt_audio_path)
            melody_is_wav = True
        elif genre is not None and auto_prompt_path is not None:
            auto_prompt = torch.load(auto_prompt_path)
            prompt_token = auto_prompt[genre][np.random.randint(0, len(auto_prompt[genre]))]
            pmt_wav = prompt_token[:,[0],:]
            vocal_wav = prompt_token[:,[1],:]
            bgm_wav = prompt_token[:,[2],:]
            melody_is_wav = False
        else:
            pmt_wav = None
            vocal_wav = None
            bgm_wav = None
            melody_is_wav = True

        generate_inp = {
            'lyrics': [lyric.replace("  ", " ")],
            'descriptions': [description],
            'melody_wavs': pmt_wav,
            'vocal_wavs': vocal_wav,
            'bgm_wavs': bgm_wav,
            'melody_is_wav': melody_is_wav,
        }

        # Removed 'streamer' arg to prevent crash
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            tokens = self.model.generate(**generate_inp, return_tokens=True)
            
        with torch.no_grad():
            # Clean up before decoding
            gc.collect()
            torch.cuda.empty_cache()
            
            if melody_is_wav:
                wav_seperate = self.model.generate_audio(tokens, pmt_wav, vocal_wav, bgm_wav, gen_type=gen_type)
            else:
                wav_seperate = self.model.generate_audio(tokens, gen_type=gen_type)

        return wav_seperate[0]

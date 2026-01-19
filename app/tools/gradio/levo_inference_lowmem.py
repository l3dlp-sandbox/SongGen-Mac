import os
import gc
import sys
import torch
import torch.nn.functional as F
import json
import numpy as np
from omegaconf import OmegaConf

from codeclm.trainer.codec_song_pl import CodecLM_PL
from codeclm.models import CodecLM
from codeclm.models import builders
from separator import Separator

# ============================================================================
# 🧠 SDPA MONKEY PATCH (The Mac Optimizer)
# Forces PyTorch to use Apple's native, memory-efficient Attention.
# ============================================================================
def apply_sdpa_patch():
    try:
        from transformers.models.llama import modeling_llama
        
        print("[PATCH] Applying SDPA (Memory Efficient Attention) for Mac...", flush=True)

        def sdpa_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
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

            # 3. Cache
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            # 4. GQA
            key_states = modeling_llama.repeat_kv(key_states, self.num_key_value_groups)
            value_states = modeling_llama.repeat_kv(value_states, self.num_key_value_groups)

            # 5. SDPA (The Fix)
            # If no mask is provided, we assume causal (autoregressive) generation
            is_causal = True if attention_mask is None else False

            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=is_causal
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value

        # Apply to both standard and "Flash" classes just in case config requests Flash
        modeling_llama.LlamaAttention.forward = sdpa_forward
        if hasattr(modeling_llama, "LlamaFlashAttention2"):
            modeling_llama.LlamaFlashAttention2.forward = sdpa_forward
            
        print("[PATCH] SDPA Patch Applied Successfully!", flush=True)

    except ImportError:
        print("[PATCH] Failed to patch LlamaAttention (Lib mismatch?)", flush=True)
    except Exception as e:
        print(f"[PATCH] Error applying SDPA patch: {e}", flush=True)

apply_sdpa_patch()
# ============================================================================


class LeVoInference(torch.nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()

        # Disable CUDNN/Benchmarks for Mac
        torch.backends.cudnn.enabled = False 
        
        OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx])
        OmegaConf.register_new_resolver("get_fname", lambda: 'default')
        OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)))

        cfg_path = os.path.join(ckpt_path, 'config.yaml')
        self.pt_path = os.path.join(ckpt_path, 'model.pt')

        self.cfg = OmegaConf.load(cfg_path)
        self.cfg.mode = 'inference'
        self.max_duration = self.cfg.max_dur

        self.default_params = dict(
            top_p = 0.0,
            record_tokens = True,
            record_window = 50,
            extend_stride = 5,
            duration = self.max_duration,
        )

    def forward(self, lyric: str, description: str = None, prompt_audio_path: os.PathLike = None, genre: str = None, auto_prompt_path: os.PathLike = None, gen_type: str = "mixed", params = dict()):
        
        # --- PREPARATION & AUDIO PROMPT ---
        if prompt_audio_path is not None and os.path.exists(prompt_audio_path):
            separator = Separator()
            audio_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint, self.cfg)
            # Use CUDA shim (which maps to MPS)
            audio_tokenizer = audio_tokenizer.eval().cuda()
            
            pmt_wav, vocal_wav, bgm_wav = separator.run(prompt_audio_path)
            pmt_wav = pmt_wav.cuda()
            vocal_wav = vocal_wav.cuda()
            bgm_wav = bgm_wav.cuda()
            
            with torch.no_grad():
                pmt_wav, _ = audio_tokenizer.encode(pmt_wav)
            
            del audio_tokenizer
            del separator
            gc.collect()
            torch.cuda.empty_cache()

            seperate_tokenizer = builders.get_audio_tokenizer_model(self.cfg.audio_tokenizer_checkpoint_sep, self.cfg)
            seperate_tokenizer = seperate_tokenizer.eval().cuda()
            with torch.no_grad():
                vocal_wav, bgm_wav = seperate_tokenizer.encode(vocal_wav, bgm_wav)
            
            del seperate_tokenizer
            melody_is_wav = False
            torch.cuda.empty_cache()
            
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

        # --- MODEL LOADING (Simplified - NO OFFLOAD PROFILER) ---
        print("[INFERENCE] Loading Main Model directly to GPU...", flush=True)
        audiolm = builders.get_lm_model(self.cfg)
        checkpoint = torch.load(self.pt_path, map_location='cpu')
        audiolm_state_dict = {k.replace('audiolm.', ''): v for k, v in checkpoint.items() if k.startswith('audiolm')}
        audiolm.load_state_dict(audiolm_state_dict, strict=False)
        audiolm = audiolm.eval()

        # Direct load to MPS (12GB model fits easily in 48GB limit)
        # We convert to Float16 immediately to save RAM
        audiolm = audiolm.cuda().to(torch.float16)

        model = CodecLM(name = "tmp",
            lm = audiolm,
            audiotokenizer = None,
            max_duration = self.max_duration,
            seperate_tokenizer = None,
        )
        params = {**self.default_params, **params}
        model.set_generation_params(**params)

        generate_inp = {
            'lyrics': [lyric.replace("  ", " ")],
            'descriptions': [description],
            'melody_wavs': pmt_wav,
            'vocal_wavs': vocal_wav,
            'bgm_wavs': bgm_wav,
            'melody_is_wav': melody_is_wav,
        }

        # --- GENERATION ---
        # Autocast might be ignored on MPS, but we already converted model to float16
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                tokens = model.generate(**generate_inp, return_tokens=True)
        
        # --- CLEANUP ---
        del model
        # Move back to CPU before deleting to help allocator? 
        # Usually just deleting and collecting is enough on Mac.
        del audiolm
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()

        # --- DECODING ---
        print("[INFERENCE] Decoding audio...", flush=True)
        seperate_tokenizer = builders.get_audio_tokenizer_model_cpu(self.cfg.audio_tokenizer_checkpoint_sep, self.cfg)
        
        # Manual device placement for decoder
        device = "cuda:0" # Maps to MPS via shim
        seperate_tokenizer.model.device = device
        seperate_tokenizer.model.vae = seperate_tokenizer.model.vae.to(device)
        seperate_tokenizer.model.model.device = torch.device(device)
        seperate_tokenizer = seperate_tokenizer.eval()
        
        # No offload profiler for decoder either
        seperate_tokenizer.model.model = seperate_tokenizer.model.model.to(device)

        model = CodecLM(name = "tmp",
            lm = None,
            audiotokenizer = None,
            max_duration = self.max_duration,
            seperate_tokenizer = seperate_tokenizer,
        )

        with torch.no_grad():
            if melody_is_wav:
                wav_seperate = model.generate_audio(tokens, pmt_wav, vocal_wav, bgm_wav, gen_type=gen_type, chunked=True)
            else:
                wav_seperate = model.generate_audio(tokens, gen_type=gen_type, chunked=True)

        gc.collect()
        torch.cuda.empty_cache()

        return wav_seperate[0]

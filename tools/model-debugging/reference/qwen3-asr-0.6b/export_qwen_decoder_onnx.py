"""Export official Qwen3-ASR thinker decoder as explicit KV ONNX graphs.

Does not use HuggingFace DynamicCache or create_causal_mask in-graph.
Does not use third-party ONNX as the oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


AUDIO_TOKEN_ID = 151676
EOS_TOKEN_IDS = (151645, 151643)
ORACLE_JFK = (
    "And so, my fellow Americans, ask not what your country can do for you; "
    "ask what you can do for your country."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path(r"N:\models\Qwen3-ASR-0.6B"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"N:\models\onnx\qwen3-asr-0.6b-official"))
    parser.add_argument("--audio", type=Path, default=Path(r"N:\github\asrjs\speech-recognition\tools\data\fixtures\audio\jfk-short.wav"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    return parser.parse_args()


def _disable_tls_verify() -> None:
    ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001


def cleanup_stray_external(output_dir: Path) -> list[str]:
    removed: list[str] = []
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith("onnx__") or path.name.endswith(".weight"):
            path.unlink()
            removed.append(path.name)
    return removed


def pack_onnx(src: Path, dest: Path, data_name: str | None = None) -> dict:
    import onnx

    model = onnx.load(str(src), load_external_data=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data_name = data_name or (dest.name + ".data")
    data_path = dest.parent / data_name
    if dest.exists():
        dest.unlink()
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(dest),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )
    if not data_path.is_file():
        raise FileNotFoundError(f"Expected external data file {data_path}")
    return {
        "onnx_path": str(dest),
        "size_bytes": dest.stat().st_size,
        "sha256": sha256_file(dest),
        "external_data": str(data_path),
        "external_size_bytes": data_path.stat().st_size,
        "external_sha256": sha256_file(data_path),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def try_export(name: str, fn) -> dict:
    try:
        details = fn()
        return {"name": name, "ok": True, **(details or {})}
    except Exception as error:  # noqa: BLE001
        return {
            "name": name,
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error)[:4000],
            "traceback": traceback.format_exc()[-4000:],
        }


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value[..., : value.shape[-1] // 2], value[..., value.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def apply_rotary(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (query * cos) + (rotate_half(query) * sin), (key * cos) + (rotate_half(key) * sin)


def repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, heads, seq, dim = hidden.shape
    if n_rep == 1:
        return hidden
    return hidden[:, :, None, :, :].expand(batch, heads, n_rep, seq, dim).reshape(batch, heads * n_rep, seq, dim)


def causal_mask(query_len: int, key_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    min_value = torch.finfo(dtype).min
    mask = torch.full((1, 1, query_len, key_len), min_value, dtype=dtype, device=device)
    if query_len == key_len:
        return torch.triu(mask, diagonal=1)
    # Step: attend to all past + current.
    return torch.zeros((1, 1, query_len, key_len), dtype=dtype, device=device)


class ExplicitKvThinker(nn.Module):
    """Official thinker weights with concatenated KV tensors, no DynamicCache."""

    def __init__(self, thinker: nn.Module) -> None:
        super().__init__()
        self.embed_tokens = thinker.model.embed_tokens
        self.layers = thinker.model.layers
        self.norm = thinker.model.norm
        self.lm_head = thinker.lm_head
        self.audio_token_id = int(thinker.config.audio_token_id)
        rotary = thinker.model.rotary_emb
        self.register_buffer("inv_freq", rotary.inv_freq.detach().clone())
        self.num_layers = len(self.layers)
        attn = self.layers[0].self_attn
        self.num_heads = attn.q_proj.out_features // attn.head_dim
        self.num_kv_heads = attn.k_proj.out_features // attn.head_dim
        self.head_dim = attn.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = attn.scaling

    def rope(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        # ASR text positions are identical on the three MRoPE axes, so 1D RoPE matches.
        flat = position_ids.reshape(-1).to(dtype=torch.float32)
        freqs = torch.outer(flat, self.inv_freq.float())
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype).unsqueeze(0)
        sin = emb.sin().to(dtype=dtype).unsqueeze(0)
        return cos, sin

    def merge_embeds(self, input_ids: torch.Tensor, audio_embeddings: torch.Tensor) -> torch.Tensor:
        text = self.embed_tokens(input_ids)
        mask = (input_ids == self.audio_token_id).unsqueeze(-1).to(dtype=text.dtype)
        # audio_embeddings is already aligned to sequence length (zeros off audio slots).
        return text * (1.0 - mask) + audio_embeddings * mask

    def layer_forward(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
        past_key: torch.Tensor | None,
        past_value: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden
        hidden = layer.input_layernorm(hidden)
        attn = layer.self_attn
        batch, seq, _ = hidden.shape
        query = attn.q_norm(attn.q_proj(hidden).view(batch, seq, -1, attn.head_dim)).transpose(1, 2)
        key = attn.k_norm(attn.k_proj(hidden).view(batch, seq, -1, attn.head_dim)).transpose(1, 2)
        value = attn.v_proj(hidden).view(batch, seq, -1, attn.head_dim).transpose(1, 2)
        query, key = apply_rotary(query, key, cos, sin)
        if past_key is not None:
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
        key_rep = repeat_kv(key, self.num_kv_groups)
        value_rep = repeat_kv(value, self.num_kv_groups)
        scores = torch.matmul(query, key_rep.transpose(2, 3)) * self.scaling
        scores = scores + attn_mask
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_out = torch.matmul(probs, value_rep).transpose(1, 2).contiguous()
        hidden = residual + attn.o_proj(attn_out.reshape(batch, seq, -1))
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        return hidden, key, value

    def decode(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        past_keys: torch.Tensor | None = None,
        past_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_len = hidden.shape[1]
        past_len = 0 if past_keys is None else past_keys.shape[3]
        attn_mask = causal_mask(query_len, past_len + query_len, hidden.dtype, hidden.device)
        cos, sin = self.rope(position_ids, hidden.dtype)
        present_keys = []
        present_values = []
        for index, layer in enumerate(self.layers):
            past_k = None if past_keys is None else past_keys[index]
            past_v = None if past_values is None else past_values[index]
            hidden, key, value = self.layer_forward(layer, hidden, cos, sin, attn_mask, past_k, past_v)
            present_keys.append(key)
            present_values.append(value)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden[:, -1:, :])
        return logits, torch.stack(present_keys, dim=0), torch.stack(present_values, dim=0)

    def prefill(self, input_ids: torch.Tensor, audio_embeddings: torch.Tensor, position_ids: torch.Tensor):
        hidden = self.merge_embeds(input_ids, audio_embeddings)
        return self.decode(hidden, position_ids)

    def step(self, input_ids: torch.Tensor, position_ids: torch.Tensor, past_keys: torch.Tensor, past_values: torch.Tensor):
        hidden = self.embed_tokens(input_ids)
        return self.decode(hidden, position_ids, past_keys, past_values)


class PrefillOnnx(nn.Module):
    def __init__(self, decoder: ExplicitKvThinker) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, input_ids: torch.Tensor, audio_embeddings: torch.Tensor, position_ids: torch.Tensor):
        return self.decoder.prefill(input_ids, audio_embeddings, position_ids)


class StepOnnx(nn.Module):
    def __init__(self, decoder: ExplicitKvThinker) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
    ):
        return self.decoder.step(input_ids, position_ids, past_keys, past_values)


def tensor_delta(left: torch.Tensor, right: torch.Tensor) -> dict:
    delta = (left.float() - right.float()).abs()
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
    }


def cache_to_stacked(past) -> tuple[torch.Tensor, torch.Tensor]:
    keys = []
    values = []
    for layer_idx in range(len(past)):
        item = past[layer_idx]
        if isinstance(item, (tuple, list)):
            key, value = item
        else:
            key, value = item
        keys.append(key)
        values.append(value)
    return torch.stack(keys, dim=0), torch.stack(values, dim=0)


def greedy_explicit(decoder: ExplicitKvThinker, input_ids, audio_embeddings, position_ids, max_new: int) -> list[int]:
    logits, keys, values = decoder.prefill(input_ids, audio_embeddings, position_ids)
    tokens: list[int] = []
    next_id = int(logits[0, -1].argmax())
    seq = int(input_ids.shape[1])
    for _ in range(max_new):
        if next_id in EOS_TOKEN_IDS:
            break
        tokens.append(next_id)
        step_ids = torch.tensor([[next_id]], dtype=input_ids.dtype, device=input_ids.device)
        step_pos = torch.tensor([[seq]], dtype=position_ids.dtype, device=position_ids.device)
        logits, keys, values = decoder.step(step_ids, step_pos, keys, values)
        next_id = int(logits[0, -1].argmax())
        seq += 1
    return tokens


def main() -> None:
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import normalize_audios, parse_asr_output

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = Qwen3ASRModel.from_pretrained(
        str(args.model_dir.resolve()),
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
    )
    inner = wrapper.model
    thinker = inner.thinker
    thinker.eval()
    processor = wrapper.processor
    decoder = ExplicitKvThinker(thinker).eval()

    prompt = wrapper._build_text_prompt(context="", force_language=None)
    waveform = normalize_audios(str(args.audio.resolve()))[0]
    inputs = processor(text=[prompt], audio=[waveform], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_features = inputs["input_features"]
    feature_attention_mask = inputs["feature_attention_mask"]
    feature_len = int(feature_attention_mask.sum().item())
    audio_token_count = int((input_ids == AUDIO_TOKEN_ID).sum().item())
    seq_len = int(input_ids.shape[1])
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    report: dict = {
        "schema_version": 1,
        "family": "qwen3-asr-0.6b",
        "source": "official qwen-asr 0.0.6 thinker with explicit KV (no DynamicCache in-graph)",
        "prompt_preview": prompt[:200],
        "seq_len": seq_len,
        "feature_len": feature_len,
        "audio_token_count": audio_token_count,
        "input_features_shape": list(input_features.shape),
        "attempts": [],
    }

    def probe_official_prefill() -> dict:
        with torch.no_grad():
            official = thinker(
                input_ids=input_ids,
                input_features=input_features,
                attention_mask=attention_mask,
                feature_attention_mask=feature_attention_mask,
                use_cache=True,
            )
            audio_features = thinker.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
            )
            aligned = torch.zeros(1, seq_len, audio_features.shape[-1], dtype=audio_features.dtype)
            audio_index = (input_ids[0] == AUDIO_TOKEN_ID).nonzero(as_tuple=False).squeeze(-1)
            aligned[0, audio_index] = audio_features[: audio_index.numel()]
            explicit_logits, explicit_keys, explicit_values = decoder.prefill(input_ids, aligned, position_ids)
            official_last = official.logits[:, -1:, :]
            cache_keys, cache_values = cache_to_stacked(official.past_key_values)
            rope_cos, rope_sin = decoder.rope(position_ids, aligned.dtype)
            official_pos = torch.arange(seq_len).view(1, 1, -1).expand(3, 1, -1)
            official_cos, official_sin = thinker.model.rotary_emb(aligned, official_pos)
        return {
            "audio_features_shape": list(audio_features.shape),
            "logits": tensor_delta(official_last, explicit_logits),
            "keys": tensor_delta(cache_keys, explicit_keys),
            "values": tensor_delta(cache_values, explicit_values),
            "rope_cos": tensor_delta(official_cos, rope_cos),
            "rope_sin": tensor_delta(official_sin, rope_sin),
            "official_first_token": int(official_last[0, -1].argmax()),
            "explicit_first_token": int(explicit_logits[0, -1].argmax()),
        }

    report["attempts"].append(try_export("explicit_prefill_vs_official", probe_official_prefill))

    aligned_audio = None
    with torch.no_grad():
        audio_features = thinker.get_audio_features(
            input_features,
            feature_attention_mask=feature_attention_mask,
        )
        aligned_audio = torch.zeros(1, seq_len, audio_features.shape[-1], dtype=audio_features.dtype)
        audio_index = (input_ids[0] == AUDIO_TOKEN_ID).nonzero(as_tuple=False).squeeze(-1)
        aligned_audio[0, audio_index] = audio_features[: audio_index.numel()]

    def probe_greedy() -> dict:
        with torch.no_grad():
            token_ids = greedy_explicit(decoder, input_ids, aligned_audio, position_ids, args.max_new_tokens)
            raw = processor.tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            language, text = parse_asr_output(raw)
        return {
            "token_count": len(token_ids),
            "raw": raw,
            "language": language,
            "text": text,
            "matches_oracle": text == ORACLE_JFK,
        }

    report["attempts"].append(try_export("explicit_greedy_jfk", probe_greedy))

    def probe_pad_crop_greedy() -> dict:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from export_qwen_onnx import StaticWindowAudioEncoder, official_token_count

        remainder_t = 1050
        hop = 160
        short_wave = waveform[: remainder_t * hop]
        short_inputs = processor(text=[prompt], audio=[short_wave], return_tensors="pt", padding=True)
        short_ids = short_inputs["input_ids"]
        short_feat = short_inputs["input_features"]
        short_mask = short_inputs["feature_attention_mask"]
        short_feat_len = int(short_mask.sum().item())
        short_seq = int(short_ids.shape[1])
        short_pos = torch.arange(short_seq, dtype=torch.long).unsqueeze(0)
        tokens = official_token_count(short_feat_len)
        pad_t = ((short_feat_len + 99) // 100) * 100
        feat_2d = short_feat[0] if short_feat.dim() == 3 else short_feat
        if feat_2d.shape[-1] < pad_t:
            feat_2d = F.pad(feat_2d, (0, pad_t - feat_2d.shape[-1]))
        elif feat_2d.shape[-1] > pad_t:
            feat_2d = feat_2d[:, :pad_t]
        static_enc = StaticWindowAudioEncoder(thinker.audio_tower).eval()
        with torch.no_grad():
            official_emb = thinker.get_audio_features(short_feat, feature_attention_mask=short_mask)
            padded_emb = static_enc(feat_2d.float())[: official_emb.shape[0]]
            delta = (official_emb.float() - padded_emb.float()).abs()
            official_aligned = torch.zeros(1, short_seq, official_emb.shape[-1], dtype=official_emb.dtype)
            padded_aligned = torch.zeros_like(official_aligned)
            audio_index = (short_ids[0] == AUDIO_TOKEN_ID).nonzero(as_tuple=False).squeeze(-1)
            n = min(int(audio_index.numel()), int(official_emb.shape[0]), int(padded_emb.shape[0]))
            official_aligned[0, audio_index[:n]] = official_emb[:n]
            padded_aligned[0, audio_index[:n]] = padded_emb[:n]
            official_ids = greedy_explicit(decoder, short_ids, official_aligned, short_pos, args.max_new_tokens)
            padded_ids = greedy_explicit(decoder, short_ids, padded_aligned, short_pos, args.max_new_tokens)
        official_raw = processor.tokenizer.decode(official_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        padded_raw = processor.tokenizer.decode(padded_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        _, official_text = parse_asr_output(official_raw)
        _, padded_text = parse_asr_output(padded_raw)
        return {
            "feature_len": short_feat_len,
            "formula_tokens": tokens,
            "official_tokens": list(official_emb.shape),
            "padded_tokens": list(padded_emb.shape),
            "emb_max_abs": float(delta.max()),
            "emb_mean_abs": float(delta.mean()),
            "official_text": official_text,
            "padded_text": padded_text,
            "text_match": official_text == padded_text,
            "matches_oracle": official_text == ORACLE_JFK and padded_text == ORACLE_JFK,
        }

    report["attempts"].append(try_export("pad_crop_greedy_t1050", probe_pad_crop_greedy))

    fp16 = args.dtype == "float16"
    prefill_path = args.output_dir / ("decoder-prefill-fp16.onnx" if fp16 else "decoder-prefill.onnx")
    step_path = args.output_dir / ("decoder-step-fp16.onnx" if fp16 else "decoder-step.onnx")
    shared_data_name = "decoder-fp16.onnx.data" if fp16 else None
    tokenizer_dir = args.output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    def save_tokenizer() -> dict:
        processor.tokenizer.save_pretrained(str(tokenizer_dir))
        files = sorted(path.name for path in tokenizer_dir.iterdir() if path.is_file())
        return {"dir": str(tokenizer_dir), "files": files}

    report["attempts"].append(try_export("save_tokenizer", save_tokenizer))

    if not args.skip_onnx:
        export_decoder = decoder
        export_audio = aligned_audio
        if fp16:
            export_decoder = ExplicitKvThinker(thinker).to(dtype=torch.float16).eval()
            export_audio = aligned_audio.to(dtype=torch.float16)
        prefill_mod = PrefillOnnx(export_decoder).eval()
        step_mod = StepOnnx(export_decoder).eval()
        dummy_step_ids = torch.tensor([[EOS_TOKEN_IDS[0]]], dtype=torch.long)
        dummy_step_pos = torch.tensor([[seq_len]], dtype=torch.long)
        prefill_work = args.output_dir / ("_export_prefill_fp16" if fp16 else "_export_prefill")
        step_work = args.output_dir / ("_export_step_fp16" if fp16 else "_export_step")
        prefill_work.mkdir(parents=True, exist_ok=True)
        step_work.mkdir(parents=True, exist_ok=True)
        prefill_raw = prefill_work / "model.onnx"
        step_raw = step_work / "model.onnx"
        report["attempts"].append(try_export("cleanup_stray_external", lambda: {"removed": cleanup_stray_external(args.output_dir)}))
        report["export_dtype"] = args.dtype

        def export_prefill() -> dict:
            import shutil

            with torch.no_grad():
                logits, keys, _values = export_decoder.prefill(input_ids, export_audio, position_ids)
                torch.onnx.export(
                    prefill_mod,
                    (input_ids, export_audio, position_ids),
                    str(prefill_raw),
                    input_names=["input_ids", "audio_embeddings", "position_ids"],
                    output_names=["logits", "present_keys", "present_values"],
                    opset_version=17,
                    dynamo=False,
                    dynamic_axes={
                        "input_ids": {1: "seq"},
                        "audio_embeddings": {1: "seq"},
                        "position_ids": {1: "seq"},
                        "present_keys": {3: "seq"},
                        "present_values": {3: "seq"},
                    },
                )
            packed = pack_onnx(prefill_raw, prefill_path, shared_data_name)
            shutil.rmtree(prefill_work, ignore_errors=True)
            return {
                **packed,
                "logits_shape": list(logits.shape),
                "kv_shape": list(keys.shape),
                "logits_dtype": str(logits.dtype),
            }

        def export_step() -> dict:
            import shutil

            with torch.no_grad():
                _, keys, values = export_decoder.prefill(input_ids, export_audio, position_ids)
                torch.onnx.export(
                    step_mod,
                    (dummy_step_ids, dummy_step_pos, keys, values),
                    str(step_raw),
                    input_names=["input_ids", "position_ids", "past_keys", "past_values"],
                    output_names=["logits", "present_keys", "present_values"],
                    opset_version=17,
                    dynamo=False,
                    dynamic_axes={
                        "past_keys": {3: "past_len"},
                        "past_values": {3: "past_len"},
                        "present_keys": {3: "present_len"},
                        "present_values": {3: "present_len"},
                    },
                )
            step_data_name = (step_path.name + ".data") if not shared_data_name else (step_path.name + ".tmp.data")
            packed = pack_onnx(step_raw, step_path if not shared_data_name else (args.output_dir / "_decoder-step-fp16.onnx"), step_data_name if not shared_data_name else "_decoder-step-fp16.onnx.data")
            shutil.rmtree(step_work, ignore_errors=True)
            if shared_data_name:
                import onnx
                from onnx.external_data_helper import uses_external_data

                tmp = args.output_dir / "_decoder-step-fp16.onnx"
                tmp_data = args.output_dir / "_decoder-step-fp16.onnx.data"
                prefill_data = args.output_dir / shared_data_name
                shared = (
                    prefill_data.is_file()
                    and tmp_data.is_file()
                    and sha256_file(prefill_data) == sha256_file(tmp_data)
                )
                model = onnx.load(str(tmp), load_external_data=False)
                if shared:
                    for tensor in list(model.graph.initializer) + list(model.graph.sparse_initializer):
                        if not uses_external_data(tensor):
                            continue
                        for entry in tensor.external_data:
                            if entry.key == "location":
                                entry.value = shared_data_name
                    if step_path.exists():
                        step_path.unlink()
                    onnx.save_model(model, str(step_path), save_as_external_data=False)
                    tmp.unlink(missing_ok=True)
                    tmp_data.unlink(missing_ok=True)
                    packed = {
                        "onnx_path": str(step_path),
                        "size_bytes": step_path.stat().st_size,
                        "sha256": sha256_file(step_path),
                        "external_data": str(prefill_data),
                        "external_size_bytes": prefill_data.stat().st_size,
                        "external_sha256": sha256_file(prefill_data),
                        "shared_weights": True,
                    }
                else:
                    packed["shared_weights"] = False
            return packed

        report["attempts"].append(try_export("decoder_prefill_onnx", export_prefill))
        report["attempts"].append(try_export("decoder_step_onnx", export_step))

        def numpy_feed(session, name: str, array: np.ndarray) -> np.ndarray:
            info = next(item for item in session.get_inputs() if item.name == name)
            if "float16" in info.type:
                return array.astype(np.float16)
            if "int64" in info.type:
                return array.astype(np.int64)
            return array

        def ort_prefill() -> dict:
            import onnxruntime as ort

            session = ort.InferenceSession(str(prefill_path), providers=["CPUExecutionProvider"])
            with torch.no_grad():
                pytorch_logits, pytorch_keys, pytorch_values = export_decoder.prefill(input_ids, export_audio, position_ids)
            feeds = {
                "input_ids": numpy_feed(session, "input_ids", input_ids.numpy()),
                "audio_embeddings": numpy_feed(session, "audio_embeddings", export_audio.float().numpy()),
                "position_ids": numpy_feed(session, "position_ids", position_ids.numpy()),
            }
            logits, keys, values = session.run(None, feeds)
            return {
                "logits": tensor_delta(pytorch_logits.float(), torch.from_numpy(np.asarray(logits, dtype=np.float32))),
                "keys": tensor_delta(pytorch_keys.float(), torch.from_numpy(np.asarray(keys, dtype=np.float32))),
                "values": tensor_delta(pytorch_values.float(), torch.from_numpy(np.asarray(values, dtype=np.float32))),
                "inputs": [f"{item.name}:{item.type}" for item in session.get_inputs()],
                "outputs": [f"{item.name}:{item.type}" for item in session.get_outputs()],
            }

        def ort_greedy() -> dict:
            import onnxruntime as ort

            prefill = ort.InferenceSession(str(prefill_path), providers=["CPUExecutionProvider"])
            step = ort.InferenceSession(str(step_path), providers=["CPUExecutionProvider"])
            logits, keys, values = prefill.run(
                None,
                {
                    "input_ids": numpy_feed(prefill, "input_ids", input_ids.numpy()),
                    "audio_embeddings": numpy_feed(prefill, "audio_embeddings", export_audio.float().numpy()),
                    "position_ids": numpy_feed(prefill, "position_ids", position_ids.numpy()),
                },
            )
            tokens: list[int] = []
            next_id = int(np.argmax(logits[0, -1]))
            seq = seq_len
            for _ in range(args.max_new_tokens):
                if next_id in EOS_TOKEN_IDS:
                    break
                tokens.append(next_id)
                logits, keys, values = step.run(
                    None,
                    {
                        "input_ids": numpy_feed(step, "input_ids", np.array([[next_id]], dtype=np.int64)),
                        "position_ids": numpy_feed(step, "position_ids", np.array([[seq]], dtype=np.int64)),
                        "past_keys": keys,
                        "past_values": values,
                    },
                )
                next_id = int(np.argmax(logits[0, -1]))
                seq += 1
            raw = processor.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            language, text = parse_asr_output(raw)
            return {
                "token_count": len(tokens),
                "raw": raw,
                "language": language,
                "text": text,
                "matches_oracle": text == ORACLE_JFK,
                "dtype": args.dtype,
            }

        prefill_packed = any(item.get("ok") and item.get("name") == "decoder_prefill_onnx" for item in report["attempts"])
        step_packed = any(item.get("ok") and item.get("name") == "decoder_step_onnx" for item in report["attempts"])
        if prefill_packed:
            report["attempts"].append(try_export("decoder_prefill_onnxruntime_cpu", ort_prefill))
        if prefill_packed and step_packed:
            report["attempts"].append(try_export("decoder_onnxruntime_greedy_jfk", ort_greedy))

    prefill_ok = any(item.get("ok") and item.get("name") == "decoder_prefill_onnx" for item in report["attempts"])
    step_ok = any(item.get("ok") and item.get("name") == "decoder_step_onnx" for item in report["attempts"])
    greedy_ok = any(
        item.get("ok") and item.get("name") == "explicit_greedy_jfk" and item.get("matches_oracle")
        for item in report["attempts"]
    )
    ort_text_ok = any(
        item.get("ok") and item.get("name") == "decoder_onnxruntime_greedy_jfk" and item.get("matches_oracle")
        for item in report["attempts"]
    )
    if prefill_ok and step_ok:
        report["failureClass"] = None
        report["status"] = "exported-decoder" if ort_text_ok or greedy_ok else "exported-decoder-parity-pending"
    else:
        blocked = next((item for item in report["attempts"] if item.get("name", "").endswith("_onnx") and not item.get("ok")), None)
        report["failureClass"] = "EXPORT_BLOCKED"
        report["status"] = "experimental-blocked"
        report["missing"] = (blocked or {}).get("error", "decoder ONNX export failed")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "failureClass": report.get("failureClass"), "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    _disable_tls_verify()
    main()

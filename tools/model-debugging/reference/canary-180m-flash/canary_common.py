from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer
from nemo.collections.common.tokenizers.canary_tokenizer import CANARY_SPECIAL_TOKENIZER

try:
    from nemo.collections.common.tokenizers.canary_tokenizer import CanaryBPETokenizer
except ImportError:  # Older NeMo environments still support Canary AggregateTokenizer.
    CanaryBPETokenizer = None  # type: ignore[assignment,misc]

DEFAULT_MODEL_ID = "nvidia/canary-180m-flash"
DEFAULT_AUDIO = (
    Path(__file__).resolve().parents[4] / "tools" / "data" / "fixtures" / "audio" / "jfk-short.wav"
)


@dataclass(frozen=True)
class PromptSettings:
    source_lang: str = "en"
    target_lang: str = "en"
    decoder_context: str = ""
    emotion: str = "<|emo:undefined|>"
    pnc: bool = True
    itn: bool = False
    timestamp: bool = False
    diarize: bool = False
    task: str = "asr"


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cpu",
    checkpoint_path: Path | None = None,
) -> ASRModel:
    local_checkpoint = checkpoint_path.expanduser() if checkpoint_path is not None else None
    if local_checkpoint is None and Path(model_id).suffix.lower() == ".nemo":
        local_checkpoint = Path(model_id).expanduser()
    if local_checkpoint is not None:
        if not local_checkpoint.is_file():
            raise FileNotFoundError(f"Local NeMo checkpoint does not exist: {local_checkpoint}")
        model = ASRModel.restore_from(str(local_checkpoint.resolve()), map_location=device)
    else:
        model = ASRModel.from_pretrained(model_id, map_location=device)
    model = model.eval()
    if hasattr(model, "to"):
        model = model.to(device)
    return model


def normalize_language_token(value: str) -> str:
    value = str(value).strip()
    if value.startswith("<|") and value.endswith("|>"):
        return value
    return f"<|{value}|>"


def normalize_boolean_token(name: str, value: bool) -> str:
    return f"<|{name}|>" if value else f"<|no{name}|>"


def parse_supported_languages(value: str | None) -> list[str] | None:
    if value is None:
        return None
    languages = [language.strip().lower() for language in value.split(",")]
    if not languages or any(not language for language in languages):
        raise ValueError("Supported languages must be a comma-separated list of non-empty codes.")
    if len(set(languages)) != len(languages):
        raise ValueError("Supported languages must not contain duplicates.")
    return languages


def build_prompt_slots(model: ASRModel, settings: PromptSettings) -> dict[str, str]:
    if str(model.prompt_format).lower() == "canary":
        task = settings.task.strip().lower()
        if task in {"asr", "transcribe"}:
            task = "asr"
        elif task in {"ast", "translate", "translation", "speech-translation", "s2t_translation"}:
            task = "translate"
        else:
            raise ValueError(f"Unsupported legacy Canary task: {settings.task!r}")
        return {
            "source_lang": settings.source_lang,
            "task": task,
            "target_lang": settings.target_lang,
            "pnc": "yes" if settings.pnc else "no",
        }

    defaults = dict(model.cfg.prompt_defaults[0].slots)
    defaults.update(
        {
            "decodercontext": settings.decoder_context,
            "emotion": settings.emotion,
            "source_lang": normalize_language_token(settings.source_lang),
            "target_lang": normalize_language_token(settings.target_lang),
            "pnc": normalize_boolean_token("pnc", settings.pnc),
            "itn": normalize_boolean_token("itn", settings.itn),
            "timestamp": normalize_boolean_token("timestamp", settings.timestamp),
            "diarize": normalize_boolean_token("diarize", settings.diarize),
            "prompt_language": CANARY_SPECIAL_TOKENIZER,
        }
    )
    return defaults


def encode_prompt(model: ASRModel, settings: PromptSettings) -> torch.Tensor:
    dialog = [{"role": "user", "slots": build_prompt_slots(model, settings)}]
    encoded = model.prompt.encode_dialog(dialog)
    return encoded["context_ids"]


def load_audio(audio_path: Path, target_sample_rate: int) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1, dtype=np.float32)

    if sample_rate == target_sample_rate:
        return mono.astype(np.float32, copy=False), sample_rate

    gcd = math.gcd(sample_rate, target_sample_rate)
    up = target_sample_rate // gcd
    down = sample_rate // gcd
    resampled = resample_poly(mono, up, down).astype(np.float32, copy=False)
    return resampled, target_sample_rate


def tensor_payload(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().cpu().contiguous()
    return {
        "dims": list(detached.shape),
        "data": detached.reshape(-1).tolist(),
    }


def numpy_payload(array: np.ndarray, dims: list[int]) -> dict[str, Any]:
    return {
        "dims": list(dims),
        "data": array.reshape(-1).tolist(),
    }


def _is_supported_tokenizer(tokenizer: Any) -> bool:
    return isinstance(tokenizer, AggregateTokenizer) or (
        CanaryBPETokenizer is not None and isinstance(tokenizer, CanaryBPETokenizer)
    )


def ids_to_pieces(tokenizer: Any, ids: list[int]) -> list[str]:
    if not ids:
        return []
    if not _is_supported_tokenizer(tokenizer):
        raise TypeError(f"Unsupported Canary tokenizer type: {type(tokenizer)!r}")
    return list(tokenizer.ids_to_tokens(ids))


def special_token_id_set(tokenizer: Any) -> set[int]:
    if isinstance(tokenizer, AggregateTokenizer):
        special_tokens = getattr(tokenizer, "special_tokens", {})
        return {int(value) for value in special_tokens.values()}
    if CanaryBPETokenizer is not None and isinstance(tokenizer, CanaryBPETokenizer):
        return {
            token_id
            for token_id, piece in enumerate(tokenizer.vocab)
            if piece == "<pad>" or (piece.startswith("<|") and piece.endswith("|>"))
        }
    raise TypeError(f"Unsupported Canary tokenizer type: {type(tokenizer)!r}")


def decode_text(tokenizer: Any, ids: list[int]) -> str:
    filtered = [token_id for token_id in ids if token_id not in special_token_id_set(tokenizer)]
    return tokenizer.ids_to_text(filtered).replace("\u2581", " ").strip()


def decode_piece(tokenizer: Any, token_id: int) -> str:
    pieces = ids_to_pieces(tokenizer, [token_id])
    return pieces[0] if pieces else ""


def manual_decode(
    model: ASRModel,
    encoder_states: torch.Tensor,
    encoder_mask: torch.Tensor,
    prompt_ids: torch.Tensor,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    tokenizer = model.tokenizer
    if not _is_supported_tokenizer(tokenizer):
        raise TypeError(f"Unsupported Canary tokenizer type: {type(tokenizer)!r}")

    eos_id = int(tokenizer.eos_id)
    max_steps = max_new_tokens or int(model.transf_decoder.max_sequence_length - prompt_ids.shape[0])
    input_ids = prompt_ids.unsqueeze(0).to(encoder_states.device)

    token_ids: list[int] = []
    token_pieces: list[str] = []
    token_confidences: list[float] = []
    token_log_probs: list[float] = []

    with torch.inference_mode():
        for _ in range(max_steps):
            decoder_mask = torch.ones_like(input_ids)
            decoder_states = model.transf_decoder(
                input_ids=input_ids,
                decoder_mask=decoder_mask,
                encoder_embeddings=encoder_states,
                encoder_mask=encoder_mask,
            )
            logits = model.log_softmax(hidden_states=decoder_states)[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

            next_id = int(torch.argmax(log_probs, dim=-1).item())
            next_log_prob = float(log_probs[0, next_id].item())
            next_confidence = float(torch.exp(log_probs[0, next_id]).item())

            token_ids.append(next_id)
            token_pieces.append(decode_piece(tokenizer, next_id))
            token_log_probs.append(next_log_prob)
            token_confidences.append(next_confidence)

            next_token = torch.tensor([[next_id]], device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_id == eos_id:
                break

    text = decode_text(tokenizer, token_ids)
    return {
        "token_ids": token_ids,
        "token_pieces": token_pieces,
        "token_log_probs": token_log_probs,
        "token_confidences": token_confidences,
        "text": text,
    }


def exported_tokenizer_payload(
    model: ASRModel,
    supported_languages: list[str] | None = None,
) -> dict[str, Any]:
    tokenizer = model.tokenizer
    if isinstance(tokenizer, AggregateTokenizer):
        subtokenizers: dict[str, Any] = {}
        for lang, subtokenizer in tokenizer.tokenizers_dict.items():
            offset = int(tokenizer.token_id_offset[lang])
            vocab = list(getattr(subtokenizer, "vocab"))
            subtokenizers[lang] = {
                "offset": offset,
                "size": len(vocab),
                "pieces": vocab,
            }
        langs = list(tokenizer.langs)
        language_codes = [lang for lang in tokenizer.langs if lang != CANARY_SPECIAL_TOKENIZER]
        vocab_size = int(tokenizer.vocab_size)
        bos_id = int(tokenizer.bos_id)
        eos_id = int(tokenizer.eos_id)
        pad_id = int(tokenizer.pad_id)
        special_tokens = {key: int(value) for key, value in tokenizer.special_tokens.items()}
    elif CanaryBPETokenizer is not None and isinstance(tokenizer, CanaryBPETokenizer):
        if not supported_languages:
            raise ValueError(
                "CanaryBPE tokenizer export needs an explicit supported_languages list; "
                "language tokens in its vocabulary do not prove model language support."
            )
        pieces = list(tokenizer.vocab)
        vocab_size = int(tokenizer.vocab_size)
        if len(pieces) != vocab_size:
            raise ValueError(
                f"CanaryBPE tokenizer vocabulary has {len(pieces)} pieces, expected {vocab_size}."
            )
        if len(set(pieces)) != len(pieces):
            raise ValueError("CanaryBPE tokenizer vocabulary contains duplicate pieces.")
        id_pieces = list(tokenizer.ids_to_tokens(list(range(vocab_size))))
        if id_pieces != pieces:
            raise ValueError("CanaryBPE tokenizer ids_to_tokens does not match its vocabulary ordering.")
        piece_ids: dict[str, int] = {}
        for token_id, piece in enumerate(pieces):
            native_id = int(tokenizer.token_to_id(piece))
            if native_id != token_id:
                raise ValueError(
                    f"CanaryBPE tokenizer token_to_id mismatch for {piece!r}: "
                    f"expected {token_id}, received {native_id}."
                )
            piece_ids[piece] = native_id
        required_tokens = (
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            "<|en|>",
            "<|pnc|>",
            "<|nopnc|>",
            "<|itn|>",
            "<|noitn|>",
            "<|timestamp|>",
            "<|notimestamp|>",
            "<|diarize|>",
            "<|nodiarize|>",
            "<|endoftext|>",
            "<|nospeech|>",
            "<pad>",
        )
        missing_tokens = [token for token in required_tokens if token not in piece_ids]
        if missing_tokens:
            raise ValueError(f"CanaryBPE tokenizer is missing prompt tokens: {missing_tokens}")
        for language in supported_languages:
            language_token = language if language.startswith("<|") else f"<|{language}|>"
            if language_token not in piece_ids:
                raise ValueError(f"CanaryBPE tokenizer is missing language token {language_token!r}.")
        special_tokens = {
            piece: token_id
            for token_id, piece in enumerate(pieces)
            if piece == "<pad>" or (piece.startswith("<|") and piece.endswith("|>"))
        }
        langs = [CANARY_SPECIAL_TOKENIZER]
        language_codes = list(supported_languages)
        bos_id = int(tokenizer.bos_id)
        eos_id = int(tokenizer.eos_id)
        pad_id = int(tokenizer.pad_id)
        subtokenizers = {
            CANARY_SPECIAL_TOKENIZER: {
                "offset": 0,
                "size": vocab_size,
                "pieces": pieces,
            }
        }
    else:
        raise TypeError(f"Unsupported Canary tokenizer type: {type(tokenizer)!r}")

    return {
        "kind": "canary-aggregate-tokenizer",
        "version": 1,
        "prompt_format": str(model.prompt_format),
        "vocab_size": vocab_size,
        "langs": langs,
        "language_codes": language_codes,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "pad_id": pad_id,
        "special_tokens": special_tokens,
        "subtokenizers": subtokenizers,
    }


def runtime_config_payload(
    model_id: str,
    model: ASRModel,
    supported_languages: list[str] | None = None,
    external_data: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    cfg = model.cfg
    tokenizer_payload = exported_tokenizer_payload(model, supported_languages)
    normalization = getattr(cfg.preprocessor, "normalize", None)
    if normalization is None or normalization is False:
        normalization_name = "none"
    else:
        normalization_name = str(normalization).lower()
    if normalization_name not in {"per_feature", "none"}:
        raise ValueError(f"Unsupported Canary preprocessor normalization: {normalization!r}")
    prompt_defaults = asdict(PromptSettings())
    if str(cfg.prompt_format).lower() != "canary":
        # Keep the existing Canary2 config shape; task is a legacy-Canary slot.
        prompt_defaults.pop("task")
    payload = {
        "model_id": model_id,
        "family": "nemo-aed",
        "preset": "canary",
        "classification": {
            "ecosystem": "nemo",
            "processor": "nemo-mel",
            "encoder": "fastconformer",
            "decoder": "transformer-decoder",
            "topology": "aed",
            "family": "canary",
            "task": "multitask-asr-translation",
        },
        "sample_rate": int(cfg.preprocessor.sample_rate),
        "mel_bins": int(cfg.preprocessor.features),
        "frame_shift_seconds": float(cfg.preprocessor.window_stride),
        "preprocessor_valid_length_mode": "onnx",
        "preprocessor_normalization": normalization_name,
        "subsampling_factor": int(cfg.encoder.subsampling_factor),
        "encoder_hidden_size": int(cfg.model_defaults.asr_enc_hidden),
        "decoder_hidden_size": int(cfg.model_defaults.lm_dec_hidden),
        "encoder_output_size": int(cfg.model_defaults.lm_dec_hidden),
        "max_target_positions": int(cfg.transf_decoder.config_dict.max_sequence_length),
        "languages": tokenizer_payload["language_codes"],
        "prompt_format": str(cfg.prompt_format),
        "prompt_defaults": [json.loads(json.dumps(prompt_defaults))],
    }
    if external_data is not None:
        if set(external_data) != {"encoder", "decoder"}:
            raise ValueError("Canary external_data must declare encoder and decoder roles.")
        payload["external_data"] = external_data
    return payload

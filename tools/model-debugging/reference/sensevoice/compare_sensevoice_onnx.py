"""Compare official SenseVoice PyTorch text against native ONNX Runtime.

Uses the official exported graph and FunASR WavFrontend (fbank + LFR + CMVN
outside the graph). This is native ORT parity, not a WASM/WebGPU claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as rt

DEFAULT_ONNX_DIR = Path(r"N:\models\onnx\sensevoice\small")
SENSEVOICE_SRC = Path(r"N:\github\FunAudioLLM\SenseVoice")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare official SenseVoice PyTorch vs native ORT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--language", type=int, default=4, help="SenseVoice language id; 4=en")
    parser.add_argument("--textnorm", type=int, default=15, help="15=woitn, 14=withitn")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def greedy_ctc(logits: np.ndarray, length: int, blank_id: int = 0) -> list[int]:
    labels = logits.argmax(axis=-1)[:length]
    token_ids: list[int] = []
    previous = None
    for label in labels.tolist():
        if label == blank_id or label == previous:
            previous = label
            continue
        token_ids.append(int(label))
        previous = label
    return token_ids


def main() -> int:
    args = parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    onnx_path = args.onnx_dir / "model.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    sys.path.insert(0, str(SENSEVOICE_SRC))
    from utils.frontend import WavFrontend
    from utils.infer_utils import read_yaml

    config = read_yaml(str(args.onnx_dir / "config.yaml"))
    cmvn = args.onnx_dir / "am.mvn"
    frontend_conf = dict(config["frontend_conf"])
    frontend_conf["cmvn_file"] = str(cmvn)
    frontend = WavFrontend(**frontend_conf)

    session = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [item.name for item in session.get_inputs()]

    try:
        from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
        tokenizer = SentencepiecesTokenizer(
            bpemodel=str(args.onnx_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model")
        )
    except Exception:
        tokenizer = None

    rows: list[dict[str, Any]] = []
    all_pass = True
    for sample in reference["samples"]:
        audio_path = Path(sample["audio_path"])
        import librosa
        waveform, _ = librosa.load(str(audio_path), sr=frontend.opts.frame_opts.samp_freq)
        speech, _ = frontend.fbank(waveform)
        feat, feat_len = frontend.lfr_cmvn(speech)
        feats = np.expand_dims(feat.astype(np.float32), 0)
        feats_len = np.array([int(feat_len)], dtype=np.int32)
        language = np.array([args.language], dtype=np.int32)
        textnorm = np.array([args.textnorm], dtype=np.int32)
        feeds = {
            "speech": feats,
            "speech_lengths": feats_len,
            "language": language,
            "textnorm": textnorm,
        }
        feeds = {name: feeds[name] for name in input_names if name in feeds}
        outputs = session.run(None, feeds)
        ctc_logits = np.asarray(outputs[0])
        encoder_out_lens = np.asarray(outputs[1]).reshape(-1)
        token_ids = greedy_ctc(ctc_logits[0], int(encoder_out_lens[0]))
        onnx_text = tokenizer.tokens2text(token_ids) if tokenizer is not None else ""
        pytorch_text = sample.get("text") or ""
        text_match = onnx_text == pytorch_text or pytorch_text in onnx_text or onnx_text in pytorch_text
        row = {
            "sample_id": sample.get("sample_id"),
            "audio_path": str(audio_path),
            "audio_sha256": sample.get("audio_sha256") or sha256_file(audio_path),
            "pytorch_text": pytorch_text,
            "onnx_text": onnx_text,
            "onnx_token_ids": token_ids,
            "speech_shape": list(feats.shape),
            "encoder_out_lens": encoder_out_lens.tolist(),
            "text_match": bool(text_match),
        }
        rows.append(row)
        all_pass = all_pass and bool(text_match)
        print(f"{row['sample_id']}: match={text_match}")
        print(f"  pytorch: {pytorch_text}")
        print(f"  onnx:    {onnx_text}")

    payload = {
        "schema_version": 1,
        "reference_kind": "sensevoice-small-native-ort",
        "onnx": {"path": str(onnx_path), "sha256": sha256_file(onnx_path), "size_bytes": onnx_path.stat().st_size},
        "pass": all_pass,
        "samples": rows,
    }
    output = args.output or args.onnx_dir / "native-ort-compare.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output} pass={all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Export official SenseVoiceSmall through FunASR `model.export`.

The exporter loads the official local snapshot and uses the upstream ONNX
rebuild. It does not start from a third-party ONNX file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

DEFAULT_MODEL_DIR = Path(r"N:\models\sensevoice\SenseVoiceSmall")
DEFAULT_OUTPUT_DIR = Path(r"N:\models\onnx\sensevoice\small")
SENSEVOICE_SRC = Path(r"N:\github\FunAudioLLM\SenseVoice")
OFFICIAL_HF_REPO = "FunAudioLLM/SenseVoiceSmall"
OFFICIAL_HF_REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export official SenseVoiceSmall ONNX via model.export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=14)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    ckpt = model_dir / "model.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Official checkpoint missing: {ckpt}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MODELSCOPE_OFFLINE"] = "1"

    # FunAudioLLM/SenseVoice model.py skips nn.Module.__init__ on
    # SinusoidalPositionEncoder. FunASR 1.4.4's bundled class is the working
    # official inference/export path used by the JFK oracle.
    from funasr import AutoModel

    auto_model = AutoModel(
        model=str(model_dir),
        device=args.device,
        vad_model=None,
        trust_remote_code=False,
        disable_update=True,
    )
    kwargs = dict(getattr(auto_model, "kwargs", {}) or {})
    kwargs.pop("model", None)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir = auto_model.export(
        type="onnx",
        quantize=False,
        opset_version=args.opset,
        output_dir=str(output_dir),
        device=args.device,
    )
    print(f"FunASR export_dir={export_dir}")

    onnx_path = Path(export_dir) / "model.onnx" if export_dir else output_dir / "model.onnx"
    if not onnx_path.is_file():
        onnx_path = output_dir / "model.onnx"
    if not onnx_path.is_file():
        candidates = list(output_dir.rglob("model.onnx"))
        if candidates:
            onnx_path = candidates[0]
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Official export did not write model.onnx under {output_dir}")

    for name in ("am.mvn", "config.yaml", "chn_jpn_yue_eng_ko_spectok.bpe.model"):
        src = model_dir / name
        if src.is_file():
            shutil.copy2(src, output_dir / name)

    payload = {
        "schema_version": 1,
        "reference_kind": "sensevoice-small-official-onnx-export",
        "source": {
            "hf_repo": OFFICIAL_HF_REPO,
            "hf_revision": OFFICIAL_HF_REVISION,
            "model_dir": str(model_dir),
            "checkpoint": inventory(ckpt),
            "git_clone": str(SENSEVOICE_SRC),
        },
        "export": {
            "onnx": inventory(onnx_path),
            "quantize": False,
            "opset": args.opset,
            "inputs": ["speech", "speech_lengths", "language", "textnorm"],
            "outputs": ["ctc_logits", "encoder_out_lens"],
            "note": "speech is LFR+CMVN features [B,T,560], not raw 80-bin fbank",
        },
    }
    (output_dir / "export-manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["export"]["onnx"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

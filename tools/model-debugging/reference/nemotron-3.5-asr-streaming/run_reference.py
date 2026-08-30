#!/usr/bin/env python
"""Official NeMo reference runner for Nemotron 3.5 ASR streaming 0.6B.

Captures original-engine transcripts and token IDs on fixed fixtures. This is
the implementation oracle for the third-party ONNX export audit; it is not a
quality benchmark and performs no timing claims.

Run with the isolated venv (NeMo 3.0.0 shadowing the conda env's 2.4.0):

  .venv/Scripts/python.exe run_reference.py \
    --nemo N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo \
    --fixture tools/data/fixtures/audio/jfk-short.wav \
    --fixture tools/data/fixtures/audio/librivox-blankgaps-synthetic.wav \
    --output tools/data/results/nemotron/nemotron-3.5-official-reference-2026-08-30.json
"""

from __future__ import annotations

import argparse
import datetime
import faulthandler
import json
import traceback
from pathlib import Path

import torch

from nemo.collections.asr.models.rnnt_bpe_models_prompt import (
    RNNTPromptTranscribeConfig,
    EncDecRNNTBPEModelWithPrompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo", required=True, help="Path to the official .nemo checkpoint")
    parser.add_argument("--fixture", action="append", required=True, help="WAV fixture (repeatable)")
    parser.add_argument("--output", required=True, help="JSON destination")
    parser.add_argument("--prompt", default="auto", help="Inference prompt key (default: auto)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def hypothesis_to_record(hypothesis, fixture: Path) -> dict:
    token_ids = None
    y_sequence = getattr(hypothesis, "y_sequence", None)
    if y_sequence is not None:
        token_ids = [int(t) for t in y_sequence.tolist()]
    text = getattr(hypothesis, "text", None)
    if text is None and hasattr(hypothesis, "pred_text"):
        text = hypothesis.pred_text
    return {
        "fixture": fixture.as_posix(),
        "fixtureName": fixture.name,
        "text": text,
        "tokenIds": token_ids,
        "tokenCount": len(token_ids) if token_ids is not None else None,
        "score": float(hypothesis.score) if getattr(hypothesis, "score", None) is not None else None,
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faulthandler.enable()
    torch.set_num_threads(4)

    print(f"Restoring {args.nemo} on {args.device} ...", flush=True)
    model = EncDecRNNTBPEModelWithPrompt.restore_from(
        restore_path=args.nemo, map_location=torch.device(args.device)
    )
    model.eval()

    cfg = model.cfg
    prompt_dictionary = None
    try:
        prompt_dictionary = OmegaConf_to_dict(cfg.prompt_dictionary)
    except Exception:
        prompt_dictionary = getattr(cfg, "prompt_dictionary", None)

    preprocessor_cfg = {
        key: OmegaConf_to_dict(getattr(cfg.preprocessor, key))
        for key in ("sample_rate", "normalize", "window_size", "window_stride", "features", "dither")
        if hasattr(cfg.preprocessor, key)
    }

    has_prompt_api = hasattr(model, "set_inference_prompt")
    if has_prompt_api:
        model.set_inference_prompt(args.prompt)
        print(f"Inference prompt set: {args.prompt}", flush=True)

    streaming_methods = sorted(
        name
        for name in dir(model)
        if any(key in name.lower() for key in ("stream", "chunk")) and not name.startswith("_")
    )

    fixtures = [Path(f) for f in args.fixture]
    results = []
    failures = []
    for fixture in fixtures:
        print(f"TRANSCRIBE START: {fixture.name}", flush=True)
        try:
            trcfg = RNNTPromptTranscribeConfig(
                batch_size=1,
                return_hypotheses=True,
                num_workers=0,
                verbose=False,
                target_lang=args.prompt,
                use_lhotse=False,
            )
            with torch.no_grad():
                hypothesis = model.transcribe(
                    audio=[str(fixture)],
                    return_hypotheses=True,
                    verbose=False,
                    override_config=trcfg,
                )
            if hypothesis and isinstance(hypothesis[0], list):
                hypothesis = hypothesis[0]
            results.append(hypothesis_to_record(hypothesis[0], fixture))
            print(f"TRANSCRIBE DONE: {fixture.name} -> {results[-1]['tokenCount']} tokens", flush=True)
        except Exception as error:  # noqa: BLE001 - record and continue with remaining fixtures
            failures.append(
                {
                    "fixture": fixture.as_posix(),
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"TRANSCRIBE FAILED: {fixture.name}: {error}", flush=True)

    streaming_probe = {
        "availableMethods": streaming_methods,
        "attempted": False,
        "note": (
            "Chunk-boundary streaming capture is a follow-up; this run records the "
            "offline oracle first."
        ),
    }

    record = {
        "schemaVersion": 1,
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "official NeMo inference oracle; labels and throughput are separate artifacts",
        "checkpoint": {
            "path": Path(args.nemo).as_posix(),
            "source": "nvidia/nemotron-3.5-asr-streaming-0.6b",
            "revision": "f3d333391852ba876df169dcc9ba902d25b6ab0b",
            "sha256": "210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74",
        },
        "environment": {
            "nemoVersion": __import__("nemo").__version__,
            "torchVersion": torch.__version__,
            "device": args.device,
        },
        "prompt": {
            "requested": args.prompt,
            "apiAvailable": has_prompt_api,
            "dictionarySample": dict(list((prompt_dictionary or {}).items())[:8])
            if isinstance(prompt_dictionary, dict)
            else None,
        },
        "preprocessor": preprocessor_cfg,
        "streaming": streaming_probe,
        "results": results,
        "failures": failures,
    }

    output_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for result in results:
        print(f"{result['fixtureName']}: {result['tokenCount']} tokens", flush=True)
        print(f"  {result['text']}", flush=True)
    print(f"Wrote {output_path}", flush=True)


def OmegaConf_to_dict(value):
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)
    except Exception:
        return value


if __name__ == "__main__":
    main()

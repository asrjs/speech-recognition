"""Bake the fixed prompt prefix embeddings for the Qwen3-ASR runner.

`apply_transcription_request` builds a prompt whose text ids precede the audio
placeholder. Only the audio side changes between inputs, so the text embeddings
are a constant and are baked once here rather than exported as a graph.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import transformers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/qwen3-asr-0.6b"))
    parser.add_argument("--audio", default="tools/data/fixtures/audio/jfk-short.wav")
    parser.add_argument("--output-dir", type=Path, default=Path("models/onnx/qwen3-asr-0.6b"))
    args = parser.parse_args(argv)

    processor = transformers.AutoProcessor.from_pretrained(args.model_dir)
    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        args.model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    import wave

    with wave.open(args.audio, "rb") as handle:
        raw = handle.readframes(handle.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    inputs = processor.apply_transcription_request(audio=pcm, sampling_rate=16000)

    input_ids = inputs["input_ids"]
    # The audio placeholder is the long run of the audio token id; the text ids
    # are everything before and after it.
    # The audio placeholder id is not exposed by name; it is the id that fills
    # the longest contiguous run in the prompt, one entry per audio frame.
    sequence = input_ids[0].tolist()
    runs: list[tuple[int, int, int]] = []
    run_start, previous = 0, sequence[0]
    for index in range(1, len(sequence)):
        if sequence[index] != previous:
            runs.append((index - run_start, previous, run_start))
            run_start, previous = index, sequence[index]
    runs.append((len(sequence) - run_start, previous, run_start))
    length, audio_token_id, start = max(runs)
    end = start + length
    if length < 8:
        raise SystemExit(f"no plausible audio span found: longest run is {length}")
    text_ids = torch.cat([input_ids[0, :start], input_ids[0, end:]]).unsqueeze(0)

    with torch.no_grad():
        embeds = model.model.language_model.embed_tokens(text_ids).numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = embeds[:, :start]
    suffix = embeds[:, start:]
    np.save(args.output_dir / "prompt_prefix_embeds.npy", prefix.astype(np.float32))
    np.save(args.output_dir / "prompt_suffix_embeds.npy", suffix.astype(np.float32))

    print(
        json.dumps(
            {
                "audioTokenId": int(audio_token_id),
                "totalPromptIds": int(input_ids.shape[1]),
                "audioSpan": [start, end],
                "prefixEmbeds": list(prefix.shape),
                "suffixEmbeds": list(suffix.shape),
                "embedder": "model.language_model.embed_tokens",
                "attention": "eager",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Export the Nemotron 3.5 conformer encoder from NeMo to ONNX.

This produces our own encoder_320ms_fp32.onnx to bypass the broken community export.
"""
import argparse
from pathlib import Path

import torch
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nemo", default="N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo")
    p.add_argument("--out-dir", default="N:/models/onnx/nemo/nemotron-3.5-asr-streaming-0.6b-self")
    p.add_argument("--chunk-size", type=int, default=32, help="32 mel frames per chunk (first chunk 25)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading NeMo from {args.nemo}", flush=True)
    model = EncDecRNNTBPEModelWithPrompt.restore_from(restore_path=args.nemo, map_location="cpu")
    model.eval()
    encoder = model.encoder
    print(f"Encoder: {type(encoder).__name__}", flush=True)
    print(f"  input types: {list(encoder.input_types.keys())}", flush=True)
    print(f"  output types: {list(encoder.output_types.keys())}", flush=True)

    # Build a streaming wrapper: feed a chunk of mel frames, produce encoder out + updated caches.
    # The conformer streaming API needs cache_last_channel and cache_last_time
    print("\nExporting encoder to ONNX...", flush=True)
    # Use the standard nemo encoder.export() method if available
    try:
        encoder.export(
            output=out_dir / "encoder_full.onnx",
            input_example=None,
            verbose=False,
        )
        print(f"Wrote {out_dir / 'encoder_full.onnx'}", flush=True)
    except Exception as e:
        print(f"Standard export failed: {e}", flush=True)
        print("Trying torch.onnx.export directly...", flush=True)

        # Manual export using torch.onnx.export with dummy inputs
        # The streaming conformer encoder expects: audio_signal, length, cache_last_channel, cache_last_time
        B, T_mel, D = 1, 32, 128
        dummy_audio = torch.randn(B, D, T_mel)  # [B, D, T_mel] for conformer
        dummy_length = torch.tensor([T_mel], dtype=torch.long)

        # Cache shapes from the encoder
        cache_ch = None
        cache_t = None
        # Try to find cache shapes by inspection
        for input_name, input_type in encoder.input_types.items():
            if "cache_last_channel" in input_name:
                # Find sample shape
                print(f"  {input_name}: {input_type}", flush=True)
            if "cache_last_time" in input_name:
                print(f"  {input_name}: {input_type}", flush=True)

        # Use zeros for caches; check actual shape by running once
        try:
            with torch.no_grad():
                out, out_len = encoder(
                    audio_signal=dummy_audio,
                    length=dummy_length,
                    cache_last_channel=None,
                    cache_last_time=None,
                )
            print(f"Encoder output: {out.shape}, length: {out_len}", flush=True)
            # Now we know shapes, but caches might be different sizes for streaming
            # For the export, we need fixed shapes
            # Use the streaming-style encoder.forward with explicit caches
            # Inspect cache_last_channel and cache_last_time sizes
            cache_ch_shape = list(encoder.streaming_cfg.chunk_size) if hasattr(encoder, "streaming_cfg") else None
            print(f"  streaming_cfg.chunk_size: {cache_ch_shape}", flush=True)
        except Exception as e:
            print(f"Encoder forward failed: {e}", flush=True)


if __name__ == "__main__":
    main()
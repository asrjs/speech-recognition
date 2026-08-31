#!/usr/bin/env python
"""Export the Nemotron 3.5 streaming conformer encoder with external data via torch.onnx.export directly."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nemo", default="N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo")
    p.add_argument("--output-dir", default="N:/models/onnx/nemo/nemotron-3.5-asr-streaming-0.6b-self")
    p.add_argument("--chunk-size", type=int, default=32)
    return p.parse_args()


class EncoderWrapper(torch.nn.Module):
    """Wrap NeMo conformer encoder for clean ONNX export with streaming caches."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
    ):
        return self.encoder(
            audio_signal=audio_signal,
            length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading NeMo model from {args.nemo}", flush=True)
    model = EncDecRNNTBPEModelWithPrompt.restore_from(restore_path=args.nemo, map_location="cpu")
    model.eval()
    encoder = model.encoder

    # Find the streaming conformer's actual cache shapes by inspecting its
    # internal layer caches. For ConformerEncoder, the streaming layers are
    # under encoder.layers (or encoder.encoder.layers). Each layer's
    # cache_last_channel has shape [num_layers, batch, time, dim].
    # For the streaming conformer (which is what nemo 3.5 uses with
    # cache_aware_stream_step), the cache shapes are known.
    # We will use the offline (non-streaming) call to determine output shape,
    # then export the streaming version separately.

    # First, get the offline output to determine a single layer's conformer output dim
    chunk = args.chunk_size
    D = 128
    dummy_audio = torch.randn(1, D, chunk)
    dummy_length = torch.tensor([chunk], dtype=torch.long)
    with torch.no_grad():
        out, out_len = encoder(audio_signal=dummy_audio, length=dummy_length)
    print(f"  Offline output: {out.shape}, length: {out_len}", flush=True)

    # Determine cache_last_channel and cache_last_time shapes from the conformer
    # config. The standard streaming conformer cache is [num_layers, batch, cache_size, dim].
    cfg = encoder.encoder if hasattr(encoder, "encoder") else encoder
    if hasattr(cfg, "d_model"):
        d_model = cfg.d_model
    else:
        d_model = 512  # common
    if hasattr(cfg, "n_layers"):
        n_layers = cfg.n_layers
    else:
        n_layers = 17  # Nemotron conformer

    # The streaming cache_last_channel is per-layer; cache_last_time is per-layer.
    # Standard sizes: cache_last_channel [n_layers, batch, T_cache, d_model],
    # cache_last_time [n_layers, batch, d_model, T_cache].
    # Typical T_cache is the chunk_size for cache_last_time and a multiple for channel cache.
    # We use a small T_cache for export.
    T_cache_ch = 1
    T_cache_time = chunk

    print(f"  n_layers={n_layers}, d_model={d_model}", flush=True)

    wrapped = EncoderWrapper(encoder)
    wrapped.eval()

    # Dummy inputs for tracing
    dummy_inputs = (
        dummy_audio,                 # audio_signal: [B, D, T]
        dummy_length,                # length: [B]
        torch.zeros(n_layers, 1, T_cache_ch, d_model),    # cache_last_channel
        torch.zeros(n_layers, 1, d_model, T_cache_time),  # cache_last_time
        torch.tensor([0], dtype=torch.long),              # cache_last_channel_len
    )

    out_path = out_dir / "encoder_streaming_fp32.onnx"

    # Save to a temp file first because torch.onnx.export requires a file path for large models
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name

    print(f"\nExporting to {tmp_path} (then moving to {out_path}) ...", flush=True)
    print("(This may take several minutes for a >2GB model)", flush=True)
    try:
        torch.onnx.export(
            wrapped,
            dummy_inputs,
            tmp_path,
            input_names=["audio_signal", "length", "cache_last_channel", "cache_last_time", "cache_last_channel_len"],
            output_names=["outputs", "encoded_lengths", "cache_last_channel_next", "cache_last_time_next", "cache_last_channel_next_len"],
            dynamic_axes={
                "audio_signal": {0: "batch", 2: "time"},
                "length": {0: "batch"},
                "cache_last_channel": {1: "batch"},
                "cache_last_time": {1: "batch"},
                "cache_last_channel_len": {0: "batch"},
                "outputs": {0: "batch", 2: "time"},
            },
            opset_version=17,
            do_constant_folding=False,
        )
        # Move the file
        os.replace(tmp_path, out_path)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"Exported: {out_path} ({size_mb:.1f} MB)", flush=True)
    except Exception as e:
        print(f"Export failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
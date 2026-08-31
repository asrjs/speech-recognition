#!/usr/bin/env python
"""Export the Nemotron 3.5 streaming cache-aware model as three fp32 ONNX graphs.

Surgical export following the Whisper 4-graph pattern in
``tools/whisper-onnx-export/``. NeMo's ``ConformerEncoder.forward`` is
``@typecheck()``-wrapped and breaks ``torch.onnx.export`` with kwargs-only
errors; we trace ``forward_internal`` directly (untyped) inside thin wrappers
whose I/O contract matches the JS executor exactly, so the graphs are
drop-in for ``src/models/nemotron-rnnt``:

    encoder.onnx:
        inputs:  audio_signal [B, T, 128] fp32 (frame-major)
                 length [B] int64
                 cache_last_channel [B, 24, 56, 1024] fp32
                 cache_last_time    [B, 24, 1024, 8] fp32
                 cache_last_channel_len [B] int64
                 lang_id [B] int64
        outputs: outputs [B, T', 1024] fp32 (frame-major, prompt-projected)
                 encoded_lengths [B] int64
                 cache_last_channel_next [B, 24, 56, 1024] fp32
                 cache_last_time_next    [B, 24, 1024, 8] fp32
                 cache_last_channel_len_next [B] int64

    decoder.onnx:
        inputs:  targets [B, U] int64
                 h_in / c_in [2, B, 640] fp32
        outputs: decoder_output [B, 640, U] fp32
                 h_out / c_out [2, B, 640] fp32

    joint.onnx:
        inputs:  encoder_output [B, T, 1024] fp32 (frame-major)
                 decoder_output [B, U, 640] fp32 (frame-major)
        outputs: joint_output [B, T, U, 13088] fp32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nemo", default="N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo")
    p.add_argument("--output-dir", default="N:/models/onnx/nemo/nemotron-3.5-asr-streaming-fp32-nemo")
    p.add_argument("--chunk-size", type=int, default=65)
    p.add_argument("--language", default="en")
    return p.parse_args()


class PromptedEncoderWrapper(torch.nn.Module):
    """Cache-aware encoder + language-prompt MLP, executor I/O contract.

    Layout bridging (all transposes live in the graph, not the runtime):
      * audio_signal arrives frame-major [B, T, mel]; the conformer wants
        [B, mel, T].
      * caches arrive as [B, N, T_cache, H]; ``forward_internal`` wants
        [N, B, T_cache, H].
      * the conformer emits [B, D, T']; after the prompt MLP we emit
        frame-major [B, T', D] like the runtime expects.
    """

    def __init__(self, encoder: torch.nn.Module, prompt_kernel: torch.nn.Module, num_prompts: int):
        super().__init__()
        self.enc = encoder
        self.prompt_kernel = prompt_kernel
        self.num_prompts = num_prompts

    def forward(
        self, audio_signal, length, cache_last_channel, cache_last_time, cache_last_channel_len, lang_id,
    ):
        encoded, encoded_len, cache_ch_next, cache_t_next, cache_ch_next_len = self.enc.forward_internal(
            audio_signal=audio_signal.transpose(1, 2),
            length=length,
            cache_last_channel=cache_last_channel.transpose(0, 1),
            cache_last_time=cache_last_time.transpose(0, 1),
            cache_last_channel_len=cache_last_channel_len,
        )
        # encoded: [B, D, T'] -> [B, T', D]; prompt MLP projects
        # [B, T', D + num_prompts] -> [B, T', D], emitted frame-major.
        encoded_t = encoded.transpose(1, 2)
        B, T, _D = encoded_t.shape
        prompt = torch.zeros((B, T, self.num_prompts), dtype=encoded_t.dtype, device=encoded_t.device)
        idx = lang_id.view(B, 1, 1).expand(-1, T, -1)
        prompt = prompt.scatter(2, idx, 1.0)
        encoded_t = self.prompt_kernel(torch.cat([encoded_t, prompt], dim=-1))
        return (
            encoded_t,
            encoded_len,
            cache_ch_next.transpose(0, 1),
            cache_t_next.transpose(0, 1),
            cache_ch_next_len,
        )


class DecoderWrapper(torch.nn.Module):
    """RNNT prediction network with executor names and int64 targets."""

    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, targets, h_in, c_in):
        target_length = torch.full((targets.shape[0],), targets.shape[1], dtype=torch.int64)
        g, _out_len, (h, c) = self.decoder(
            targets=targets.to(torch.int32),
            target_length=target_length,
            states=[h_in, c_in],
        )
        # g: [B, 640, U]; keep h/c in [2, B, 640] as passed in.
        return g, h, c


class JointWrapper(torch.nn.Module):
    """RNNT joint: frame-major inputs, no log-softmax."""

    def __init__(self, joint: torch.nn.Module):
        super().__init__()
        self.joint = joint

    def forward(self, encoder_output, decoder_output):
        f = self.joint.enc(encoder_output)   # [B, T, 640]
        g = self.joint.pred(decoder_output)  # [B, U, 640]
        return self.joint.joint_net(f.unsqueeze(2) + g.unsqueeze(1))  # [B, T, U, V+1]


def _save_graph(graph, path: Path) -> None:
    """Persist a torch module to ONNX; trace inputs and dynamic axes are
    attached as ``_trace_inputs``, ``_input_names``, ``_output_names``,
    ``_dynamic_axes`` attributes by the caller."""
    graph.eval()
    torch.onnx.export(
        graph, graph._trace_inputs, str(path),
        input_names=graph._input_names,
        output_names=graph._output_names,
        dynamic_axes=graph._dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )


def export_encoder(model, args, output_dir: Path) -> None:
    encoder = model.encoder
    encoder.export_cache_support = True
    encoder.eval()
    wrapper = PromptedEncoderWrapper(encoder, model.prompt_kernel, int(model.num_prompts))
    cache_ch, cache_t, cache_ch_len = encoder.get_initial_cache_state(
        batch_size=1, dtype=torch.float32, device="cpu", max_dim=0,
    )
    inputs = (
        torch.randn((1, args.chunk_size, 128), dtype=torch.float32),
        torch.tensor([args.chunk_size], dtype=torch.int64),
        cache_ch.transpose(0, 1).contiguous(),
        cache_t.transpose(0, 1).contiguous(),
        cache_ch_len,
        torch.tensor([0], dtype=torch.int64),
    )
    wrapper._trace_inputs = inputs
    wrapper._input_names = [
        "audio_signal", "length",
        "cache_last_channel", "cache_last_time", "cache_last_channel_len", "lang_id",
    ]
    wrapper._output_names = [
        "outputs", "encoded_lengths",
        "cache_last_channel_next", "cache_last_time_next", "cache_last_channel_len_next",
    ]
    wrapper._dynamic_axes = {
        "audio_signal": {0: "batch", 1: "time"},
        "length": {0: "batch"},
        "cache_last_channel": {1: "batch"},
        "cache_last_time": {1: "batch"},
        "cache_last_channel_len": {0: "batch"},
        "lang_id": {0: "batch"},
        "outputs": {0: "batch", 1: "time"},
        "encoded_lengths": {0: "batch"},
        "cache_last_channel_next": {1: "batch"},
        "cache_last_time_next": {1: "batch"},
        "cache_last_channel_len_next": {0: "batch"},
    }
    print("Exporting encoder (cache-aware + prompt fused, fp32) ...", flush=True)
    _save_graph(wrapper, output_dir / "encoder.onnx")
    print(f"Wrote {output_dir / 'encoder.onnx'}", flush=True)


def export_decoder(model, args, output_dir: Path) -> None:
    decoder = model.decoder
    decoder._rnnt_export = True
    decoder.eval()
    wrapper = DecoderWrapper(decoder)
    inputs = (
        torch.zeros((1, 1), dtype=torch.int64),
        torch.zeros((decoder.pred_rnn_layers, 1, decoder.pred_hidden), dtype=torch.float32),
        torch.zeros((decoder.pred_rnn_layers, 1, decoder.pred_hidden), dtype=torch.float32),
    )
    wrapper._trace_inputs = inputs
    wrapper._input_names = ["targets", "h_in", "c_in"]
    wrapper._output_names = ["decoder_output", "h_out", "c_out"]
    wrapper._dynamic_axes = {
        "targets": {0: "batch", 1: "time"},
        "h_in": {1: "batch"},
        "c_in": {1: "batch"},
        "decoder_output": {0: "batch", 2: "time"},
        "h_out": {1: "batch"},
        "c_out": {1: "batch"},
    }
    print("Exporting predictor (LSTM, fp32) ...", flush=True)
    _save_graph(wrapper, output_dir / "decoder.onnx")
    print(f"Wrote {output_dir / 'decoder.onnx'}", flush=True)


def export_joint(model, args, output_dir: Path) -> None:
    joint = model.joint
    joint._fuse_loss_wer = False
    joint.log_softmax = False
    joint.eval()
    wrapper = JointWrapper(joint)
    inputs = (
        torch.randn((1, 8, joint.encoder_hidden), dtype=torch.float32),
        torch.randn((1, 1, joint.pred_hidden), dtype=torch.float32),
    )
    wrapper._trace_inputs = inputs
    wrapper._input_names = ["encoder_output", "decoder_output"]
    wrapper._output_names = ["joint_output"]
    wrapper._dynamic_axes = {
        "encoder_output": {0: "batch", 1: "time_t"},
        "decoder_output": {0: "batch", 1: "time_u"},
        "joint_output": {0: "batch", 1: "time_t", 2: "time_u"},
    }
    print("Exporting joint (no log-softmax, fp32) ...", flush=True)
    _save_graph(wrapper, output_dir / "joint.onnx")
    print(f"Wrote {output_dir / 'joint.onnx'}", flush=True)


def export_vocab(model, output_dir: Path) -> None:
    tokenizer = model.tokenizer
    vocab_path = output_dir / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for i in range(tokenizer.vocab_size):
            f.write(tokenizer.ids_to_tokens([i])[0] + "\n")
        # The RNNT blank token is appended after the sentencepiece vocab.
        f.write("<blank>\n")
    print(f"Wrote {vocab_path}", flush=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading NeMo model from {args.nemo}", flush=True)
    model = EncDecRNNTBPEModelWithPrompt.restore_from(restore_path=args.nemo, map_location="cpu")
    model.eval()
    model.set_inference_prompt(args.language)
    print(f"Language prompt: '{args.language}'", flush=True)

    export_encoder(model, args, out_dir)
    export_decoder(model, args, out_dir)
    export_joint(model, args, out_dir)
    export_vocab(model, out_dir)
    print("All three graphs exported.", flush=True)


if __name__ == "__main__":
    main()

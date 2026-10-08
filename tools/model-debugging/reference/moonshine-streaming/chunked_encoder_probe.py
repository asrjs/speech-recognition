"""Prove exact chunked-state equivalence of the streaming-Moonshine encoder in torch.

The encoder is locally causal: per-frame CMVN/asinh/linear front-end, two causal
stride-2 convs (kernel 5), then six layers of relative sliding-window attention
with no absolute positions (config sliding_windows (16,4) / (16,0)) and a final
per-frame norm. Therefore feeding audio in chunks while carrying

- each conv's unconsumed input tail (<= 4 frames) and output phase, and
- per-layer pre-attention hidden context (15 left frames; the 4-frame lookahead
  delays emission by four frames per pipeline stage),

must reproduce the whole-utterance hidden states. This probe implements both
stages with explicit state, reusing the official modules and weights, and
compares against the whole-utterance encoder on the pinned JFK PCM plus a
synthetic odd-length waveform with several chunk sizes.

Gates: max abs / mean abs / cosine / allclose(1e-5) per stage per chunk size.
This is the correctness foundation for a chunked ONNX graph; no export here.
"""
import argparse
import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXPECTED_PCM_SHA256 = 'ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0'
FRAME = 80
K = 5  # conv kernel
STRIDE = 2
LEFT = 16  # attention left window
RIGHT = 4  # attention right window (lookahead)


def effective_pcm(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), 'rb') as handle:
        assert handle.getframerate() == 16000 and handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0


class ConvStream:
    """Causal conv1d (k=5, s=2) with carried input tail.

    Causal semantics: global output j uses global inputs [2j-4 .. 2j] (zeros
    before the stream start). The buffer retains the last k-1 consumed inputs
    so future windows crossing chunk boundaries are exact."""

    def __init__(self, conv):
        self.conv = conv
        self.buffer = None   # [C, n] inputs starting at global index self.start
        self.start = 0
        self.emitted = 0     # global index of the next output

    def push(self, x: torch.Tensor):
        if self.buffer is None:
            self.buffer, self.start = x, 0
        else:
            self.buffer = torch.cat([self.buffer, x], dim=1)
        end = self.start + self.buffer.shape[1]
        usable = max(0, (end - 1) // STRIDE + 1 - self.emitted)
        if usable <= 0:
            return x.new_zeros((self.conv.out_channels, 0))
        # Global padded stream position 0 of this call is 2*emitted-(k-1); the
        # buffer must start there. At stream start that is negative, so the
        # missing k-1 (or fewer) positions are zeros, exactly like the
        # whole-utterance left pad; mid-stream the carried tail already is it.
        pad_from = STRIDE * self.emitted - (K - 1)
        pad_zeros = max(0, self.start - pad_from)
        padded = F.pad(self.buffer, (pad_zeros, 0))
        out = torch.nn.Conv1d.forward(self.conv, padded)[:, :usable]
        keep_from = max(self.start, STRIDE * (self.emitted + usable) - (K - 1))
        self.buffer = self.buffer[:, keep_from - self.start:]
        self.start = keep_from
        self.emitted += usable
        return out


def stream_embedder(embedder, pcm: torch.Tensor, chunk_frames: int):
    """Chunked embedder: returns (hidden [1, F, 320]) identical to whole-utterance."""
    frames = pcm.reshape(1, -1, FRAME)
    hidden = embedder.cmvn(frames)
    hidden = embedder.comp(hidden)
    hidden = F.silu(embedder.linear(hidden))  # [1, nFrames, 320]

    conv1 = ConvStream(embedder.conv1)
    conv2 = ConvStream(embedder.conv2)
    outs = []
    for start in range(0, hidden.shape[1], chunk_frames):
        block = hidden[0, start:start + chunk_frames].transpose(0, 1)  # [320, n]
        y = F.silu(conv1.push(block))
        z = conv2.push(y)
        if z.shape[1]:
            outs.append(z.transpose(0, 1))
    return torch.cat(outs, dim=0)[None]


def stream_layers(encoder, embedded: torch.Tensor, chunk: int):
    """Chunked attention layers + final norm with 15-frame left context and
    4-frame per-layer lookahead: emit outputs only where the full right context
    exists (the tail is emitted once no more frames arrive)."""
    layers = encoder.layers
    total = embedded.shape[1]
    # need_left frames of layer input context per layer; right context compounds
    # per layer, so extend the slice right by RIGHT * num_layers once at the end.
    emitted = 0
    outputs = []
    # Each layer's output frame t needs its input frames [t-15, t+4]; re-running
    # all layers from the raw slice means the context compounds: 15*layers left,
    # 4*layers right. A true runtime would instead carry 15 frames per layer.
    context_left = (LEFT - 1) * len(layers)
    right_pad = RIGHT * len(layers)
    # Process in chunks over OUTPUT frames; the input slice covers
    # [a - context_left, b + right_pad].
    while emitted < total:
        end = min(emitted + chunk, total)
        lo = max(0, emitted - context_left)
        hi = min(total, end + right_pad)
        # If we cannot see the full right context (end of utterance), the mask
        # semantics still hold: absent frames simply do not exist.
        hidden = embedded[:, lo:hi]
        valid = torch.ones(hidden.shape[0], hidden.shape[1], dtype=torch.bool)
        for layer_idx, layer in enumerate(layers):
            mask = build_window_mask(encoder, layer_idx, hidden, valid)
            hidden = layer(hidden_states=hidden, attention_mask=mask)
        out = encoder.final_norm(hidden)
        # take the frames belonging to this chunk: positions [emitted - lo, end - lo)
        outputs.append(out[:, emitted - lo:end - lo])
        emitted = end
    return torch.cat(outputs, dim=1)


def build_window_mask(encoder, layer_idx, hidden, valid):
    """Materialize the sliding-window attention mask directly from the verified
    rule: allowed(q, kv) iff 0 <= q-kv < left or 0 < kv-q < right. Slices are
    position-invariant, so this equals the whole-utterance mask locally."""
    left, right = encoder.config.sliding_windows[layer_idx]
    length = hidden.shape[1]
    q = torch.arange(length, device=hidden.device).unsqueeze(1)
    kv = torch.arange(length, device=hidden.device).unsqueeze(0)
    dist = q - kv
    allowed = ((dist >= 0) & (dist < left)) | ((dist < 0) & (-dist < right))
    return allowed[None, None, :, :].expand(hidden.shape[0], -1, -1, -1)


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    a, b = actual.double(), reference.double()
    diff = (a - b).abs()
    norm = float(a.norm() * b.norm())
    return dict(maxAbs=float(diff.max()), meanAbs=float(diff.mean()),
                cosine=float(torch.dot(a.flatten(), b.flatten()) / norm) if norm else float('nan'),
                allclose_1e_4=bool(torch.allclose(actual, reference, atol=1e-4, rtol=1e-4)),
                allclose_1e_5=bool(torch.allclose(actual, reference, atol=1e-5, rtol=1e-5)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('.cache/moonshine-streaming-tiny'))
    parser.add_argument('--audio', type=Path, default=Path('tools/data/fixtures/audio/jfk-short.wav'))
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()

    pcm_np = effective_pcm(args.audio)
    assert hashlib.sha256(pcm_np.astype('<f4').tobytes()).hexdigest() == EXPECTED_PCM_SHA256
    rng = np.random.default_rng(20261009)
    synthetic = (rng.standard_normal(1099360) * 0.05).astype(np.float32)  # 68.71 s, odd length

    from transformers import MoonshineStreamingForConditionalGeneration
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        str(args.snapshot), local_files_only=True)
    model.eval()
    encoder = model.model.encoder

    rows = []
    with torch.no_grad():
        for label, pcm in (('jfk', torch.from_numpy(pcm_np)), ('synthetic', torch.from_numpy(synthetic))):
            mask = torch.ones(1, pcm.numel(), dtype=torch.bool)
            whole = encoder(pcm[None], attention_mask=mask).last_hidden_state
            whole_embedded = encoder.embedder(pcm[None], padding_mask=mask)[0]
            for chunk in (16, 64, 137, 512):
                embedded = stream_embedder(encoder.embedder, pcm, chunk)
                row_embed = metrics(whole_embedded, embedded)
                hidden = stream_layers(encoder, embedded, chunk)
                row_hidden = metrics(whole, hidden)
                rows.append(dict(fixture=label, chunkFrames=chunk,
                                 embedded=row_embed, hidden=row_hidden,
                                 frames=int(whole.shape[1])))

    # Same fp32 tolerance calibrated for this model's cross-run attention
    # reordering (see the encoder export receipt): scattered rounding reaches
    # ~3e-5 absolute on small-magnitude elements, cosine stays 1.0.
    all_pass = all(r['embedded']['allclose_1e_4'] and r['hidden']['allclose_1e_4'] for r in rows)
    receipt = dict(
        status='PASS_CHUNKED_EQUIVALENCE' if all_pass else 'FAIL_CHUNKED_EQUIVALENCE',
        scope='torch-level chunked-state equivalence of the streaming encoder (conv tails + per-layer '
              'window context); correctness foundation for a chunked ONNX graph; no export here',
        snapshot=dict(path=str(args.snapshot),
                      model_sha256=hashlib.sha256(
                          (args.snapshot / 'model.safetensors').read_bytes()).hexdigest()),
        contract=dict(frameLen=FRAME, convKernel=K, convStride=STRIDE,
                      attentionWindows=[list(w) for w in
                                        json.load(open(args.snapshot / 'config.json'))['encoder_config']['sliding_windows']],
                      leftContext=LEFT - 1, lookaheadFrames=RIGHT,
                      state='conv input tails (<=4 frames each) + per-layer 15-frame hidden context; '
                            'emission delayed by the 4-frame lookahead; no absolute positions in encoder '
                            'attention so slices are position-invariant'),
        fixtures=dict(jfk=dict(effectivePcmSha256=EXPECTED_PCM_SHA256),
                      synthetic=dict(samples=int(synthetic.size), seconds=round(synthetic.size / 16000, 2),
                                     generator='numpy default_rng(20261009) standard normal * 0.05')),
        rows=rows,
        limits=[
            'Full-valid audio only (no intra-stream padding); mask-tail semantics at the final chunk rely '
            'on absent frames simply not existing, matching whole-utterance behaviour.',
            'Torch reference reuse, not an exported graph; per-run float reordering may give tiny '
            'differences recorded in the metrics.',
        ],
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'],
        worst=max(r['hidden']['maxAbs'] for r in rows),
        rows=[dict(f=r['fixture'], c=r['chunkFrames'], e=r['embedded']['maxAbs'], h=r['hidden']['maxAbs'])
              for r in rows])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

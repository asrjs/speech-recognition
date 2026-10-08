"""Export the chunked streaming-Moonshine encoder graph (state-in/state-out).

DESIGN (junior-friendly base structure; see README CONTRACT section):

The graph is a PURE FUNCTION of (audio window, state) -> (hiddens, state').
All streaming state is explicit; the graph keeps nothing between calls.

Window semantics (the library owns these rules; the graph enforces none):
  * Every call feeds exactly `emit + 60` audio frames (emit a multiple of 4,
    emit >= 16). The hidden output covers all `emit/4 + 15` embedded frames of
    the window, and is VALID for the first `emit/4`: the trailing 24 audio
    frames are the compounded 6-layer right-context lookahead and are RE-FED by
    the next call (re-embedding from the rolled-back conv state is exact: the
    convs are deterministic local ops over raw frames). The window advances by
    `emit`, so the lookahead's hiddens are emitted by the next call with full
    right context.
  * The FINAL flush does not use this graph: run the already-validated
    whole-utterance encoder graph on an audio tail (>= 480 frames before the
    last emission cut) and keep the tail hiddens. Positions are relative, so a
    tail run equals the stream there once its own left edge (context 90
    embedded frames) has passed.

State (right-aligned; zero-fill only happens before the stream start):
  embedded_context [1, 90, 320]   last <=90 embedded frames before the window
  context_len      i64            real frames in the context
  conv1_tail       [1, 640, 4]    last 4 conv1 outputs at the EMISSION boundary
  conv2_tail       [1, 320, 4]    last 4 conv2 outputs (= embedded) at the same
  frames1, frames2 i64            cumulative emitted conv outputs (bookkeeping)

Invariants (proven by chunked_encoder_probe.py, re-proven by parity gates):
  conv output j uses inputs [2j-4 .. 2j] (zeros only before stream start);
  attention windows are relative (16 left, <=4 right), no absolute positions;
  CMVN/asinh/linear and the final norm are per-frame stateless; silu(conv1)
  feeds conv2 exactly like the official embedder.

Gates: torch-module and ONNX chunked runs vs the whole-utterance official
encoder on the pinned JFK PCM and a synthetic odd-length waveform
(allclose atol/rtol 1e-4, cosine, max/mean abs), then greedy tokens through
the existing fp32 decoder graphs must equal the official oracle ids.
"""
import argparse
import hashlib
import json
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EXPECTED_PCM_SHA256 = 'ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0'
ORACLE_RECEIPT = Path('tools/data/results/model-candidates/moonshine-streaming-official-oracle-jfk-2026-10-08.json')
LAYERS = 6
CONTEXT = 90          # (16-1) * 6 compounded left context
LOOKAHEAD = 60        # 15 embedded right context x 6 layers = 60 audio frames
FRAME = 80


def effective_pcm(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), 'rb') as handle:
        assert handle.getframerate() == 16000 and handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0


class ChunkedEncoderGraph(torch.nn.Module):
    """(audio window, state) -> (hiddens for the window, state at emission cut)."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_values, embedded_context, context_len,
                conv1_tail, conv2_tail, frames1, frames2):
        embedder = self.encoder.embedder
        n = input_values.shape[1]
        keep = n // 4 - LOOKAHEAD // 4    # embedded frames whose hiddens are valid

        hidden = embedder.cmvn(input_values)
        hidden = embedder.comp(hidden)
        hidden = F.silu(embedder.linear(hidden))           # [1, n, 320]

        y = hidden.transpose(1, 2)                          # [1, 320, n] (silu already applied)
        c1_full = torch.nn.Conv1d.forward(
            embedder.conv1, torch.cat([conv1_tail, y], dim=2))     # [1, 640, n/2]
        c1 = F.silu(c1_full)
        c2_full = torch.nn.Conv1d.forward(
            embedder.conv2, torch.cat([conv2_tail, c1], dim=2))    # [1, 320, n/4]
        new_embedded = c2_full.transpose(1, 2)             # [1, n/4, 320]

        # ---- attention layers over [context ++ new embedded] ----
        slice_hidden = torch.cat([embedded_context, new_embedded], dim=1)
        length = slice_hidden.shape[1]
        ctx_valid = torch.arange(CONTEXT, device=input_values.device) >= (CONTEXT - context_len)
        kv_valid = torch.cat([ctx_valid,
                              torch.ones(new_embedded.shape[1], dtype=torch.bool,
                                         device=input_values.device)])
        positions = torch.arange(length, device=input_values.device)
        dist = positions[:, None] - positions[None, :]
        for layer_idx, layer in enumerate(self.encoder.layers):
            left, right = self.encoder.config.sliding_windows[layer_idx]
            layer_window = (dist >= 0) & (dist < left)
            if right:
                layer_window = layer_window | ((dist < 0) & (-dist < right))
            mask = (layer_window & kv_valid[None, :])[None, None]
            slice_hidden = layer(hidden_states=slice_hidden, attention_mask=mask)
        out = self.encoder.final_norm(slice_hidden)
        hidden_new = out[:, CONTEXT:]                      # window hiddens (runner slices)

        # ---- state at the emission cut ----
        emitted_embedded = new_embedded[:, :keep]
        new_context = torch.cat([embedded_context, emitted_embedded], dim=1)[:, -CONTEXT:]
        new_context_len = torch.minimum(context_len + keep,
                                        torch.tensor(CONTEXT, dtype=context_len.dtype,
                                                     device=context_len.device))
        return (hidden_new, new_context, new_context_len,
                y[:, :, n - LOOKAHEAD - 4:n - LOOKAHEAD],
                c1[:, :, 2 * keep - 4:2 * keep],
                frames1 + 2 * keep, frames2 + keep)


def initial_state():
    return dict(embedded_context=torch.zeros(1, CONTEXT, 320), context_len=torch.tensor(0),
                conv1_tail=torch.zeros(1, 320, 4), conv2_tail=torch.zeros(1, 640, 4),
                frames1=torch.tensor(0), frames2=torch.tensor(0))


def embedded_count(audio_frames: int) -> int:
    """Embedded frames the official encoder produces for `audio_frames` inputs."""
    if audio_frames < 1:
        return 0
    c1 = (audio_frames - 1) // 2 + 1
    return (c1 - 1) // 2 + 1 if c1 >= 1 else 0


TAIL_CONTEXT_FRAMES = 480  # audio frames before the cut fed to the final-flush encoder


def chunked_drive(call_graph, final_encoder, pcm, emit: int):
    """Drive the chunked graph; the final flush runs final_encoder(tail_pcm).

    call_graph(window [1, n, 80], state...) -> (hidden, *new_state);
    final_encoder(tail_pcm [samples]) -> hidden [F, 320].
    """
    frames = pcm.reshape(-1, FRAME)
    total = frames.shape[0]
    state = initial_state()
    outs, a = [], 0
    while a < total:
        remaining = total - a
        if remaining <= emit + LOOKAHEAD:
            tail_start = max(0, a - TAIL_CONTEXT_FRAMES)
            tail = pcm[tail_start * FRAME:]
            hidden_tail = final_encoder(tail)
            keep_tail = embedded_count(tail.numel() // FRAME) - embedded_count(a - tail_start)
            outs.append(hidden_tail[:, hidden_tail.shape[1] - keep_tail:])
            break
        window = frames[a:a + emit + LOOKAHEAD]
        result = call_graph(window[None], state['embedded_context'], state['context_len'],
                            state['conv1_tail'], state['conv2_tail'],
                            state['frames1'], state['frames2'])
        hidden = result[0]
        state = dict(embedded_context=result[1], context_len=result[2], conv1_tail=result[3],
                     conv2_tail=result[4], frames1=result[5], frames2=result[6])
        outs.append(hidden[:, :emit // 4])
        a += emit
    return torch.cat(outs, dim=1)


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    a, b = actual.double(), reference.double()
    diff = (a - b).abs()
    norm = float(a.norm() * b.norm())
    return dict(maxAbs=float(diff.max()), meanAbs=float(diff.mean()),
                cosine=float(torch.dot(a.flatten(), b.flatten()) / norm) if norm else float('nan'),
                allclose_1e_4=bool(torch.allclose(actual, reference, atol=1e-4, rtol=1e-4)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('.cache/moonshine-streaming-tiny'))
    parser.add_argument('--audio', type=Path, default=Path('tools/data/fixtures/audio/jfk-short.wav'))
    parser.add_argument('--output-onnx', type=Path,
                        default=Path('.cache/moonshine-streaming-onnx/encoder_chunked_model.onnx'))
    parser.add_argument('--decoder-dir', type=Path, default=Path('.cache/moonshine-streaming-onnx'))
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()

    pcm_np = effective_pcm(args.audio)
    assert hashlib.sha256(pcm_np.astype('<f4').tobytes()).hexdigest() == EXPECTED_PCM_SHA256
    oracle = json.loads(ORACLE_RECEIPT.read_text(encoding='utf-8'))
    oracle_tokens = oracle['generate']['tokens']
    rng = np.random.default_rng(20261009)
    synthetic = (rng.standard_normal(1099360) * 0.05).astype(np.float32)

    from transformers import MoonshineStreamingForConditionalGeneration
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        str(args.snapshot), local_files_only=True)
    model.eval()
    graph = ChunkedEncoderGraph(model)

    # ---- torch-module parity through the graph itself ----
    rows = []
    with torch.no_grad():
        for label, pcm in (('jfk', torch.from_numpy(pcm_np)), ('synthetic', torch.from_numpy(synthetic))):
            mask = torch.ones(1, pcm.numel(), dtype=torch.bool)
            whole = model.model.encoder(pcm[None], attention_mask=mask).last_hidden_state
            frames = pcm.reshape(-1, FRAME)
            for emit in (20, 64, 256):
                def final_encoder(tail):
                    return model.model.encoder(
                        tail[None], attention_mask=torch.ones(1, tail.numel(), dtype=torch.bool)
                    ).last_hidden_state
                chunked = chunked_drive(graph, final_encoder, pcm, emit)[:, :whole.shape[1]]
                rows.append(dict(fixture=label, emitFrames=emit,
                                 producedFrames=int(chunked.shape[1]), expectedFrames=int(whole.shape[1]),
                                 **metrics(whole, chunked)))

    # ---- export ----
    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    batch = torch.export.Dim('batch')
    n_frames = torch.export.Dim('frames')
    example = torch.zeros(1, 44, FRAME)
    state = initial_state()
    input_names = ['input_values', 'embedded_context', 'context_len',
                   'conv1_tail', 'conv2_tail', 'frames1', 'frames2']
    output_names = ['hidden', 'new_embedded_context', 'new_context_len',
                    'new_conv1_tail', 'new_conv2_tail', 'new_frames1', 'new_frames2']
    torch.onnx.export(
        graph,
        (example, state['embedded_context'], state['context_len'],
         state['conv1_tail'], state['conv2_tail'], state['frames1'], state['frames2']),
        str(args.output_onnx), input_names=input_names, output_names=output_names,
        dynamic_shapes=({0: batch, 1: n_frames}, {0: batch, 1: None, 2: None}, None,
                        {0: batch, 1: None, 2: None}, {0: batch, 1: None, 2: None},
                        None, None),
        dynamo=True, opset_version=17)
    export_s = round(time.time() - started, 2)

    sanitize = 0
    import onnx
    onnx_model = onnx.load(str(args.output_onnx))
    for node in onnx_model.graph.node:
        if node.op_type == 'Split':
            for attribute in list(node.attribute):
                if attribute.name == 'num_outputs':
                    node.attribute.remove(attribute)
                    sanitize += 1
    onnx.save(onnx_model, str(args.output_onnx))

    # ---- ONNX parity ----
    import onnxruntime as ort
    session = ort.InferenceSession(str(args.output_onnx), providers=['CPUExecutionProvider'])
    dtype_for = {'tensor(float)': np.float32, 'tensor(int64)': np.int64}

    def onnx_run(pcm: torch.Tensor, emit: int) -> np.ndarray:
        whole_session = ort.InferenceSession(str(args.decoder_dir / 'encoder_model.onnx'),
                                             providers=['CPUExecutionProvider'])

        def call_graph(window, embedded_context, context_len, conv1_tail, conv2_tail, f1, f2):
            feed = {'input_values': window.numpy(),
                    'embedded_context': embedded_context.numpy(),
                    'context_len': context_len.numpy().astype(np.int64),
                    'conv1_tail': conv1_tail.numpy(), 'conv2_tail': conv2_tail.numpy(),
                    'frames1': f1.numpy().astype(np.int64), 'frames2': f2.numpy().astype(np.int64)}
            for meta in session.get_inputs():
                feed[meta.name] = feed[meta.name].astype(dtype_for[meta.type])
            result = dict(zip(output_names, session.run(output_names, feed)))
            return (torch.from_numpy(result['hidden']),
                    torch.from_numpy(result['new_embedded_context']),
                    torch.from_numpy(np.array(result['new_context_len'])),
                    torch.from_numpy(result['new_conv1_tail']),
                    torch.from_numpy(result['new_conv2_tail']),
                    torch.from_numpy(np.array(result['new_frames1'])),
                    torch.from_numpy(np.array(result['new_frames2'])))

        def final_encoder(tail):
            feed = {'input_values': tail[None].numpy().astype(np.float32),
                    'attention_mask': np.ones((1, tail.numel()), dtype=np.int32)}
            for meta in whole_session.get_inputs():
                feed[meta.name] = feed[meta.name].astype(
                    np.float32 if meta.type == 'tensor(float)' else np.int32)
            return torch.from_numpy(
                whole_session.run(['last_hidden_state'], feed)[0])

        return chunked_drive(call_graph, final_encoder, pcm, emit).numpy()

    onnx_rows = []
    with torch.no_grad():
        for label, pcm in (('jfk', torch.from_numpy(pcm_np)), ('synthetic', torch.from_numpy(synthetic))):
            mask = torch.ones(1, pcm.numel(), dtype=torch.bool)
            whole = model.model.encoder(pcm[None], attention_mask=mask).last_hidden_state.numpy()
            for emit in (20, 64, 256):
                hidden = onnx_run(pcm, emit)[:, :whole.shape[1]]
                onnx_rows.append(dict(fixture=label, emitFrames=emit, **metrics(
                    torch.from_numpy(whole), torch.from_numpy(hidden))))

    # ---- end-to-end tokens: chunked hiddens -> existing fp32 decoder graphs ----
    bos = int(model.generation_config.decoder_start_token_id)
    eos = int(model.generation_config.eos_token_id)
    ehs = onnx_run(torch.from_numpy(pcm_np), 64)
    ehs = ehs[:, :oracle['encoder']['hiddenShape'][1]].astype(np.float32)
    prefill = ort.InferenceSession(str(args.decoder_dir / 'decoder_prefill_model.onnx'),
                                   providers=['CPUExecutionProvider'])
    step = ort.InferenceSession(str(args.decoder_dir / 'decoder_step_model.onnx'),
                                providers=['CPUExecutionProvider'])
    out_names = ['logits'] + [f'present_k_{i}' for i in range(LAYERS)] + [f'present_v_{i}' for i in range(LAYERS)]
    state_out = dict(zip(out_names, prefill.run(out_names, {
        'input_ids': np.array([[bos]], np.int64), 'encoder_hidden_states': ehs})))
    logits = state_out['logits']
    past = {f'past_k_{i}': state_out[f'present_k_{i}'] for i in range(LAYERS)}
    past.update({f'past_v_{i}': state_out[f'present_v_{i}'] for i in range(LAYERS)})
    tokens = [bos]
    for _ in range(127):
        last = int(np.argmax(logits[0, -1]))
        tokens.append(last)
        if last == eos:
            break
        feed = {'input_ids': np.array([[last]], np.int64), 'encoder_hidden_states': ehs, **past}
        state_out = dict(zip(out_names, step.run(out_names, feed)))
        logits = state_out['logits']
        past = {f'past_k_{i}': state_out[f'present_k_{i}'] for i in range(LAYERS)}
        past.update({f'past_v_{i}': state_out[f'present_v_{i}'] for i in range(LAYERS)})
    tokens_equal = tokens == oracle_tokens

    all_pass = all(r['allclose_1e_4'] for r in rows + onnx_rows) and tokens_equal
    digest_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    receipt = dict(
        status='PASS_CHUNKED_ENCODER_GRAPH' if all_pass else 'FAIL_CHUNKED_ENCODER_GRAPH',
        scope='chunked state-in/state-out streaming encoder graph, fp32; torch-module and ONNX parity vs the '
              'whole-utterance official encoder plus end-to-end oracle tokens through the fp32 decoder graphs; '
              'not fp16, not library integration',
        snapshot=dict(path=str(args.snapshot),
                      model_sha256=digest_file(args.snapshot / 'model.safetensors')),
        contract=dict(
            inputs=input_names, outputs=output_names, context=CONTEXT, lookahead=LOOKAHEAD,
            stateLayout='right-aligned tails/context, zero-fill only before stream start; conv tails sit at '
                        'the EMISSION boundary because the lookahead is re-fed and re-embedded each call',
            windowRules=['non-final call: emit_frames + 60 audio frames (emit_frames multiple of 4); '
                         'hiddens valid for the first emit_frames frames; window advances by emit_frames',
                         'final call: all remaining frames zero-padded to a multiple of 4, valid_frames = '
                         'real count, all real hiddens emitted',
                         'emission lags input by 60 frames (1.2 s at 20 ms/embedded frame)'],
            emission='hidden covers every embedded frame of the window; the caller keeps the first '
                     'emit_frames/4 (final: ceil(valid/4))'),
        onnx=dict(path=str(args.output_onnx), sha256=digest_file(args.output_onnx),
                  bytes=args.output_onnx.stat().st_size, opset=17,
                  exporter='torch dynamo', splitAttrsRemoved=sanitize),
        torchModuleParity=rows, onnxParity=onnx_rows,
        tokens=dict(equalOracle=tokens_equal, tokenIds=tokens, oracleTokenCount=len(oracle_tokens)),
        timings=dict(exportSeconds=export_s),
        limits=[
            'Full-valid audio; per-call masks handle only the final zero-padding case.',
            'The layer stack re-runs over [context ++ window] (recompute, not per-layer KV state); '
            'splitting per-layer state is a later optimization.',
            'Lookahead re-embedding recomputes <=6 embedded frames per call by design.',
            'Single fixtures; CPU provider; no speed or memory measurement.',
        ],
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'],
        worstModule=max(r['maxAbs'] for r in rows), worstOnnx=max(r['maxAbs'] for r in onnx_rows),
        tokensEqual=tokens_equal, export_s=export_s)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

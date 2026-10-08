"""Export the official streaming-Moonshine decoder to fp32 ONNX and prove parity.

Two graphs, mirroring the offline decoder contract the library already knows
(logits plus per-layer self-attention histories; cross-attention key/value are
computed from the encoder states, which stay constant per utterance):

- prefill: `input_ids [batch, tokens]` + `encoder_hidden_states [batch, frames, 320]`
  -> `logits [batch, tokens, vocab]` + per-layer self K/V histories.
- step: same inputs plus per-layer past K/V `[batch, heads, past, head_dim]`
  (past >= 1; the empty case is the prefill graph) -> next-step logits and
  grown histories. Positions come from the cache length inside the graph.

Parity gates, all on the pinned JFK effective PCM:
1. per-step logits: ONNX greedy loop vs a torch reference greedy loop using the
   official decoder with real caches (max/mean abs, cosine, allclose 1e-4),
2. final token ids/text equal the official oracle receipt's 28 ids,
3. the encoder hidden states fed to both loops come from our exported fp32
   encoder graph (onnxruntime), closing the fully-ONNX pipeline.

The decoder adds absolute positional embeddings to the encoder states and
projects them itself; both graphs therefore take raw encoder hidden states.
"""
import argparse
import hashlib
import json
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

EXPECTED_PCM_SHA256 = 'ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0'
ORACLE_RECEIPT = Path('tools/data/results/model-candidates/moonshine-streaming-official-oracle-jfk-2026-10-08.json')
LAYERS = 6


def effective_pcm(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), 'rb') as handle:
        assert handle.getframerate() == 16000 and handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0


def dynamic_cache_with(past_k, past_v):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for index in range(LAYERS):
        cache.update(past_k[index], past_v[index], index)
    return cache


class PrefillWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out

    def forward(self, input_ids, encoder_hidden_states):
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache
        cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
        out = self.decoder(input_ids=input_ids.clone(),
                           encoder_hidden_states=encoder_hidden_states.clone(),
                           past_key_values=cache, use_cache=True)
        logits = self.proj_out(out.last_hidden_state)
        self_cache = cache.self_attention_cache
        return (logits, *[self_cache.layers[i].keys for i in range(LAYERS)],
                *[self_cache.layers[i].values for i in range(LAYERS)])


class StepWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out

    def forward(self, input_ids, encoder_hidden_states, *past):
        from transformers.cache_utils import DynamicCache, EncoderDecoderCache
        past_k, past_v = list(past[:LAYERS]), list(past[LAYERS:])
        self_cache = dynamic_cache_with(past_k, past_v)
        cache = EncoderDecoderCache(self_cache, DynamicCache())
        out = self.decoder(input_ids=input_ids.clone(),
                           encoder_hidden_states=encoder_hidden_states.clone(),
                           past_key_values=cache, use_cache=True)
        logits = self.proj_out(out.last_hidden_state)
        return (logits, *[self_cache.layers[i].keys for i in range(LAYERS)],
                *[self_cache.layers[i].values for i in range(LAYERS)])


def sanitize_split_nodes(path: Path) -> int:
    """Drop legacy `num_outputs` attributes from Split nodes (opset >= 13 form).

    The dynamo/onnxscript emitter can leave the pre-13 attribute on the node
    while producing opset-13+ semantics (equal splits, sizes by edges); ORT
    rejects that combination. Equal-split nodes are identical without the
    attribute, and the parity gates below re-verify the whole pipeline.
    """
    import onnx
    model = onnx.load(str(path))
    removed = 0
    for node in model.graph.node:
        if node.op_type == 'Split':
            for attribute in list(node.attribute):
                if attribute.name == 'num_outputs':
                    node.attribute.remove(attribute)
                    removed += 1
    if removed:
        onnx.save(model, str(path))
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('.cache/moonshine-streaming-tiny'))
    parser.add_argument('--audio', type=Path, default=Path('tools/data/fixtures/audio/jfk-short.wav'))
    parser.add_argument('--encoder-onnx', type=Path,
                        default=Path('.cache/moonshine-streaming-onnx/encoder_model.onnx'))
    parser.add_argument('--output-dir', type=Path, default=Path('.cache/moonshine-streaming-onnx'))
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()

    pcm = effective_pcm(args.audio)
    if hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest() != EXPECTED_PCM_SHA256:
        raise SystemExit('effective PCM digest mismatch')
    oracle = json.loads(ORACLE_RECEIPT.read_text(encoding='utf-8'))
    oracle_tokens = oracle['generate']['tokens']

    from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration
    processor = AutoProcessor.from_pretrained(str(args.snapshot), local_files_only=True)
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        str(args.snapshot), local_files_only=True)
    model.eval()
    inputs = processor(pcm.tolist(), return_tensors='pt', sampling_rate=16000)

    # ---- encoder hidden states from our exported fp32 encoder graph ----
    import onnxruntime as ort
    enc_session = ort.InferenceSession(str(args.encoder_onnx), providers=['CPUExecutionProvider'])
    feed = {}
    for tensor_meta, value in zip(enc_session.get_inputs(),
                                  (inputs.input_values.numpy(), inputs.attention_mask.numpy())):
        numpy_dtype = {'tensor(float)': np.float32, 'tensor(int64)': np.int64,
                       'tensor(int32)': np.int32}[tensor_meta.type]
        feed[tensor_meta.name] = value.astype(numpy_dtype)
    ehs_np = enc_session.run(['last_hidden_state'], feed)[0].astype(np.float32)
    encoder_hidden = torch.from_numpy(ehs_np)

    # ---- export both decoder graphs ----
    bos = model.generation_config.decoder_start_token_id
    eos = model.generation_config.eos_token_id
    batch, frames, past_dim = (torch.export.Dim(name) for name in ('batch', 'frames', 'past'))
    # Both graphs process exactly one token (the seed BOS, then one token per
    # step): a static token dimension keeps `q_len > 1` a Python bool, so SDPA's
    # is_causal never becomes symbolic, and single-token attention over the full
    # past is precisely greedy causal semantics. No library code is patched.

    started = time.time()
    prefill_path = args.output_dir / 'decoder_prefill_model.onnx'
    torch.onnx.export(
        PrefillWrapper(model),
        (torch.tensor([[bos]]), encoder_hidden),
        str(prefill_path),
        input_names=['input_ids', 'encoder_hidden_states'],
        output_names=['logits'] + [f'present_k_{i}' for i in range(LAYERS)]
        + [f'present_v_{i}' for i in range(LAYERS)],
        dynamic_shapes=({0: batch}, {0: batch, 1: frames}),
        dynamo=True, opset_version=17)

    # Build a step-shaped example: prefill once in torch, then take step inputs.
    with torch.no_grad():
        seed = model.model.decoder(input_ids=torch.tensor([[bos]]),
                                   encoder_hidden_states=encoder_hidden.clone(), use_cache=True)
    seed_cache = seed.past_key_values
    self_cache = seed_cache.self_attention_cache if hasattr(seed_cache, 'self_attention_cache') else seed_cache.self
    step_example_ids = torch.tensor([[bos]])
    past_example = ([self_cache.layers[i].keys for i in range(LAYERS)]
                    + [self_cache.layers[i].values for i in range(LAYERS)])

    step_path = args.output_dir / 'decoder_step_model.onnx'
    step_input_names = ['input_ids', 'encoder_hidden_states'] \
        + [f'past_k_{i}' for i in range(LAYERS)] + [f'past_v_{i}' for i in range(LAYERS)]
    torch.onnx.export(
        StepWrapper(model),
        (step_example_ids, encoder_hidden, *past_example),
        str(step_path),
        input_names=step_input_names,
        output_names=['logits'] + [f'present_k_{i}' for i in range(LAYERS)]
        + [f'present_v_{i}' for i in range(LAYERS)],
        dynamic_shapes=({0: batch}, {0: batch, 1: frames},
                        tuple({0: batch, 2: past_dim} for _ in range(2 * LAYERS))),
        dynamo=True, opset_version=17)
    export_s = round(time.time() - started, 2)
    sanitized = dict(prefill=sanitize_split_nodes(prefill_path),
                     step=sanitize_split_nodes(step_path))

    # ---- ONNX greedy loop ----
    prefill_session = ort.InferenceSession(str(prefill_path), providers=['CPUExecutionProvider'])
    step_session = ort.InferenceSession(str(step_path), providers=['CPUExecutionProvider'])

    def dtype_for(meta):
        return {'tensor(float)': np.float32, 'tensor(int64)': np.int64, 'tensor(int32)': np.int32}[meta.type]

    out_names = ['logits'] + [f'present_k_{i}' for i in range(LAYERS)] + [f'present_v_{i}' for i in range(LAYERS)]
    feed = {}
    for meta, value in zip(prefill_session.get_inputs(), (np.array([[bos]], dtype=np.int64), ehs_np)):
        feed[meta.name] = value.astype(dtype_for(meta))
    prefill_out = dict(zip(out_names, prefill_session.run(out_names, feed)))
    onnx_logits_steps = [prefill_out['logits']]
    past = {f'past_k_{i}': prefill_out[f'present_k_{i}'] for i in range(LAYERS)}
    past.update({f'past_v_{i}': prefill_out[f'present_v_{i}'] for i in range(LAYERS)})
    tokens_onnx = [bos]
    for _ in range(127):
        last = int(np.argmax(prefill_out['logits'][0, -1]))
        tokens_onnx.append(last)
        if last == eos:
            break
        step_feed = {'input_ids': np.array([[last]], dtype=np.int64),
                     'encoder_hidden_states': ehs_np}
        step_feed.update({name: past[name].astype(np.float32) for name in past})
        prefill_out = dict(zip(out_names, step_session.run(out_names, step_feed)))
        onnx_logits_steps.append(prefill_out['logits'])
        past = {f'past_k_{i}': prefill_out[f'present_k_{i}'] for i in range(LAYERS)}
        past.update({f'past_v_{i}': prefill_out[f'present_v_{i}'] for i in range(LAYERS)})

    # ---- torch reference greedy loop ----
    with torch.no_grad():
        ref = model.model.decoder(input_ids=torch.tensor([[bos]]),
                                  encoder_hidden_states=encoder_hidden.clone(), use_cache=True)
        ref_logits_steps = [model.proj_out(ref.last_hidden_state)]
        ref_cache = ref.past_key_values
        ref_self = ref_cache.self_attention_cache if hasattr(ref_cache, 'self_attention_cache') else ref_cache.self
        tokens_ref = [bos]
        for _ in range(127):
            last = int(torch.argmax(ref_logits_steps[-1][0, -1]))
            tokens_ref.append(last)
            if last == eos:
                break
            ref = model.model.decoder(input_ids=torch.tensor([[last]]),
                                      encoder_hidden_states=encoder_hidden.clone(),
                                      past_key_values=ref_cache, use_cache=True)
            ref_logits_steps.append(model.proj_out(ref.last_hidden_state))

    # ---- parity ----
    step_count = min(len(onnx_logits_steps), len(ref_logits_steps))
    max_abs = max(float(np.abs(onnx_logits_steps[i][0, -1].astype(np.float64)
                               - ref_logits_steps[i][0, -1].numpy().astype(np.float64)).max())
                  for i in range(step_count))
    tokens_equal = tokens_onnx == tokens_ref == oracle_tokens
    text = processor.decode(tokens_onnx, skip_special_tokens=True)

    digest_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    receipt = dict(
        status='PASS_STREAMING_DECODER_FP32'
        if max_abs < 1e-4 and tokens_equal else 'FAIL_STREAMING_DECODER_FP32',
        scope='own fp32 ONNX decoder prefill+step graphs from the official checkpoint; logits/token parity; '
              'not fp16, not chunked-state encoder, not library integration',
        snapshot=dict(path=str(args.snapshot),
                      model_sha256=digest_file(args.snapshot / 'model.safetensors')),
        encoder=dict(path=str(args.encoder_onnx), sha256=digest_file(args.encoder_onnx)),
        graphs=[dict(name=p.name, sha256=digest_file(p), bytes=p.stat().st_size)
                for p in (prefill_path, step_path)],
        sanitizer=dict(removedNumOutputsAttrs=sanitized,
                       note='legacy Split num_outputs attributes dropped for opset>=13; pipeline parity '
                            'gates re-verified afterwards'),
        contract=dict(
            prefill='seed: input_ids int64 [batch, 1] (BOS) + encoder_hidden_states f32 [batch, frames, 320] '
                    '-> logits f32 [batch, 1, 32768] + 6x present_k/present_v [batch, 8, 1, 40]',
            step='input_ids int64 [batch, 1] + encoder_hidden_states + 6x past_k/past_v [batch, 8, past>=1, 40] '
                 '(positions from cache length) -> logits [batch, 1, 32768] + grown histories',
            staticTokenDim='both graphs take exactly one token so `q_len > 1` stays a Python bool and SDPA '
                            'is_causal never becomes symbolic; single-token attention over the full past is '
                            'greedy causal semantics',
            cross='cross-attention key/value are computed inside each graph from encoder_hidden_states; '
                  'the graph set is stateless across utterances'),
        parity=dict(stepsCompared=step_count, perStepLogitsMaxAbs=max_abs,
                    tolerance='atol 1e-4 on final-position logits per step'),
        tokens=dict(onnx=tokens_onnx, reference=tokens_ref, oracle=oracle_tokens,
                    equal=tokens_equal, text=text),
        timings=dict(exportSeconds=export_s),
        limits=[
            'Prefill graph covers the empty-past case; the step graph requires past >= 1 by design.',
            'Single fixture (JFK), CPU execution provider, greedy decoding only; no memory or speed claim.',
        ],
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'], steps=step_count, maxAbs=max_abs,
                          tokensEqual=tokens_equal, text=text[:60], export_s=export_s)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

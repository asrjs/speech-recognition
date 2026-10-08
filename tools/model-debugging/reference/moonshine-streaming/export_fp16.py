"""Export the streaming-Moonshine graph set in fp16 from the official checkpoint.

Follows the repo's measured precedent: ORT's float16 converter broke the Qwen
graphs, so fp16 comes from a direct torch export (model loaded with
dtype=float16) rather than post-conversion. The graph contracts match the
validated fp32 set exactly (encoder: waveform + INT32 mask; decoder seed/step:
one token, per-layer past K/V, in-graph cross attention, positions from cache
length); the same Split sanitizer is applied.

Gates, all on the pinned JFK effective PCM:
1. the fully-fp16 pipeline (encoder -> seed -> step greedy loop) must reproduce
   the official oracle's exact 28 token ids and canonical text,
2. numerical drift vs the validated fp32 graphs is recorded for the encoder
   hidden states and every per-step final-position logit vector (max/mean abs,
   cosine) — evidence, with token equality as the acceptance gate per the
   fp16-after-fp32 parity rule.

If token equality fails here, the controlled fallback is mixed precision
(fp32 norms/softmax), which this script does not silently do.
"""
import argparse
import hashlib
import importlib.util
import json
import time
import wave
from pathlib import Path

import numpy as np
import torch

EXPECTED_PCM_SHA256 = 'ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0'
ORACLE_RECEIPT = Path('tools/data/results/model-candidates/moonshine-streaming-official-oracle-jfk-2026-10-08.json')
HERE = Path(__file__).parent
LAYERS = 6


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def effective_pcm(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), 'rb') as handle:
        assert handle.getframerate() == 16000 and handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('.cache/moonshine-streaming-tiny'))
    parser.add_argument('--audio', type=Path, default=Path('tools/data/fixtures/audio/jfk-short.wav'))
    parser.add_argument('--fp32-dir', type=Path, default=Path('.cache/moonshine-streaming-onnx'))
    parser.add_argument('--output-dir', type=Path, default=Path('.cache/moonshine-streaming-onnx-fp16'))
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()

    encoder_module = load_module('ms_encoder', HERE / 'export_streaming_encoder.py')
    decoder_module = load_module('ms_decoder', HERE / 'export_streaming_decoder.py')

    pcm = effective_pcm(args.audio)
    if hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest() != EXPECTED_PCM_SHA256:
        raise SystemExit('effective PCM digest mismatch')
    oracle = json.loads(ORACLE_RECEIPT.read_text(encoding='utf-8'))
    oracle_tokens = oracle['generate']['tokens']

    from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration
    processor = AutoProcessor.from_pretrained(str(args.snapshot), local_files_only=True)
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        str(args.snapshot), local_files_only=True, dtype=torch.float16)
    model.eval()
    inputs = processor(pcm.tolist(), return_tensors='pt', sampling_rate=16000)
    input_values = inputs.input_values.to(torch.float16)
    attention_mask = inputs.attention_mask

    bos = model.generation_config.decoder_start_token_id
    eos = model.generation_config.eos_token_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch, frames, past_dim = (torch.export.Dim(name) for name in ('batch', 'frames', 'past'))

    started = time.time()
    encoder_path = args.output_dir / 'encoder_model.onnx'
    torch.onnx.export(
        encoder_module.EncoderWrapper(model), (input_values, attention_mask), str(encoder_path),
        input_names=['input_values', 'attention_mask'], output_names=['last_hidden_state'],
        dynamic_shapes=({0: batch, 1: None}, {0: batch, 1: None}),
        dynamo=True, opset_version=17)

    with torch.no_grad():
        seed = model.model.decoder(input_ids=torch.tensor([[bos]]),
                                   encoder_hidden_states=torch.zeros(
                                       1, 8, model.config.hidden_size, dtype=torch.float16),
                                   use_cache=True)
    seed_self = seed.past_key_values.self_attention_cache
    past_example = ([seed_self.layers[i].keys for i in range(LAYERS)]
                    + [seed_self.layers[i].values for i in range(LAYERS)])

    ehs_example = torch.zeros(1, 64, model.config.hidden_size, dtype=torch.float16)
    prefill_path = args.output_dir / 'decoder_prefill_model.onnx'
    torch.onnx.export(
        decoder_module.PrefillWrapper(model), (torch.tensor([[bos]]), ehs_example),
        str(prefill_path),
        input_names=['input_ids', 'encoder_hidden_states'],
        output_names=['logits'] + [f'present_k_{i}' for i in range(LAYERS)]
        + [f'present_v_{i}' for i in range(LAYERS)],
        dynamic_shapes=({0: batch}, {0: batch, 1: frames}),
        dynamo=True, opset_version=17)

    step_path = args.output_dir / 'decoder_step_model.onnx'
    step_input_names = ['input_ids', 'encoder_hidden_states'] \
        + [f'past_k_{i}' for i in range(LAYERS)] + [f'past_v_{i}' for i in range(LAYERS)]
    example_ehs = ehs_example
    torch.onnx.export(
        decoder_module.StepWrapper(model), (torch.tensor([[bos]]), example_ehs, *past_example),
        str(step_path),
        input_names=step_input_names,
        output_names=['logits'] + [f'present_k_{i}' for i in range(LAYERS)]
        + [f'present_v_{i}' for i in range(LAYERS)],
        dynamic_shapes=({0: batch}, {0: batch, 1: frames},
                        tuple({0: batch, 2: past_dim} for _ in range(2 * LAYERS))),
        dynamo=True, opset_version=17)
    export_s = round(time.time() - started, 2)

    sanitized = {p.name: decoder_module.sanitize_split_nodes(p)
                 for p in (encoder_path, prefill_path, step_path)}

    # ---- sessions ----
    import onnxruntime as ort
    def session_for(path):
        return ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])

    def dtype_map(meta):
        return {'tensor(float)': np.float32, 'tensor(float16)': np.float32,  # run fp16 IO as float32 views
                'tensor(int64)': np.int64, 'tensor(int32)': np.int32}[meta.type]

    def feed_for(session, values):
        feed = {}
        for meta, value in zip(session.get_inputs(), values):
            array = value if isinstance(value, np.ndarray) else value.numpy()
            if meta.type == 'tensor(float16)':
                feed[meta.name] = array.astype(np.float16)
            else:
                feed[meta.name] = array.astype(dtype_map(meta))
        return feed

    fp16_encoder = session_for(encoder_path)
    fp32_encoder = session_for(args.fp32_dir / 'encoder_model.onnx')
    enc_feed_names = [m.name for m in fp16_encoder.get_inputs()]
    base_feed = [input_values.numpy(), attention_mask.numpy()]
    fp16_ehs = fp16_encoder.run(['last_hidden_state'], feed_for(fp16_encoder, base_feed))[0]
    fp32_ehs = fp32_encoder.run(['last_hidden_state'],
                                {m.name: v.astype(np.float32 if m.type == 'tensor(float)' else
                                                  np.int64 if m.type == 'tensor(int64)' else np.int32)
                                 for m, v in zip(fp32_encoder.get_inputs(), base_feed)})[0]

    def hidden_metrics(a, b):
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        diff = np.abs(a64 - b64)
        norm = np.linalg.norm(a64) * np.linalg.norm(b64)
        return dict(maxAbs=float(diff.max()), meanAbs=float(diff.mean()),
                    cosine=float(np.dot(a64.ravel(), b64.ravel()) / norm) if norm else float('nan'))

    encoder_metrics = hidden_metrics(fp16_ehs, fp32_ehs)

    # ---- fully-fp16 greedy pipeline ----
    prefill_session = session_for(prefill_path)
    step_session = session_for(step_path)
    out_names = ['logits'] + [f'present_k_{i}' for i in range(LAYERS)] + [f'present_v_{i}' for i in range(LAYERS)]
    prefill_out = dict(zip(out_names, prefill_session.run(
        out_names, feed_for(prefill_session, [np.array([[bos]], dtype=np.int64), fp16_ehs]))))
    logits_steps = [prefill_out['logits'].astype(np.float32)]
    past = {f'past_k_{i}': prefill_out[f'present_k_{i}'] for i in range(LAYERS)}
    past.update({f'past_v_{i}': prefill_out[f'present_v_{i}'] for i in range(LAYERS)})
    tokens = [bos]
    for _ in range(127):
        last = int(np.argmax(logits_steps[-1][0, -1]))
        tokens.append(last)
        if last == eos:
            break
        step_feed = {'input_ids': np.array([[last]], dtype=np.int64), 'encoder_hidden_states': fp16_ehs}
        step_feed.update({name: past[name] for name in past})
        prefill_out = dict(zip(out_names, step_session.run(out_names, feed_for(step_session, [
            step_feed['input_ids'], step_feed['encoder_hidden_states'],
            *[step_feed[f'past_k_{i}'] for i in range(LAYERS)],
            *[step_feed[f'past_v_{i}'] for i in range(LAYERS)]]))))
        logits_steps.append(prefill_out['logits'].astype(np.float32))
        past = {f'past_k_{i}': prefill_out[f'present_k_{i}'] for i in range(LAYERS)}
        past.update({f'past_v_{i}': prefill_out[f'present_v_{i}'] for i in range(LAYERS)})

    # ---- fp16 vs fp32 per-step logits via a parallel fp32 loop ----
    fp32_prefill = session_for(args.fp32_dir / 'decoder_prefill_model.onnx')
    fp32_step = session_for(args.fp32_dir / 'decoder_step_model.onnx')
    fp32_out = dict(zip(out_names, fp32_prefill.run(
        out_names, feed_for(fp32_prefill, [np.array([[bos]], dtype=np.int64), fp32_ehs]))))
    fp32_logits_steps = [fp32_out['logits'].astype(np.float32)]
    fp32_past = {f'past_k_{i}': fp32_out[f'present_k_{i}'] for i in range(LAYERS)}
    fp32_past.update({f'past_v_{i}': fp32_out[f'present_v_{i}'] for i in range(LAYERS)})
    fp32_tokens = [bos]
    for _ in range(127):
        last = int(np.argmax(fp32_logits_steps[-1][0, -1]))
        fp32_tokens.append(last)
        if last == eos:
            break
        fp32_out = dict(zip(out_names, fp32_step.run(out_names, feed_for(fp32_step, [
            np.array([[last]], dtype=np.int64), fp32_ehs,
            *[fp32_past[f'past_k_{i}'] for i in range(LAYERS)],
            *[fp32_past[f'past_v_{i}'] for i in range(LAYERS)]]))))
        fp32_logits_steps.append(fp32_out['logits'].astype(np.float32))
        fp32_past = {f'past_k_{i}': fp32_out[f'present_k_{i}'] for i in range(LAYERS)}
        fp32_past.update({f'past_v_{i}': fp32_out[f'present_v_{i}'] for i in range(LAYERS)})

    steps = min(len(logits_steps), len(fp32_logits_steps))
    logits_metrics = hidden_metrics(
        np.concatenate([s[0, -1] for s in logits_steps[:steps]]),
        np.concatenate([s[0, -1] for s in fp32_logits_steps[:steps]]))

    tokens_equal = tokens == oracle_tokens
    text = processor.decode(tokens, skip_special_tokens=True)
    tokens_equal_fp32 = tokens == fp32_tokens

    digest_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    graphs = []
    for path in (encoder_path, prefill_path, step_path):
        row = dict(name=path.name, sha256=digest_file(path), bytes=path.stat().st_size)
        sidecar = Path(str(path) + '.data')
        if sidecar.is_file():
            row['externalDataSha256'] = digest_file(sidecar)
            row['externalDataBytes'] = sidecar.stat().st_size
        graphs.append(row)

    receipt = dict(
        status='PASS_STREAMING_FP16' if tokens_equal and tokens_equal_fp32 else 'FAIL_STREAMING_FP16',
        scope='direct fp16 torch export of the streaming graph set; oracle token gate plus fp32-drift evidence; '
              'not browser WebGPU measurement, not mixed-precision, not library integration',
        snapshot=dict(path=str(args.snapshot),
                      model_sha256=digest_file(args.snapshot / 'model.safetensors')),
        policy='fp16 comes from direct torch export (dtype=float16), not ORT post-conversion, per the measured '
               'Qwen converter failure; every op including norms/softmax is fp16 in this pass — if tokens had '
               'diverged, the controlled fallback is fp32-sensitive-op mixed precision, not silently applied',
        graphs=graphs, sanitizer=dict(removedNumOutputsAttrs=sanitized),
        parity=dict(encoderHidden=encoder_metrics, perStepLogits=logits_metrics, stepsCompared=steps),
        tokens=dict(fp16=tokens, fp32=fp32_tokens, oracle=oracle_tokens,
                    equalOracle=tokens_equal, equalFp32=tokens_equal_fp32, text=text),
        timings=dict(exportSeconds=export_s),
        limits=[
            'CPU execution provider; fp16 graph IO accepted here as float16 tensors; no WebGPU run, no '
            'speed or memory measurement, no browser claim.',
            'Single fixture (JFK), greedy decoding only.',
        ],
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'], equalOracle=tokens_equal, equalFp32=tokens_equal_fp32,
                          encoderMaxAbs=encoder_metrics['maxAbs'], logitsMaxAbs=logits_metrics['maxAbs'],
                          steps=steps, export_s=export_s)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

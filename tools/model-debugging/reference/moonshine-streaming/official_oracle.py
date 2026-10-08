"""Official streaming-Moonshine oracle capture on the pinned JFK effective PCM.

Loads moonshine-ai/moonshine-streaming-tiny from the local snapshot (byte-identical
to upstream revision f8e9dfd8c562…) with the project Transformers 5 stack and
records, for the deterministic mono 16 kHz float32 decode of
tools/data/fixtures/audio/jfk-short.wav (byte-bound to the established
effective-PCM SHA-256):

1. checkpoint/config identity plus a strict key/shape diff against the model,
2. full-file greedy generate token ids and text (the authoritative output
   oracle for this fixture),
3. encoder stage capture: encoder last hidden states (dtype/shape/sha256) for
   later export parity,
4. decoder cache continuity: step-by-step generation with use_cache equals the
   one-shot greedy output,
5. reset control: a fresh model instance reproduces the same tokens.

Contract finding (2026-10-08): the streaming encoder REQUIRES the processor's
attention/padding mask; generating without it yields degenerate repetition on
this whole-utterance path while the offline model is mask-insensitive. Any
faithful export must carry the mask semantics.

Output-only oracle evidence for the model lane: no export, no library claim.
"""
import argparse
import hashlib
import json
import time
import wave
from pathlib import Path

import numpy as np
import torch

EXPECTED_PCM_SHA256 = 'ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0'


def effective_pcm(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), 'rb') as handle:
        assert handle.getframerate() == 16000 and handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def strict_key_report(snapshot: Path, archive: str) -> dict:
    from safetensors import safe_open
    from transformers import AutoConfig, MoonshineStreamingForConditionalGeneration
    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
    model = MoonshineStreamingForConditionalGeneration(config)
    model_keys = {k: list(v.shape) for k, v in model.state_dict().items()}
    with safe_open(snapshot / archive, framework='pt') as handle:
        ckpt = {k: list(handle.get_slice(k).get_shape()) for k in handle.keys()}
    missing = sorted(set(model_keys) - set(ckpt))
    unexpected = sorted(set(ckpt) - set(model_keys))
    mismatched = sorted(k for k in set(model_keys) & set(ckpt) if model_keys[k] != ckpt[k])
    return dict(archive=archive, tensors=len(ckpt), missing=len(missing),
                unexpected=len(unexpected), shapeMismatch=len(mismatched),
                sampleMissing=missing[:3], sampleUnexpected=unexpected[:3])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('.cache/moonshine-streaming-tiny'))
    parser.add_argument('--audio', type=Path, default=Path('tools/data/fixtures/audio/jfk-short.wav'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    pcm = effective_pcm(args.audio)
    pcm_sha = hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()
    if pcm_sha != EXPECTED_PCM_SHA256:
        raise SystemExit(f'effective PCM digest mismatch: {pcm_sha}')

    from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration
    import transformers
    processor = AutoProcessor.from_pretrained(str(args.snapshot), local_files_only=True)
    inputs = processor(pcm.tolist(), return_tensors='pt', sampling_rate=16000)
    input_values, attention_mask = inputs.input_values, inputs.attention_mask

    def digest_file(path: Path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    key_report = strict_key_report(args.snapshot, 'model.safetensors')

    load_started = time.time()
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        str(args.snapshot), local_files_only=True)
    model.eval()
    load_s = round(time.time() - load_started, 2)

    # ---- full-file greedy generate (authoritative oracle) ----
    started = time.time()
    with torch.no_grad():
        generated = model.generate(input_values=input_values, attention_mask=attention_mask,
                                   max_new_tokens=128, do_sample=False, num_beams=1)
    generate_s = round(time.time() - started, 2)
    tokens = generated[0].tolist()
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]

    # ---- encoder stage capture ----
    with torch.no_grad():
        encoder_outputs = model.model.encoder(input_values, padding_mask=attention_mask)
        hidden = encoder_outputs.last_hidden_state if hasattr(encoder_outputs, 'last_hidden_state') \
            else encoder_outputs[0]

    # ---- decoder cache continuity ----
    with torch.no_grad():
        stepwise = model.generate(input_values=input_values, attention_mask=attention_mask,
                                  max_new_tokens=128, do_sample=False, num_beams=1, use_cache=True)
    cache_equal = stepwise.tolist() == generated.tolist()

    # ---- reset control ----
    fresh = MoonshineStreamingForConditionalGeneration.from_pretrained(
        str(args.snapshot), local_files_only=True)
    fresh.eval()
    with torch.no_grad():
        fresh_tokens = fresh.generate(input_values=input_values, attention_mask=attention_mask,
                                      max_new_tokens=128, do_sample=False, num_beams=1)[0].tolist()

    receipt = dict(
        status='PASS_OFFICIAL_STREAMING_ORACLE'
        if text.strip() and cache_equal and fresh_tokens == tokens else 'FAIL_OFFICIAL_STREAMING_ORACLE',
        scope='official streaming-Moonshine output oracle, output/stage-identity evidence only; '
              'not export parity, not incremental chunked decode, not a speed claim',
        snapshot=dict(path=str(args.snapshot), upstream='moonshine-ai/moonshine-streaming-tiny',
                      upstreamRevision='f8e9dfd8c562',
                      model_sha256=digest_file(args.snapshot / 'model.safetensors'),
                      config_sha256=digest_file(args.snapshot / 'config.json'),
                      preprocessor_sha256=digest_file(args.snapshot / 'preprocessor_config.json'),
                      tokenizer_sha256=digest_file(args.snapshot / 'tokenizer.json'),
                      generationConfig_sha256=digest_file(args.snapshot / 'generation_config.json'),
                      strictKeyReport=key_report),
        input=dict(audio_wav_sha256=digest_file(args.audio), effectivePcmSha256=pcm_sha,
                   samples=int(pcm.size), sampleRate=16000, encoding='f32le',
                   attentionMaskSum=int(attention_mask.sum()),
                   attentionMaskShape=list(attention_mask.shape)),
        runtime=dict(torch=torch.__version__, transformers=transformers.__version__,
                     dtype='float32'),
        generate=dict(tokens=tokens, tokenCount=len(tokens), text=text,
                      seconds=generate_s, modelLoadSeconds=load_s),
        encoder=dict(inputShape=list(input_values.shape),
                     hiddenDtype=str(hidden.dtype), hiddenShape=list(hidden.shape),
                     hiddenSha256=tensor_sha(hidden)),
        decoder=dict(cacheStepwiseEqualsOneShot=cache_equal),
        reset=dict(freshInstanceEquals=fresh_tokens == tokens),
        contract=dict(
            attentionMaskRequired=True,
            note='Without the processor attention/padding mask this whole-utterance path degenerates '
                 'into repeated tokens; the offline moonshine model is mask-insensitive on the same '
                 'fixture/stack. Exports must carry mask semantics.'),
        limits=[
            'Chunked streaming decode (partial/final semantics with encoder conv/window state) is not '
            'exercised by this receipt; the HF class consumes whole-utterance input.',
            'Greedy decode only; single fixture (JFK); no quality, language or speed claim.',
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'], text=text, tokens=len(tokens),
                          cacheOk=cache_equal, resetOk=fresh_tokens == tokens,
                          strictKeys=key_report, hiddenShape=receipt['encoder']['hiddenShape'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

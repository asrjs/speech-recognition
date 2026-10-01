# LibriVox Domain Parity Playbook

This playbook documents the debugging workflow for the `LibriVox.org` domain
split issue:

- incorrect browser output looked like `LibriVox. org.`
- deterministic Node output produced the correct `LibriVox.org`

## What We Learned

The issue was not explained by the TDT decode loop alone.

The strongest signal came from comparing:

1. original 22.05 kHz WAV
2. pre-resampled 16 kHz WAV
3. browser-side audio preparation
4. deterministic Node WAV loading + linear resampling

`@asrjs/speech-recognition` in Node produced the correct domain string for both:

- [asrjs-librivox-node-direct.json](N:\github\asrjs\speech-recognition\tools\data\results\nemo-tdt\asrjs-librivox-node-direct.json)
- [asrjs-librivox-node-direct-16k.json](N:\github\asrjs\speech-recognition\tools\data\results\nemo-tdt\asrjs-librivox-node-direct-16k.json)

That narrowed the remaining suspect to browser audio preparation.

## Workflow

### 1. Reproduce with the original WAV

Fixture:

- [librivox.org.wav](N:\github\asrjs\speech-recognition\tools\data\fixtures\audio\librivox.org.wav)

### 2. Reproduce with the shared 16 kHz WAV

Fixtures:

- [librivox.org.16k.wav](N:\github\asrjs\speech-recognition\tools\data\fixtures\audio\librivox.org.16k.wav)
- [librivox.org.16k.pcm16.wav](N:\github\asrjs\speech-recognition\tools\data\fixtures\audio\librivox.org.16k.pcm16.wav)

### 3. Run the `@asrjs/speech-recognition` Node inspector

Script:

- [node-asrjs-nemo-inspect.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-asrjs-nemo-inspect.mjs)

Example:

```powershell
node tools/model-debugging/scripts/node-asrjs-nemo-inspect.mjs `
  --audio tools/data/fixtures/audio/librivox.org.wav `
  --model-dir N:\models\onnx\nemo\parakeet-tdt-0.6b-v2-onnx `
  --output tools/data/results/nemo-tdt/asrjs-librivox-node-direct.json
```

### 4. Compare against native/reference scripts

Useful references:

- [node-nemo-shared-audio-parity.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\hf-parakeet-port\node-nemo-shared-audio-parity.mjs)
- [python-nemo-inspect.py](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\hf-parakeet-port\python-nemo-inspect.py)
- [python-onnx-asr-inspect.py](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\hf-parakeet-port\python-onnx-asr-inspect.py)
- [python-resample-wav.py](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\hf-parakeet-port\python-resample-wav.py)

### 5. If Node is correct but browser is wrong

Check the audio prep path before touching model code:

- WAV parser vs `AudioContext`
- native-rate decode vs target-rate `AudioContext`
- linear resampler vs browser resampler behavior

## Recent Fixes Triggered By This Investigation

This debugging pass directly led to:

- deterministic WAV parsing in [media.ts](N:\github\asrjs\speech-recognition\src\runtime\media.ts) for the native-rate browser path
- Node support for local `file://` tokenizer vocab loading in [tokenizer.ts](N:\github\asrjs\speech-recognition\src\models\nemo-tdt\tokenizer.ts)
- Node-safe ORT WASM path setup and file-path normalization in [ort.ts](N:\github\asrjs\speech-recognition\src\models\nemo-tdt\ort.ts)
- Node-safe local preprocessor model loading in [preprocessor.ts](N:\github\asrjs\speech-recognition\src\models\nemo-tdt\preprocessor.ts)

## Rule To Reuse

When transcript quality looks suspicious:

1. validate with deterministic Node audio prep
2. validate with pre-resampled shared audio
3. only then blame model execution or decoding

## 2026-10-01: Orukeet same-input control

The original 22.05 kHz speech/silence fixture produced 69 official tokens versus
76 library tokens when the inspector used linear resampling. Official NeMo file
inference uses Lhotse's windowed-sinc preparation; the same raw WAV hash did not
establish the same effective input. Preserve the original failed receipt rather
than attributing it to the decoder or removing the fixture.

The official preprocessor-input capture now records 649,600 mono float32 samples
at 16 kHz, valid length 649,600, no right padding, f32le SHA-256
`8f4293517daf07f433b307a4a971f7c17dc9ed348b5f72999945b217331ec3df`.
Instrumentation left all six output controls unchanged. The public library in
native Node and native Bun, using `--input-mode oracle-pcm`, matched the complete
69-token sequence, frames, TDT durations and text, with successful awaited
cleanup. Receipts under `tools/data/results/orukeet/`:

- `official-speech-silence-waveform-summary-2026-10-01.json`
- `library-node-native-fp32-speech-silence-same-pcm-2026-10-01.json`
- `library-bun-native-fp32-speech-silence-same-pcm-2026-10-01.json`

These remain `PARTIAL_PASS_OUTPUT_ONLY`. They isolate input preparation for this
fixture; they do not pass feature numerics, prove a production resampling fix,
or certify Orukeet WebGPU. The existing inspector rejects mismatched effective
PCM metadata/hash and unsupported padded captures; legacy input captures remain
UNVERIFIED. Keep ordinary source-audio/resampling tests as a separate lane.

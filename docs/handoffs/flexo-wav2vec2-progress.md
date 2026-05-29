# Flexo — WAV2VEC2 CTC Model Progress

Branch: `feat/asr-pipeline-output-formats`
Last updated: 2026-05-31
Author: Flexo (gpt-5.5, home P520)

## Current Status

WAV2VEC2 base-960h is now wired as a real model family and preset, and the Node/WASM smoke path has been validated against a real ONNX model.

Completed:
- Shared `src/ctc/` module for CTC argmax/collapse/timing/word building.
- `src/models/wav2vec2/` core model implementation.
- `src/models/wav2vec2/model.ts` model factory.
- `src/models/wav2vec2/mapping.ts` native→canonical transcript mapping.
- `src/presets/wav2vec2/` manifest + preset factory.
- Built-in runtime registration and descriptor catalog entry.
- Real ONNX Node/WASM smoke script and run.

## Commit History (WAV2VEC2 + CTC related)

```text
b25e9a6  feat: wire context conditioning (contains Wav2Vec2 model factory/preset files + smoke script)
64c1cea  feat: add formatTranscript() (contains Wav2Vec2 ORT external data, built-in registration, descriptor wiring)
be7f1c9  refactor: extract shared CTC module, migrate wav2vec2 + lasr-ctc executors
94ceb99  feat: add WAV2VEC2 CTC model family — types, config, tokenizer, ORT glue, executor
```

## Architecture

```text
Raw Float32 PCM @ 16 kHz
  → Wav2Vec2 single ONNX graph
  → logits [batch, frames, vocab]
  → src/ctc CtcDecoder helpers
     - argmaxAndSelectedLogProbs()
     - ctcCollapseWithSpans()
     - addTimesToTokenSpans()
     - buildUtteranceTiming()
     - buildSentenceTimings()
     - buildWordsFromCharSpans()
  → Wav2Vec2NativeTranscript
  → mapWav2Vec2NativeToCanonical()
  → TranscriptResult
```

Important boundary: Wav2Vec2 is raw-waveform, not mel-based. The convolutional feature extractor is inside the ONNX graph.

## Implemented Files

### Model family

| File | Purpose |
|------|---------|
| `src/models/wav2vec2/types.ts` | Config, artifact sources, transcript options, native output, executor deps |
| `src/models/wav2vec2/config.ts` | Default and parsed Wav2Vec2 config |
| `src/models/wav2vec2/tokenizer.ts` | Character-level CTC tokenizer (`|` → space) |
| `src/models/wav2vec2/ort.ts` | ORT Web session creation, HF/direct artifact resolution, external data support |
| `src/models/wav2vec2/executor.ts` | Real ONNX inference → shared CTC decode → native transcript |
| `src/models/wav2vec2/model.ts` | `createWav2Vec2ModelFamily()` + `SpeechModel`/`SpeechSession` wrapper |
| `src/models/wav2vec2/mapping.ts` | Native Wav2Vec2 transcript → canonical transcript |
| `src/models/wav2vec2/index.ts` | Barrel export |
| `src/models/wav2vec2.ts` | Package subpath shim |

### Presets and built-ins

| File | Purpose |
|------|---------|
| `src/presets/wav2vec2/manifest.ts` | `facebook/wav2vec2-base-960h` preset manifest + aliases |
| `src/presets/wav2vec2/factory.ts` | `createWav2Vec2PresetFactory()` |
| `src/presets/wav2vec2/index.ts` | Preset barrel export |
| `src/presets/wav2vec2.ts` | Package subpath shim |
| `src/runtime/builtins.ts` | Registers Wav2Vec2 model family + preset |
| `src/presets/descriptors.ts` | Built-in catalog descriptor for Wav2Vec2 CTC |
| `src/presets/index.ts` | Exports Wav2Vec2 preset helpers |

### Tests and smoke

| File | Purpose |
|------|---------|
| `tests/wav2vec2-model.test.ts` | Family support, stub fallback, preset resolution, built-in registration |
| `tests/preset-descriptors.test.ts` | Wav2Vec2 descriptor/catalog metadata |
| `tests/smoke/wav2vec2-node-wasm-smoke.mjs` | Real ONNX Node/WASM smoke script |

## ONNX Model Details

Local smoke artifact:

```text
/tmp/wav2vec2-base-960h.onnx
/tmp/wav2vec2-base-960h.onnx.data
```

Specs:
- Source model: `facebook/wav2vec2-base-960h`
- Input: `input_values` `[1, samples]` float32 raw waveform
- Output: `logits` `[1, frames, 32]` float32
- Output stride: 320 samples (`16000 / 320 ≈ 50 fps`)
- CTC blank token ID: 0 (`<pad>`)
- Vocabulary: 32 char-level tokens, `|` as word separator
- Local ONNX graph uses external data; Node/WASM must pass `externalData` explicitly.

Critical smoke-source pattern:

```js
source: {
  kind: 'direct',
  artifacts: {
    modelUrl: pathToFileURL('/tmp/wav2vec2-base-960h.onnx').href,
    modelDataUrl: pathToFileURL('/tmp/wav2vec2-base-960h.onnx.data').href,
    modelDataFilename: 'wav2vec2-base-960h.onnx.data',
    tokenizerUrl: 'https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json',
  },
  cpuThreads: 1,
}
```

## Validated Smoke Output

Command:

```bash
node tests/smoke/wav2vec2-node-wasm-smoke.mjs --expect country --expect ask
```

Observed output:

```text
wav2vec2 node/wasm smoke passed
model=/tmp/wav2vec2-base-960h.onnx
audio=/home/steam/github/asrjs/speech-recognition/tests/fixtures/jfk2.en.wav
sampleRate=16000 duration=11.000s elapsed≈8.6s
words=22 tokens=105
and so my fellow americans ask not what your country can do for you ask what you can do for your country
```

## Verification

Latest gate:

```bash
npx vitest run tests/wav2vec2-model.test.ts tests/preset-descriptors.test.ts tests/exports.test.ts
npm run typecheck
npm run lint
npm test
npm run build
node tests/smoke/wav2vec2-node-wasm-smoke.mjs --expect country --expect ask
```

Results:
- Focused Wav2Vec2/preset/export tests: 16 passed
- Typecheck: clean
- Lint: 0 errors, 5 existing max-lines warnings
- Full tests: 100 files, 568 tests passed
- Build: clean
- Node/WASM Wav2Vec2 smoke: passed

## Design Decisions

- Browser/runtime default for Wav2Vec2 descriptor is WASM first. WebGPU is listed as available but not validated here.
- Preset manifest currently does not force a hub ONNX source; local smoke uses an explicit direct source. This avoids pretending the asrjs-owned Wav2Vec2 ONNX repo is published before it is.
- Stub fallback remains only for tests/no-source development. Real inference activates whenever `options.source` is provided.
- Backward compatibility is no longer a project constraint before release. The `lasr-ctc/ctc.ts` wrapper can be deleted when MedASR is rewritten.

## Remaining Work

1. Publish/host Wav2Vec2 ONNX artifact if we want `useManifestSources: true` to load without local direct paths.
2. Implement CTC Viterbi forced alignment in `src/alignment/ctc-viterbi.ts`.
3. Add Wav2Vec2 aligner wrapper in `src/alignment/wav2vec2-aligner.ts`.
4. Add alignment fixture tests.
5. Optional: add an npm script for `tests/smoke/wav2vec2-node-wasm-smoke.mjs` if recurring.
6. Remove MedASR backward-compat wrappers when rewriting MedASR.

## Resume Instructions

Next useful task: start Phase D CTC Viterbi alignment.

Start by writing failing tests for a tiny synthetic CTC lattice, then implement a model-agnostic aligner under `src/alignment/` or `src/ctc/` depending on whether it is generic CTC path logic or transcript-alignment-specific orchestration.

# ASR Pipeline Roadmap Implementation Plan

> For Hermes/Flexo: this is the durable handoff for `asrjs/speech-recognition` branch `feat/asr-pipeline-output-formats`. Use `asrjs-dev` + `test-driven-development` before changing code. Keep implementation modular, reusable, and pluggable.

Goal: build a framework-neutral ASR library pipeline that can combine model decoding, VAD, windowing, sentence/word normalization, sidecar output generation, subtitle formatting, and future Whisper stages without hardcoding one monolithic flow.

Architecture: low-level model sessions stay direct and family-specific. High-level pipeline stages are reusable building blocks in `src/pipeline/`, operating on `PipelineContext` and returning partial updates (`transcript`, `sidecars`, `options`). Model limits and strategies remain catalog-driven.

Tech stack: TypeScript, Vitest, ONNX Runtime Web/Node, browser + Node targets, WebGPU/WASM backends.

---

## Current branch state

Branch: `feat/asr-pipeline-output-formats`

Relevant commits:
- `ef3709d feat: add catalog-driven ASR windowing pipeline`
- `a3a777c feat: normalize ASR output detail and segment mapping`
- `bc741d5 feat: add sentence and subtitle transcript outputs`
- `824e8a4 feat: add pluggable transcript pipeline stages`

Verification status at `824e8a4`:
- `npm run typecheck` passed
- `npm run lint` passed with 3 pre-existing max-lines warnings
- `npm test` passed: 56 files, 292 tests
- `npm run build` passed
- `node tests/smoke/offline-output-smoke.mjs` passed

Known working tree noise:
- `.serena/` is untracked agent state. Leave it alone.

Git LFS:
- `git-lfs` is installed on Flexo. Normal `git push` works; no `--no-verify` needed.

---

## Implemented building blocks

### Transcript/detail/output surface

Files:
- `src/types/transcript.ts`
- `src/types/runtime.ts`
- `src/pipeline/output-options.ts`
- `src/models/nemo-common/mapping.ts`

Implemented:
- `TranscriptSentence`
- `SubtitleCue`
- `TranscriptResult.sentences`
- `TranscriptMeta.sentenceCount`
- detail levels: `text`, `segments`, `sentences`, `words`, `sentences+words`, `detailed`
- `returnTimestamps: 'sentence' | 'sentences'`
- output format option placeholders: `json`, `srt`, `vtt`

### Sentence and subtitle utilities

Files:
- `src/pipeline/sentence-segmenter.ts`
- `src/pipeline/subtitles.ts`
- `tests/transcript-output-formats.test.ts`
- `tests/smoke/offline-output-smoke.mjs`

Implemented:
- `partitionWordsIntoSentences`
- legacy-compatible `partitionWordsIntoSegments`
- `formatSubtitleTimestamp`
- `transcriptToSrt`
- `transcriptToVtt`
- `transcriptToSubtitleCues`
- `cuesToSrt`
- `cuesToVtt`

### Pipeline composition API

Files:
- `src/pipeline/composition.ts`
- `src/pipeline/index.ts`
- `tests/pipeline-composition.test.ts`

Implemented:
- `createPipelineContext`
- `runPipelineStages`
- `createSubtitleSidecarStage`
- `PipelineContext`
- `PipelineStage`
- `PipelineStageResult`
- `PipelineSidecars`
- `PipelineStageError`
- `PipelineAbortedError`

Design rules:
- stages are reusable side-effect-light modules
- context snapshots are copied between stages
- sidecars are merged, not overwritten wholesale
- abort is checked before every stage
- stage failures are wrapped with `PipelineStageError(stageId, cause)`

---

## Reference implementations already inspected

Local:
- `/home/steam/github/ysdede/transformers.js`
  - branch/reference fork contains Nemo Conformer TDT pipeline and Whisper tokenizer/pipeline logic
- `/home/steam/github/ysdede/parakeet.js`
- `/home/steam/github/ysdede/tdt-webgpu-demo`
- `/home/steam/github/asrjs/browser-demo`
- `/home/steam/github/asrjs/streaming-demo`
- `/mnt/n/github/ysdede/parakeet.js-demo`

Cloned public refs:
- `/tmp/asrjs-refs/whisper.cpp`
- `/tmp/asrjs-refs/whisperX`
- `/tmp/asrjs-refs/ten-vad`

Important patterns:
- WhisperX architecture: VAD -> ASR decode -> forced alignment -> optional diarization -> output writers
- whisper.cpp exposes VAD thresholds, max speech duration, subtitle writers, timestamp outputs
- transformers.js Whisper pipeline uses 30s chunks, stride, timestamp-token decoding, longest-common-sequence overlap merge
- tdt-webgpu-demo exposes direct rich output: words, tokens, metrics, frame/logprob/TDT-step details

---

## Non-negotiable design constraints

1. Do not hardcode Parakeet constants globally.
   - Use `ModelInferenceLimits` and built-in descriptors.
   - Whisper max window is 30s.
   - Parakeet TDT recommended/max is 90s/180s.

2. Keep low-level model sessions direct.
   - Family-specific sessions should expose native-capable direct transcription.
   - High-level handles can opt into windowing/pipeline composition.

3. Keep pipeline stages model-neutral where possible.
   - Stage inputs/outputs should be `PipelineContext`, `TranscriptResult`, words/sentences/sidecars.
   - Model-specific behavior should live in strategy functions or family adapters.

4. Tests first.
   - For every new stage: write failing Vitest coverage first, verify RED, implement, verify GREEN.

5. No browser demo dependency for core behavior.
   - Add Node/offline smoke tests for formatting, chunking, stage composition, and later fixture-audio model smoke tests.

---

## Next implementation tasks

### Task 1: Add explicit built-in transcript sidecar/output stage — DONE

Objective: make JSON/SRT/VTT sidecar generation a reusable pipeline stage driven by requested output formats.

Files:
- Created: `src/pipeline/output-sidecars.ts`
- Modified: `src/pipeline/composition.ts`
- Modified: `src/pipeline/index.ts`
- Test: `tests/pipeline-output-sidecars.test.ts`

Implemented:
- `createTranscriptOutputStage({ formats: ['json', 'srt', 'vtt'] })`
- `createTranscriptSidecars(transcript, formats)`
- compatibility: `createSubtitleSidecarStage` delegates to the generic output stage

Verified:
- `npm test -- tests/pipeline-output-sidecars.test.ts tests/pipeline-composition.test.ts --run`
- `npm run typecheck`

Commit:
- `feat: add transcript output sidecar stage`

### Task 2: Add windowing as a stage adapter

Objective: let high-level users compose long-audio windowing into pipelines without hardwiring it into all `transcribe()` calls.

Files:
- Create: `src/pipeline/windowing-stage.ts`
- Modify: `src/pipeline/index.ts`
- Test: `tests/pipeline-windowing-stage.test.ts`

API sketch:
```ts
createWindowingStage({
  inference,
  transcribeWindow,
  sampleRate,
})
```

Expected behavior:
- reads `context.input`, `context.options`, `context.signal`
- calls existing `transcribeWithWindowing`
- returns `{ transcript }`
- respects abort signal and passes progress callback through options

Tests:
- long audio calls multiple windows and returns merged transcript
- disabled windowing calls direct fallback if provided
- missing input throws a stage error with useful message

Commit:
- `feat: expose long-audio windowing as pipeline stage`

### Task 3: Add sentence normalization as an explicit stage

Objective: allow non-NeMo models or user pipelines to add sentence spans from words after ASR/alignment.

Files:
- Create: `src/pipeline/sentence-stage.ts`
- Modify: `src/pipeline/index.ts`
- Test: `tests/pipeline-sentence-stage.test.ts`

API sketch:
```ts
createSentenceSegmentationStage({
  source: 'words',
  updateSegments?: boolean,
  gapThresholdSeconds?: number,
})
```

Behavior:
- if transcript has words, produce `sentences`
- optionally update `segments` for legacy apps
- preserve existing words/tokens/text
- update `meta.sentenceCount`

Commit:
- `feat: add sentence segmentation pipeline stage`

### Task 4: Add VAD segment schema and merge helpers before full VAD stage

Objective: define reusable VAD data structures and pure merge functions before wiring actual TEN-VAD/FireRed backends.

Files:
- Create: `src/pipeline/vad-segments.ts`
- Modify: `src/pipeline/index.ts`
- Test: `tests/pipeline-vad-segments.test.ts`

Types:
```ts
interface SpeechSegment {
  index: number;
  startTime: number;
  endTime: number;
  confidence?: number;
}
```

Helpers:
- `padSpeechSegments`
- `mergeNearbySpeechSegments`
- `splitLongSpeechSegments`
- `speechSegmentsToWindows`

Defaults inspired by WhisperX/whisper.cpp:
- threshold: 0.5 is backend-level, not here
- min speech: 250ms
- min silence: 100ms
- pad: 30ms
- max speech/window: from model catalog, often 30s for Whisper

Commit:
- `feat: add VAD speech segment helpers`

### Task 5: Add Whisper chunk/stride planning helpers

Objective: implement the model-neutral part of Whisper chunking before decoding real models.

Files:
- Create: `src/pipeline/whisper-chunking.ts`
- Modify: `src/pipeline/index.ts`
- Test: `tests/pipeline-whisper-chunking.test.ts`

Helpers:
- `planWhisperChunks(audioLengthSamples, sampleRate, chunkLengthSeconds, strideLengthSeconds)`
- validate stride < chunk/2 if using left+right stride
- produce chunks with sample offsets and stride metadata

Reference:
- `/home/steam/github/ysdede/transformers.js/packages/transformers/src/pipelines/automatic-speech-recognition.js`
- `/home/steam/github/ysdede/transformers.js/packages/transformers/src/models/whisper/tokenization_whisper.js`

Commit:
- `feat: add Whisper chunk planning helpers`

### Task 6: Add Whisper timestamp-token merge helpers

Objective: prepare SOTA-ish Whisper output merging independently of actual ONNX decoder implementation.

Files:
- Create: `src/pipeline/whisper-timestamps.ts`
- Modify: `src/pipeline/index.ts`
- Test: `tests/pipeline-whisper-timestamps.test.ts`

Helpers:
- timestamp token detection abstraction
- decode timestamp spans from token stream
- overlap merge via longest common sequence
- word timestamp collation tests for punctuation/CJK handling

Do not implement full model decode here.

Commit:
- `feat: add Whisper timestamp merge helpers`

### Task 7: Add offline fixture-audio smoke harness

Objective: support user-provided audio samples for repeatable offline smoke tests without browser demos.

Files:
- Create: `tests/smoke/transcribe-fixture.mjs`
- Create: `tests/fixtures/README.md`
- Modify: `package.json`

Behavior:
- accepts local audio path and expected text snippets
- supports cached/local model artifacts if already available
- skips gracefully if model assets are absent unless env var forces run

Potential scripts:
- `npm run test:fixture-smoke -- --audio path --model parakeet-tdt-0.6b-v2 --expect "..."`

Commit:
- `test: add offline fixture transcription smoke harness`

---

## Verification gate after every task

Run targeted tests first, then:

```bash
npm run typecheck
npm run lint
npm test
npm run build
node tests/smoke/offline-output-smoke.mjs
```

Expected known lint warnings:
- `src/presets/parakeet/compat.ts` max-lines
- `src/runtime/browser-waveform.ts` max-lines
- `src/runtime/streaming-detector.ts` max-lines

---

## Handoff prompt for next session

If session resets, start with:

```
Load skill asrjs-dev. Open /home/steam/github/asrjs/speech-recognition/docs/plans/asr-pipeline-roadmap.md. Continue branch feat/asr-pipeline-output-formats from latest origin. Follow TDD. Do not touch .serena/. Continue next unchecked task in the plan.
```

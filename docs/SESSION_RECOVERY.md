# Session Recovery — Flexo (2026-05-30 02:07 UTC+3)

## Branch: `feat/asr-pipeline-output-formats`
Main merged + pushed. Working on feature branch.

## Resume
```bash
cd ~/github/asrjs/speech-recognition
git checkout feat/asr-pipeline-output-formats
npm test  # verify: 596/597 pass (1 pre-existing browser failure)
npm run build  # verify: clean
```

## Status

### Completed
- [x] Whisper Vanilla Core (pure decode loop)
- [x] Enhanced Executor (VAD + 4 gates + temp fallback + drift + context)
- [x] Standalone Modules: quality/ chunking/ post-processing/ alignment/
- [x] CTC Viterbi forced alignment (14 tests)
- [x] WAV2VEC2 alignment backend (8 tests)
- [x] WAV2VEC2 model factory + preset (Flexo-glm5.1)
- [x] formatTranscript() — sentence boundary + normalization
- [x] ProductionWhisperPipeline — end-to-end wrapper (7 tests)
- [x] Main branch merged + pushed

### Architecture
```
Audio → VAD → Chunking → Whisper Enhanced Executor
  (4 quality gates + temp fallback + drift + context)
  → Segment Merge → Post-Processing (sentences, dedup, normalize)
  → Production Pipeline (subtitles SRT/VTT, metrics)
  → clean ProductionTranscript
```

### Standalone imports
```ts
import { compressionRatioGate, withTemperatureFallback } from '@asrjs/speech-recognition/quality';
import { DriftHandler, mergeVadSegments, FixedWindowChunker } from '@asrjs/speech-recognition/chunking';
import { mergeSegments, formatTranscript } from '@asrjs/speech-recognition/post-processing';
import { ctcForceAlign, createWav2Vec2Aligner } from '@asrjs/speech-recognition/alignment';
import { ProductionWhisperPipeline, createWhisperProductionPipeline } from '@asrjs/speech-recognition/pipeline';
```

### Key numbers
- 597 tests total (596 pass, 1 pre-existing browser flaky)
- Typecheck clean
- 3 standalone modules + 1 pipeline module
- WAV2VEC2 ONNX smoke verified by Flexo-glm5.1

### Next: End-to-end smoke test
Needs:
- Whisper large-v3-turbo ONNX split-graph model (available at /tmp/hf-publish/)
- Run: load model → transcribe jfk2.en.wav → EnhancedWhisperExecutor → Production pipeline
- Verify: 4 quality gates active, sentence output correct, SRT/VTT generated
- Blueprint: `scripts/whisper-e2e.ts`

### ONNX models available
- /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph/q8/ (1.4GB, 4 ONNX files)
- /tmp/whisper-tiny-onnx/ (merged decoder, needs split)
- /tmp/whisper-tiny-ts-onnx/ (config-present, use loadSpeechModel)

### Flexo-glm5.1 handoff
- `docs/handoffs/flexo-wav2vec2-progress.md` — WAV2VEC2 DONE
- `docs/AGENT_TASKS.md` — coordination between agents

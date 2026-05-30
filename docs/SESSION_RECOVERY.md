# Session Recovery — Flexo (2026-06-01 02:19 UTC+3)

## Branch: `feat/asr-pipeline-output-formats`
Main merged + pushed. Feature branch ahead of origin.

## Resume
```bash
cd ~/github/asrjs/speech-recognition
git checkout feat/asr-pipeline-output-formats
npm test  # 597/597 pass
npm run build  # clean
```

## Completed (all validated)

### Whisper Production Engine
- [x] Vanilla core (greedy decode loop)
- [x] Enhanced executor (VAD + 4 gates + temp fallback + drift + context)
- [x] Beam search (whisperBeamDecode with KV-cache-per-beam)
- [x] WhisperX params: numBeams, bestOf, patience, lengthPenalty
- [x] ProductionWhisperPipeline (SRT/VTT + metrics, 7 tests)
- [x] formatTranscript (sentence boundary + normalization)

### Alignment Phase D
- [x] CTC Viterbi forced alignment (ctcForceAlign, 14 tests)
- [x] WAV2VEC2 alignment backend (createWav2Vec2Aligner, 8 tests)
- [x] groupCharAlignmentToWords (char → word timestamps)

### Standalone Modules (all exportable)
- [x] @asrjs/speech-recognition/quality (7 files, 13 tests)
- [x] @asrjs/speech-recognition/chunking (7 files, 11 tests)
- [x] @asrjs/speech-recognition/post-processing (3 files, 4 tests)
- [x] @asrjs/speech-recognition/alignment (4 files, 22 tests)
- [x] @asrjs/speech-recognition/pipeline (16 files, 7 tests)

### Infrastructure
- [x] WAV2VEC2 model factory + ONNX smoke (Flexo-glm5.1)
- [x] Shared CTC module (src/ctc/)
- [x] ORT URL/path unified (bare paths + file:// + HTTP)
- [x] Main merged + pushed (backup: backup/main-20260530-0200)

### Key commits (most recent)
```
87e5e6a fix: unified URL/path handling — tokenizer fetchText accepts bare file paths
783dfd1 feat: add WhisperX params — numBeams, lengthPenalty, patience, bestOf
d2ce555 feat: wire beam search into executor
a0bdb9e feat: add beam search decode to Whisper core
e6f734d feat: add ProductionWhisperPipeline
4ab5e89 feat: WAV2VEC2 alignment backend
33cc27b feat: CTC Viterbi forced alignment
```

## Architecture

```
Audio → VAD (TenVAD/FireRed)
  → Whisper Enhanced Executor
    (onTokenLogits → 4 quality gates + temp fallback + drift + context)
  → Segment Merge + Word Dedup
  → formatTranscript (sentences + normalize)
  → Production Pipeline (SRT/VTT + metrics)
  → ProductionTranscript
```

## ONNX Models
- /tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph/q8/ (1.4GB, 4 graphs)
- /tmp/whisper-tiny-onnx/ (merged decoder, ~40MB int8)
- /tmp/whisper-base-4graph/ (256MB q8, fast iteration)

## Known Issues
1. loadSpeechModel direct-source path: materializeHuggingFaceArtifacts manipulates URLs even for kind='direct'. Workaround: direct session creation (like validation smoke tests).
2. 1 pre-existing browser test flaky (browser-realtime.test.ts)
3. CTC Viterbi on real WAV2VEC2 ONNX model not yet tested

## New Fixture (2026-06-01)
- `tests/fixtures/end-of-chapter-4.en.mp3` — 2m47s, 22050Hz mono, 64kbps MP3
- `tests/fixtures/end-of-chapter-4.en.txt` — reference transcription (2622 bytes)
- Purpose: long audio stitching smoke test (> Whisper 30s limit)

## Next Tasks (from AGENT_TASKS.md)
1. Fix loadSpeechModel direct-source path (materializeHuggingFaceArtifacts)
2. Run `loaded.transcribeMonoPcm(endOfChapter4, sr, opts)` — auto-windowed
3. Compare output vs reference transcription
4. Verify long audio stitching quality (word dedup, no hallucinations)
5. WAV2VEC2 HF publish + CTC Viterbi integration test

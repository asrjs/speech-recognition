# Session Recovery — Flexo-DSV4Pro (2026-05-31)

## Branch & Repo
- **Branch:** `feat/asr-pipeline-output-formats`
- **Repo:** `/home/steam/github/asrjs/speech-recognition`
- **Remote:** `asrjs/speech-recognition`
- **Tests:** 562/567 pass (5 pre-existing WAV2VEC2 failures)

## Resume Instructions
```bash
cd ~/github/asrjs/speech-recognition
git checkout feat/asr-pipeline-output-formats
npm test  # verify state
```

---

## What We're Building

**Production Whisper inference engine** using best practices from:
- **WhisperX** — VAD pre-segmentation (70% quality), disabled context conditioning (20%)
- **whisper.cpp** — drift correction, entropy gate, compression ratio, temperature fallback
- **faster-whisper** — logprob gate, no-speech detection, quality gate composition

**Architecture: Vanilla → Enhanced → Standalone Modules**

```
Layer 0: core.ts              ← pure decode loop (ONNX-agnostic)
Layer 1: executor.ts          ← ONNX bridge + vanilla pipeline
Layer 2: EnhancedWhisperExecutor ← VAD + gates + temp fallback + drift + context + merge
Layer 3: standalone modules   ← model-agnostic, reusable by any ASR model
```

---

## Key Documents

| Document | Path | Lines | Purpose |
|----------|------|-------|---------|
| Production Techniques | `docs/references/whisper-production-techniques.md` | 467 | Hallucination suppression, alignment, long audio |
| Master Guide | `docs/plans/enhanced-asr-master-guide.md` | 484 | 6-layer architecture, 7-phase plan (A→G) |
| Standalone Modules Plan | `docs/plans/standalone-nlp-alignment-modules.md` | 375 | Module architecture, exports |
| Vanilla+Enhanced Architecture | `docs/plans/whisper-vanilla-enhanced-architecture.md` | 273 | Vanilla vs Enhanced split |
| Enhanced Implementation | `docs/plans/whisper-enhanced-implementation-plan.md` | 624 | 11-phase plan |
| WAV2VEC2 Model | `docs/plans/wav2vec2-model-and-alignment.md` | 407 | Dual-purpose ASR + alignment |
| Agent Tasks | `docs/AGENT_TASKS.md` | 155 | Coordination between agents |

---

## Commits (Flexo-DSV4Pro, most recent first)

```
b25e9a6 feat: wire context conditioning — extraPromptTokens + token collection
bc634eb feat: wire onTokenLogits from options→executor→core, enable all 4 quality gates
3df750a feat: production-ready EnhancedWhisperExecutor — VAD+gates+fallback+drift+merge
0b1b948 feat: add alignment/ module — cross-attention DTW extracted (T4)
1db314c feat: VAD backends + fixed-window chunker + post-processing extras (T1-T3)
960a97b docs: comprehensive task redesign with blocker chain
1e3edfc feat: package.json exports for quality/chunking/post-processing
bdfef2a refactor: wire enhanced-executor to standalone modules (Phase F)
5d31de4 refactor: extract post-processing/ module (Phase C)
0ec5fba refactor: extract chunking/ module (Phase B)
5474991 refactor: extract quality/ module (Phase A)
7c85cdb feat: EnhancedWhisperExecutor composition (Phase 8)
696f79c feat: VAD segmenter + segment merger (Phases 6-7)
ae8a8b6 feat: drift-handler (Phase 5)
32dae1d feat: chunk-context builder (Phase 4)
42f5494 feat: temperature fallback (Phase 3)
e808e15 feat: quality gates — compression, logprob, entropy, no-speech (Phase 2)
708aac9 feat: onTokenLogits callback + enhanced types (Phase 1)
1efddda refactor: extract vanilla Whisper core with pure decode loop
```

---

## Standalone Modules (importable)

```ts
import { compressionRatioGate, logProbGate, withTemperatureFallback } from '@asrjs/speech-recognition/quality';
import { DriftHandler, mergeVadSegments, FixedWindowChunker } from '@asrjs/speech-recognition/chunking';
import { mergeSegments, deduplicateWords, normalizeText, buildSentences } from '@asrjs/speech-recognition/post-processing';
import { crossAttentionDtwTimestamps } from '@asrjs/speech-recognition/alignment';
```

---

## Files Created (Flexo-DSV4Pro)

```
src/quality/types.ts, compression-ratio.ts, log-probability.ts, entropy.ts, no-speech.ts, temperature-fallback.ts, index.ts
src/quality.ts (entry stub)
src/chunking/types.ts, drift-handler.ts, vad-segmenter.ts, fixed-window.ts, index.ts, backends/ten-vad.ts, backends/firered-vad.ts
src/chunking.ts (entry stub)
src/post-processing/segment-merger.ts, extras.ts, index.ts
src/post-processing.ts (entry stub)
src/alignment/cross-attention-dtw.ts, index.ts
src/alignment.ts (entry stub)
src/models/whisper-seq2seq/enhanced-executor.ts (production pipeline)
tests/quality-gates.test.ts, chunking.test.ts, post-processing.test.ts, chunking-backends.test.ts, chunking-post-extras.test.ts, alignment-dtw.test.ts, whisper-enhanced-executor.test.ts
```

---

## What's NOT Ours (Flexo-glm5.1)

```
src/ctc/              — CTC module (DONE)
src/models/wav2vec2/  — WAV2VEC2 model (PARTIAL, model.ts+factory+pending)
src/presets/wav2vec2/ — presets (NEW)
```

**Blocker chain:** WAV2VEC2 model factory incomplete → CTC Viterbi alignment blocked

---

## WhisperX Quality Pipeline (checklist)

- [x] VAD pre-segmentation (70% quality) — TenVAD/FireRed backends wired
- [x] Context conditioning off by default (20% quality) — extraPromptTokens + ChunkContextBuilder
- [x] Compression ratio gate — catches "the the the" hallucinations
- [x] Log probability gate — catches low-confidence output
- [x] Entropy gate — catches uncertain distributions
- [x] No-speech gate — skips silence segments
- [x] Temperature fallback [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] — escapes hallucination loops
- [x] Drift correction — whisper.cpp-style seek counter
- [x] Segment merge + word dedup — clean multi-chunk output
- [x] Fixed-window chunker — fallback when VAD unavailable
- [x] onTokenLogits callback — gate'lere gerçek logits akışı
- [x] DTW alignment — cross-attention based word timestamps

---

## Remaining (production polish)

1. **Sentence boundary + text normalization in output** — `buildSentences()` and `normalizeText()` exist but not called in executor output
2. **End-to-end smoke test** — real audio → VAD → Whisper → gates → formatted output
3. **No-speech probability from first token** — gate exists but needs real inference to test
4. **WAV2VEC2 forced alignment** — blocked on Flexo-glm5.1 model factory
5. **Parakeet/MedASR enhanced executors** — same modules, different models

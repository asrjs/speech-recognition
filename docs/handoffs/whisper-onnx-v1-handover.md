# Whisper ONNX Integration v1 — Handover Summary

**Date:** 2026-05-28
**Branch:** `feat/asr-pipeline-output-formats`
**Commit:** `de6fb6c` — feat: add real Whisper ONNX inference support
**Agent:** Flexo (P520, WSL2)

---

## What was accomplished

Whisper went from a completely stubbed scaffold to real ONNX inference with encoder/decoder sessions, HF tokenizer loading, greedy generation loop, and a log-mel frontend.

### New files created

| File | Purpose |
|------|---------|
| `src/models/whisper-seq2seq/ort.ts` | ONNX artifact resolution, `initWhisperOrt()`, `createWhisperOrtSession()` |
| `src/models/whisper-seq2seq/tokenizer.ts` | `WhisperTokenizer` loads `tokenizer.json` from HF, special tokens, timestamp detection |
| `src/models/whisper-seq2seq/executor.ts` | `WhisperOnnxExecutor`: encoder → prompt tokens → greedy decoder loop with KV-cache |
| `tests/whisper-integration.test.ts` | 14 tests: manifests, tokenizer, mel, model factory, regression |

### Files modified

| File | What changed |
|------|-------------|
| `src/audio/whisper-mel.ts` | Real `WhisperMelProcessor`: 80/128-bin log10 mel, Slaney scale, 16kHz, Hann window, Cooley-Tukey FFT |
| `src/models/whisper-seq2seq/types.ts` | Added `WhisperArtifactSource`, `WhisperExecutor`, `WhisperDecodeContext` |
| `src/models/whisper-seq2seq/config.ts` | Updated defaults (melBins: 80, vocabSize: 51865) |
| `src/models/whisper-seq2seq/model.ts` | Wires `WhisperOnnxExecutor` when `source` present; keeps stub fallback |
| `src/models/whisper-seq2seq/index.ts` | Exports new modules |
| `src/presets/whisper/manifest.ts` | 4 presets with real HF sources (onnx-community/whisper-tiny/base/small/large-v3-turbo) |
| `src/presets/whisper/factory.ts` | Passes manifest `source` into model options via `useManifestSource` |
| `docs/plans/asr-pipeline-roadmap.md` | Task 8 marked DONE, Tasks 9-14 added |

### Models configured

| Preset | Params | Use case |
|--------|--------|----------|
| `whisper-tiny` | ~39M | Smoke tests, low-memory devices |
| `whisper-base` | ~74M | **Default** multilingual baseline |
| `whisper-small` | ~244M | Better quality, heavier |
| `whisper-large-v3-turbo` | ~809M | Experimental desktop/WebGPU only |

### Verification at commit time

- `tsc --noEmit` ✓ clean
- `npm test` ✓ 64 files, 334 tests (was 63/320 before)
- `npm run build` ✓ clean

---

## Architecture decisions

1. **Followed NeMo TDT pattern**: `resolveWhisperArtifacts()` → `WhisperOnnxExecutor` → session, mirroring `resolveNemoTdtArtifacts` → `OrtNemoTdtExecutor`.
2. **Merged decoder only**: Uses `decoder_model_merged.onnx` (handles both first-step and cache-branch) rather than separate `decoder_model` + `decoder_with_past_model`.
3. **Stub fallback preserved**: When no `source` is configured, the session still returns stub output. This keeps tests and offline usage passing without downloading multi-MB ONNX files.
4. **Default preset changed**: From `openai/whisper-base` (no ONNX artifacts) to `onnx-community/whisper-base`.
5. **Quantization auto-selection**: WebGPU encoder → fp16, WASM decoder → int8. User can override via `encoderQuant`/`decoderQuant` in source options.
6. **Prompt tokens**: SOT (50258) → language (e.g. `<|tr|>` = 50268) → task (`<|transcribe|>` = 50359) → optional `<|notimestamps|>` (50363).
7. **Timestamp support**: Segment-level only, via timestamp token splitting in `buildSegments()`. Word-level not implemented yet.

---

## Known limitations — Tasks 9-14

These are documented in `docs/plans/asr-pipeline-roadmap.md` and detailed in `~/.hermes/skills/mlops/asrjs-dev/references/whisper-onnx-integration.md`.

### Task 9: Validate mel processor against OpenAI reference
The `WhisperMelProcessor` is a clean-room implementation. It has NOT been validated for bit-exact matching against OpenAI's `whisper.log_mel_spectrogram()`. Next agent should:
- Generate reference mel features with Python `openai-whisper` for a test signal.
- Compare against `WhisperMelProcessor` output.
- Fix discrepancies in windowing, padding, mel scale, or log clamping.

**Refs:** OpenAI `whisper/audio.py`, HF `WhisperFeatureExtractor`, `whisper.cpp` `whisper_mel_calc`.

### Task 10: Implement full BPE tokenizer encode
Current `WhisperTokenizer.encode()` only handles exact special token matches and falls back to naive character-level encoding. A full BPE encoder is needed that:
- Parses `model.merges` from `tokenizer.json`.
- Applies GPT-2 style byte-level BPE (`Ġ` prefix for word boundaries).
- Supports `pre_tokenizer` rules if present.

**Refs:** OpenAI `whisper/tokenizer.py`, HF `WhisperTokenizer`, `tokenizers` Rust bindings.

### Task 11: Implement beam search decoding
Only greedy decoding exists. Beam search should:
- Track `num_beams` hypotheses.
- Use log-prob scores from `confidenceFromLogits`.
- Read defaults from `generation_config.json`.
- Default remains greedy; beam search is opt-in.

**Refs:** HF `BeamSearchScorer`, `faster-whisper` beam search, `whisper.cpp` `whisper_decode_internal`.

### Task 12: Implement word-level timestamps
Current output only has segment-level timestamps. Word-level requires:
- Cross-attention weights from decoder (may need re-exported ONNX with attention outputs).
- DTW or median-filtered alignment between attention and encoder frames.
- Fallback: timestamp token interpolation.

**Refs:** OpenAI `whisper/timing.py`, HF `WhisperTokenTimestampDecoder`, `whisperX` alignment, `faster-whisper` word timestamps.

### Task 13: Wire long-audio chunking into Whisper executor
Audio > 30s is not handled. Should:
- Use existing `src/pipeline/whisper-chunking.ts` `planWhisperChunks()`.
- Run encoder + decoder per chunk.
- Merge with existing `src/pipeline/whisper-timestamps.ts` `mergeWhisperTokenSequences()`.

**Refs:** HF `pipelines/automatic_speech_recognition.py`, `faster-whisper` chunking, `whisper.cpp` CLI.

### Task 14: Add real ONNX fixture smoke test
No end-to-end ONNX inference test exists. Should:
- Download `onnx-community/whisper-tiny` artifacts once.
- Run on short Turkish audio (or synthetic PCM).
- Assert output is NOT stub text.
- Skip gracefully unless `ASRJS_FIXTURE_SMOKE=1`.

---

## SOTA reference implementations to study

Priority order for the next agent:

1. **OpenAI whisper (Python)** — Reference
   - `whisper/audio.py` — mel, STFT, padding
   - `whisper/tokenizer.py` — BPE, special tokens
   - `whisper/decoding.py` — greedy, beam search, temperature
   - `whisper/timing.py` — word-level timestamps
   - `whisper/transcribe.py` — chunking, stride, merging

2. **Hugging Face transformers (Python)** — Most complete open implementation
   - `models/whisper/modeling_whisper.py`
   - `models/whisper/tokenization_whisper.py`
   - `pipelines/automatic_speech_recognition.py`
   - `generation/utils.py`

3. **whisper.cpp (C++)** — Fastest CPU
   - `whisper.cpp` — `whisper_mel_calc`, `whisper_decode_internal`
   - `examples/main/main.cpp`

4. **faster-whisper (Python)** — Optimized CTranslate2
   - `faster_whisper/transcribe.py`
   - `faster_whisper/tokenizer.py`
   - `faster_whisper/feature_extractor.py`

5. **WhisperX (Python)** — Alignment + diarization
   - `whisperx/alignment.py`
   - `whisperx/asr.py`

6. **Transformers.js (JavaScript)** — Browser ONNX
   - `src/models/whisper/whisper_decoder.js`
   - `src/models/whisper/whisper_feature_extractor.js`
   - `src/pipelines/automatic_speech_recognition.js`
   - Local fork: `~/github/ysdede/transformers.js` (branch `v4-nemo-conformer-tdt-main-r3`)

7. **onnx-community Whisper exports** — Our artifact source
   - `onnx-community/whisper-*` on Hugging Face

---

## How to resume

```bash
cd /home/steam/github/asrjs/speech-recognition
git checkout feat/asr-pipeline-output-formats
git pull origin feat/asr-pipeline-output-formats  # if needed
npm run typecheck && npm test && npm run build
```

Then:
1. Load skill `asrjs-dev`.
2. Read `docs/plans/asr-pipeline-roadmap.md`.
3. Read `~/.hermes/skills/mlops/asrjs-dev/references/whisper-onnx-integration.md`.
4. Pick Task 9 (or whichever is highest priority).
5. Write failing test first, implement, verify, commit.
6. Update roadmap and reference file before finishing.

---

## Key contacts / context

- **Repo:** `asrjs/speech-recognition`
- **Branch:** `feat/asr-pipeline-output-formats`
- **Skill:** `asrjs-dev`
- **Reference file:** `~/.hermes/skills/mlops/asrjs-dev/references/whisper-onnx-integration.md`
- **Roadmap:** `docs/plans/asr-pipeline-roadmap.md`
- **User preference:** TDD (write failing test first). No PRs for small fixes. Do not name private repos in commits.

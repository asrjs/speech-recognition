# Session Recovery — Flexo (2026-06-01)

## Branch: `feat/asr-pipeline-output-formats`
Main merged + pushed. Feature branch ahead of origin by ~75 commits.

## Resume
```bash
cd ~/github/asrjs/speech-recognition
git checkout feat/asr-pipeline-output-formats
npm test           # 598/599 pass (1 pre-existing browser flaky)
npm run build      # clean
```

## Critical Fix: Splitgraph KV Cache Bridge
**Commit: `cef005f`** — Root cause of all "invalid data location" / "missing in feeds" errors.

Three bugs in `transcribeWithSplitGraph` bridge:
1. **dims: []** → tensors without shape → ORT rejection
2. **Prefix mismatch** → init outputs `present.*`, step expects `past_key_values.*`
3. **Encoder KV lost** → step outputs only decoder KV, must preserve from init

After fix: whisper-base fp32 transcribes 167s audio in 73s, 84.8% overlap, zero hallucinations.

## Key Learnings
- Splitgraph `decoder_init` has ONLY 2 inputs (input_ids + encoder_hidden_states) — NOT 25 like merged decoder
- Cross-window KV cache NOT needed — each 30s window is independent. Context is TEXT tokens
- WhisperX DISABLES condition_on_previous_text by default (error cascading risk)
- Tensor data must be CLONED when passing between ONNX sessions (ORT WASM limitation)

## Architecture
```
Audio → VAD (TenVAD/FireRed)
  → 30s Window Chunking (transcribeWithWindowing)
    → Whisper Splitgraph (init→step→...→done)
      (onTokenLogits → 4 quality gates + temp fallback)
  → Segment Merge + Word Dedup
  → formatTranscript (sentences + normalize)
  → Production Pipeline (SRT/VTT + metrics)
  → ProductionTranscript
```

## Completed Features
- [x] Greedy + Beam search decode (whisperDecode dispatch)
- [x] Long audio windowing (auto 30s windows, word-gap cursor)
- [x] Timestamp tokens + token suppression
- [x] Quality gates (compression, logprob, entropy, no-speech)
- [x] Temperature fallback [0.0→0.2→...→1.0]
- [x] Context conditioning (extraPromptTokens)
- [x] Mel dimension auto-detect (numMelBins from manifest)
- [x] SRT/VTT subtitle export
- [x] VRAM optimization (skip merged decoder, defer alignment)
- [x] URL/path unified (fetchText handles bare + file:// + HTTP)

## ONNX Models
- `/tmp/whisper-base-4graph/fp32/` — 756MB, verified working
- `/tmp/whisper-base-4graph/q8/` — 260MB, q8 data location bug
- `/tmp/hf-publish/whisper-large-v3-turbo-onnx-4graph/q8/` — 1.4GB

## Next Tasks
1. **Large-v3-turbo smoke test** — needs model that fits VRAM (fp16 2.3GB or q8)
2. **bestOf independent decodings** — run N decodes, pick best scoring
3. **patience beam search** — early stopping on consecutive EOS
4. **WAV2VEC2 HF publish** — upload ONNX to HuggingFace
5. **CTC Viterbi integration test** — real WAV2VEC2 model alignment
6. **Batched encoder** — parallel window encoding (WhisperX-style, Phase 9)

## Fixtures
- `tests/fixtures/end-of-chapter-4.en.mp3` (167s, 22050Hz) + .en.txt (reference)
- `tests/fixtures/jfk2.en.wav` (11s) — quick smoke
- `tests/fixtures/JFK_Short.en.wav` (17s) — minimal smoke

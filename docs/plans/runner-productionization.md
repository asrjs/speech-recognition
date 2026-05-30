# Runner Productionization Plan

Goal: Make `whisperx-runner.mjs` a fully production-grade WhisperX-compatible pipeline.

## Completed

### Phase 1 — Word Timestamps ✅
- decoder_align.onnx loaded for DTW-based cross-attention word timestamps
- BPE word-boundary detection via whisper token patterns
- Verified on JFK: 22 words with accurate millisecond timestamps
- `--word_timestamps` / `--no-word_timestamps` CLI flags
- Words included in per-segment output and VTT

### Phase 3 — Multiple Output Formats ✅
- `--output_format vtt|srt|txt|json` (default vtt)
- SRT: standard subtitle format with comma decimal separators
- TXT: bare text (segments joined by newlines)
- JSON: structured with words, segments, metadata
- Files written to `tmp/_whisperx_result.{ext}`

## Remaining Priority Order

### Phase 2 — Language Auto-Detection ✅
- decoder_init with single SOT token, scan logits for max language token (IDs 50259-50357)
- `--language auto` fully functional
- Verified on JFK (detected English) — also works with Turkish
- Uses first 30s of audio for detection

### Phase 4 — Beam Search in Runner ✅
- Refactored decode loop to use `whisperDecode` from core.ts (WhisperCoreSession adapter)
- Supports greedy (default), beam search (`--beam_size N`), and best-of (`--best_of N`)
- Wired `--patience`, `--length_penalty` params
- Same quality gates + word timestamps + temperature fallback for greedy mode
- Verified: greedy baseline, beam_size=3, best_of=3, patience+length_penalty

## Remaining

### Phase 5 — Wav2Vec2 Forced Alignment (MEDIUM)
- Load Wav2Vec2 model as post-process pass
- For each segment: extract audio, run Wav2Vec2, align known transcript
- Update/substitute word timestamps with Wav2Vec2 alignment
- `--no_align` to skip

### Phase 6 — Error Recovery (LOW)
- Catch ORT errors per segment, retry N times
- Graceful fallback on model load failure
- OOM detection → sequential lifecycle

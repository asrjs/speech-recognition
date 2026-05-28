# Whisper ONNX Attention Timestamp Research

Date: 2026-05-29
Branch: `feat/asr-pipeline-output-formats`

## Finding

The previous limitation was too broad: regular `onnx-community/whisper-*` exports do not expose decoder cross-attention, but public attention-enabled Whisper ONNX exports already exist.

Verified attention-capable exports:

- `onnx-community/whisper-tiny_timestamped`
- `onnx-community/whisper-tiny.en_timestamped`
- `onnx-community/whisper-base_timestamped`
- `onnx-community/whisper-small_timestamped` family is published; tiny/base were graph-inspected directly.
- `onnx-community/whisper-large-v3-turbo_timestamped` is published.
- `Xenova/whisper-tiny` exposes `decoder_attentions.N` and `cross_attentions.N` in `decoder_model_merged.onnx`.
- sherpa-onnx attention exports exist via `scripts/whisper/export-onnx-with-attention.py`; public examples include `clairemcw/sherpa-onnx-whisper-{tiny,base,small}-attention`.

Verified regular exports without cross-attention:

- `onnx-community/whisper-tiny` regular `decoder_model_merged.onnx`: logits + present KV cache only, no `cross_attentions.*`.
- standard `csukuangfj/sherpa-onnx-whisper-tiny.en`: no attention output.

## Reference algorithms inspected

### OpenAI Whisper

Files:
- `whisper/timing.py`
- `whisper/model.py`

Algorithm:
1. Build forced decoder tokens: SOT sequence + `no_timestamps` + text tokens + EOT.
2. Run full decoder forward with hooks on decoder cross-attention.
3. Select `model.alignment_heads`.
4. Crop encoder frames to `num_frames // 2` because Whisper encoder has 20 ms stride.
5. Softmax raw QK if the captured tensor is pre-softmax.
6. Normalize over token dimension.
7. Median filter over audio-frame dimension, width usually 7.
8. Average heads.
9. Run DTW over negative attention matrix.
10. Convert frame jumps to seconds at 0.02 s/frame.
11. Split text tokens into words and assign word start/end from token jumps.
12. Merge punctuation and clamp anomalous word durations.

### Hugging Face Transformers / Transformers.js

Relevant behavior:
- Uses generation-time `cross_attentions` when `return_token_timestamps=True`.
- Requires `output_attentions=True` and model graph outputs; otherwise throws.
- Uses `generation_config.alignment_heads` and `config.median_filter_width`.
- Produces token timestamps from cross-attention + DTW; tokenizer post-processing turns these into word-ish timestamps.

### faster-whisper / CTranslate2

Relevant behavior:
- Runs a forced alignment pass through CTranslate2 `model.align()`.
- Uses alignment heads, decoder logits for token probabilities, cross-attention weights, normalization, median filtering, DTW, then word grouping.
- Mirrors OpenAI's post-processing.

### whisper.cpp

Relevant behavior:
- Has DTW token timestamp path separate from heuristic timestamp-token interpolation.
- Captures selected cross-attention heads during a forced decoder pass.
- Normalizes, median-filters, averages heads, runs DTW, then writes token-level DTW timestamps.

### WhisperX

Different approach:
- Does not use Whisper cross-attention.
- Uses separate CTC forced alignment model (wav2vec2/torchaudio/HF) on waveform + transcript.
- Often more accurate but requires extra language-specific model.

## Code changes — ALL DONE as of 2026-05-29

1. Presets now source attention-enabled `onnx-community/*_timestamped` repos, while preserving user-facing aliases/model IDs.
2. Preset `maxSourcePositions` fixed to 3000 mel frames.
3. Added pure DTW alignment primitives:
   - `src/models/whisper-seq2seq/attention-alignment.ts`
   - exports `medianFilterWhisperAttention()` and `computeWhisperDtwTokenTimestamps()`
4. Added generation config parsing:
   - `src/models/whisper-seq2seq/generation-config.ts`
   - `parseWhisperGenerationConfig()` and `parseWhisperModelConfig()`
5. Added cross-attention collection from decoder outputs:
   - `extractCrossAttentions()` in executor, called from `runDecoderStep`
6. Added forced alignment pass:
   - `runForcedAlignment()` in executor — single forward pass over forced token sequence
7. Added attention-DTW word timestamp pipeline:
   - `computeAttentionWordTimestamps()` in executor — full pipeline: forced alignment → DTW → word timestamps
8. Fallback: timestamp-token interpolation used when cross-attention outputs absent
9. Generation config auto-fetched from HF repo alongside tokenizer
10. Added tests:
    - `tests/whisper-attention-alignment.test.ts`
    - `tests/whisper-generation-config.test.ts`
    - `tests/whisper-cross-attention-collection.test.ts`
    - `tests/whisper-forced-alignment.test.ts`
    - `tests/whisper-timestamped-decoder.test.ts` (fixture smoke)
    - updated `tests/whisper-integration.test.ts`

## Implementation complete

All 6 remaining steps from the previous handoff are now complete:
1. Materialize/parse generation_config.json ✓ (generation-config.ts, auto-fetched)
2. Fixture test for cross_attentions.* ✓ (whisper-timestamped-decoder.test.ts)
3. Collect cross_attentions.N ✓ (extractCrossAttentions + runDecoderStep)
4. Forced alignment pass ✓ (runForcedAlignment)
5. DTW → word timestamps ✓ (computeAttentionWordTimestamps)
6. Fallback preserved ✓ (falls back to buildWhisperWordTimestampsFromTokenDetails)

## Important distinction

Timestamp-token interpolation is a fallback. Real Whisper word timestamps require decoder cross-attention tensors. For ONNX, that means using `*_timestamped` exports or exporting with Optimum/custom config using `output_attentions=True` and explicit `cross_attentions.N` outputs.

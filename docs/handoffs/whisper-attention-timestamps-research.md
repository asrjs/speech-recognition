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

## Code changes started

1. Presets now source attention-enabled `onnx-community/*_timestamped` repos, while preserving user-facing aliases/model IDs.
2. Preset `maxSourcePositions` fixed to 3000 mel frames. Whisper encoder expects 30s x 100 fps mel frames; encoder output is downsampled to 1500 frames.
3. Added pure DTW alignment primitives:
   - `src/models/whisper-seq2seq/attention-alignment.ts`
   - exports `medianFilterWhisperAttention()`
   - exports `computeWhisperDtwTokenTimestamps()`
4. Added tests:
   - `tests/whisper-attention-alignment.test.ts`
   - updated `tests/whisper-integration.test.ts`

## Remaining implementation plan

1. Load `generation_config.json` from the same HF repo as tokenizer/model artifacts.
   - Parse `alignment_heads`.
   - Parse `median_filter_width` from `config.json` if available, default 7.
2. Extend `WhisperDirectArtifacts` / `WhisperHuggingFaceSource` to optionally include config/generation config URLs.
3. Collect `cross_attentions.N` outputs from attention-enabled `decoder_model_merged.onnx` runs.
   - ONNX shape: `[batch, heads, decoder_sequence_length, encoder_sequence_length]`.
   - For generation-time alignment, collect one slice per emitted text token.
4. Prefer OpenAI/CTranslate2 forced alignment for accuracy:
   - After decoding final tokens, rerun decoder over `SOT + lang + task + no_timestamps + text + EOT` without timestamp tokens.
   - Request/use all cross-attention outputs from the ONNX graph.
   - Compute DTW timestamps with the helper added here.
5. Convert token timestamps to word timestamps using tokenizer split/grouping. Current fallback `buildWhisperWordTimestampsFromTokenDetails()` can stay as fallback when attention outputs are absent.
6. Add a fixture smoke test using `onnx-community/whisper-tiny_timestamped` direct local artifacts and assert `cross_attentions.*` are present.

## Important distinction

Timestamp-token interpolation is a fallback. Real Whisper word timestamps require decoder cross-attention tensors. For ONNX, that means using `*_timestamped` exports or exporting with Optimum/custom config using `output_attentions=True` and explicit `cross_attentions.N` outputs.

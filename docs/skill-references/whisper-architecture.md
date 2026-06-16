# Whisper Architecture & Pitfalls (condensed from SKILL.md)

## Critical pitfalls

### Splitgraph decoder_init has ONLY 2 inputs
`input_ids` + `encoder_hidden_states`. No `past_key_values`, no `use_cache_branch`. Never feed zero-filled KV to splitgraph init.

### Splitgraph step outputs decoder KV only
`decoder_step` outputs ONLY `present.{i}.decoder.{key,value}`. Encoder KV from init must be preserved across all step iterations.

### Tensor dims must be preserved through bridge
The core decode loop strips tensor dimensions. Store actual dims from init output, reuse for step input. `dims: []` causes ORT validation failures.

### Cross-session tensor reuse
Default CPU/WASM tensor bridge: do not pass ORT tensors from one session into
another. Extract raw data, preserve dims, and create a new `ort.Tensor`.

Experimental WebGPU KV bridge: when decoder init/step sessions are both WebGPU
and were created with `preferredOutputLocation: 'gpu-buffer'`, keep KV tensors
on GPU and feed them directly into the next decoder step. Do not touch `.data`
for those tensors. Prefer a per-output location map so KV outputs stay on GPU
while logits stay on CPU until logit processing moves to GPU. Dispose replaced
GPU KV tensors.

The current measured fast path intentionally keeps logits on CPU. Do not append
a raw `ArgMax(logits)` output to the decoder graph as a shortcut: Whisper token
selection depends on `suppress_tokens`, `begin_suppress_tokens`, no-timestamp
suppression, and timestamp-state rules. A GPU ArgMax experiment must be masked,
limited first to greedy `noTimestamps=true`, and published as an alternate
artifact.

### Mel dimension from manifest
`config.json` lacks `num_mel_bins`. Read from `generation_config.json` or `manifest.json`. Large-v3-turbo = 128 mel, whisper-base/tiny = 80 mel.

### External data auto-detect
Co-located `.onnx.data` files need explicit `{data: path, path: filename}` in session options.

### VRAM: skip merged decoder for splitgraph
When `isSplitGraph`, skip loading merged `decoderSession`. Defer `decoderAlignSession`. Peak: 3 sessions vs 5.

### Node.js HF source — FIXED
`materializeHuggingFaceArtifacts` downloads `.onnx`/`.onnx.data` to temp before passing to ORT. ORT Node backend can't open HTTP URLs.

### ONNX external data filename must match internal reference
Uploaded filenames MUST match ONNX graph's internal `external_data.location`. Check with `onnx.load(model, load_external_data=False)`.

### 4-graph KV-cache export pitfalls
- HF 5.x `EncoderDecoderCache` yields 6-element layer tuples
- `aten::diff` has NO ONNX lowering — use manual decoder block iteration
- `decoder_step` does NOT need `encoder_hidden_states` as input
- Export tool venv: `tools/whisper-onnx-export/.venv/`

## Whisper Vanilla + Enhanced Architecture

- **Vanilla**: `core.ts` (pure greedy decode) + `executor.ts` (ONNX bridge)
- **Enhanced**: `EnhancedWhisperExecutor` wrapping vanilla, adding quality gates, temp fallback, VAD chunking
- **Production Pipeline**: `ProductionWhisperPipeline` — VAD pre-segmentation → per-chunk Whisper → 4 gates → temp fallback → context conditioning → drift → merge → format

## Backend strategy

- **Native ORT** (`onnxruntime-node`): Full system RAM, persistent multi-session. Use for local dev + large models.
- **WASM** (`onnxruntime-web`): ~1.5 GB heap limit, single-session constraint. Use for browser + small models.
- **WebGPU**: Browser only. fp16 models load directly, inline weights only.
- **Sequential lifecycle** for large models on 8GB: load encoder→run→dispose, then decoders→run→dispose. Peak ~2.5GB.
- **Encoder graph capture**: `experimentalWebGpuEncoderGraphCapture` is an
  opt-in WebGPU-only experiment. It applies ORT `enableGraphCapture` to the
  encoder session only; do not enable it for dynamic decoder-step sessions
  without a measured A/B.
- **ArgMax graph output**: future experiment only. It must preserve logit
  processor semantics, use ORT fetches to avoid requesting full logits, and be
  rejected on any token mismatch.
- **GPU encoder→decoder bridge** (`experimentalGpuKvCache` + WebGPU):
  Encoder session set `preferredOutputLocation: 'gpu-buffer'`. Encoder output
  stays on GPU. **Preferred approach:** strip the final `Cast(f16→f32)` from the
  fp16_iofp32 encoder (see `tools/whisper-onnx-export/strip_encoder_cast.py`),
  output fp16 directly. Paired with the original fp16-input `decoder_init.onnx`,
  this gives zero dtype casts in the pipeline. Encode: 6.9× faster (1900→277ms).
  RTFx: 4.8→27.6×. Step P50: 80→9.5ms.
- **Fast mel spectrogram** (`fastFft` option, default true): Uses N_FFT=512
  (power-of-2) with zero-padded Hann window instead of Bluestein FFT for N_FFT=400.
  Preprocess: 2.9× faster (237→81ms). Matches parakeet.js approach.
- **WebGPU optimization lessons**: See `docs/Whisper-Optimizations.md` §Lessons
  Learned for 14 documented pitfalls covering ONNX surgery, WebGPU execution,
  optimization methodology, and Vite dev server.

## Beam search
`whisperBeamDecode` in core.ts. `numBeams`, `lengthPenalty`, `bestOf`, `patience` params. Wired into splitGraphDecodeLoop. Default `numBeams=1` (greedy).

The experimental WebGPU GPU-KV bridge is greedy-only: `numBeams=1`,
`bestOf=1`, and `temperature=0`. Beam search remains supported on the stable
splitgraph path, but it is not part of the measured WebGPU `11x` path until a
batched beam graph and GPU KV reorder path exist.

Default greedy decoding does not track cumulative log-probability scores.
`trackScore: true` is used only for `bestOf`, where ranking multiple candidate
decodes requires a score. Do not reintroduce full-vocabulary score work into
ordinary greedy decoding without an end-to-end benchmark showing it is needed.

## Reproducibility harness
Two modes: feature-input (100% token match, Python mel features) and wav-input (≥80%, TS mel frontend). Env vars: `WHISPER_REFERENCE_JSON`, `WHISPER_SPLITGRAPH_FIXTURE_DIR`.

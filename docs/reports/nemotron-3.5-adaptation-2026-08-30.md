# Nemotron 3.5 streaming 0.6B — acquisition and graph audit (2026-08-30)

First slice of the ADAPT decision recorded in
`docs/reports/hf-multilingual-asr-candidate-survey-2026-08-30.md`. This report
covers artifact acquisition and graph-contract audit only. It makes **no**
quality, parity, or performance claim yet; the official NVIDIA reference chain
and the native/WASM/WebGPU parity ladder are still open gates.

## Acquisition

Local HF-compatible folder:
`N:/models/onnx/nemo/nemotron-3.5-asr-streaming-0.6b-onnx` (community export
[codavidgarcia/nemotron-3.5-asr-streaming-0.6b-onnx](https://huggingface.co/codavidgarcia/nemotron-3.5-asr-streaming-0.6b-onnx),
built from official revision `f3d333391852ba876df169dcc9ba902d25b6ab0b`,
license OpenMDW-1.1 with the license/NOTICE files stored alongside).

- Every LFS file's SHA-256 was verified against the Hugging Face API's
  published `lfs.sha256` after download.
- `encoder_320ms_first_fp16.onnx.data` and `encoder_320ms_fp16.onnx.data` are
  the same LFS blob (`159887b7…`, 1,228,512,000 bytes); the second is an NTFS
  hard link to the first, so both encoder variants resolve their external data
  from one 1.17 GB copy.
- Provenance manifest with all file hashes:
  `tools/data/results/nemotron/nemotron-3.5-artifact-provenance-2026-08-30.json`.

## Graph audit

`node-audit-onnx-artifact.mjs` result: **4/4 ONNX graphs load on native CPU
ORT** (`ok: true`), full JSON at
`tools/data/results/nemotron/nemotron-3.5-onnx-audit-2026-08-30.json`.

| Graph | Inputs (count) | Key input shapes | Outputs (count) | Key output shapes |
| --- | --- | --- | --- | --- |
| `decoder.onnx` (LSTM predictor step) | 3 | `token` int64 `[1,1]`; `h_in`/`c_in` fp32 `[2,1,640]` | 3 | `decoder_out` fp32 `[1,640]`; `h_out`/`c_out` `[2,1,640]` |
| `encoder_320ms_first_fp16.onnx` (first chunk) | 78 | features fp32 `[1,25,128]`; `prompt_ids` int64 `[1]`; `cache_mask` `[1,1,1,60]`; 24×K/V caches `[1,8,56,128]`; 3 conv2d + 24 conv1d caches | 76 | `encoder_out` fp32 `[1,4,640]` + updated caches |
| `encoder_320ms_fp16.onnx` (continuation) | 78 | features fp32 `[1,32,128]`; same cache layout | 76 | `encoder_out` fp32 `[1,4,640]` + updated caches |
| `joiner.onnx` | 2 | `encoder_frame` `[1,640]`; `decoder_out` `[1,640]` | 1 | `logits` fp32 `[1,13088]` |

Contract facts, all consistent with the published `nemotron_onnx_config.json`:

- fp16 weights with fp32 graph I/O (ORT converts internally on CPU), so the
  ORT Web dtype boundary looks favorable; WebGPU is unproven until run.
- Cache-aware streaming encoder: 24 transformer layers, per-layer K/V cache
  `[1,8,56,128]` (left context 56 frames), 27 convolution caches; 320 ms audio
  chunk → 4 encoder frames at 80 ms frame time.
- Predictor is a 2-layer LSTM, hidden 640 — the same shape family as the
  existing NeMo RNNT predictor machinery in `src/models/nemo-rnnt`.
- Vocab 13088, blank id 13087, language prompt ids from the config
  (`auto=101`, `tr=18`, `en=0`), max 10 symbols per step, 128-bin mel at
  16 kHz (hop 160, n_fft 512).
- First-chunk encoder consumes 25 mel frames; continuation consumes 32 mel
  frames; both emit 4 encoder frames. Exact feature-centering/trimming rules
  must come from the official reference runner, not guesswork.
- Benign CPU constant-folding warning on `/encode_positions/MatMul`; no load
  failures.

## Environment readiness

- `conda env nemo` has NeMo 2.4.0 + torch 2.6.0+cu118, so the official
  reference chain can run locally without new installs.
- The official `.nemo` checkpoint (~3.8 GB) has **not** been downloaded yet;
  no original-engine output exists on this host.

## Next steps (in order)

1. Download the official `nvidia/nemotron-3.5-asr-streaming-0.6b` checkpoint;
   run official NeMo inference on jfk-short plus a speech/silence streaming
   fixture; capture reference transcripts/tokens (labeled oracle separate from
   throughput runs).
2. Port/verify the JS 128-bin mel against the official feature pipeline before
   any graph execution claims.
3. Native-ORT parity of the audited graphs against step 1, then WASM, then
   Chrome headless real-WebGPU (keep the ORT Web entry-point alias invariant).
4. Design the library adapter: a Nemotron RNNT executor reusing NeMo RNNT
   predictor/joint machinery where proven shared, with the cache-aware chunked
   encoder as a model-specific concern (not generic runtime code).
5. Streaming latency benchmark (first-partial, per-chunk, steady-state RTFx,
   memory) only after exact-token parity on the shared window.

## Reusable lessons

- Check LFS blob identity before re-downloading multi-variant exports; two
  graph variants often share one external-data payload (saved 1.17 GB here).
- Audit before integrating: the audit run surfaced the full cache tensor
  inventory (24 K/V + 27 conv caches, 78 inputs) that no README stated.

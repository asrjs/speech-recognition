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
- The official checkpoint is downloaded to
  `N:/models/nemo/nemotron-3.5-asr-streaming-0.6b/` (`nemotron-3.5-asr-streaming-0.6b.nemo`,
  2,258.6 MB, SHA-256 `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`
  verified against the HF-published LFS hash, revision
  `f3d333391852ba876df169dcc9ba902d25b6ab0b`). No original-engine output has
  been captured yet.

## Official reference capture (2026-08-30, this turn)

Step 3a complete. The official NeMo reference runner
(`tools/model-debugging/reference/nemotron-3.5-asr-streaming/run_reference.py`)
was executed on both committed fixtures using an isolated venv (NeMo 3.0.0
shadowing conda 2.4.0, `use_lhotse=False` to avoid the Windows
`WinError 267` temp-dir + Lhotse bug). Results in
`tools/data/results/nemotron/nemotron-3.5-official-reference-2026-08-30.json`:

| Fixture | Tokens | Score | Text |
| --- | ---: | ---: | --- |
| jfk-short.wav | 48 | −7.06 | And so my fellow Americans ask not what your country can do for you. <en-US> Ask what you can do for your country. <en-US> |
| librivox-blankgaps-synthetic.wav | 111 | −10.12 | Preface of a year with the birds this is a Librivox recording. <en-US> All Librivox recording. <en-US> Links are in the public domain. <en-US> For more information or to volunteer, please visit librivox dot org read by Olivia. <en-US> |

Key findings:

- The prompt system injects `<en-US>` language-ID tags at chunk boundaries;
  this is expected cache-aware streaming behavior, not a tokenizer artifact.
- The official NeMo preprocessor uses 128-bin mel, 25 ms window, 10 ms
  stride, NA normalization, dither 1e-5. The JS frontend must match these
  exact parameters before any ONNX graph execution claims.
- The prompt dictionary has 128 entries (`auto=101`, `tr=18`, `en=0`),
  matching the community export's `nemotron_onnx_config.json`.
- Streaming API is available: `conformer_stream_step`,
  `transcribe_simulate_cache_aware_streaming`. The offline runner is a
  stepping stone, not the final streaming adapter.

Reusable lessons:

- NeMo 2.4.0 lacks the prompt-RNNT class (`rnnt_bpe_models_prompt`);
  the required module only exists on NeMo `main` / PyPI 3.0.0. Create an
  isolated venv with `--system-site-packages` to inherit torch/numpy from
  conda, then `pip install --no-deps` the 3.0.0 wheel. This avoids
  upgrading the conda env and breaking other projects.
- NeMo's `transcribe()` writes a temp `manifest.json` for the Lhotse
  dataloader; this path is broken on Windows (`WinError 267`). Always
  pass `use_lhotse=False` in the Windows reference runner.
- For per-fixture transcription, pass a single-element list and read
  `hypotheses[0]` (NeMo returns a list of lists).

## Next steps (in order)

1. Run official NeMo inference on jfk-short plus a speech/silence streaming
   fixture; capture reference transcripts/tokens (labeled oracle separate from
   throughput runs). The checkpoint is local and hash-verified.
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

# Parakeet TDT v3 WebGPU EP probe — 2026-08-29

## Scope

This handoff records the controlled browser probe for the Parakeet TDT v3
`decoder_joint-model.onnx` GRU graph after upgrading `onnxruntime-web` from
the 1.27 nightly to stable 1.29.0. The probe used the existing
`N:\github\asrjs\webgpu-agent-test` Chrome headless harness, a real NVIDIA
Blackwell adapter, and the local model artifact at
`N:\models\onnx\nemo\parakeet-tdt-0.6b-v3-onnx`.

The graph emits `[1, 1, 1, 8198]`: 8193 tokenizer-vocabulary logits followed
by five TDT duration logits. Token selection must slice the first 8193 values;
the raw argmax is not a valid token ID for the next decoder step.

## Corrected A/B result

The first harness version had two evidence defects: spike mode imported the
full library before the direct ORT probe, and the spike logger called an
undefined `push` function. It also compared raw logits without excluding the
duration head. Those defects were fixed before accepting the measurements;
the runner now recognizes the two per-EP records emitted by spike mode.

Three corrected runs per ORT version/backend completed with finite outputs and
the same vocabulary-sliced token sequence (`8192` for all five steps):

| ORT Web      | EP     | Session-load median | Warm-step mean range | Result   |
| ------------ | ------ | ------------------: | -------------------: | -------- |
| 1.27 nightly | WebGPU |              1.41 s |         14.0–32.2 ms | 3/3 pass |
| 1.27 nightly | WASM   |              0.63 s |           5.5–5.8 ms | 3/3 pass |
| 1.29.0       | WebGPU |              1.46 s |          9.9–14.9 ms | 3/3 pass |
| 1.29.0       | WASM   |              0.68 s |           5.5–6.5 ms | 3/3 pass |

The earlier “1.29 session-create hang” was therefore a false classification;
the corrected graph loads and runs on both browser execution providers.
The production default remains encoder-WebGPU/decoder-WASM because the full
model has not yet established a transcript-quality and lifecycle win for a
WebGPU decoder.

## GPU-state experiment

An opt-in direct ORT run set
`preferredOutputLocation: { outputs: 'cpu', output_states_1: 'gpu-buffer',
output_states_2: 'gpu-buffer' }`. ORT 1.29 reported both recurrent state
outputs as `gpu-buffer` on every step, and the vocabulary-sliced sequence
remained identical. Three WebGPU repetitions produced warm-step means of
9.01, 8.26, and 7.78 ms. Comparable CPU-state runs measured 12.73 and
13.14 ms, so the decoder-only microbenchmark suggests roughly 29–41% lower
warm-step time.

This is a hypothesis, not a promoted library path. The experiment does not
include the FastConformer encoder, real transcript quality, repeated model
transcriptions, or safe disposal of recurrent GPU tensors. In particular,
the initial version that disposed GPU state handles surfaced ORT's `null
function` failure; the spike was rerun without per-step disposal only to
measure execution. A production implementation must prove ownership and
disposal (or a bounded fallback) before enabling GPU-state retention.

## Decode hot-path optimization

The library decoder consumed each float32 logits tensor synchronously before
disposing the ORT output, but still allocated a full `Float32Array` copy for
every step. The TDT path now borrows the typed-array view and only normalizes
non-float32 views. In the same change, `confidenceFromLogits` no longer runs
the entropy traversal that belongs to `tokenQualityFromLogits`; it retains the
same max/softmax/log-probability arithmetic while doing only the work its
callers request. TDT model/session disposal also removes redundant async
wrappers.

The reproducible `benchmark:hot-paths` harness now includes an 8,193-entry
TDT-vocabulary scenario. On Node v26.2.0 / win32-x64 with 10,000 iterations,
the confidence-only path measured `0.0849 ms` p50 versus `0.3970 ms` p50 for
the full entropy path (about 79% lower). Borrowing the logits view measured
`0.0928 ms` p50 versus `0.1114 ms` p50 for the copy control (about 17% lower).
These are post-processing microbenchmarks, not end-to-end RTFx claims; the
existing executor tests and full suite remain the correctness gate.

## Remaining work

1. Repeat the full-model browser pass with deterministic `native-rate` linear
   audio preparation across browsers and the available v3 artifacts. The
   default target-rate AudioContext path produced only 74 tokens, but native-
   rate restored exact 91-token parity for fp16/WebGPU encoder + WASM decoder
   (16–19x RTFx), the full WebGPU decoder (three-run median 5.64x RTFx), and
   int8/WebGPU encoder + int8/WASM decoder (2.02x RTFx). This makes audio
   preparation the leading cause of the earlier mismatch and confirms ORT
   1.29 decoder correctness without a decoder speed win. Details are in
   `docs/reports/parakeet-tdt-v3-browser-full-model-2026-08-29.md`.
   The repeatable runner now supports `--repeat=N`: eight same-session runs
   were exact for both compositions. Hybrid transcribe times warmed to
   729–802 ms (~25x RTFx), while full WebGPU decoding stayed around
   3,044–3,353 ms (~6x RTFx). Chrome heap snapshots are GC-sensitive (the
   full-WebGPU run dropped from ~1.33 GB to ~38 MB after collection), so this
   is a baseline rather than a leak verdict.
2. The earlier “more than ten minutes” fp16 session-create statement was an
   invalid harness run: the page failed before valid model creation. The
   corrected harness reaches model-ready in 7–12 seconds. The fp32 external-
   data browser control is separately blocked by `Module.MountedFiles is not
   available`.
3. The library now exposes an opt-in `decoderStateOutputLocation` source/config
   option for hub, local, and direct Parakeet loading. It maps only
   `output_states_1/2` to the requested location while keeping decoder logits
   on CPU; Node sessions defensively map the diagnostic request back to CPU.
   Exercise this path in the browser library entry point, then require
   end-to-end transcript parity, repeated-transcription memory checks, and
   disposal behavior before promotion.
4. Keep backend placement workload-specific: the direct spike is faster on
   WASM for this one-frame GRU workload, while GPU-state retention is a
   promising optimization for larger decode loops.

## Reproduction

Start Vite in `N:\github\asrjs\webgpu-agent-test` with the repository's ORT
package aliased by `vite.config.js`, then run:

```text
npm run dev -- --force
node scripts/run-parakeet-tdt-webgpu.mjs --mode=spike
node scripts/run-parakeet-tdt-webgpu.mjs --mode=spike --gpu-state
```

The browser runner uses `--enable-unsafe-webgpu`, Vulkan/D3D11 ANGLE flags,
and writes JSON evidence to `webgpu-agent-test/_results/`. The Parakeet runner
defaults to deterministic `native-rate` linear WAV preparation; pass
`--audio-strategy=target` only for the explicit AudioContext-resampler control.

## Lifecycle refresh after Qwen optimization work (2026-08-29)

The corrected Chrome harness was rerun after the Qwen WebGPU graph probes. A
three-repeat full-model all-WebGPU run (fp16 encoder, fp32 decoder, JS
preprocessor, native-rate audio) produced the exact 91-token LibriVox
transcript on every run and completed the normal model-disposal `finally`
path. Transcription times were `3608.435`, `3400.030`, and `3420.915 ms`
(median `3420.915 ms`, `5.477x` RTFx); model load was `13623.170 ms`.
Browser JS-heap snapshots rose from about `1.350 GB` to `1.366 GB` across the
three same-session runs, so this is a repeatability signal rather than a leak
verdict. The raw capture is retained in the sibling harness and summarized in
`docs/reports/parakeet-tdt-v3-webgpu-lifecycle-refresh-2026-08-29.json`.

The earlier GPU-state diagnostic was also repeated with explicit disposal of
each replaced GPU state tensor. Two fresh runs completed all five WebGPU
steps with `gpu-buffer/gpu-buffer` state outputs, vocabulary argmax parity,
and no `null function` error; the paired WASM controls remained exact. This
narrowly localizes the old failure to an earlier harness/runtime condition,
but it does not prove that full-model GPU-state retention is production-safe.
Keep the default encoder-WebGPU/decoder-WASM composition and require an
opt-in library path plus repeated full-model parity and disposal checks before
promoting decoder GPU-state output.

## Library-entry GPU-state A/B (2026-08-29)

The new `decoderStateOutputLocation` option was exercised through the actual
`loadSpeechModel` library path, not only a direct ORT spike. On the same
Chrome headless/NVIDIA Blackwell/ORT 1.29.0/native-rate fixture, fp16 WebGPU
encoder + fp32 WebGPU decoder produced the exact 91-token transcript for all
three runs in both controls:

| Decoder state output | Warm transcribe runs (ms) | Median | Median RTFx | Load (ms) |
| -------------------- | -------------------------: | ------: | ----------: | --------: |
| `cpu` (control)       | 3808.445 / 3474.620 / 3443.575 | 3474.620 | 5.3918 | 10698.940 |
| `gpu-buffer` (opt-in) | 3114.505 / 2871.875 / 2811.170 | 2871.875 | 6.5244 | 10550.080 |

The opt-in path therefore reduced median latency by `17.3471%` and increased
median RTFx by `21.006%`; the 107 decode iterations and 91 emitted tokens
matched on every run. JS-heap snapshots were comparable (about 1.345–1.364
GB), and the executor's normal replacement/disposal path completed without
error. The raw captures are retained in the sibling harness and summarized in
`docs/reports/parakeet-tdt-v3-library-gpu-state-ab-2026-08-29.json`.

This is a meaningful single-adapter end-to-end win, but it remains opt-in.
Repeat it on another browser or adapter and perform a longer repeated-
transcription/heap/disposal soak before changing the production default.
Chrome Vulkan and D3D12 ANGLE attempts on this host returned
`WEBGPU_NO_ADAPTER`; those negative controls are retained in the report and
must not be interpreted as model or graph failures.

## ORT 1.29 storage-buffer cache sweep (2026-08-29)

ORT Web 1.29 exposes model-specific WebGPU buffer-cache modes. The library
now forwards these options through the Parakeet source, preset, local adapter,
and NeMo TDT executor, while leaving them unset by default. A matched Chrome
headless/NVIDIA Blackwell/native-rate fixture used the same fp16 WebGPU
encoder, fp32 WebGPU decoder, JavaScript preprocessor, artifact, and three
warmed transcriptions per condition:

| Storage cache mode | State output | Median transcribe (ms) | Median RTFx |
| ------------------ | ------------ | ----------------------: | ----------: |
| `bucket` (explicit control) | `cpu` | 3498.960 | 5.3538 |
| `simple` | `cpu` | 3525.735 | 5.3133 |
| `disabled` | `cpu` | 3482.320 | 5.3804 |
| `lazyRelease` | `cpu` | 3506.150 | 5.3441 |
| `simple` | `gpu-buffer` | 2886.670 | 6.4920 |

Every condition preserved the exact 91-token transcript. The cache-only
variants show no repeatable win over ORT's bucket default; the small disabled
delta is within run variance. Combining `simple` with the positive GPU-state
path was 0.5149% slower than the prior bucket-default GPU-state control. Keep
the ORT bucket default and the cache knobs opt-in until a longer,
model-specific sweep demonstrates a stable latency or memory benefit.
Evidence: `docs/reports/parakeet-tdt-v3-webgpu-cache-sweep-2026-08-29.json`.

## Five-run GPU-state lifecycle soak (2026-08-29)

The same Chrome headless/NVIDIA Blackwell/native-rate fixture was repeated
five times through `loadSpeechModel` for both CPU and GPU recurrent-state
outputs. Every run emitted the exact 91-token transcript and completed 107
decode iterations. The bucket-default control had a 3,491.575 ms median
transcription (5.3655x RTFx); `gpu-buffer` state had a 2,907.200 ms median
(6.4454x RTFx), a 16.7367% latency reduction and 20.1267% RTFx increase. GPU
state loaded 7.3665% faster in this repeat (9,794.690 ms vs 10,573.590 ms).

The browser harness invokes both model and runtime disposal, records teardown
errors, and reported none for either condition. This is still not proof of GPU
resource reclamation. Heap readings rose for the first four runs and dropped
sharply on the fifth in both conditions; treat them as diagnostic samples,
not a leak verdict. Keep GPU-state placement opt-in until the result is
repeated on another browser or adapter.
Evidence: `docs/reports/parakeet-tdt-v3-webgpu-lifecycle-soak-2026-08-29.json`.
The additional Chrome SwiftShader adapter attempt also returned
`WEBGPU_NO_ADAPTER` (raw capture retained in the sibling harness), so this
host still cannot supply a second usable adapter for promotion testing.

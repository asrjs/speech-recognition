# Nemotron FP32 package repair and v7 browser evidence — 2026-10-24

## Result

A bounded, reversible repack produced self-contained FP32 and mixed-precision candidate folders without modifying the source weights under `N:\models`. The FP32 export's ONNX graphs referenced only the second half of each oversized external-data file. The packaged duplicate first halves were unused. The repacker copied every referenced byte range, verified its SHA-256, and validated the resulting graphs by path.

The surgery addresses packaging and WebGPU graph constraints only. It does **not** make the current transcript pass. All four v7 app file flows with the mixed candidate, plus a full-FP32 Showcase control, returned the same strict-oracle mismatch: `...for you at what...` where the unchanged expected text is `...for you ask what...`. Those flows loaded the model and dispatched real WebGPU work. Each is recorded as **FAIL**; no microphone, VAD or Playground recording PASS is claimed because the file gate did not pass.

## Repack proof

| Graph | Source live tensor bytes | Original external file bytes | Repacked sidecar bytes | External references |
| --- | ---: | ---: | ---: | ---: |
| Encoder | 2,454,405,120 | 4,908,810,240 | 2,454,405,120 | 640 |
| Decoder | 59,760,640 | 119,521,280 | 59,760,640 | 7 |
| Joint | 37,822,592 | 75,645,184 | 37,822,592 | 6 |

All references start at the old file's midpoint and end at its end. The repack receipt records the source range, length, destination offset and SHA-256 for each tensor; it confirms exact source/destination range hash equality and that the source graph hash did not change during copying. Source graph SHA-256 values were encoder `58cf678d66c7c36461e204861dd59e4eea01a0c3c0805dd9e0ae004bd5ef53f3`, decoder `523cb8402312720e458286d793118e43245e07739e44321dcbc5f9bdc993de44`, and joint `38678494d52a30d68f11799a9a74de2ceec1bb643f8247e70f3d0cbd411fdee8`.

The surgical encoder changes only `/Concat_4` and `/Concat_5`, each with 24 inputs, into ordered Concat trees with at most six inputs per node. Outputs and order are preserved. The FP32 candidate graph SHA-256 is `03d6cd3d33434133534e94d7954db40a9a318c6039d3761763ccbb839a94264e`; its packed external-data SHA-256 is `7584f85df76bc9ae6fbdfa53aa8d97b07a842525d1c501d536d77fd9e4f57ac7`.

The mixed package uses the surgically split FP16 encoder, and the repacked FP32 decoder and joint. A filename-conforming copy (`encoder_fp16.onnx` and its `.data` sidecar) lets the built local adapter recognize actual precision. The graph's external-data location was updated to match the renamed sidecar; its weights were not changed.

## CPU component gate

The original FP32 encoder and the repacked+surgically split FP32 encoder ran with ONNX Runtime's `CPUExecutionProvider` only, 2 intra-op/inter-op threads, and graph optimizations disabled. Using the stored NeMo JFK frozen features, both the first 65-frame chunk and the continuation chunk using each side's returned cache produced exact equality for all five outputs; maximum absolute difference was 0. This is `PASS_GRAPH_EQUIVALENCE_ONLY`, not transcript or original-checkpoint parity.

`tools/data/fixtures/audio/jfk-short.wav` and `tests/fixtures/jfk2.en.wav` are byte-identical (SHA-256 `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`). Torchaudio decoded both to the same 16 kHz mono 176,000-frame float32 PCM (SHA-256 `ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0`).

## v7 file-flow results

The controlled Chromium harness used the unchanged JFK fixture hash and exact text oracle, a dedicated `tmp/real-e2e/nemotron-gpu-profile`, local candidate files, and four app routes. Each receipt verifies the selected model and nonzero WebGPU dispatch. All four return the same mismatch, and each remains **FAIL**:

| App | Encoder / decoder selection | File WebGPU dispatches | Transcript result |
| --- | --- | ---: | --- |
| Showcase | WebGPU FP16 / WASM FP32 | 26,928 | FAIL: `at` instead of `ask` |
| Realtime | WebGPU FP16 / WASM FP32 | 26,928 | FAIL: `at` instead of `ask` |
| Playground | WebGPU FP16 / WASM FP32 | 26,933 | FAIL: `at` instead of `ask` |
| Benchmark | WebGPU hybrid, FP16 / FP32 | 26,933 | FAIL: `at` instead of `ask` |

Realtime has no separate quantization controls. Its captured local-load diagnostics confirm the selected graph names `encoder_fp16.onnx` and `decoder.onnx`; its folder inspection reports FP16 and FP32. No app had page errors or console errors in these runs.

A full-FP32 Showcase control (FP32 encoder, FP32 decoder/joint, same graph surgery, same fixture and oracle) also loaded and ran on WebGPU (27,234 file dispatches) and returned the same `at` mismatch. Precision is therefore not a demonstrated remedy for this word. The old saved FP32 pipeline also emits `at`, while the saved official NeMo reference emits a different, repeated “Ask what…” sequence. That official receipt uses the same fixture and checkpoint SHA-256 `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`, but requests `auto` prompt, whereas the browser run selected English. The fresh CPU NeMo restore did not reach inference within the 15-minute bound, so no new official text/token oracle was obtained.

## Evidence files

Small receipts and bounded logs are in [`evidence/nemotron-fp32-repack-2026-10-24`](evidence/nemotron-fp32-repack-2026-10-24). They include the three per-tensor repack receipts, encoder CPU parity, four mixed-precision app receipts, the full-FP32 Showcase receipt, and the two cached reference receipts. No model weights or audio payloads are copied there.

Candidate model folders remain under `tmp/nemotron-fp32-repack-20261024` for replay. The source files under `N:\models\onnx\nemo\nemotron-3.5-asr-streaming-fp32-nemo` were read-only throughout.

## Additional read-only frontend diagnosis

The WAV and decoded float32 PCM exactly match the stored NeMo JFK fixture. The frozen reference has 1101 mel frames but declares only 1100 valid frames. Comparing the current JS frontend to all 128 × 1101 cached values:

- `validLengthMode: onnx` gives 1100 valid frames and a zeroed padded last frame; the full-array comparison is max abs `0.0005131`, mean abs `0.000001535`, cosine similarity `0.99999999999977` (the small drift is consistent with NeMo's recorded dither).
- The pre-fix registered Nemotron config selected `validLengthMode: centered`, which treats all 1101 as valid. Its last frame differs from the frozen NeMo reference by max abs `13.7683`.

This establishes a concrete terminal-frame frontend contract mismatch at the earliest stage. The source config has now been changed to `onnx`, backed by a failing-before/fixed-after JFK padding regression and 15999/16000/16001 sample-boundary cases. The pre-Canary v8 source suite passed 1,896 tests with 36 skipped; deployment and real v8 browser validation are still pending. No claim is made that this fixes the mid-utterance `at` token.

The cached old FP32 pipeline receipt also exposes the earlier false-pass route: its recorded `oracle.text` is identical to its own `...you at what...` output, and its token equality PASS therefore does not establish agreement with the independent official NeMo receipt. The official cached run uses the same checkpoint hash and exact PCM but requests prompt `auto`; its token sequence diverges from the legacy FP32 pipeline at token 28. These receipts do not authorize changing the current strict oracle.

## Fresh native reference and windowing repair: v11

The restore hold above was resolved by a process-local writable Numba cache, without editing installed packages. A fresh English-prompt native CPU reference now emits the correct JFK words. Its raw output contains `<en-US>` markers, so raw canonical text equality is false; the 48 native token IDs match the cached official token sequence. The actual native cache-aware schedule uses first input 49 frames without dropping output, then inputs of 9 previous + 56 new frames with a two-row drop, and a final 53-frame input. It produces 139 encoder rows across 20 calls. The legacy runtime instead advanced by 65 and the graph unconditionally dropped two rows even on the first call.

A surgical graph candidate makes only that drop conditional on the initial empty cache. All five continuation outputs are bit-exact against the original graph across 19 calls when **both receive identical native feeds**. The earlier receipt that fed native caches to only one side was rejected as an invalid control. A separate candidate-chained comparison still fails the existing `atol=rtol=1e-4` cache-time gate on steps 1 and 2 (maximum about `5.4e-4`); that gate remains failed. Encoder outputs satisfy the existing comparison, and the faithful greedy ONNX chain matches all 48 native token IDs. Token agreement does not erase the cache-time discrepancy.

The tracked builder produces encoder graph SHA-256 `c5f75200a708852901bdf9d4447f9731ea4b4ca943e359f2745c459be39749cd`; its sidecar SHA-256 is `7584f85df76bc9ae6fbdfa53aa8d97b07a842525d1c501d536d77fd9e4f57ac7`. Original weights remain unchanged. Its self-contained bundle declares optional validated `encoderWindowing` metadata (49 first, 56 shift, 9 left context). SDK v11 applies this schedule in offline and incremental execution; bundles without this declaration retain legacy behavior.

Actual v11 file inference now matches the full normalized canonical transcript in all four demos. Realtime controlled manual, TEN-VAD-only and FireRed-VAD-only captures, plus Playground controlled recording, also match. Playground native streaming executes 18 steps and matches the full transcript. Parent verification independently checked all nine completed receipts, the immutable fixture SHA, complete oracle text, nonzero inference dispatches and empty verifier/page/console error arrays. Showcase file dispatches: 32,040; Realtime file: 32,040, manual: 60,881, TEN-VAD: 35,244, FireRed: 36,846; Playground stream: 32,045. Showcase/Realtime/Playground request FP32 encoder WebGPU and decoder WASM. Benchmark round 2 uses its aggregate WebGPU Hybrid control and has 32,045 file dispatches; its individual component assignments are not exposed. The first Benchmark attempt failed at a nonexistent component control before loading, and is preserved as a harness failure. Physical microphone remains unverified, and the separate native cache-time numeric gate above remains failed.

Native CPU evidence remains in task-local `C:/Users/steam/AppData/Local/Temp/nemotron-native-cpu-reference-20261024/` and the verified CLI bundle in `nemotron-conditional-drop-cli-real-20261024/`. Browser receipts: `demos/tmp/real-e2e/remaining-models-20261024/nemotron-v11-conditional-drop/`. This follow-up supersedes the earlier restore hold, without rewriting the preserved v7/v8 failures.

The [durable CPU port evidence manifest](evidence/model-ports-v11-2026-10-24/manifest.json) preserves 27 small native/export/storage receipts and replay scripts across Nemotron and Canary, each independently hash checked after copying. It explicitly labels the rejected unequal-feed control and retained numeric parity failure; no weights or audio are included.

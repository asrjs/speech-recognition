# Original Whisper Large v3: own export and runtime validation

This port uses **original `openai/whisper-large-v3`**, not Turbo. Native CPU and our own four-graph CPU ONNX chain produce the same complete JFK content tokens. The original v13 WebGPU session-load attempt failed with `std::bad_alloc`; a later v15 Realtime single-cycle run completed the full JFK file and controlled manual/TEN-VAD/FireRed-VAD flows. A repeated same-worker stress run then failed inside FireRed inference after WebAssembly memory reached `4,294,901,760` bytes, so memory stability remains unresolved. The v16 source cleanup is packaged, but a current arena-off browser run changes served JavaScript only and cannot certify the package.

## Source and own export

The [original source checkpoint](https://huggingface.co/openai/whisper-large-v3/tree/06f233fe06e710322aca913c1bc4249a0d71fce1) is pinned to revision `06f233fe06e710322aca913c1bc4249a0d71fce1`. Parent independently rehashed `model.safetensors`: `a8e94b85976e5864ba3e9525c7e6c83b2a1eca42d4b797a0c7c24d778e40fd95`. The selective download and export used task-local storage; no original cached weights or global environment settings were edited.

Checkpoint geometry is 32 encoder and 32 decoder layers, hidden 1,280, 20 heads, 128 mel bins, vocabulary 51,866, source positions 1,500 (runtime frontend 3,000 mel frames), target positions 448. The existing configurable four-graph exporter handled this geometry. The SDK local adapter now checks registered decoder geometry and rejects cross-loading Turbo's four decoder layers as original Large v3, or vice versa. No fabricated Hub ONNX source was registered.

The FP16 export contains encoder, decoder-init, decoder-step and alignment graphs. Structural audit completed 50 checks without failure, including ONNX path checking and CPU ORT session creation. These are structural checks, not transcript acceptance. Substantial weights remained inline in the ONNX protobufs despite the partial sidecars:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `encoder_model.onnx` | 1,274,338,448 | `e851950b5ed4469139b41e154c0675df6e210425f96aa2ab958a79d2e9b5b692` |
| `decoder_init.onnx` | 1,681,057,757 | `cff7cf63e9afd52b9bc7d00a68b5aa1cd07848dc49dcbe9a740d3d629451a0f6` |
| `decoder_init.onnx.data` | 265,553,920 | `d2ffb83fbb738500320a79acfa1014575eafcc2f6aa6e321d25f9c088ef7dd94` |
| `decoder_step.onnx` | 1,471,279,733 | `79318815baee72eea03085e269f214eaafe53d1d66d6977d785272c96b555738` |
| `decoder_align.onnx` | 1,333,125,476 | `98dc3603b0daaa98bfb5d2f4b1ba114fcbdca81db96b081def074faf404fc622` |
| `decoder_step.onnx.data` and `decoder_align.onnx.data`, each | 132,776,960 | `30bd232b7a138f8c12623f4dc74a528683751d5c9816e9348b92abe110fd02ae` |

## Native and CPU ONNX reference

The unchanged JFK fixture SHA-256 is `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`: 16 kHz, 176,000 samples, 11 seconds. English transcription prompt is `[50258,50259,50360,50364]`.

Native and ONNX greedy content IDs agree exactly, producing “And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.” The native no-timestamps sequence includes terminal EOS 50257; the ONNX helper omits that token from the saved list while reporting `eos_stopped=true`. Parent compared all saved tokens after removing only native terminal EOS. Raw sequence equality is false by that declared convention. Native timestamp generation produces the same words; ONNX alignment executed with shape `[1,10,29,1500]` and text crop `[10,25,1500]`. Finite alignment logits are execution evidence, not timing-accuracy certification.

Two reusable tool/runtime contract issues were corrected:

- Checkpoint model config PAD is 50256; effective generation config and tokenizer PAD are 50257. Exported generation metadata now follows generation config with a model-config fallback. The original manifest is preserved. Parent independently verified that PAD is the only manifest difference and all seven graph/sidecar hashes remain unchanged. Native transcript success alone was not used to erase the metadata discrepancy.
- The new decoder-step graph declares INT64 rank-one `cache_position`. The reference helper previously omitted it and failed before an ONNX result; that failed attempt remains labeled `INVALID_HELPER`. Both helper and shared runtime now bind this input only when declared, using the decoder self-cache sequence length. Multi-token steps use consecutive positions; multiple beams with a one-token input share a single position. Parent/reviewer caught and corrected an initial beam-count/sequence-length mix-up. Legacy graphs that do not declare the input retain their feed contract.

Parent reran three reference-helper tests, the PAD-precedence regression, 17 focused runtime tests, and five production split-graph ownership tests. The v15 source suite passed 1,943 tests / 36 skipped; the subsequent v16 Omnilingual normalization addition brings the source suite to 1,947 / 36. Typecheck passed, lint had zero errors / 14 warnings, static root imports remain 44 (under the unchanged 45-module ceiling). All four demo builds and 68 demo tests passed. These software gates do not establish memory stability or physical-microphone acceptance.

## Browser gate and evidence

The v13 Showcase attempt used the real local export and requested WebGPU. Session creation failed with `std::bad_alloc` before ready/inference; the receipt does not identify which graph failed. The lossless external-data candidate was verified separately by parent: 7,157 tensor payload hashes and four stripped-IR hashes match, with sidecar bytes fully covered without gaps or overlap. That graph/data proof reduces the inline-protobuf packaging concern, but it did not by itself resolve runtime memory pressure.

The v15 Realtime single-cycle run completed file, controlled manual, TEN-VAD-only and FireRed-VAD-only flows with the exact JFK transcript. A later same-page repeat-stress run passed its first manual and TEN-VAD flows, then FireRed failed with ONNX Runtime `OrtRun` `std::bad_alloc` at the worker’s 4 GiB WASM limit. The failing VAD segment was 3.38 seconds after many earlier calls; linear-memory capacity traces show repeated ~100 MiB growth, but do not reveal live allocations or prove a leak. Source review confirms per-transcription owned encoder tensors, decoder feeds/outputs and KV caches are disposed in `finally`; v15 added focused lifecycle coverage. A second stress cycle left the worker unresponsive; its incomplete receipt is retained, and parent stopped only that test process and descendants. No repeat-stability PASS is claimed.

A narrow v16 diagnostic is testing `enableCpuMemArena:false` for the Whisper WebGPU session through a Playwright response transform. It records pre/post code hashes and is explicitly marked transformed-code experiment with `packageAcceptance:false`. It must not be described as shipped-package acceptance. If it resolves the failure, the same setting still needs a normal source change, a new package, and an unmodified-package E2E rerun.

- [Durable source/export/native/ORT evidence and parent verification](evidence/model-ports-v13-2026-10-24/whisper-large-v3/parent-verification.json).
- Original model/export: `C:/Users/steam/AppData/Local/Temp/asrjs-whisper-large-v3-20261024/`.
- SDK v16 archive SHA-256: `597b1f9d75ea5d2c012b399ba82868eb02e8df23c22fbbca2fb9a356687f8740`; parent checked all 37 recorded source hashes and all 1,655 installed/archive distribution hashes. Full source suite: 1,947 passed / 36 skipped; typecheck passed; lint 0 errors / 14 warnings; static root imports 44. All four demo builds and 68 demo tests passed.
- Externalization parent proof: `evidence/model-ports-v14-2026-10-24/whisper-externalization/parent-verification.json`; the failed candidate that externalized Constant tensor attributes was discarded and must not be conflated with the verified candidate.
- v15 single-cycle receipt: `demos/tmp/real-e2e/remaining-models-20261024/whisper-large-v3-v15-owned/realtime/`; repeated stress failure and incomplete repeat: `demos/tmp/real-e2e/remaining-models-20261024/whisper-large-v3-v15-stress2/realtime/`.
- v16 arena-off transformed experiment: `demos/tmp/real-e2e/remaining-models-20261024/whisper-large-v3-v16-arena-off/realtime/`. This is diagnostic code, not the immutable v16 package. The result was pending when this report was refreshed.

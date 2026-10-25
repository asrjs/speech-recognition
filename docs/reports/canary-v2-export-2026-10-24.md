# Canary 1B v2 CPU export and component validation — 2026-10-24

## Result and scope

The official `nvidia/canary-1b-v2` checkpoint was restored locally, exported as FP32 ONNX, and compared with a CPU NeMo reference on the unchanged JFK fixture. Encoder outputs, lengths and binary mask matched the reference gates; decoder token IDs and text matched exactly. The real export path produced a compact single-file encoder sidecar.

The hardened verifier replay returned exit code `0` and `PASS_GATED_COMPONENTS`. This is checkpoint-backed ONNX component evidence with frozen reference frontend features. It does not certify the JS frontend, a browser runtime, WASM/WebGPU, microphone capture, VAD or streaming.

## Checkpoint and model contract

| Property | Evidence |
| --- | --- |
| Model/revision | `nvidia/canary-1b-v2` at `cc46424723391883dbf1e2516631f5c177d7b263` |
| Download | Public pinned `.nemo`, HTTP 200, anonymous; 6,358,958,080 bytes; SHA-256 `ae5ef1bf06812a95a1594a8f5f0ee9c51f35418e5ba96939fa6b98ab00431094`; no HF cache used |
| Embedded NeMo config | NeMo checkpoint config version `2.3rc5`; `canary2`; 16 kHz; 128 mel bins; 10 ms hop; per-feature normalization; 32-layer, 1024-wide encoder; 8-layer, 1024-wide decoder; 16,384 vocabulary |
| Language metadata | Export used the model card’s 25 language codes: `bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk`. Token presence alone was not treated as proof of language support. |
| Tokenizer check | All 16,384 embedded SentencePiece IDs matched NeMo’s unified Canary BPE mapping; no ID/piece mismatches. Byte fallback is disabled in the checkpoint. |

The downloader receipt is at `tmp/canary-1b-port-20261024/download-receipt.json`; checkpoint metadata and native reference details are in `tmp/canary-1b-port-20261024/checkpoint-inspection.json`. No model weights or audio payloads are included in this report.

## Export and bundle

The pinned local checkpoint was restored on CPU in the existing NeMo 3.0 overlay environment (Python 3.10.12, PyTorch 2.6.0+cu118, ONNX 1.20.1, ONNX Runtime 1.23.2). Export used the existing Canary exporter with `--device cpu --skip-preprocessor`, the local checkpoint path and the explicit supported-language list. The decoder trace prompt was obtained from the restored model’s prompt encoder rather than hard-coded language-token IDs.

PyTorch wrote the encoder’s 487 external tensors as many per-tensor files. The exporter’s publisher consolidates them with the existing streaming ONNX repacker into one sibling sidecar and emits `external_data`, `preprocessor_valid_length_mode: "onnx"` and `preprocessor_normalization: "per_feature"` in `config.json`. The ONNX frame-length mode is independent of the selected frontend backend. The browser bundle must still select the JS frontend because this export skipped `nemo128.onnx`.

The auto-published real-graph bundle is in the task-local temp folder `asrjs-canary-v2-port-20261024/onnx-bundle-publisher`. It contains:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `encoder-model.onnx` | 4,013,476 | `af8a77ac3502e6471b87bfdf1cf0344f80b39a1e101dfa19a4aba461bef7741d` |
| `encoder-model.onnx.data` | 3,282,000,896 | `9904f1a89bdfe20e04f2828b2955d2bb91cb308c8a3e28bfcf68a88d8433d9aa` |
| `decoder-model.onnx` | 676,338,494 | `1d283495fc0b791886f0d1251e32c3f4ed14bbca42dfb9973e732a5227036845` |
| `tokenizer.json` | 342,397 | `2f74972b146840e829bc6fcd8f9086c8f2b76d520fa5d72b2642a74f8b58b4ae` |
| `config.json` | 1,347 | `0311edf848f5a37e39e13060505d2d36614813d8c8598afb69265fb6b1c2a7ab` |

The auto-published bundle was compared against the manually assembled, CPU-validated bundle. All five file sizes and SHA-256 values matched, including independently hashed encoder sidecars. Path-based ONNX checks passed; the encoder references all resolve to `encoder-model.onnx.data`, while the decoder has no external tensors.

## Reference and ONNX Runtime gates

The native reference used `tests/fixtures/jfk2.en.wav` (SHA-256 `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`), 11 seconds, on CPU. NeMo and the manual decoder produced the same canonical text. Their raw token-list lengths differ by one terminal EOS: the manual helper includes EOS and NeMo greedy omits it. The 31 content-token IDs and text match; raw token arrays were not reported as identical.

The hardened ONNX verifier ran with frozen NeMo features, ONNX Runtime `CPUExecutionProvider`, and 2 intra-op / 1 inter-op threads:

| Gate/measurement | Result |
| --- | --- |
| Preprocessor | `SKIPPED_REFERENCE_FEATURES`; frontend parity not claimed |
| Encoder output shape and valid lengths | Match |
| Encoder mask | Shape and values match exactly; max/mean absolute difference `0` |
| Encoder states | Shapes match; max absolute difference `4.461407661437988e-05`, mean `5.490412036124326e-07`; reported `MEASURED_NOT_GATED` because no numeric tolerance is defined |
| Decoder token IDs and text | Exact match |
| Browser, WASM/WebGPU, microphone, VAD/streaming | `NOT_RUN` |

The verifier now exits nonzero on shape, length, exact binary-mask, token-ID or text mismatches. Runtime errors emit structured JSON and exit `2`; skipped frontend work is never reported as a frontend pass.

## Changes and regression coverage

- `canary_common.py`: added explicit local `.nemo` restore with no Hub fallback for a missing path; supports the checkpoint’s unified `CanaryBPETokenizer` and emits the preprocessor contract.
- `export_canary_onnx.py`: added local-checkpoint/language options, model-derived decoder prompt IDs, explicit external-data export, path validation and streaming single-sidecar publication. Output directories must be empty to avoid overwriting or mixing bundles.
- `verify_canary_onnx.py`: added fail-closed gates, explicit scope and nonzero exit behavior while leaving numeric differences measured but ungated.
- `tools/model-debugging/tests/test_canary_common.py`: **9 tests passed**, including the real embedded tokenizer, external-data publication, mismatched decoder exit code, exact-mask mismatch and skipped-frontend scope.

The first export attempt exposed a Numba cache loop in `tempfile._mkstemp_inner` during librosa import. It was stopped and retried with a task-local `NUMBA_CACHE_DIR`; both bundled models restored and export completed. No global Python cache or environment settings were changed.

Detailed machine-readable evidence is in `tmp/canary-1b-port-20261024/onnx-export-verification.json`.

## Browser follow-up: v10

The v9 browser attempt failed at session creation because the AED executor did not forward the external-data descriptor. After that shared loader fix, v10 completed exact normalized JFK file transcription in all four demos. Realtime manual, TEN-VAD-only and FireRed-VAD-only controlled Chromium captures, and Playground record-then-transcribe, also matched. Parent verification independently checked the nine completed receipts, their fixture identity and empty error arrays. These are fixture-specific checks; physical microphone remains **UNVERIFIED**.

Showcase, Realtime and Playground requested encoder WebGPU / decoder WASM. Benchmark completed GPU-backed inference, but its receipt does not expose effective component assignments; no decoder-WASM claim is made for Benchmark. Playground native streaming reported **UNSUPPORTED_OFFLINE_FALLBACK** and failed streaming acceptance. Offline Canary transcription does not become native streaming through that fallback.

Both all-WASM FP32 load attempts failed with `std::bad_alloc`, before transcription: the original inline decoder bundle and a lossless external-decoder packaging variant. Parent comparison verified all 674 decoder tensor payload hashes and the graph structure unchanged for the variant; this establishes storage equivalence, not successful WASM execution. WASM acceptance remains failed at model loading.

Browser receipts are under `demos/tmp/real-e2e/remaining-models-20261024/canary-v2-v10-hybrid/` and `canary-v2-v11-hybrid/`. The latter folder's name is historical: its manifest correctly records **served SDK v10**, not v11. The all-WASM failures are in `canary-v2-v10-wasm/` and `canary-v2-v10-wasm-external-decoder/`. The interrupted older Playground file receipt is incomplete; the fresh `playground-allflows-round2/` receipts supply completed file/record results and the separate unsupported-streaming result.

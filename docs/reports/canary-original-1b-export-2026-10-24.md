# Original Canary 1B: native reference, own export and browser validation

This is **`nvidia/canary-1b`**, distinct from Canary 1B v2 and 180M Flash. It now completes the pinned full-JFK file test in all four demos using SDK v12 and its own FP32 ONNX export. Realtime controlled manual/TEN-VAD/FireRed captures and Playground record-then-transcribe also match the complete normalized oracle. Native streaming remains unsupported; physical microphone remains unverified.

## Source and checkpoint contract

The [original checkpoint](https://huggingface.co/nvidia/canary-1b/blob/5113160c7a3e52a7e5de1d904810d7aeabe6135b/canary-1b.nemo) is pinned to revision `5113160c7a3e52a7e5de1d904810d7aeabe6135b`, 4,071,127,040 bytes, SHA-256 `b0284183a9a1e039a2fff39427e2991fa4df0b9612a3447fc33ff82b20fdfb5a`. Parent independently rehashed the checkpoint and all nine source/bundle/reference files listed in the port receipt. Public download used task-local temporary storage; original model folders and global Python environments were not modified.

The restored checkpoint identifies NeMo 1.23.0 and `EncDecMultiTaskModel`, the legacy `canary` prompt, aggregate tokenizer over `spl_tokens/en/de/es/fr`, 4,128 output classes, 24 encoder and 24 decoder layers, 1,024 hidden dimensions, 8 decoder heads and target limit 512. It uses 128 mel bins, 16 kHz, per-feature normalization. English ASR prompt IDs are `[3,4,8,4,11]`: start, English, transcribe, English, punctuation/capitalization. The copied 5,248-token/Canary2 metadata was corrected from this actual checkpoint.

Native CPU JFK inference and the manual decoder match all 38 content token IDs and the canonical words. Their raw arrays differ because manual output includes terminal EOS `2`; native output also has leading whitespace. Raw equality is not claimed. The fixture SHA-256 is `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`.

## Export and CPU gates

The existing Canary exporter now dispatches native prompt slots by declared format. Legacy `canary` uses source/task/target/PnC; `canary2` retains its existing slots. Three focused reference-tool regressions passed both agent and parent reruns. The exporter consolidates encoder sidecars and publishes a self-contained bundle with checkpoint-derived tokenizer/config metadata.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Encoder graph | 3,037,416 | `4f5d345a5a1dd1924dbf04ed6663dffcbc2858362bb1384f85082f403e9359c1` |
| Encoder sidecar | 2,476,071,936 | `0072b872b63ddc6738cd90bfb224cb6e9261908b37a8a7805fa442fbed823028` |
| Decoder graph, inline weights | 1,649,466,511 | `1f383ae2c422b2d6d1fcd8bcca9d3e4a73e2e034ea904529f7a55e215c8cfedf` |
| Tokenizer | 71,560 | `cfbfe38a3bdffc47021d0c64eec2aebeba6ff0b12ea9dafcc83386f458d27874` |
| Runtime config | 1,131 | `c64e1ad80554436eb1a475257f796ed5becdadce80a1b7b83b32cfff94f67bb5` |

The fail-closed CPU ONNX verifier returned `PASS_GATED_COMPONENTS`: encoder shapes, lengths and exact mask match; decoder token IDs and text match. Encoder max absolute difference `5.394220e-6`, mean `3.575599e-7` are **MEASURED_NOT_GATED**, without an invented tolerance. Frontend parity is **SKIPPED_REFERENCE_FEATURES** because this comparison uses native features. Browser acceptance is a separate actual run using the JS frontend.

## Actual v12 browser matrix

| Flow | Result | Inference GPU dispatches |
| --- | --- | ---: |
| Showcase file | Exact normalized full oracle | 1,530 |
| Realtime file | Exact | 1,530 |
| Realtime controlled manual capture | Exact | 1,535 |
| Realtime TEN-VAD-only capture | Exact | 6,120 |
| Realtime FireRed-VAD-only capture | Exact | 7,650 |
| Playground file | Exact | 1,535 |
| Playground controlled recording | Exact | 1,530 |
| Benchmark file, round 2 | Exact | 1,535 |
| Playground native streaming | **FAIL: UNSUPPORTED_OFFLINE_FALLBACK**, zero native steps | Fallback GPU work is not a native-streaming pass |

Showcase, Realtime and Playground request encoder WebGPU / decoder WASM. Benchmark exposes aggregate WebGPU Hybrid; component assignments are not separately exposed. Nonzero dispatch proves GPU work, not complete graph coverage. Parent independently checked all eight accepted receipts against model identity, immutable fixture SHA, complete normalized oracle and empty verifier/page/console error arrays. Controlled captures are Chromium fake-audio tests; physical microphone is **UNVERIFIED**.

The first Benchmark attempt stopped before loading because its single-source UI has no source toggle and the runner required descriptor confirmation. The actual SDK descriptor was independently checked as local=true/Hub=false; a fresh output folder reran with `--descriptor-sources local`. The initial failed receipt remains. There is no verified default Hub ONNX source for this model; catalog Hub support is disabled until one exists. Legacy prompt metadata no longer advertises unsupported ITN/timestamp controls.

## Evidence

- [Durable CPU port receipts and independent hashes](evidence/model-ports-v12-2026-10-24/canary-original/parent-hash-verification.json).
- Native/bundle files remain in task-local `C:/Users/steam/AppData/Local/Temp/asrjs-canary-1b-port-20261024/`; no weights are included in the repository report.
- Browser receipts: `demos/tmp/real-e2e/remaining-models-20261024/canary-original-v12-hybrid/`, including `parent-matrix-verification.json` and `served-v12-provenance.json` (all four served HTML/entry-script hashes checked against disk).
- SDK v12 archive SHA-256: `14e37ce25da6ed7ef1988c4ebc8447bbd92ea33c9ff56b08b0ecca0c0f327ad9`; source suite 1,926 passed / 36 skipped, root graph 44, demo suite 68 passed and all four builds passed. These local code gates do not certify other models or backends.

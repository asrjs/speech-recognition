# ASR.js Release Status

## Release Gates

| Gate | Milestone | Status | Deliverables |
|---|---|---|---|
| G0 | Baseline Evidence Package | COMPLETE | `speech-recognition/release-baseline.json`, `STATUS.md`, Git & PR reconciliation |
| G2 | Reliability, Streaming Contracts & Soak Qualification | COMPLETE | TEN-VAD leak fix (#254), telemetry counters, synthetic PCM soak, `release-reliability-2026-10-31.md` / `.json` |
| G3 | Performance, Accuracy & Reproducible Benchmarks | COMPLETE | Benchmark JSON export, warm/cold execution tracking, GPU dispatch audit, `benchmark-parakeet-v3-baseline-2026-10-31.md` / `.json` |
| G4 | Developer Experience & SDK Productization | COMPLETE | Golden Paths 1-4, capability descriptors, tarball smoke test (1,781 files), root import graph (44 static modules), documentation sync |
| G5 | Demo Suite Polish & Showcase | COMPLETE | UI polish across suite portal and 4 demos (`realtime`, `showcase`, `playground`, `benchmark`), 124 demo tests pass, dist build |
| G6 | Release Readiness & Checklist | COMPLETE | Comprehensive Go/No-Go report (`release-readiness-2026-10-31.md`), safety compliance, PR readiness (`READY_FOR_RELEASE`) |

## Repository and Pull Request State

| Repository | Active Branch | HEAD Commit | Open PR | Status | Notes |
|---|---|---|---|---|---|
| `speech-recognition` | `feature/maintenance-and-release-fixes` | `a4dd0947` | PR #255 | OPEN | Includes fixes for R1 (whisper <=783 lines), R4 docs links, P0-221..224 |
| `demos` | `feature/ort-deduplication` | `052a243` | PR #9 | OPEN | ONNX Runtime asset deduplication to root `dist/ort/` (saves 250.8 MB) |
| `speech-recognition` (VAD) | `fix/ten-vad-destroy-memory-leak-3034539744886247313` | `3034539` | PR #254 | OPEN | Fixes WASM memory leak in `ten-vad-worker.ts` on destroy error; unmerged |

### Safety Constraints
- `docs/reports/handover-2026-10-30.md`: Verified untracked and gitignored (`.gitignore:21:docs/reports/handover-*.md`).
- `NODE_TLS_REJECT_UNAUTHORIZED`: Explicitly unset for all network/git operations.
- Publishing restriction: No npm publish or Hugging Face push without explicit approval.

## Verified Baseline Matrix

| Combination | Status | Claim | Evidence Path | Notes |
|---|---|---|---|---|
| Parakeet v3 FP16 WebGPU + INT8 WASM | `PASS` | Full pipeline streaming ASR on WebGPU (encoder) and WASM (decoder/joint) with verified accuracy | `speech-recognition/docs/reports/evidence/microphone-composition-2026-10-25/real-run-final/receipt.json` | Verified on Chrome hardware WebGPU JFK fixture and real microphone input |
| Parakeet v3 WASM fallback | `PARTIAL` | CPU WASM fallback for both encoder and decoder/joint when WebGPU is unavailable | `speech-recognition/tests/nemo-tdt-native-ort.test.ts` | Functional fallback paths implemented and unit-tested; cold/warm benchmark performance receipts pending G3 |
| Whisper 4-Graph | `PARTIAL` | Whisper 4-Graph architecture for Tiny/Base/Large-v3-Turbo with externalized graphs | `speech-recognition/docs/reports/whisper-large-v3-turbo-variant-validation.json` | PR #255 fixed line count constraint (<=783 lines); models exported; regression anchor pending |
| Moonshine streaming tiny | `ARTIFACT-GATED` | Streaming Moonshine tiny encoder with cross-KV caching | `speech-recognition/models/onnx/moonshine-streaming-tiny-fp32/export-receipt.json` | Experimental architecture with local ONNX weights; public weights not published to HuggingFace |
| X-ASR | `ARTIFACT-GATED` | Zipformer2 causal chunk streaming with 160ms chunk size | `speech-recognition/src/runtime/experimental-families.ts` | Causal streaming contracts verified; public model weights gated |
| Nemotron RNNT | `ARTIFACT-GATED` | Nemotron 0.6B RNNT streaming model with bounded context | `speech-recognition/docs/reports/evidence/model-ports-v11-2026-10-24/nemotron-native/official-reference-jfk2-en-prompt-en.json` | Official reference parity demonstrated; model port paused and artifact-gated |
| Qwen3-ASR 0.6B | `ARTIFACT-GATED` | Dynamic audio frontend with decoder step ONNX graphs in FP16/FP32 | `speech-recognition/tests/smoke/qwen3-asr-node-wasm-benchmark.mjs` | Verified in Node WASM on JFK fixture; browser WebGPU execution requires public weights distribution |
| Native Node/Bun Resource Lifecycle | `PASS` | Native ONNX runtime resource management, session disposal, and environment creation | `speech-recognition/tests/native-resource-smoke.test.ts` | 125-byte Identity ONNX graph passes cleanly in Node and Bun smoke tests without leaks |
| Native Node ASR Inference | `UNTESTED` | End-to-end ASR speech transcription inside pure Node.js runtime | `speech-recognition/tests/smoke/transcribe-fixture.mjs` | Native ORT binding and audio decoders are untested for full pipeline inference parity; planned for post-release |
| TEN-VAD Worker Cleanup (#254) | `PASS` | WASM pointer deallocation and lifecycle cleanup on ten-vad worker termination | `speech-recognition/docs/reports/release-reliability-2026-10-31.md` | Remediated in G2: try...finally deallocation, private terminateWorker() on error/init/dispose, zero leaks verified |

## Approved Models and Artifact Hashes

### Primary Production Profile: `parakeet-tdt-0.6b-v3`
- Profile ID: `webgpu-fp16-wasm-int8-v1`
- Hugging Face Repository: `ysdede/parakeet-tdt-0.6b-v3-onnx`
- Pinned Revision: `f88260fa0777fe0868dda6df85d1a98f012a4a7a`
- Status: `approved`

| Component | File Path | Size (Bytes) | SHA-256 | Backend | Precision |
|---|---|---|---|---|---|
| Encoder | `encoder-model.fp16.onnx` | 1,238,960,452 | `a2bdeeb99cb7e5548818e823127b33854dd0c26f5d0c8da91effdd895ea0e717` | WebGPU | FP16 |
| Decoder / Joint | `decoder_joint-model.int8.onnx` | 18,202,004 | `eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70` | WASM | INT8 |
| Config | `config.json` | 101 | `562aedb0b804acbeb4e04215de190846c37e0bb239b696376df69d555785373a` | - | - |
| Vocabulary | `vocab.txt` | 102,132 | `ba8e4007c65f4bb4358ffe2ecc13d9ccc7a10351151065242b5c3a943e685742` | - | - |

## Audio Fixtures Inventory (35 Files)

| # | Relative Path | Size (Bytes) | Format / SR / Ch / Duration | SHA-256 | License | Gold Transcript / Reference Source |
|---|---|---|---|---|---|---|
| 1 | `tests/fixtures/00a74da8fdcf346733fb3186ba622b66298714d6b8e51717680151a6ae31abcc_04.en.wav` | 846,444 | WAV / 16000 Hz / 1 ch / 26.450s | `58ce74b97dfb2c459966baf899a98e217d14130f23dc431b8b86aba121da4335` | MIT (Kokoro-82M TTS) | Functionally, the lower leg is supplied by two vessels: ay tee pee and fibular artery... |
| 2 | `tests/fixtures/019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav` | 596,552 | WAV / 16000 Hz / 1 ch / 18.641s | `b2b2a48e917527bda2ba97f161e642943d63fbd1c703c6b51fae882857f9f363` | CC-BY (XTTS v1 TTS) | Bulaşıcı hastalıkların beklenmedik zamanlarda yaptıkları salgınlar... |
| 3 | `tests/fixtures/0b89c2ac3fc76616b77b958030f97546b503ac842d9e51ad5173d54ea6811458_01.en.wav` | 322,444 | WAV / 16000 Hz / 1 ch / 10.075s | `ff7d3472ccd412e26e4f134ff60f51091d71b56f8d0e2fea0d1484454916a9a2` | MIT (Kokoro-82M TTS) | No evidence of periaortic collections... |
| 4 | `tests/fixtures/12_dans.tr.m4a` | 10,713,797 | M4A / compressed | `8b6a99ce7fa43ed2acb5c1211f44d4eb614e22a62bf5f7d4d7cebe193749353a` | Open Access (Turkish Fairy Tale / LibriVox TR) | `tests/fixtures/12_dans.tr.vtt` |
| 5 | `tests/fixtures/common_voice_tr_38277367.tr.mp3` | 11,961 | MP3 / compressed | `02011be4bfd08302fd662a9b8821cda5a06280910ddf70b06596cbaefa4fdef4` | CC0-1.0 (Mozilla Common Voice) | Burada yazı artık okunmaz bir şekil alıyordu. |
| 6 | `tests/fixtures/common_voice_tr_38284290.tr.mp3` | 5,949 | MP3 / compressed | `da5eb2459b73e002abd6189c331450193139d6090344191eea8b5923381f2c70` | CC0-1.0 (Mozilla Common Voice) | Daha sonra bu görevi bir süre vekâleten yürüttü. |
| 7 | `tests/fixtures/common_voice_tr_38284299.tr.mp3` | 15,957 | MP3 / compressed | `25b5670e9958e23c2d6bff404340eb2020da1648d8f49ee4b6ee3d64f9980f4e` | CC0-1.0 (Mozilla Common Voice) | Bu durumun tek istisnası, daha sonra değineceğimiz gibi, çöl kurbağasıdır. |
| 8 | `tests/fixtures/common_voice_tr_38289682.tr.mp3` | 31,941 | MP3 / compressed | `a284496198fcded7aa531e84f98250cf14b4aeddf943c2c9776f103f8fa321f6` | CC0-1.0 (Mozilla Common Voice) | Ailesi ona 'küçük prenses' derdi. |
| 9 | `tests/fixtures/end-of-chapter-4.en.mp3` | 1,340,192 | MP3 / compressed | `bcb1544bedab93c7ec97734dce50316bbaef5ca59377f75d0f9e1eff4dac784c` | Public Domain (LibriVox) | The very close examination thus afforded me... |
| 10 | `tests/fixtures/ItsLifeJim.en.wav` | 100,314 | WAV / 16000 Hz / 1 ch / 3.133s | `720029790d0718aff094b0e1c353d7890bce1c9feba0029f935cd82b3a804e66` | Fair Use / Benchmark sample | It's life Jim but not as we know it |
| 11 | `tests/fixtures/jfk2.en.wav` | 352,078 | WAV / 16000 Hz / 1 ch / 11.000s | `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e` | Public Domain (US Gov) | and so my fellow Americans ask not what your country can do for you ask what you can do for your country |
| 12 | `tests/fixtures/JFK_Short.en.wav` | 548,504 | WAV / 16000 Hz / 1 ch / 17.139s | `77468f8e15fd3b4132d7ecb75a3a6794cde6e071704770485da68a744aa40876` | Public Domain (US Gov) | and so my fellow Americans ask not what your country can do for you ask what you can do for your country |
| 13 | `tests/fixtures/librivox.org-1600hz.en.wav` | 599,086 | WAV / 16000 Hz / 1 ch / 18.714s | `799de7f940b9086723ced5691cfc7aed5b3af59f3df74b170a6518751db4ba7c` | Public Domain (LibriVox) | LibriVox 16kHz speech sample |
| 14 | `tests/fixtures/train_000015.tr.mp3` | 87,093 | MP3 / compressed | `898bfc0a59beef008e240601f498aa8938687df19a93bbef628bad16204d8b79` | CC-BY-SA-4.0 | Turkish speech training/eval corpus |
| 15 | `tests/fixtures/train_000017.tr.mp3` | 199,557 | MP3 / compressed | `c8a48f643a009188553f3e594add6d08aa187d060d04bdb56a224f906d4834cc` | CC-BY-SA-4.0 | Turkish speech training/eval corpus |
| 16 | `tests/fixtures/train_000058.tr.mp3` | 33,345 | MP3 / compressed | `15fe1f9093e3eadd9af8561c2b5e6b059abe9c2c8549c35ad18c657d649332fc` | CC-BY-SA-4.0 | Turkish speech training/eval corpus |
| 17 | `tests/fixtures/train_000085.tr.mp3` | 111,141 | MP3 / compressed | `966d50fb9a2f316a060eeea911a1df0b0884fe9c047810b372930759d3607154` | CC-BY-SA-4.0 | Turkish speech training/eval corpus |
| 18 | `tests/fixtures/train_000094.tr.mp3` | 246,681 | MP3 / compressed | `0767e92c1d81dfe9156ac9665b9d3ae8403c4773dc2186faa0eeb29a9dcec1bc` | CC-BY-SA-4.0 | Turkish speech training/eval corpus |
| 19 | `tools/data/fixtures/audio/Harvard-L2-1.ogg` | 18,235 | OGG / compressed | `6e8379d6a016c3b9606f952d2ef5049740476dc98f2050e9bf30769af610935c` | Public Domain (IEEE) | The birch canoe slid on the smooth planks. |
| 20 | `tools/data/fixtures/audio/jfk-short.wav` | 352,078 | WAV / 16000 Hz / 1 ch / 11.000s | `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e` | Public Domain (US Gov) | and so my fellow Americans ask not what your country can do for you ask what you can do for your country |
| 21 | `tools/data/fixtures/audio/JFK.ogg` | 434,941 | OGG / compressed | `ab126000032326b5ac98faf187bc8f8506e78c6fd7a1f9600c0c945e4e52284c` | Public Domain (US Gov) | and so my fellow Americans ask not what your country can do for you ask what you can do for your country |
| 22 | `tools/data/fixtures/audio/librivox-blankgaps-synthetic.wav` | 1,790,504 | WAV / 22050 Hz / 1 ch / 40.600s | `3479a461cfca404165c71905e9b66a77faaee685434134786e8fed5161e60ba7` | Public Domain (LibriVox / Synthetic) | - |
| 23 | `tools/data/fixtures/audio/librivox-doubled.wav` | 1,677,468 | WAV / 22050 Hz / 1 ch / 38.037s | `36d8e707d5bb6413fcdcdd69aafa84d109fb7d3e7fe6acd9137e0a4c01274413` | Public Domain (LibriVox / Concatenated) | - |
| 24 | `tools/data/fixtures/audio/librivox.org.16k.pcm16.wav` | 598,894 | WAV / 16000 Hz / 1 ch / 18.714s | `ab27bd101385c5e78fad1207a46b079723ccafa76fb6b2c8047ea54aabebd6b9` | Public Domain (LibriVox) | - |
| 25 | `tools/data/fixtures/audio/librivox.org.16k.wav` | 1,197,780 | WAV / 16000 Hz / 1 ch / 18.714s | `d4ce45c6391f2bd4195baa0bb2f8d412f83d8599ba9756bd323725ef80cf3f88` | Public Domain (LibriVox) | - |
| 26 | `tools/data/fixtures/audio/librivox.org.wav` | 825,526 | WAV / 22050 Hz / 1 ch / 18.714s | `2f6886c1956765b56b996bd6fbb00c5e4001368db8cb4be94987ca2e3166b8b4` | Public Domain (LibriVox) | - |
| 27 | `tools/data/fixtures/audio/omnilingual/de.wav` | 121,388 | WAV / 22050 Hz / 1 ch / 2.752s | `36d3c4845b9808a1656a2a2e92d884590e2db94389e6fe559643291ae0cd3710` | CC-BY-4.0 / Open | - |
| 28 | `tools/data/fixtures/audio/omnilingual/en.wav` | 184,608 | WAV / 24000 Hz / 1 ch / 3.845s | `148b936b43ce7c546a866e64da059f0458aee2d65e617f16e9d94f06e8d99ed6` | CC-BY-4.0 / Open | - |
| 29 | `tools/data/fixtures/audio/omnilingual/es.wav` | 235,052 | WAV / 22050 Hz / 1 ch / 5.329s | `49fd2cfa4b62db7068143c582b35de9d31ec2733495ece3611105131d21de06c` | CC-BY-4.0 / Open | - |
| 30 | `tools/data/fixtures/audio/omnilingual/fr.wav` | 219,180 | WAV / 22050 Hz / 1 ch / 4.969s | `b59be4349b92d344fb903677165eaf4694025d1ab119c608726ecbcb3164b528` | CC-BY-4.0 / Open | - |
| 31 | `examples/demo/public/audio/jfk.ogg` | 434,941 | OGG / compressed | `ab126000032326b5ac98faf187bc8f8506e78c6fd7a1f9600c0c945e4e52284c` | Public Domain (US Gov) | and so my fellow Americans ask not what your country can do for you ask what you can do for your country |
| 32 | `demos/apps/benchmark/public/assets/life_Jim.wav` | 224,248 | WAV / 44100 Hz / 1 ch / 2.542s | `7cedf38d2c2b669ebf8c9345aacda4f0bea3b79cfb4c5c4d6256761603c212e4` | Fair Use / Benchmark sample | It's life Jim but not as we know it |
| 33 | `demos/apps/playground/public/assets/Harvard-L2-1.ogg` | 18,235 | OGG / compressed | `6e8379d6a016c3b9606f952d2ef5049740476dc98f2050e9bf30769af610935c` | Public Domain (IEEE) | The birch canoe slid on the smooth planks. |
| 34 | `demos/apps/realtime/public/assets/Harvard-L2-1.ogg` | 18,235 | OGG / compressed | `6e8379d6a016c3b9606f952d2ef5049740476dc98f2050e9bf30769af610935c` | Public Domain (IEEE) | The birch canoe slid on the smooth planks. |
| 35 | `demos/apps/showcase/public/assets/Harvard-L2-1.ogg` | 18,235 | OGG / compressed | `6e8379d6a016c3b9606f952d2ef5049740476dc98f2050e9bf30769af610935c` | Public Domain (IEEE) | The birch canoe slid on the smooth planks. |

## Known Issues and Resolution Status

### Milestone G2 Resolution: TEN-VAD Worker Pointer Leak (PR #254) — RESOLVED
- **Defect**: In `src/runtime/ten-vad-worker.ts`, `handleDispose()` previously risked leaking WASM memory if `_ten_vad_destroy()` threw an error.
- **Resolution**: Wrapped `_ten_vad_destroy()` in `try ... finally` block ensuring unconditional freeing of all WASM pointers (`[audioPtr, probPtr, flagPtr, handlePtr]`), added `terminateWorker()` helper on init/error/dispose in `ten-vad-browser.ts`, and verified zero leaks across all error paths in `tests/ten-vad-worker-abi.test.ts` and `tests/browser-ten-vad.test.ts`. Status: **RESOLVED & VERIFIED**.

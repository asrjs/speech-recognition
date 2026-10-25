# Native controls for remaining quality failures

These fresh CPU controls separate canonical transcript quality from agreement with the source model. A native model making the same word error does not turn that error into a quality PASS. Parent independently checked 26 source/audio/gold hashes and the saved token/text comparisons. [Durable receipts and parent verification](evidence/native-quality-controls-2026-10-24/parent-verification.json).

| Model / fixture | Fresh native result | Scope |
| --- | --- | --- |
| Parakeet HF CTC 110M / full JFK | `askk not`, 34 collapsed IDs matching the saved official oracle | Canonical quality **FAIL**; this word also occurs in the native model output |
| Turkish XLS-R 300M / Common Voice 38277367 | `burada yazı artık okulmaz bir şekil alıyordu` | Canonical quality **FAIL** (`okunmaz` expected); native/browser words agree, numerical parity not claimed |
| MedASR / labeled 10.45 s medical fixture | Native and CPU ONNX both match the gold; frame IDs and collapsed IDs agree | One-fixture native/ONNX quality and token agreement; broad medical accuracy not claimed |

## Parakeet CTC 110M

The fresh run used pinned local source weights offline, two CPU threads, PyTorch 2.10.0 CPU and Transformers 5.3.0.dev0. Input features were `[1,1101,80]`, with 1100 valid attention frames; logits `[1,138,1025]`, blank/pad ID 1024. The emitted pieces around the error are `▁ask`, `k`, `▁not`, which correctly concatenate to `askk not`. Native IDs match the stored 34-token oracle. The current four-demo audit also reports `askk`; no decoder text patch or gold rewrite was applied.

The first environment attempt lacked librosa. Subsequent task-local runners first encountered a denied Numba cache write and then invalid assumptions about processor blank-ID attributes. Those stopped before inference and remain listed as environmental/harness failures. The completed run derives and checks blank ID from the model config and tokenizer. Parent rehashed the checkpoint, config, tokenizer and exact JFK WAV.

## Turkish XLS-R

Source snapshot: `mpoyraz/wav2vec2-xls-r-300m-cv7-turkish`, `708639f50559d7970f462e13ec64d3f059ca89f6`. The immutable WAV SHA is `9cbbc554318fa4d06cdf3be659485cb09bdbf82dacd8d68fa1d45a78eb4b1c34`; the gold text SHA is `703f6299e65713656d7c5e3e5bdf4de270b381a82e184cfc24832dbe887eaa88`.

Native FP32 CPU inference used the standard feature extractor and greedy CTC tokenizer/processor, 24 layers, hidden size 1024 and vocabulary 38; logits were `[1,169,38]`. The snapshot advertises a processor with language-model decoding, but its optional `pyctcdecode` dependency is absent. The control therefore explicitly uses greedy CTC, matching the browser's decoder; it is not an LM-decoder comparison. Native output contains the same `okulmaz` error as the browser. Parent verified source hashes before/after and the unchanged gold; no vocabulary substitution was used to hide the error.

## MedASR medical control and real demos

The labeled fixture is `tools/model-debugging/reference/medasrjs/upstream-tests/fixtures/sanity_sample.wav`, SHA `719445c26e8e35d1dd9c61a3d49512ef4f3585f4cb3e873118bf14e66d07ab95`, 16 kHz mono, 167,200 samples. Its adjacent label SHA is `e5b1a27f8ba65adcd61863685c34679997132b176e26a3ab7fbbcce7d12e3b79`:

> The articular cartilage of the femoral trochlea and the medial and lateral articular surfaces of the patella appear normal. No abnormalities are noted in the popliteal fossa.

Cached `google/medasr` source revision `7383e5e461baa820bdf9060652fe51b333bafba5` was compared to the existing FP32 export without rewriting either. Native and ORT use features `[1,1043,128]`, mask `[1,1043]`, logits `[1,258,512]`, blank ID 0; both texts match the label, and frame/collapsed IDs agree. Maximum absolute logit difference is `2.47955322265625e-05`, measured without inventing a numeric acceptance tolerance. Parent independently checked the source/export/gold hashes and token comparisons.

The preset's stale vocabulary 32 / blank 31 metadata was corrected to 512 / 0. The tokenizer file also contains added-token rows after model output ID 511; those are not extra output classes. The shared local adapter validates config, frontend and contiguous token IDs before binding the graph, its exact external-data filename and tokenizer. All four demos use that common adapter.

Actual SDK v14 WASM results on the medical fixture:

- All four file-transcription flows: exact normalized full label, zero inference GPU dispatches, no runtime/verifier errors.
- Realtime controlled manual capture and Playground record-then-transcribe: exact.
- Realtime TEN-VAD-only and FireRed-VAD-only: **FAIL**; missing first word and suffix fragments at independently decoded chunk boundaries.
- Playground native streaming: **UNSUPPORTED_OFFLINE_FALLBACK**, zero native steps; no native-streaming PASS.
- Physical microphone: **UNVERIFIED**. Captures are Chromium fake-audio tests.

The opt-in `--capture-mic-pcm` harness now preserves the actual float32 buffers sent to the ASR worker before transfer, with request IDs, frame counts, on-disk sizes and SHA checks. A synthetic Chrome worker test verifies unchanged request/transfer semantics. The first capture attempt's naming-collision failure is preserved; the corrected rerun uses a new evidence directory.

Native CPU ORT on the **actual captured buffers** reproduces the browser's per-chunk words, including missing `The` and suffix `s`/`s of` at window edges. Full captured manual audio remains exact. This locates the symptom in the segmentation/context presented to the recognizer rather than a demonstrated WASM-versus-native decoder difference. It does not establish that a text-merging patch or arbitrary preroll change is safe. Forced-duration crops and continuation overlap can bisect a word; treating their independently decoded text as complete utterances loses context.

The manual model-bound capture has 327,936 frames, versus the nominal padded fixture's 327,200, and alignment differs by about 70 ms. Earlier fixture-index crop reconstruction is therefore retained as an unproven approximation, not byte-identical captured input. The actual `.f32le` captures are the provenance for the native crop control. Audio stays in task-local temporary evidence, not the committed JSON archive.

The original `medical-control.json` had literal backslash-n characters after an otherwise complete JSON document. Parent preserved its raw bytes as `medical-control.invalid-json.txt` and wrote `medical-control.validated.json` by replacing only that trailer with a newline; all data fields match. The repair and both hashes are recorded, rather than silently accepting malformed evidence.

Browser evidence: `demos/tmp/real-e2e/remaining-models-20261024/medasr-medical-v14-wasm/` and `medasr-medical-v14-wasm-pcm-capture-retry/realtime/`. These results supersede neither other model variants nor untested backends; the earlier JFK MedASR failures remain separate non-medical tests.

The [durable captured-PCM metadata/native-control bundle](evidence/native-quality-controls-2026-10-24/medasr-captured-pcm/manifest.json) contains the small retry and failed-attempt JSONs only. Parent independently rechecked all 18 durable hashes and all eight original PCM file hashes and sizes (`bytes = frames × 4`). No audio payload was copied into documentation. These controls reproduce the chunk words; the TEN/FireRed quality gates remain FAIL.

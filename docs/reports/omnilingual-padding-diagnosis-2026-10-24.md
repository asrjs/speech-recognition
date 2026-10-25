# Omnilingual CTC padding sensitivity — 2026-10-24

## Finding

The browser file path and CPU ONNX path produce different transcripts for the same JFK2 speech when the model input sequence is extended with silence. The controlled CPU test isolates sequence/context length as a sufficient cause of the change: adding one second of zero samples **after normalization**, without rescaling or changing any original speech sample, changed the transcript from Arabic gibberish to mostly English. Across the original 549 aligned output frames, 83 argmax IDs changed (15.1%).

This establishes input-length sensitivity in the tested exported graph. It does **not** establish that padding is required, that the graph diverges from the original checkpoint, or that adding silence is a valid repair. No padding workaround was implemented, and the existing file-flow failure remains a failure.

## Reproduction evidence

All runs used the local INT8 ONNX graph with CPU providers. The Python trace used ONNX Runtime 1.29.0 / `CPUExecutionProvider`; the public-library Node trace used `loadSpeechModel` with its WASM CPU backend. The exact local artifacts were:

| Input / artifact | Evidence |
| --- | --- |
| Original `tests/fixtures/jfk2.en.wav` | SHA-256 `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`; 16 kHz mono; 176,000 samples; 549 graph frames. |
| `model.int8.onnx` | SHA-256 `38a6cc569d46eb382f27d85b3208b883e834ff281b2057a606ec79e4ae501545`. |
| `tokens.txt` | SHA-256 `7d99997ef207ff14c2cfe825f2aa037528ea250113cc3c6392bfe49326884ba6`. |
| Saved browser capture `jfk2.en-padded-5s.wav` | SHA-256 `f712d5bee48ce06fa60c292c3bc21d1210bc5a39fdf25078b435c4c406c195df`; 16 kHz mono; 336,000 samples. Its middle 176,000 PCM samples are byte-identical to the original; both ends are exactly 80,000 zero samples. Source: `N:\github\asrjs\demos\tmp\real-e2e\remaining-models-20261024\flows-round3\omnilingual\realtime\jfk2.en-padded-5s.wav`. |

| CPU input | Result |
| --- | --- |
| Original, library normalization (`eps=1e-7`) | `  ل م            `; 549 frames. |
| Exact original plus 1 second trailing zeros, normalized with the same library path | `and so my fellow america ask not wat yor country can do for yo ask wat you can do for your contry`; 599 frames. |
| Exact original plus 5 seconds trailing zeros | Readable, near-full English transcript. |
| Exact original plus 5 seconds leading zeros | Readable English with minor word errors. |
| Saved browser capture with 5 seconds of zeros at both ends | Readable, near-full English transcript. |
| Original-normalized PCM plus 1 second trailing zeros appended **after normalization** | Transcript changes to readable English; 83/549 aligned frame argmax IDs change. This holds speech scale fixed and isolates sequence extension from re-normalization. |
| Earlier epsilon note (superseded by the native oracle below) | The earlier diagnostic receipt reported no changed frame IDs. The exact native-v2 counterfactual below corrects that: `eps=1e-5` versus `eps=1e-7` changes two INT8 argmax frames (65, 75), while the overall decoded Arabic character remains the same. |

The public-library Node WASM CPU path also returned readable English for the exact original plus five seconds of trailing zeros. Its original unpadded result was the same Arabic text as the Python CPU run. Saved traces contain waveform and normalized-waveform hashes, statistics, graph shape, first-frame logits/top IDs where captured, collapsed IDs, and transcripts.

## Official input contract reviewed

The [official Meta inference pipeline](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/inference/pipeline.py#L57) sets a 40-second maximum, [resamples to 16 kHz and normalizes input](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/inference/pipeline.py#L414-L431), and does not specify a minimum input length or padding-to-multiple rule. Its [CTC path](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/inference/pipeline.py#L282-L301) builds a `BatchLayout` with source sequence lengths, runs argmax, collapses consecutive IDs, then uses the tokenizer decoder with special-token skipping. The [official inference guide](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/inference/README.md) says models were mostly trained on clips up to 30 seconds and currently accepts inputs under 40 seconds; JFK2 is 11 seconds.

The [official audio helper](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/datasets/utils/audio.py#L20-L38) defines normalization as `torch.nn.functional.layer_norm(waveform, waveform.shape)` with PyTorch's default epsilon (`1e-5`) and averages channels to mono. The shared Wav2Vec2 normalizer defaults to `eps=1e-7`; the subsequent v16 change makes Omnilingual use its own default `eps=1e-5`. The old diagnostic note incorrectly stated the epsilon control caused no frame changes; the exact comparison in the new section below shows two INT8 frame argmax changes. The ONNX graph's feature-extractor layer-normalization `ReduceMean` nodes reduce on the final per-frame channel axis, not over the time axis. I found no evidence in the reviewed graph for temporal GroupNorm explaining the padding result.

In the official pipeline, batch collation uses zero as the audio pad value while retaining sequence lengths for `BatchLayout`. For the single-file case there is no documented extra padding requirement. Its CTC implementation does not state a numeric blank ID; it collapses adjacent argmax IDs and delegates special-token removal to the tokenizer. The tested graph exposes only raw waveform input `x` and no length/mask input; in the unpadded single-clip run every sample is valid. In the padding controls, appended zeros are therefore processed as additional audio context.

## Limits and status

- **Original checkpoint parity: UNVERIFIED in the padding-only runs below.** A later, separate native Meta v2 CPU control is documented after this historical analysis. It provides one-fixture FP32 numerical comparison; it does not make broad quality or browser/GPU claims.
- The earlier same-graph Python baseline's generating script was not present in the checkout. The included Python CPU script reconstructs its logged decode steps; it is not an upstream inference run.
- The earlier EN baseline receipt is self-consistency evidence for a different WAV, not an upstream oracle. That baseline WAV is 24 kHz and was fed directly to a graph declaring 16 kHz, so it cannot certify the correctly prepared JFK2 path.
- The specific cause within the exported model (attention/context effects, export behavior, or another architecture detail) remains unresolved. The controlled test shows the sequence extension changes logits, but does not identify an implementation defect.
- **Padding-flow acceptance: FAIL / unresolved.** No padding workaround was implemented, no oracle was lowered, and no source/runtime code was changed in that investigation.

## Reproduction files

The bounded evidence archive is [evidence/omnilingual-padding-2026-10-24](evidence/omnilingual-padding-2026-10-24/). It includes CPU JSON traces and small scripts only; it does not copy WAV fixtures or model/tokenizer weights. Main reproductions:

```powershell
python tmp\omnilingual-diagnosis\make-padding-variants.py
python tmp\omnilingual-diagnosis\python-cpu-trace.py tests\fixtures\jfk2.en.wav
python tmp\omnilingual-diagnosis\python-cpu-trace.py tmp\omnilingual-diagnosis\jfk2-right1.wav
python tmp\omnilingual-diagnosis\python-postnorm-pad-test.py
node tmp\omnilingual-diagnosis\node-public-api-cpu.mjs tests\fixtures\jfk2.en.wav
```

These are diagnostic CPU runs, not acceptance tests. The public Node script uses the repository's already-built `dist`; its model loading goes through the public `loadSpeechModel` API.

## Native Meta v2 CPU oracle and epsilon control — 2026-10-25

The previous padding-only trace did not compare the exported graph with the original checkpoint. This follow-up ran the official `facebookresearch/omnilingual-asr` source at commit [`81f51e224ce9e74b02cc2a3eaf21b2d91d743455`](https://github.com/facebookresearch/omnilingual-asr/tree/81f51e224ce9e74b02cc2a3eaf21b2d91d743455), model card `omniASR_CTC_300M_v2`, on CPU. The task-local WSL environment used Python 3.12.3, Torch/Torchaudio 2.8.0 CPU, fairseq2/fairseq2n 0.6, and ONNX Runtime 1.30.0. The v2 checkpoint SHA-256 is `8ce340ada22435d189908a8af67e3bb04899b6f2f5329d036248b9c3e38d2b50`; the matching `omniASR_tokenizer_written_v2.model` SHA-256 is `8aa11a1092142ef472537476ef6e76541123e2f0d789b79f3ebd119008240b1e`. These are v2 assets; the different v1 checkpoint/tokenizer were not used.

The FP32 ONNX file was retrieved from the pinned OpenVoiceOS repository commit and its bytes were verified. That repository card claims the v2 export, but the exact source checkpoint/export-run provenance was not independently attested; the one-fixture native comparison below is the available parity evidence.

The exact fixture WAV remains `tests/fixtures/jfk2.en.wav`, SHA-256 `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`. Meta's preprocessing produced 176,000 float32 samples at 16 kHz with float32-LE content SHA `ed1689e6d2515827b26aeb57f837915a6b34018693ea6df00d43adc6c0d6733d`. Its output equals `torch.nn.functional.layer_norm(raw_waveform, raw_waveform.shape, eps=1e-5)` bit-for-bit. With `eps=1e-7`, the normalized waveform differs by up to `0.0013494492` and has content SHA `3fc5629db0a1588a300ca77f0b910d33e279d7e41862442c8faa1df350cb6152`.

The native v2 model and both ONNX artifacts returned transcript `م`, not the immutable English JFK gold. Native and FP32 logits had shape `[1,549,10288]`; across 5,648,112 logits the FP32 graph's maximum absolute difference from native was `0.0004775524`, mean absolute difference `0.00000767374`, cosine similarity `0.9999999999989317`, and argmax agreement `549/549`. The independent verifier applied `atol=1e-3, rtol=1e-4` and confirmed `allclose=true`; this is a **scoped native-vs-FP32 numerical PASS for this fixture and these tolerances only**. The INT8 graph had max absolute difference `7.7892001`, mean absolute difference `0.26682373`, cosine similarity `0.9989255136`, and argmax agreement `547/549` against native under the official waveform; the same verifier found `allclose=false`, so INT8 numerical parity is **FAIL** under that gate.

The controlled INT8 epsilon counterfactual used the same Linux ONNX Runtime 1.30 session and graph for all three waveforms: official `eps=1e-5` and a separate `eps=1e-5` Torch calculation were bit-identical and yielded identical logits/IDs. `eps=1e-7` changed frames 65 and 75 (547/549 frame agreement), with maximum logit difference `4.4431852` and mean difference `0.06867649`. This establishes epsilon sensitivity in that controlled session, but it does **not** isolate the actual SDK-vs-native mismatch. A parent-run same-ORT 1.29 comparison of the compiled JS `eps=1e-5` waveform against the exact Meta waveform passed `allclose` at `atol=1e-6, rtol=1e-6` (maximum difference `4.7683716e-7`, mean `2.7264699e-8`). On INT8, that same-ORT input comparison changed frame 52; on the exact Meta waveform, changing only ORT from Linux 1.30 to Windows 1.29 changed frames 52, 65, and 75. The actual SDK INT8 replay against Linux ORT 1.30 INT8 logits on the exact Meta waveform failed `allclose` at `atol=1e-3, rtol=1e-4` (max difference `4.5597684`, mean `0.05998983`, cosine `0.9998951083`, argmax `547/549`). FP32 remained `allclose` at `atol=1e-3, rtol=1e-4` with no argmax changes. Thus the observed SDK-vs-native INT8 difference crosses runtime versions and both input and runtime effects matter; it cannot be assigned to epsilon alone.

The follow-up v16 code now exposes an optional family-level `normalizationEpsilon` and sets Omnilingual's default to `1e-5`; the shared Wav2Vec2 default remains `1e-7`. A focused regression checks the real JFK waveform against the official preprocessing summary and verifies the override. Library validation reported 1,947 passed / 36 skipped, root-import count 44, and demo validation 68 passed with 4 builds. The initial four-demo retest did not establish pure-WASM acceptance: Showcase and Realtime retained a WebGPU encoder despite the WASM runtime selection, and Benchmark stopped before loading because the harness lacked its single-source descriptor confirmation. Playground executed with wrong text and unsupported native streaming. All failures are preserved; an explicit-backend retry is required. The earlier CPU oracle itself made no code changes; the subsequent v16 fix is a separate implementation step. The normalization counterfactual also does not explain the Arabic output or the earlier padding/context sensitivity.

The explicit-backend v16 retry then completed all four file flows, Realtime manual/TEN-VAD/FireRed captures and Playground recording. Parent independently checked all eight receipts: loading and inference executed, inference GPU dispatches were zero, no page/console errors occurred, and every transcript failed the unchanged full-JFK oracle. File output differs slightly between apps (`م ل م` versus `ل م`); numerical equality across browser app configurations is not established. Playground native streaming is unsupported/offline fallback, not a native streaming PASS. Evidence: `demos/tmp/real-e2e/remaining-models-20261024/omnilingual-v16-wasm-explicit/parent-matrix-verification.json`. Physical microphone remains unverified; all captures used Chromium fake audio.

**Acceptance remains FAIL for this fixture's transcript quality**: native, FP32, and INT8 CPU controls output `م` instead of the immutable English text, and the eight browser flows also fail. Broader quality remains **UNVERIFIED**. The native CPU oracle itself did not test microphones, streaming, or any other fixture. The invalid initial backend-selection attempt remains separate from the explicit pure-WASM retry. A strict receipt and small reproduction scripts are in [evidence/omnilingual-fp32-control-2026-10-24](evidence/omnilingual-fp32-control-2026-10-24/). Full `.npy` arrays and model files remain task-local; the evidence folder has only their hashes, shapes, and task-local paths.

# Product Tasks & Engine Milestones (Summary)

> [!NOTE]
> The full historical product task log (2,882 lines across library architecture, multi-model engine milestones, and demo evolution) has been archived to:
> [`docs/archive/PRODUCT_TASKS-2026-10-17.md`](file:///N:/github/asrjs/speech-recognition/docs/archive/PRODUCT_TASKS-2026-10-17.md).
>
> All active sprint tasks and verification gates are maintained in:
> [`docs/ACTIVE_SPRINT.md`](file:///N:/github/asrjs/speech-recognition/docs/ACTIVE_SPRINT.md).
>
> Deep architecture audit and future technical roadmap are in:
> [`docs/reports/deep-review-2026-10-17.md`](file:///N:/github/asrjs/speech-recognition/docs/reports/deep-review-2026-10-17.md).

---

## Executive Summary of Engine Milestones

1. **Model Family Implementations (15 Families):**
   - **NeMo TDT / RNNT / AED:** `nemo-tdt` (Parakeet, Orukeet), `nemo-rnnt`, `nemo-aed` (Canary-180m-flash), `nemotron-rnnt` (Nemotron 3.5 streaming).
   - **Whisper Seq2Seq:** 4-graph decoupled ONNX export (encoder, decoder, decoder-with-past, aligner) with word-level timestamps.
   - **CTC Family:** `lasr-ctc`, `parakeet-ctc`, `omnilingual-ctc` (multilingual MMS/Wav2Vec2), `gigaam-ctc` (Russian GigaAM), `wav2vec2`.
   - **Streaming & Autoregressive:** `moonshine` (realtime streaming encoder-decoder), `x-asr` (chunk-stateful streaming), `qwen-asr` (Qwen3-ASR dynamic audio front + decoder step), `gigaam-rnnt`, `sensevoice`.

2. **Frontend & Audio Feature Pipelines:**
   - Multi-engine FFT implementations (PocketFFT, PFFFT, Web Audio API).
   - Dynamic Mel filterbank, pre-emphasis, windowing, and log compression.
   - Verified numerical parity against PyTorch torchaudio and NeMo golden tensors.

3. **Streaming & Voice Activity Detection (VAD):**
   - Multi-mode `StreamingSpeechDetector` supporting energy-based `RoughSpeechGate` and neural VAD (`TEN-VAD`, `FireRedVAD`).
   - Browser microphone streaming controllers and chunked window aggregators.

4. **Active Sprint Focus (Wave 1):**
   - **T1:** Decouple `@asrjs/speech-recognition` root bundle (`dist/index.js` zero-model-leak gate).
   - **T2:** Repair runtime resolution wiring for `moonshine`, `qwen3-asr`, `wav2vec2`, `nemotron`, `canary`.
   - **T3:** Standardize `owned-session` lifecycle and canonical transcript metadata across all 15 families.
   - **T4:** Purge placeholder backends (`webgl`, `webnn`) and enforce honest backend reporting.
   - **T5:** Validate Moonshine & Qwen3-ASR in real Chrome WebGPU and reclaim ~10 GB disk space.
   - **T6:** Integrate neural VAD probabilities into real endpointing decisions to eliminate word clipping.
   - **T7:** Link sibling demo applications to the root package and maintain repo hygiene.

For details on completed items and past PR milestones, see the [historical archive](file:///N:/github/asrjs/speech-recognition/docs/archive/PRODUCT_TASKS-2026-10-17.md).

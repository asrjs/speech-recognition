# Product Tasks & Engine Milestones (Summary)

## Current product direction — 2026-10-25

Build one easy-to-integrate offline and streaming speech library for web and
Node.js/Bun developers. Preserve established Parakeet TDT optimization and
Whisper custom-graph work. Model ports, VAD/waveform demos and verification
tools support the library's public API and developer experience.

Current work follows the [project charter](PROJECT_CHARTER.md) and
[developer documentation](README.md). This is the product queue; the dated
engine milestones below preserve earlier scope.

1. **Consumer workflows:** maintain small public-API recipes for PCM/files,
   microphone/VAD, native streaming and windowed offline inference. The current
   recipes are indexed in [examples/README.md](../examples/README.md); source
   typechecking is separate from real-model/environment acceptance.
2. **Shared integration:** keep model loading and capability decisions in the
   library, common app behavior in demo-runtime, and model math in its family.
   Use the four maintained apps to expose concrete API friction and remove
   duplicated behavior after checking existing contracts.
3. **Deployment:** make model/engine assets, caching, workers and local runtime
   setup explicit. Validate supported browser and Node/Bun combinations without
   assuming one environment's success proves another's.
4. **Established models and useful new models:** preserve measured accuracy,
   speed, graph optimizations and lifecycle behavior while extending support.
   Prioritize product value and reusable architecture over catalog count.

## Consumer workflow checkpoint — 2026-10-25

- File/Blob/URL transcription now has a one-shot root API and a reusable
  controller action. The library resolves the loaded model's sample rate,
  decodes, downmixes, resamples, observes cancellation and owns cleanup.
- React/Svelte/Vue file examples use the controller action. Bindings preserve
  custom controller capabilities without changing the existing PCM contract.
- Microphone actions now bind the loaded model to the existing capture/VAD
  controller, preserve context revisions, await pending inference on stop and
  release capture before model disposal. React/Svelte/Vue examples use these
  actions. A controlled Chrome run with local Parakeet TDT v3 FP16/INT8 matched
  the JFK text for file, manual microphone and TEN-VAD-only capture. Physical
  microphones and other model/backend/device combinations remain separate gates.
- Reference comparison found two TDT refinements needing model-owned acceptance:
  Parakeet.js's audible-tail re-scan and decoder-state continuation between
  calls. Neither is replaced by generic VAD/windowed inference. Keep those
  changes separate from the microphone composition and validate with fixtures
  before promoting a port or claiming reference parity.

Validation details live in the [dated report](reports/audio-source-api-2026-10-25.md); the items above describe scope,
not an unconditional model/environment compatibility claim.

## Historical engine milestones and sprint routing

> [!NOTE]
> The full historical product task log (2,882 lines across library architecture, multi-model engine milestones, and demo evolution) has been archived to:
> [`docs/archive/PRODUCT_TASKS-2026-10-17.md`](archive/PRODUCT_TASKS-2026-10-17.md).
>
> The October 17 sprint tasks and verification gates were recorded in:
> [`docs/archive/ACTIVE_SPRINT.md`](archive/ACTIVE_SPRINT.md).
>
> Deep architecture audit and future technical roadmap are in:
> [`docs/reports/deep-review-2026-10-17.md`](reports/deep-review-2026-10-17.md).

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

4. **Historical Sprint Focus (Wave 1, 2026-10-17):**
   - **T1:** Decouple `@asrjs/speech-recognition` root bundle (`dist/index.js` zero-model-leak gate).
   - **T2:** Repair runtime resolution wiring for `moonshine`, `qwen3-asr`, `wav2vec2`, `nemotron`, `canary`.
   - **T3:** Standardize `owned-session` lifecycle and canonical transcript metadata across all 15 families.
   - **T4:** Purge placeholder backends (`webgl`, `webnn`) and enforce honest backend reporting.
   - **T5:** Validate Moonshine & Qwen3-ASR in real Chrome WebGPU and reclaim ~10 GB disk space.
   - **T6:** Integrate neural VAD probabilities into real endpointing decisions to eliminate word clipping.
   - **T7:** Link sibling demo applications to the root package and maintain repo hygiene.

For details on completed items and past PR milestones, see the [historical archive](archive/PRODUCT_TASKS-2026-10-17.md).

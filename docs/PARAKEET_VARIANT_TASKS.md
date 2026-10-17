# Parakeet & Orukeet Variant Tasks (Summary)

> [!NOTE]
> The full historical execution log (4,154 lines across all porting, calibration, and parity waves) has been archived to:
> [`docs/archive/PARAKEET_VARIANT_TASKS-2026-10-17.md`](file:///N:/github/asrjs/speech-recognition/docs/archive/PARAKEET_VARIANT_TASKS-2026-10-17.md).
>
> All active sprint tasks and verification gates are maintained in:
> [`docs/ACTIVE_SPRINT.md`](file:///N:/github/asrjs/speech-recognition/docs/ACTIVE_SPRINT.md).
>
> Strategic architecture analysis and comprehensive findings are in:
> [`docs/reports/deep-review-2026-10-17.md`](file:///N:/github/asrjs/speech-recognition/docs/reports/deep-review-2026-10-17.md).

---

## Executive Summary of Completed Milestones

1. **Model Provenance & Conversion:**
   - Source NeMo checkpoint: `orukeet-v0.1.0.nemo` (TDT 0.6B architecture, Turkish CTC/RNNT/TDT acoustic model).
   - Exported to ONNX graphs: `encoder-model.onnx`, `decoder_joint-model.onnx`, and `vocab.txt`.
   - Standardized layout under `models/orukeet/onnx/fp32-packed-2026-10-01/`.

2. **Precision & Quantization Porting:**
   - `fp32`: Golden numerical baseline across all stages (filterbank, Mel, encoder, predictor, joint).
   - `fp16`: Optimized WebGPU encoder path validated against oracle transcripts.
   - `int8`: Dynamic quantization for WASM decoder and joint network to achieve interactive latencies.

3. **Feature Preprocessor Validation:**
   - Multi-stage verification comparing PyTorch NeMo feature extraction against TypeScript runtime implementations (PocketFFT, PFFFT, Web Audio FFT).
   - Established strict ULP bounds and max-abs tolerance for audio preprocessing frames.

4. **Browser & Headless Verification:**
   - Hybrid WebGPU (encoder) + WASM (decoder/joint) pipeline validated in real headless Chrome on JFK and long synthetic speech fixtures.
   - Verified offline asset resolution and memory disposal lifecycle.
   - Preserved canonical verification receipts under `tools/data/results/canonical/`.

5. **Current Status & Active Focus:**
   - Parakeet/Orukeet core engine is mature and verified.
   - Active work has shifted to:
     - Root package core decoupling (Card T1)
     - Neural VAD soft-splitting and endpointing (Card T6)
     - Demo workspace unification and cross-model selection (Card T7)

For full detail on prior task cards (C1–C94) and diagnostic traces, refer to the [archive log](file:///N:/github/asrjs/speech-recognition/docs/archive/PARAKEET_VARIANT_TASKS-2026-10-17.md).

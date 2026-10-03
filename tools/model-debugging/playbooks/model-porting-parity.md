# Model Porting Parity Workflow

Use this playbook when adding a new model family and you need reproducible parity checks.

## Goals

1. Prove stage-by-stage behavior against an original/reference implementation.
2. Keep lightweight checks in CI.
3. Keep heavyweight parity scripts and artifacts available for on-demand troubleshooting.

## Recommended Flow

1. Start with a copied reference suite under `tools/model-debugging/reference/<model>/`.
2. Classify tests into:
   - CI-safe helpers (no external model downloads, no heavy runtime deps)
   - on-demand parity checks (requires local models/artifacts)
3. Compare final transcript text before comparing WER/CER if the stacks use different scoring or normalization pipelines.
4. Port core logic checks into `tests/*.test.ts` for CI coverage.
5. Preserve original tests and artifacts in `tools/model-debugging/reference/` as debugging references.
6. For stage captures, use `node-compare-stage-captures.mjs` and keep the
   `sample_id` plus audio identity in both captures. Treat its
   `first_failed_stage` as the next debugging boundary; do not join rows by
   order or transcript text.
7. Document entry points in:
   - `tools/model-debugging/README.md`
   - `tools/model-debugging/reference/README.md`
   - model-specific `README.md` under that reference folder
8. Before declaring a port verified, run at least one end-to-end clip that is
   **longer than the encoder's feature dimension in frames**, on the same code
   path as the shipped configuration. NeMo TDT exports carry 1024 features with
   a 10 ms frame shift and 8x subsampling, so the threshold is about 82 seconds
   of audio: a longer clip makes `encoder_outputs` `[1, 1024, T]` with `T > 1024`,
   which is where layout, slicing and lifetime defects surface. Short fixtures
   hide them; the recorded Orukeet library runs that missed this all used an 11 s
   clip. A repeated-fixture WAV built from an existing 16 kHz PCM source is
   enough, and needs no new weights.

## MedASR Example

- Reference suite copy:
  - `tools/model-debugging/reference/medasrjs/upstream-tests/`
- CI-safe helper tests:
  - `tests/lasr-ctc-medasr-port-helpers.test.ts`

This split keeps CI stable while still preserving deep parity materials for future model ports.

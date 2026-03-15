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
3. Port core logic checks into `tests/*.test.ts` for CI coverage.
4. Preserve original tests and artifacts in `tools/model-debugging/reference/` as debugging references.
5. Document entry points in:
   - `tools/model-debugging/README.md`
   - `tools/model-debugging/reference/README.md`
   - model-specific `README.md` under that reference folder

## MedASR Example

- Reference suite copy:
  - `tools/model-debugging/reference/medasrjs/upstream-tests/`
- CI-safe helper tests:
  - `tests/lasr-ctc-medasr-port-helpers.test.ts`

This split keeps CI stable while still preserving deep parity materials for future model ports.

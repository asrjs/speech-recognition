# Product improvement queue

This queue complements `PARAKEET_VARIANT_TASKS.md`; it does not replace the
Orukeet-first model queue or the whole-product goal in `GOAL_PROMPT_RECOMMENDED.md`.

## C01 — Opt-in native ORT installation

Status: INVESTIGATED, implementation and consumer acceptance UNTESTED.

Purpose: browser-only consumers should not need to install a native engine
that the runtime does not automatically register.

Evidence as of 2026-10-01: `package.json` requires
`onnxruntime-node@1.25.0-dev.20260228-6e72d31970`. Its local installed tree has
237 files / 223,239,822 bytes (212.9 MiB), including multiple platform assets.
This is an installed-tree measurement, not a published archive-size claim.
The architecture already keeps native CPU composition explicit and browser
imports free of eager native bindings.

Next bounded step: inspect supported package-manager optional-peer behavior
and current native loading/error contracts, then choose an opt-in dependency
declaration with a development-only engine for this repository's checks.
Do not silently install an engine on a missing-binding failure. Do not change
the public API or fork model executors by host.

Acceptance:

- A packed-package consumer without native ORT imports root/browser surfaces.
- Explicit native probe/load reports a clear missing-engine error.
- A native-present consumer runs the same Node/Bun local example and cleanup.
- Browser root import closure and package tarball smoke checks pass.
- Native resource lifecycle tests pass; affected docs explain opt-in setup.
- Measure comparable install/package footprints; do not equate lazy imports
  with lower install size. Package-manager or artifact checks unavailable in
  this environment remain explicitly UNTESTED.

Relevant existing checks: `tests/smoke/browser-root-import-graph.mjs`,
`tests/smoke/package-tarball-smoke.mjs`, `tests/package-tarball-contract.test.ts`,
`tests/native-cpu-backend.test.ts`, `tests/native-resource-smoke.test.ts`, and
`examples/nemo-tdt-native-local.mjs`.

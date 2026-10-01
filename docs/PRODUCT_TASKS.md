# Product improvement queue

This queue complements `PARAKEET_VARIANT_TASKS.md`; it does not replace the
Orukeet-first model queue or the whole-product goal in `GOAL_PROMPT_RECOMMENDED.md`.

## C01 — Opt-in native ORT installation

Status: PARTIAL PASS — optional-peer metadata and resolution-isolated smoke.
Real packed-consumer installation and footprint comparison remain UNTESTED.

Purpose: browser-only consumers should not need to install a native engine
that the runtime does not automatically register.

Baseline evidence as of 2026-10-01: `package.json` previously required
`onnxruntime-node@1.25.0-dev.20260228-6e72d31970`. Its local installed tree has
237 files / 223,239,822 bytes (212.9 MiB), including multiple platform assets.
This is an installed-tree measurement, not a published archive-size claim.
The architecture already keeps native CPU composition explicit and browser
imports free of eager native bindings.

Implemented: the unchanged pinned native version is an optional peer and a
development dependency. Offline lockfile-only refresh ran with install scripts
disabled; no package versions changed. Native loading remains lazy and explicit.
`npm run test:native-optional-peer` verifies metadata, built root/browser imports
with exact native package resolution blocked, unavailable probe, typed rejection
without fallback, and an installed CPU binding positive control. This process-local
simulation is not proof of a real packed-consumer installation or model inference.

Next bounded step: test isolated packed consumers (runtime imports and TypeScript
declarations), compare actual install footprints, and repeat native Node/Bun local
example/resource inference controls before marking this task complete.
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

# Product improvement queue

This queue complements `PARAKEET_VARIANT_TASKS.md`; it does not replace the
Orukeet-first model queue or the whole-product goal in `GOAL_PROMPT_RECOMMENDED.md`.

## C01 — Opt-in native ORT installation

Status: PARTIAL PASS — optional-peer metadata, resolution-isolated smoke and
actual packed-archive consumption. Package-manager installation and footprint
comparison remain UNTESTED.

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

Packed consumer control (2026-10-01): `npm run test:packed-native-absent`
creates a real local archive with scripts disabled, extracts it outside the
checkout and links only existing browser dependencies/type helpers. With native
ORT physically absent, actual package root/browser imports, explicit native
unavailable rejection and strict NodeNext public declarations passed (no
`skipLibCheck`). The public `runtime.loadModel()` control explicitly requests
native CPU with an available WASM backend also registered: it rejects with
`backend-unavailable` before model creation or asset resolution, without fallback.
No model inference or browser GPU execution is certified.
Fixtures are retained in the printed temporary directory for inspection.

Next bounded step: compare actual package-manager install footprints and repeat native Node/Bun local
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

### C01 resource refresh (2026-10-02)

The existing real native resource harness was rerun explicitly with Node
v26.2.0 and Bun 1.3.14 against the same small ONNX Identity fixture. Both
successful acquisition (two sessions) and partial acquisition failure (one
session) pass exact acquired/release-attempt/released counts. Repeated dispose
does not add releases. These are resource-only controls, not model inference,
numerical parity, or install-footprint acceptance. Small evidence pointer:
`tools/data/results/harness/native-resource-refresh-handoff-2026-10-02.json`.

Next portability gap: the prerequisite Vitest tests currently spawn
`process.execPath` (Node), and the manually used Identity fixture lives under
a local ONNX virtual environment. Add an explicit opt-in Bun prerequisite path
and a portable repository-owned fixture; retain requested-missing-fixture FAIL
and optional-unrequested SKIP semantics. Existing manual Bun evidence is valid
but does not make that automatic test path implemented.

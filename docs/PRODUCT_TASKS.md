# Product improvement queue

This queue complements `PARAKEET_VARIANT_TASKS.md`; it does not replace the
Orukeet-first model queue or the whole-product goal in `GOAL_PROMPT_RECOMMENDED.md`.

## C01 — Opt-in native ORT installation

Status: PARTIAL PASS — optional-peer metadata, resolution-isolated smoke and
actual packed-archive consumption. The offline package-manager comparison has
been attempted but fails on missing cached dependencies; completed installation
footprints remain UNVERIFIED.

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

### C01 portable automatic resource controls (2026-10-02)

Implemented the bounded portability slice: own 125-byte Identity graph and
synthetic vocabulary, temporary exclusive writes and hash/overwrite controls,
Node/Bun opt-in real executor checks, and the same four prerequisite negatives
under explicitly requested Bun. No local model/Python fixture dependency or
native engine install was introduced. Strict changed-test compilation passes.
Both requested hosts pass all 10 smoke checks; opt-ins unset gives 4 PASS /
6 SKIP. Explicitly requested Bun without an executable produces 5 FAIL, never
an accidental PASS/SKIP (negative-run JSON retained locally). Fixture unit
control passes; shared gate regression gives 8 PASS / 1 intentional SKIP.
See `tests/fixtures/README.md` for build prerequisite and opt-in commands.
This closes the automatic resource-fixture gap, not C01 install-footprint or
full model/native/browser acceptance.

### C01 offline install comparison harness (2026-10-02)

`tests/smoke/install-footprint.mjs` packs one actual built archive and installs
it into two fresh temporary consumers: package alone versus the same package
plus its exact optional native peer. npm is invoked without a shell, offline,
with install scripts disabled. Receipts hash tool/metadata/archive, preserve
both consumers, bind lockfile versions and require actual native resolution
identity. A failed install produces no footprint delta; existing receipts are
refused before npm runs. Seven tests (including actual CLI negative controls)
pass; strict changed-test compilation and build pass. Related gates give
38 PASS / 6 intentional SKIP; browser import graph and 1,401-file package
tarball contracts pass. This is not native/model/browser execution acceptance.

Actual Node v26.2.0/npm 11.13.0 offline runs fail both consumers with
`ENOTCACHED` for `@types/node@26.4.0`; partial module trees and npm cleanup
warnings are retained, not measured as successful installs. The packed archive
is 1,093,345 bytes compressed / 5,152,384 bytes unpacked. These archive sizes
are not installed footprint savings. See
`tools/data/results/harness/offline-install-footprint-handoff-2026-10-02.json`.

Next obtain an authorized cache-complete environment and rerun with a new
receipt. Do not switch to network fetching or lower success checks silently.
Even a future PASS measures script-disabled logical file bytes only: native
assets generated/downloaded by the native install script and allocated disk
blocks remain a separate footprint/runtime gate. C01 stays PARTIAL.

## C02 — Local text/JSON cancellation consistency

Status: PASS — bounded Blob read checkpoint fix; not universal load cancellation.

`BlobAssetHandle.readText()` previously ignored its request signal, unlike
`readBytes()`. Pre-aborted text reads returned success; abort during text reads
returned late success, and JSON reads could report a parse error instead of
cancellation. Three focused regression cases reproduced these failures.

Text reads now check the existing model-neutral abort helper before and after
`Blob.text()`. JSON reads inherit the same checks before parsing. Successful
reads and genuine Blob read errors retain their original semantics. This does
not interrupt the platform's Blob materialization or add a new public API,
model branch, native dependency, timer or framework state.

Verification: strict changed-test compilation, five Blob cases and all 39
related IO/provider/tokenizer/ORT-abort tests pass; build and browser root
import graph pass (297 modules / 27 entries). The built public IO provider
with real Blob objects passes two successful reads and four cancelled reads
on Node v26.2.0 and Bun 1.3.14 separately. Real browser/app cancellation and
full model inference were not run for this slice. Companion source/API changes
are unnecessary; broader cancellation integration remains a separate gate.

## C03 — Segmented transcription respects explicit finalization

Status: PASS — bounded controller lifecycle fix; live microphone/model and
sibling application acceptance remain separate.

`transcribeUtterance()` previously bypassed the finalized guard used by
`pushAudio()` and `flush()`. After explicit `finalize()` it could call the model
and reset/mutate merger and latency state. Two regressions failed before the
fix: submission after completed finalization and submission queued behind
finalization. The existing guard now runs inside the serialized operation,
after stale-generation rejection and before any utterance state changes.
Reset restores use; repeated segmented utterances before explicit finalization
remain supported. A segment's `reason: 'stop'` does not itself finalize the
whole controller. No public types, model mathematics or framework glue changed.

Verification: all 11 controller tests and 39 related realtime/browser-controller
simulation tests pass. Source and TS-example compilation, strict changed-test
compilation with `skipLibCheck=false`, built browser import closure (297 modules /
27 entries), and built public realtime API checks under Node v26.2.0 and Bun
1.3.14 pass. Those public checks use deterministic callbacks, not ASR inference.
The existing abort test had a narrower-than-public callback parameter; it now
accepts `AbortSignalLike` and asserts/narrows the actual controller-owned native
signal before listening. Runtime cancellation semantics are unchanged.

Full suite: 1,317 PASS / 33 SKIP / 0 FAIL in
`tools/data/results/harness/realtime-finalized-guard-full-suite-tls-2026-10-02.json`.
This run explicitly sets `NODE_TLS_REJECT_UNAUTHORIZED=1` only in the test child
environment; the first run inherited a disabled-TLS setting and is retained
separately. No process-wide or user environment setting was changed.

Affected consumer audit: streaming-demo routes detected segments through this
controller, resets it on capture start, and normally uses stop/flush rather
than explicit transcription-controller finalization. Its ordinary flow needs
no source migration. Direct consumers that call explicit `finalize()` must call
`reset()` before submitting another utterance, consistently with push/flush.
Playground and benchmark-demo have no matching source calls in the scoped
audit. Sibling build and real browser/microphone execution are UNTESTED here:
streaming-demo's build writes to its read-only checkout. No sibling edits were
made. Continue the Orukeet FFT boundary work alongside broader resource and
application gates; this lifecycle slice does not close those workstreams.

## C04 — Disposal during browser microphone/VAD startup

Status: PASS — deterministic lifecycle coverage only; no production fix was
needed for these paths. Real browser microphone/model acceptance stays open.

Added `tests/browser-controller-dispose.test.ts` through the public browser API.
Controlled Promise gates verify disposal while microphone acquisition is
pending and while VAD initialization is pending (both late success and late
rejection). Tests require immediate inactive state, VAD abort, exactly-once
capture stop, starter stop with `flush:false`, subscription cleanup, monitor
and starter disposal, and rejection of a subsequent start. The manual case
also verifies late chunk/error callbacks cannot mutate state or emit an
utterance, plus repeated disposal does not stop/dispose resources again.
No new model-specific branch, framework binding, dependency or API is added.

Verification: all three cases pass; strict standalone changed-test compilation
with `skipLibCheck=false` passes; the browser controller, browser realtime
starter and transcription controller suites together pass 27 tests. These use
injected capture/starter/monitor resources, not a real device, VAD model, ASR
or WebGPU. Source implementation is unchanged, so no companion migration is
required; sibling/browser hardware integration remains UNTESTED.

Next product gate: exercise the same lifecycle under the existing browser
loop with real capture/model resources where available. Keep Orukeet's open
mel2/frame1027 numerical investigation separate from these lifecycle checks.

## C05 — Capture cleanup survives release failures

Status: PASS — bounded resource-owner fix with deterministic injected resources;
real microphone/browser failure integration remains UNTESTED.

Four new regressions failed before the fix: an early disconnect skipped later
releases, a rejected context close skipped owned tracks, a concurrent stop
returned before cleanup completed, and startup cleanup swallowed its own failure.
Capture now shares one cleanup promise, attempts every independent release,
preserves a single original cleanup error, and aggregates multiple errors.
Startup failures retain both initialization and cleanup errors. Repeated stop
observes the same completed or rejected cleanup rather than rerunning releases.
Borrowed streams remain untouched unless track ownership is explicitly opted in.
PCM, sample-rate, resampling and model behavior are unchanged.

Verification: seven new cases and all 41 related capture/browser/realtime tests
pass; source/example compilation, strict changed-test compilation with
`skipLibCheck=false`, and built browser import closure (297 modules / 27 entries)
pass. These checks use injected resources, not live capture or model inference.
No companion edits or new dependencies were made. Callers must still handle
`stop()` rejection; broader controller disposal after a capture cleanup failure
is a separate acceptance gate, not proven by this owner-level fix.

The late starter-event gate is now covered by C06 below. Next run current-source
real-browser capture lifecycle using a root-owned server
or separately authorized sibling cache writes. Do not validate against the
stale sibling build or claim mocked tests as microphone/WebGPU acceptance.

## C06 — Ignore starter events once disposal begins

Status: PASS — bounded event-delivery lifecycle fix; actual VAD/browser and
model inference acceptance remain separate.

Two regressions failed before the fix: retained starter events changed status
and error state while disposal waited for VAD startup or capture shutdown.
The subscription callback now returns immediately when disposed, before reading
segment PCM, publishing an utterance, calling transcription or changing state.
Tests also invoke the retained callback after cleanup has completed. The guard
does not depend on microphone-active state: ordinary speech-detect stop sets
the mic inactive before flushing its last segment, which must still be delivered.
A positive flushed-stop test and existing active-segment transcription coverage
remain passing. No decoder, framework state or public contract changes.

Verification: two RED-to-GREEN regressions and one positive stop-flush case;
all 44 related capture/browser/realtime tests pass. Source/example compilation,
strict changed-test compilation with `skipLibCheck=false`, scoped lint and built
browser import closure (297 modules / 27 entries) pass. Tests use controlled
starter/capture resources and a real transcription-controller instance, not a
VAD model, physical microphone, ASR inference or WebGPU. No sibling writes.

Next product gate: current-source real-browser capture lifecycle, then cleanup
of controller-owned resources when capture stop itself rejects (C05 only proves
the capture owner's independent releases). Keep these separate from Orukeet's
unresolved feature and recurrent WebGPU parity gates; no model promotion.

## C07 — Real Chromium capture owner lifecycle

Status: PASS — real browser/fake WAV device, capture owner only. Controller,
VAD/ASR, physical device, soak and cleanup-failure browser gates remain open.

Added a root-owned Python Playwright runner and small JS probe importing the
built public `dist/browser.js` entry. No sibling Vite/cache writes or server
are required: page HTTP requests are locally fulfilled from `dist` plus the
explicit existing `pako` ESM dependency; unexpected requests fail. The initial
attempt stopped before capture because bare `pako` lacked an import map. Adding
the verified local package entry fixed harness resolution, without core edits.
Python/Playwright/Chrome were already installed; nothing was downloaded.

After refreshing the library build, Chrome 154.0.8037.93 passed two independent
capture sessions: four contiguous 256-frame chunks each, finite nonzero PCM,
48 kHz device/context and 16 kHz output, one actual AudioWorklet module load,
shared concurrent stop, context `closed` and owned track `ended`. No extra
chunks arrived during each 150 ms post-stop observation. This uses real browser
media/audio APIs, not injected fake audio nodes. WAV content may be changed by
browser capture processing; no exact PCM or speech-quality claim is made.

Evidence: `tools/data/results/harness/real-browser-capture-owner-reviewed-2026-10-02.json`
hashes the probe, runner, capture source, fixture and all 269 served resources
(267 library modules, probe and pako). Inputs were stable before/after the run.
Python/JS syntax and help checks pass. Existing-output preflight rejects before
browser launch and preserves the receipt hash. README provides build/replay
commands and requires a fresh build; source-to-build equivalence is not
automatically certified by the runner. The receipt is exclusively created.

Next: extend this actual-browser path to the public microphone controller's
stop/restart/dispose and pending-handoff lifecycle, preserving capture-owner
controls. Independently cover controller cleanup when capture stop rejects.
Orukeet feature/recurrent parity and model promotion are still unresolved.

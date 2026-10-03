# Product improvement queue

This queue complements `PARAKEET_VARIANT_TASKS.md`; it does not replace the
Orukeet-first model queue or the whole-product goal in `GOAL_PROMPT_RECOMMENDED.md`.

## C26 — Frontend conformance gate policy for future ports

Status: POLICY RECORDED from existing receipts. No new inference, no weight
downloads, no official-capture refresh, no runtime or tolerance change.

The `atol=1e-5, rtol=1e-4` feature-coordinate gate on Orukeet was never met by
any NeMo port shipped from this repository. Recorded official-versus-library
frontend comparisons: `parakeet-realtime-eou-120m-v1` max `4.0801039` /
mean `9.019221e-4` with exact token and text parity, `canary-180m-flash`
asrjs `2.1461394` / `8.918627e-4`, meljs `0.7285282` / `9.190078e-4`, and
Orukeet `3.4236908e-4` / `1.226810e-6` with oracle-matching 38-token output.

Orukeet's frontend is the closest to its official oracle of any port here:
6268x, 2128x and 11917x smaller maximum absolute error than those three, and
727x, 749x and 735x smaller mean absolute error. `3.42e-4` therefore is not a
port defect and Orukeet is not harder to port than the models already shipping.
The gate stays recorded as FAIL and no tolerance moves; it is informational
conformance data, not port acceptance.

Port acceptance for the next family is transcript, token, decoder and
end-to-end parity plus the browser gates, with frontend deviation reported on
the same max-abs/mean-abs/rmse scale. The [focus goal](GOAL_MODEL_PORT_HARNESS.md#the-coordinate-gate-is-not-a-port-acceptance-criterion)
and the [NeMo porting playbook](../tools/model-debugging/playbooks/nemo-rnnt-porting.md#what-the-frontend-metric-means-for-a-port)
carry the table so the next port does not open another FFT lane.

Pure silence remains a separate recorded divergence: an official masked fp32
mean reduction on a constant row, not spectrum fidelity.

Receipt: `tools/data/results/orukeet/frontend-conformance-family-comparison-2026-10-03.json`
(aggregation of six existing receipts with their SHA-256). Contracts differ
between rows and no official stock v2/v3 feature capture is stored locally, so
this compares magnitude classes rather than identical contracts.

## C25 — Existing Qwen harness reuse and bounded numerical disposition

Status: PASS for the focused reuse exercise; Orukeet production-feature
conformance remains **DEFERRED / FAIL**, not model acceptance. No runtime,
family math, tolerance, exporter or shared-control implementation changed.

The existing Qwen Node/WASM CLI ran local official-export artifacts on JFK
with dynamic encoder/fp16 decoder and produced 25 tokens, WER/CER zero and
normalized exact match against the saved native reference. Cold elapsed time
was 111899.6057 ms. It exited successfully through its awaited disposal path.
This did not rebuild the library, refresh official features or prove browser,
performance or resource-reclamation acceptance.

The unchanged shared attribution control also ran the real Qwen dynamic
encoder. Reference features came from the current library processor, not a
fresh official-engine capture. A deliberate same-shape frame-major/mel-major
misinterpretation was localized to `input_features`; replay reproduced the
candidate output with zero numerical error. The receipt binds the waveform,
prepared feed and seven unchanged source/artifact hashes. Three actual
encoder calls, one six-line callback; zero new inference loops, exporters,
weight downloads, shared-attribution changes or permanent adapter scripts.
The one-input diagnostic proves real cross-family reuse, not a production bug
or multi-input discrimination.

Receipts under `tools/data/results/qwen/`:
`goal-harness-existing-port-jfk-2026-10-03.json` and
`goal-harness-encoder-layout-attribution-jfk-2026-10-03.json`.
The existing [Qwen guide](../tools/model-debugging/reference/qwen3-asr-0.6b/README.md#shared-encoder-boundary-control)
now exposes the tested control recipe and its limits.

The [focus goal](GOAL_MODEL_PORT_HARNESS.md#bounded-orukeet-numerical-disposition)
records the bounded Orukeet disposition: official-complex/native-real-FFT
controls pass downstream feature gates, portable full-512 FFT already fails
end-to-end, and pure silence has a separate measured reduction mismatch.
No rejected FFT experiment was repeated or numerical gate relaxed. Browser
real-hybrid scores retain their existing fixture-specific PASS. Complete
Orukeet feature and full ASR acceptance remain unproven.

## C24 — Python reference environment inventory

Status: AUDITED — read-only environment discovery, real import/CPU smoke and
dependency checks. Not model parity, CUDA acceptance or environment migration.
User requested project-owned environments per runtime/reference library and
inspection of venv, Conda and uv installations.

Eleven interpreters were inspected. Six reference stacks passed actual imports
and Torch CPU arithmetic. Only the isolated project Qwen environment passed
`pip check` among those six. The project NeMo environment is an overlay:
NeMo 3.0.0 is local, while Torch/ONNX/ORT/NumPy come from Conda `nemo`, whose
own NeMo version is 2.4.0. Both import ASR successfully but have dependency
conflicts; they are not interchangeable oracle identities. The sibling
GigaAM environment also inherits from system Python and is not project-owned.

Conda is installed but absent from `PATH`; explicit invocation locates `nemo`,
`medasr_env`, `asrds_py311`, `vgtts` and base. uv supplies a managed Python
3.13.15 interpreter, but no uv tool environment or project uv manifest/lockfile
was found. Its visible NumPy comes from the user site, not an isolated ASR stack.

The [tool map](../tools/README.md#python-reference-environments), local skill
and [workstation inventory](WORKSPACE_CONTEXT.md#python-reference-environment-inventory)
now expose interpreter selection, package provenance, isolation and actual
dependency-check outcomes. The recommended unit is a version-pinned reference
dependency stack, shared across compatible models, with separate environments
for incompatible reference versions. No package was installed or upgraded;
no environment or inherited Nemotron change was edited.

Receipt: `tools/data/results/python-runtime-environments-2026-10-03.json`.
Imports/CPU arithmetic passed for project NeMo, project Qwen, Conda NeMo,
Conda MedASR, Conda onnx-asr and sibling sherpa/FunASR. Dependency-check failures
are preserved, not suppressed. Provisioning isolated replacements remains
separate from this completed inventory.

## C23 — Project-local skill and tool discovery

Status: PASS — documentation discovery and CLI-help smoke; not model inference
or numerical acceptance. User requested easier discovery after agents missed
existing model-porting skills, reference engines and local artifacts.

The tracked startup now exposes the project-local Model Debugging Skill and
the [task-to-tool map](../tools/README.md#task-to-tool-map) before long task
records. The map routes ports, feature contracts, original captures, stage
comparison, attribution, artifact audits and browser checks to existing tools.
No registry, second runner, inference loop or global skill install was added.
Core runtime and the twelve inherited Nemotron changes remain untouched.

Toolkit startup links and the family-playbook index now use relative paths.
The script index exposes the existing offline Qwen library CLI and correctly
groups frontend options with the frontend tool, not the artifact packer.
It warns about the legacy frontend's sibling default, missing help handler and
different reference schema. The Qwen guide's PowerShell capture command is
repaired; its offline runner section names actual supported arguments.
The workspace map supplies exact local NeMo, onnx-asr and sherpa-helper paths.
Local environments and paths are discovery leads, not fresh-clone guarantees.

Two fresh-agent, read-only exercises started at `docs/AGENT_ENTRYPOINT.md`.
Both located the local skill, family workflow, original reference helpers and
existing runners. Qwen discovery also located local ONNX artifacts and the
already implemented `qwen-asr` family. NeMo discovery found the project-local
reference interpreter instead of treating a default-Python import failure as
missing tooling. Their CLI/schema findings were checked against source.
The Qwen guide was rechecked after its command and runner-route corrections.

Validation: 122 local Markdown file/heading links resolve across the ten edited
discovery/goal guides. `node tests/smoke/qwen3-asr-node-wasm-benchmark.mjs --help`
exits successfully with the documented flags. The stage comparator's `--help`
also exits successfully. The discovered reference interpreter imports NeMo
3.0.0 and Torch 2.6.0+cu118. Formatting and whitespace checks pass.
No real model inference was run for this documentation-only change.

The focus goal record retracts the unsupported claim that Qwen needs a new
family/download merely to run local ONNX artifacts. Its historical original
PyTorch snapshot path is absent, which is a separate prerequisite for fresh
official-engine capture. Existing ports do not automatically demonstrate the
focus goal's cheaper-port/cross-family-attribution criterion. Orukeet feature
parity remains FAIL; all original tolerances and model ordering stay unchanged.

## C22 — Real-provider disposal drain and post-C21 native refresh

Status: PASS — actual onnxruntime-node inference through the public example
CLI and inspector; not feature-parity, browser, fp16 or measured-memory
acceptance.

C18-C21 shipped with simulated/deferred-call evidence and explicitly left real
provider shutdown open. This slice records actual native inference at the
current HEAD and closes the "accepted work during disposal" question with the
real executor rather than an injected one.

Refreshed public-consumer output on a freshly source-bound build (rebuilt
`dist`, build snapshot hash `1d74ddfc3b463007`, 1403 files, identical across
all four runs). All four public example CLI runs exit 0 with awaited cleanup
PASS, actual `native-cpu`/`onnxruntime-node`/`cpu` identity, `fp32` precision
and pinned node ORT `1.25.0-dev.20260228-6e72d31970`: Node 26.2.0 and Bun
1.3.14 each match the official oracle on JFK (38 tokens) and on the
same-captured-PCM speech/silence control (69 tokens, waveform comparison PASS).
Receipts under `tools/data/results/orukeet/`:
`public-example-node-jfk-post-c21-2026-10-03.json`,
`public-example-bun-jfk-post-c21-2026-10-03.json`,
`public-example-node-speech-silence-post-c21-2026-10-03.json`,
`public-example-bun-speech-silence-post-c21-2026-10-03.json`. All remain
**PARTIAL_PASS_OUTPUT_ONLY**.

Lifecycle: real mid-decode abort plus two oracle repeats pass again
(`native-cancel-reuse-post-c21-2026-10-03.json`; abort observed at decode
1/508, handle identity unchanged). New `--dispose-drain` control
(`native-disposal-drain-post-c21-2026-10-03.json`) requests disposal while a
call is in flight, on a directly created family session first and then on the
loaded public handle. Both phases settle with oracle-matching output before
disposal resolves (dispose 3341.73ms / 3373.48ms), the next call on each is
rejected, and `createSession()` after disposal is rejected. The control fails
closed if disposal settles before the accepted call, if the accepted output
diverges, if disposal itself fails, or if a post-disposal call is accepted.
Seven new simulated regressions cover those behaviors and the exclusive
loaded-session requirement (`tests/nemo-inspector-disposal-drain.test.ts`);
they are not real provider acceptance.

Full regression: 1,462 PASS / 33 SKIP / 0 FAIL; receipt:
`tools/data/results/harness/post-c21-disposal-drain-full-suite-2026-10-03.json`.
TypeScript (library and examples) and scoped ESLint pass; the only lint output
is pre-existing `max-lines` warnings. TLS verification/system CA were enabled
only for that test child.

No runtime, decoder internals, family math, tolerance or public API changed: the
new control lives in the existing inspector adapter.

The disposal record now carries the observed settle order
(`["acceptedCall","dispose"]` on both real-provider phases in
`native-disposal-drain-settle-order-post-c21-2026-10-03.json`) instead of an
asserted boolean; the derived flag is computed from that observation.

Follow-up on the open browser gate: the head call 20 normalized-score failure
is now attributed to the real-encoder feed, not the WASM head. Perturbing only
the recorded encoder delta (`1.4566e-6`) on official reference inputs
reproduces the failure exactly (`5.8079e-4`, same 3 mismatches, same first
channel); predictor-state deltas contribute nothing, and the head response is
linear at about 399x gain. Details and the two read-only probes are in
`PARAKEET_VARIANT_TASKS.md`. The gate remains FAIL at unchanged tolerances.

Still open, with the unchanged failing gates cited by receipt: production
feature parity remains FAIL at the unchanged `atol=1e-5, rtol=1e-4`
(JFK 313/140,928, `filterbank-feature-jfk-2026-10-03.json`; speech/silence
74/519,808 on the default path versus 73 under the diagnostic width-8
normalizer, `production-raw-width8-speech-silence-2026-10-03.json`; pure
silence 12,800/12,928, `filterbank-feature-pure-silence-2026-10-03.json`).
Also open: the browser real-hybrid normalized-score failure at head call 20,
source-audio resampling acceptance, held-out quality, fp16 promotion, a hung
provider still delaying disposal, and measured memory/VRAM recovery after
teardown.

## C21 — Direct public TDT session shutdown

Status: PASS — public runtime/model/session lifecycle with injected executor;
not real ORT/WebGPU teardown, raw executor or cross-family certification.

High-level model handles already used an operation lease, but sessions returned
by direct model.createSession() could dispose their executor during transcription.
Two new deferred-call regressions reproduced premature executor disposal for
session and model owners. The TDT session now consumes the existing shared
runtime operation lease: accepted transcription (including canonical mapping)
settles before executor teardown, new calls reject once session disposal enters,
and errors release their lease. No decoder internals or shared helper changed.
Dependency direction remains shared runtime ownership primitive -> family session.

Seven new lifecycle cases include the public root runtime.loadModel().createSession()
path: session/model/runtime shutdown with successful or failing inference, plus
multiple active calls and an executor cleanup failure. Runtime disposal first
settles inflight loads and reaches model/session cleanup asynchronously; tests
observe actual session-dispose entry, not a fixed microtask count or an instant
closure guarantee for all runtime-owned models. A faulty timing-assumption test
extension was reverted before rebuilding that synchronization. Direct session
disposal closes its own lease synchronously. Repeated cleanup retains the same
failure and disposes the executor once; callers retain their inference error.

Eight session tests and seventy-five related executor/initialization/high-level/
streaming-owner tests pass with TLS verification enabled. TypeScript, scoped
ESLint and formatting pass. Full suite: 1,455 PASS / 33 SKIP / 0 FAIL; receipt:
`tools/data/results/harness/nemo-tdt-public-session-disposal-full-suite-2026-10-03.json`.
TLS verification/system CA are test-child-only settings.

Real provider shutdown and resource recovery still need an actual inference
check. Raw OrtNemoTdtExecutor callers and other family sessions are not covered
by this change; a never-settling inference still delays disposal. Orukeet parity
remains FAIL and no model promotion, tolerance change or speed claim follows.

## C20 — Preprocessor active-run disposal ownership

Status: PASS — deterministic deferred-run/session-creation simulations;
not real provider teardown, whole executor or long-session certification.

Two regressions reproduced disposal completing while a preprocessor run was
still pending, for both successful and failing inference. Active calls are now
registered synchronously and drained through their tensor cleanup before the
processor releases its session. New calls reject once disposal begins, and a
post-acquisition check prevents a warm-session run from starting after disposal.
Accepted runs already executing retain their result/error; disposal waits, it
does not interrupt the provider. A never-settling provider run still prevents
disposal completion; no unsafe forced release or timeout was added.

Six new tests cover success/failure, cleanup-before-release, multiple active
runs, pending creation success/failure, warm acquisition and repeated disposal.
Thirty-one processor and fifty-six processor/native-bridge/RNNT/lifecycle tests
pass; TypeScript, scoped ESLint, formatting and independent review pass.
Full regression: 1,448 PASS / 33 SKIP / 0 FAIL; receipt:
`tools/data/results/harness/nemo-preprocessor-run-disposal-full-suite-2026-10-03.json`.
TLS verification/system CA were enabled only for that test child.

Ownership remains local to the NeMo processor; no model math, shared runtime,
decoder or public API changed. Broader executor encoder/decoder disposal during
active transcription remains open and must be checked through public sessions
before claiming whole-model shutdown safety. Orukeet feature parity remains FAIL.

## C19 — Preprocessor cleanup failure isolation

Status: PASS — injected tensor-disposal failures and session reuse;
not real native/browser memory-leak or concurrent-run certification.

Following C18, a throwing output disposer still skipped later outputs and both
inputs; a throwing input disposer skipped the next input. Cleanup could also
mask an inference or validation error. Eight new regression cases reproduced
the failures before the production fix; one cleanup control already passed.
The processor now attempts each unique transient tensor once despite earlier
disposal failures. Single errors preserve identity; multiple errors are reported
in an AggregateError, with the original operation failure first and as cause
when present (including a thrown undefined). Aliased outputs are not disposed
twice. No shared helper, decoder, feature mathematics or public API changed.

Twenty-five processor tests and fifty processor/native-bridge/RNNT/lifecycle
tests pass. TypeScript, scoped ESLint, formatting and independent review pass.
Full regression: 1,442 PASS / 33 SKIP / 0 FAIL; receipt:
`tools/data/results/harness/nemo-preprocessor-disposal-cleanup-full-suite-2026-10-03.json`.
TLS verification/system CA were enabled only for that test child. These are
mocked fault-injection regressions, not actual ORT allocation/disposal failures
or measured GPU/CPU memory recovery. Concurrent process/dispose remains open.

## C18 — Preprocessor partial input allocation cleanup

Status: PASS — injected ORT input-allocation failure/reuse ownership;
not actual out-of-memory, browser/native VRAM or full lifecycle certification.

NeMo TDT's ONNX processor allocated both inputs before its cleanup scope.
If the second Tensor constructor threw, the first input was never disposed;
the executor's session teardown did not own those per-call tensors. A regression
reproduced the missing dispose call. Both allocations now occur inside the
existing try/finally; optional references release whichever inputs succeeded.
Original error identity is preserved, inference does not start on allocation
failure, later calls reuse the session, and repeated disposal releases it once.
No model mathematics, public API, backend selection or generic runtime changed.

Forty-one processor/native-bridge/RNNT/lifecycle tests (including 16 processor
tests) pass; TypeScript, scoped ESLint, formatting and diff checks pass.
Full regression: 1,433 PASS / 33 SKIP / 0 FAIL. Full regression
receipt: `tools/data/results/harness/nemo-preprocessor-allocation-cleanup-full-suite-2026-10-03.json`.
TLS verification/system CA were enabled only for that test child. Real native
allocation-failure injection, cleanup errors during tensor disposal and
concurrent processor-run/disposal behavior remain separate acceptance work.

## C17 — Repeatable real-browser local lifecycle smoke

Status: PASS — bounded stock-v3 hybrid cancellation/reuse/disposal smoke;
not Orukeet parity, GPU-kernel interruption or long-session soak acceptance.

`npm run test:browser-local-lifecycle -- --consumer-root <browser-demo-root> --model-root <stock-v3-model-root> --audio <fixture> --expected-phrase "<full phrase>" --encoder-precision fp16 --output <new-receipt.json>`
uses the existing family-owned browser page and public library inference API,
with cooperative decode cancellation enabled. Paths and expected text are
required, never guessed or downloaded. Requires existing Python/Playwright,
Chrome, Node, the sibling's installed Vite toolchain and local model files.
It builds the harness in memory from the current source aliases; no dist
rebuild, sibling edit, dependency installation or public model promotion.
Use the direct Python entry point if selecting a particular interpreter.

The runner now independently requires two matching nonempty results, actual
WebGPU encoder/WASM decoder metrics, inference/disposal evidence and no
execution/cleanup errors. Requested cancellation must reject with
`PipelineAbortedError` at a partially completed decode, without completion.
Full expected-transcript normalization stays in the existing shared browser
smoke comparator, not a second Python decoder or ASR quality metric.
`npm run test:browser-local-lifecycle:unit` runs ten fast preflight/receipt tests;
they do not launch model inference. Invalid PASS labels and partial lifecycle
evidence are rejected. The real browser command is opt-in, not part of Vitest.

Fresh Chrome 154 / ORT Web 1.29.0 run through the npm command passed:
two identical Harvard transcripts, decode aborted at 5 / 55 units, successful
reuse, awaited disposal, no blocked requests/page errors, verified engine/build
bytes, unchanged inputs, preview process stopped and port closed.
Receipt: `tools/data/results/harness/local-lifecycle-npm-2026-10-03.json`.
Existing ORT node-placement warnings remain; session provider metrics are not
proof that every graph node ran on GPU. No production runtime code changed.

Post shared-cleanup refresh (2026-10-03): the same npm opt-in with fresh receipt
`tools/data/results/harness/local-lifecycle-shared-closure-2026-10-03.json`
passes real Chrome154/ORT Web1.29.0 stock-v3 fp16-encoder/fp32-head inference.
Decode cancellation rejects at5/55 units with PipelineAbortedError, without a
complete event. Both successful results are exactly
"The boy was there when the sun rose." with WebGPU encoder/WASM decoder metrics;
awaited disposal, unchanged protected inputs, served build/engine byte hashes,
zero blocked requests/page errors and stopped preview all pass. Shared closure
probe explicitly reports ECONNREFUSED. Single-run times2223.65ms and339.79ms
are cold/reused observations, not a matched optimization benchmark or soak.
This closes real lifecycle revalidation of the helper reuse, not app UI model
loading, Orukeet numerical parity or every-node GPU placement.

## C15 — Preserve requested remote artifact revisions

Status: PASS — network fetch revision contract; existing cache authentication
and real remote/browser downloads are not certified by these simulated tests.

Removed the shared asset handle's silent HF revision-404 fallback to `main`.
Byte reads, streamed reads and blob locators now reject the requested missing
artifact without fetching or caching a different revision. Provider controls
preserve successful branch/commit reads and the omitted-revision `main` default.
The original failure was reproduced on all three paths before the fix.
54 focused IO/external-data/runtime/builtin tests passed; full suite: 1,414
passed, 33 skipped, zero failed. Receipt:
`tools/data/results/harness/hf-revision-strict-full-suite-2026-10-03.json`.
TypeScript and scoped ESLint passed. Migration and old-cache limitations are
documented in `architecture.md`; no cache entries or sibling files were deleted.

Legacy built-in cache isolation (2026-10-03): a prepopulated wrong-main entry
under a pinned key reproduced the old-cache hole before implementation. The
built-in HF helper now uses a versioned JSON-tuple key for non-main revisions;
main/default keys and all caller-owned provider keys are untouched. A network
read populates the new key and the second read reuses it; tests prove the old
entry remains and no delete occurs. All 35 targeted HF/IO/IndexedDB tests and
source typecheck pass. Status: PARTIAL PASS pending independent review and wider
regression. This uses injected cache/network controls, not an actual remote
download or persistent browser migration; first materialized pinned reads may
require downloading again. No handle-level override of custom cache keys was
retained. Namespace isolation does not authenticate arbitrary cache contents.

Wider regression checkpoint: fresh full suite with TLS verification enabled in
the test child process passes 1,431 tests /33 SKIP /0 FAIL. Receipt:
`tools/data/results/harness/hf-legacy-cache-isolation-full-suite-tls-2026-10-03.json`.
The preceding run inherited an environment TLS-disable flag and is not used as
TLS-enabled evidence; no persistent environment setting was changed. Scoped
ESLint passes. New TS hunks were formatted; unrelated baseline test formatting
was restored and eight HF tests re-passed. Independent final review and commit
remain next; actual remote/browser cache migration is not certified here.

Independent final review: PASS, no correctness findings. Reviewer reran the
32 HF/handle tests and source/examples typechecks; recorded full-suite counts
were verified. Baseline formatting warnings outside new test hunks remain
unchanged. Built-in key isolation is accepted; real browser persistence remains
an explicit next integration rung, not implied by injected cache tests.

Real Chrome persistence rung (2026-10-03): PASS using
`python tests/smoke/hf-cache-browser-smoke.py --output <new-receipt.json>`.
Chrome154 fresh context seeds legacy wrong pinned bytes and compatible main
bytes, intercepts one tiny pinned response, then reloads the same origin with
all remote requests blocked. Pinned text persists in real IndexedDB, main text
is reused, and both old entries remain unchanged. No blocked request/page error;
owned preview closure confirms ECONNREFUSED. Receipt:
`tools/data/results/harness/hf-cache-real-chrome-bound-2026-10-03.json`.
Source, tool and in-memory bundle hashes are bound; no model download or sibling
mutation. This exercises internal production helpers, not published imports,
real HF/TLS, model inference, cross-browser or browser-restart persistence.
Existing-output preflight rejects before build/browser; optimized Python is
rejected because it would disable assertions.

## C16 — Caller cancellation during a coalesced model load

Status: PASS — cooperative pipeline wait/cache cancellation contract;
prompt cancellation and real browser integration are not certified here.

Two deterministic tests reproduced aborted joined/cached callers receiving
handles before the fix. `src/runtime/load.ts` now checks the caller signal
before mutation/cache access, after mutation/disposal waits and after a joined load.
A cancelled joiner rejects without cancelling the live owner or disposing its
shared handle. Pre-aborted forced reload does not evict an existing handle.
The creator still owns the underlying load signal: its cancellation rejects
live joiners too; explicit retry creates a fresh usable handle. A third test
records that ownership/cleanup contract, without silent retries or transfers.
40 focused lifecycle/API/builtin tests passed; TypeScript passed. ESLint has
zero errors and two max-lines warnings. These are simulated generic runtime
tests, not real model/browser acceptance. Boolean-only `AbortSignalLike` is
checked cooperatively at await boundaries, not immediately during a pending
promise. Generation/forced-reload/disposal regressions stay in the same suite.
Final full regression: 1,418 passed / 33 skipped / zero failed; independent
read-only review passed after adding the post-forceReload-disposal guard.
The test environment emitted an existing TLS-disabled warning; no global TLS
settings were changed, and these results do not certify secure remote loading.

Post-commit package verification (2026-10-03): `npm run build`,
`node tests/smoke/node-import-smoke.mjs`,
`node tests/smoke/browser-root-import-graph.mjs` and
`node tests/smoke/packed-native-absent-smoke.mjs` passed on the current working
tree (including preserved paused Nemotron edits; not a clean-commit artifact).
The browser static closure covers 298 modules / 27 entry points. The isolated
offline packed consumer imports without native ORT, passes strict public
declaration checks and rejects an explicitly unavailable native backend before
asset/model creation, without silently substituting available WASM.
The retained fixture is `C:\Users\steam\AppData\Local\Temp\asrjs-packed-consumer-kZkUuL`.
These are packaging/import and unavailable-backend controls, not actual browser
inference, ASR parity, native execution or a packed-consumer cancellation test.

## C14 — Companion in-memory browser preview

Status: PARTIAL PASS — actual browser-demo UI boot/settings; model inference,
local file-picker/load, service-worker behavior and other companion UI remain open.

`node tests/smoke/companion-build.mjs <consumer-root> --serve` now serves the
existing in-memory build on an ephemeral 127.0.0.1 port. It writes no sibling
build/config/cache files. Only the explicit build outputs and bounded top-level
public files enter its immutable route map; directories and symlinks are not
recursively traversed, individual public files above 10 MiB fail. COOP/COEP,
exact Content-Length, WASM MIME and no-store headers support browser checks.
This is not a general Vite preview replacement or a file/model directory server.
Default build-only behavior is unchanged; three existing CLI negatives pass.
Syntax, focused diff checks and actual browser-demo source-alias build pass.

Actual Chrome154.0.8037.93 control (2026-10-03): fresh context, service workers
blocked and external requests rejected; title/Load Model/Idle state render,
Parakeet v3 selection keeps WebGPU encoder + WASM decoder, and Local folder
selection exposes the local-model panel. No page errors or ONNX weight fetches.
Four HTTP missing/traversal/source/malformed-path negatives return 400/404;
the main JS served bytes match build SHA256
`314ec796f0bdb0bb159778ec933a7ce42d83ee2aa0848c30c1e9a3d7cae4be53`.
Config SHA256 remains `05b797e775018c124b38ca310802453e5bb72854bcc7aebf68d525ffbd0dbbcb`.
The app attempts font/badge and Hugging Face metadata requests at startup/model
selection; this bounded control blocks them rather than claiming offline support.
Evidence is terminal output, not an automated persisted browser receipt.

Automation refresh (2026-10-03): the existing
`python tests/smoke/browser-demo-smoke.py --consumer-root <browser-demo-root> --output <new-receipt.json>`
already implements this scenario and verifies served HTML/application bytes
against its in-memory build receipt. Fresh Chrome154 run passed boot, v3 hybrid
selection, WASM selection, local-source panel and missing-folder recovery with
no page errors or weight requests. Font/badge/HF metadata requests remain
blocked and recorded; this is not a claim that the app is fully offline.
Receipt: `tools/data/results/harness/browser-demo-ui-closure-2026-10-03.json`.

Fixed a cleanup evidence defect: socket errors of any kind no longer count as
closed-port proof. UI and local inference smoke runners share a bounded Node
probe that accepts only explicit ECONNREFUSED; reachable ports, timeouts,
unknown errors and malformed/contradictory evidence cannot pass cleanup.
Two new regressions failed before implementation;8 UI-runner and10 local
lifecycle tests now pass, with Python syntax checks. The real UI receipt records
the stopped preview and `{closed:true, code:'ECONNREFUSED'}`. No sibling source
or published runtime changed; local inference was not rerun for this tool-only
helper reuse. Next cover real app-local artifact loading/cancellation and
transcription separately from the existing public-library lifecycle harness.
UI-selected hybrid settings do not prove graph placement or ASR correctness.

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

## C08 — Real Chromium manual microphone controller lifecycle

Status: PASS — real browser/fake WAV device, explicit manual mode only.

Extended the offline runner with a separate controller scenario, preserving the
capture-owner control. The public controller, default starter and monitor are
real library instances. Instrumentation wraps actual capture handles and actual
AudioContext/AudioWorklet resources; it does not substitute fake audio nodes.

Chrome 154.0.8037.93 passed stop/flush with exact delivery of accepted captured
PCM, restart, active disposal without an extra utterance, stable disposed state
and rejection of start after disposal. A separate controlled handoff delays the
return of an already-live capture handle: disposal accepts no PCM and closes
the context and owned track after handoff. All three sessions loaded one actual
worklet, stopped through the controller once, ended tracks and closed contexts.
This is not a microphone permission-dialog race or exact source-WAV parity.

Evidence: `tools/data/results/harness/real-browser-manual-controller-2026-10-02.json`.
The separate capture-owner regression also passed:
`tools/data/results/harness/real-browser-capture-controller-control-2026-10-02.json`.
Receipts bind source/probe/runner/audio and served-module hashes. Fresh source
build, JS/Python syntax, CLI help and diff checks pass. No downloads or sibling
writes. Speech-detect/VAD, ASR, WebGPU, physical-device and cleanup-failure
acceptance remain open; no Orukeet/model promotion follows from these checks.

Next: prove controller-owned cleanup when capture stop rejects, independently
of capture-owner cleanup and model-parity work.

## C09 — Controller disposal after release failures

Status: PASS — controlled resource-failure regressions; real-browser healthy
manual lifecycle control also passes. Browser failure injection is untested.

A rejecting capture stop previously skipped starter unsubscription, monitor
disposal and starter disposal. Since the controller was already marked disposed,
a second dispose could not complete those releases. The new single-failure
regression failed at the missing unsubscription before the implementation fix.

Dispose now attempts each owned release independently in the original order,
clears buffered manual PCM and listeners, preserves a single error by identity,
and aggregates multiple failures in release order. Two tests cover a capture
failure and simultaneous capture/unsubscribe/monitor/starter failures, including
safe repeated disposal. Normal stop/flush behavior is deliberately unchanged.

Verification: 46 related capture/browser/realtime tests pass; independent review
reran all eight disposal tests. Source and example compilation, strict changed-
test compilation, scoped lint, formatting and diff checks pass. Fresh-build
Chromium manual-controller healthy control passed all three real capture sessions:
`tools/data/results/harness/real-browser-controller-disposal-control-2026-10-02.json`.
That browser receipt does not certify injected cleanup failures, VAD or ASR.

Next: separately specify and test normal stop when capture stop rejects; return
to Orukeet's earliest feature divergence without relaxing numerical acceptance.

## C10 — Logical stop after capture cleanup failure

Status: PASS — simulated failure matrix and real-browser healthy manual control.

Previously a rejected capture stop skipped capture-info reset, speech-detect
starter stop/manual flush, stopped status and monitor finalization. Four new
manual/speech-detect × flush true/false regressions failed before the fix.
Stop now independently attempts those operations, preserving the existing
flush option and surfacing errors afterward: single errors retain identity,
multiple failures are aggregated in operation order. Buffered manual PCM is
delivered once only when requested; flush false discards it. Speech-detect
still delivers the final starter segment when flushing despite capture failure.

Six focused cases cover that matrix, capture/starter/monitor simultaneous
failure, and disposal overtaking a pending failing stop. Review caught late
manual publication in the first implementation; that source edit was reverted
before adding the race guard. Disposal now suppresses buffered publication and
late status/monitor updates while queued disposal owns remaining cleanup.
The revised implementation passed independent review (20 focused tests).
Monitor mocks are cleared after startup to ensure stop finalization is
actually observed. All 52 related capture/browser/realtime tests pass. Source
and example compilation, strict changed-test types, scoped lint, format and diff
checks pass. Current-build Chromium healthy manual lifecycle passed three real
sessions: `tools/data/results/harness/real-browser-controller-stop-reviewed-2026-10-02.json`.
This is not browser cleanup-failure injection, physical-device, VAD-model or ASR
acceptance. Sibling consumer execution remains untested; no sibling edits.

Next: bounded package/runtime acceptance and cancellation/repeated-session
checks, while Orukeet's portable feature and recurrent browser gates stay open.

## C11 — Cached pipeline reuse after cancellation

Status: PARTIAL PASS — generic controlled-model regression; real-model host
reuse remains untested.

The existing cancellation test now also proves a fresh-signal transcription
succeeds after an aborted call, using one cached model and disposing it once.
Both calls use an explicit stable cacheKey: an AbortSignal-bearing request is
not automatically serializable, so expecting automatic cache reuse would test
the wrong contract. No production implementation changed.

Verification: 28 high-level API/lifecycle tests, scoped lint, formatting and
diff checks pass. Standalone strict test compilation reports a generic
TranscriptResponse mismatch in the unchanged createProbeRuntime mock at line
60; that mock typing remains open and this is not a clean typecheck claim.
This test does not certify real ONNX session recovery or browser GPU reuse.

Typing follow-up (2026-10-03): reproduced the line-60 factory error. The mock
session returned canonical output for every generic response flavor. It now
uses actual public input/options/conditional response types and returns native,
canonical or envelope output as requested. Three public loaded-handle cases
independently assert exact single/batch response shapes and once-only disposal.
Strict standalone NodeNext compilation with skipLibCheck disabled now passes;
31 high-level API/lifecycle tests, scoped lint, format and diff checks pass.
No production API or type was weakened. Cancellation reuse remains a controlled
model result, not real ONNX recovery acceptance.

## C12 — Global window timestamp contract

Status: PARTIAL PASS — controlled two-window public API regression.

Token-detail work in progress (2026-10-03): detailed window requests now retain
tokens, offset available seconds, assign global token indices and remap word
references without mutating window-local outputs. Only matching ID, text,
raw text and positive time intervals (within 1e-6 seconds) identify duplicates;
matching is one-to-one. Untimed or unmatched tokens inside audio overlap are
retained with `pipeline.windowing.token-overlap-unresolved`, not guessed away.
Missing word references are omitted with the same warning. Tokens can therefore
contain unresolved window evidence rather than a fully reconciled final sequence.
Model-local frame/step metadata is preserved, not converted to global indices.
The direct planner also resolves `returnTokens` via the existing canonical
detail resolver, preserving explicit detail precedence.

Current source typecheck, scoped ESLint, diff check and 22 focused windowing
tests pass. Final independent review approved the corrected ambiguity behavior.
The final full suite passed 1,427 tests with 33 skips, with TLS verification
enabled only in the child process. Formatting was confined to new code; existing
unrelated formatting and paused Nemotron changes were preserved.
Next: real model/browser token-detail acceptance and bounded long-audio memory
measurement. This conservative merge is not numerical ASR parity acceptance.

Real stock-v3 token-detail control (2026-10-03): the existing local Chrome
public API harness now has opt-in `--verify-window-tokens`. The same decoded
short speech fixture is placed in two disjoint, silence-padded 20-second windows.
Independent validation requires detailed output, two actual windows, contiguous
global token/word indices, finite global seconds in both windows, valid word-token
references and repeated identical detail. This is no-overlap real-model control,
not overlap reconciliation, numerical parity, Orukeet or long-session acceptance.

The first real run correctly failed: windowed metrics discarded actual component
backend identity. A failing-first regression exposed the same generic defect;
the shared string accumulator now retains encoder/decoder/joint backend identity
and emits `mixed` when windows disagree. Source typecheck, scoped lint and 23
windowing tests pass;11 Python harness tests and JS/Python syntax checks pass.
The second real Chrome154 run exited0 with two repeated 40-second results, each
22 tokens and16 words, WebGPU encoder/WASM decoder, awaited disposal, unchanged
inputs, no page errors/blocked requests and explicit ECONNREFUSED cleanup proof.
Receipts (preserved separately):
`tools/data/results/harness/local-window-tokens-v3-2026-10-03.json` (FAIL),
`tools/data/results/harness/local-window-tokens-v3-providers-2026-10-03.json` (PASS).
Next: finish independent harness review, full regression/static checks and
scoped commit, then actual overlap and bounded-memory controls.

Review checkpoint: runtime provider telemetry is committed separately as
`2641f9e`, with 1,428 full-suite PASS /33 SKIP,23 targeted tests, typecheck and
scoped lint. The harness extension remains uncommitted pending two regressions:
reject oversized decoded fixtures before allocating the doubled window buffer;
reject in-range word-token references whose token timing belongs to the wrong
window. Existing successful real receipt is preserved, but these validation
holes must close before accepting/committing the harness extension.

Resolved checkpoint (2026-10-03): both review regressions now pass. The fixture
rejects empty or over-19-second mono PCM before allocating a fixed 40-second
repeated buffer (2,560,000 bytes); this does not bound the original media decode
allocation or certify VRAM/soak behavior. Word-token references must also match
the word's timing interval, rejecting an in-range reference into another window.
Independent read-only review approved the corrected code. Twelve Python runner
tests and two Node fixture tests pass. The corrected actual Chrome hybrid run
exits 0 with receipt
`tools/data/results/harness/local-window-tokens-v3-bounded-2026-10-03.json`:
two repeat transcriptions each retain 22 tokens, 16 words and two windows, with
identical canonical detail, WebGPU encoder/WASM decoder, no page errors or
blocked requests, unchanged input hashes, awaited disposal and ECONNREFUSED
preview closure. This remains stock-v3 no-overlap smoke, not Orukeet parity,
overlap reconciliation, timestamp accuracy or a long-session memory benchmark.

Actual shared-speech overlap diagnostic (2026-10-03): opt-in
`--verify-window-tokens --window-token-overlap` places the short speech once at
15 seconds in a bounded 34-second PCM fixture. Two 20-second windows overlap
by six seconds, sharing the complete speech; source PCM must be at most five
seconds. A failing-first Node fixture test now passes, together with all three
fixture tests and 13 Python runner tests. CLI rejects overlap without requested
token detail; the validator requires the explicit overlap policy and still
checks canonical indices, timing/reference associations and repeated detail.
Receipt `tools/data/results/harness/local-window-tokens-v3-overlap-2026-10-03.json`
exits 0: both actual Chrome hybrid results contain the single expected phrase,
eight words and 15 tokens, with unchanged inputs and successful cleanup.
Status is PARTIAL PASS: word/text duplication control passes, but token overlap
remains unresolved. Four context-shifted occurrences remain beyond the 11-token
phrase; both runs emit `pipeline.windowing.token-overlap-unresolved`. Global
token indices therefore do not imply a unique chronologically ordered token
sequence. Do not hide this evidence by fuzzy timing deduplication or call it
complete reconciliation. Next: review this diagnostic, then determine a safe
canonical token ownership contract using the captured real ambiguity before
changing generic merging; long-session memory measurements remain separate.

Review correction: two windows alone did not prove overlap was applied. A
failing-first negative removes the overlap warning from an otherwise valid
two-window result; the diagnostic now requires the captured ambiguous-overlap
warning in both repeats. All 13 Python tests pass. The existing real receipt
passes the tightened validator by offline post-processing; this is not a new
browser run with the updated runner hash. This fixture deliberately certifies
the unresolved-overlap diagnostic, not all future successfully reconciled output.
Public transcript JSDoc now explains token-array indices, known word references
and why retained ambiguous tokens must not be joined to reconstruct final text.
No token selection/deduplication behavior or model mathematics was changed.

Added word and segment-only cases through root-exported
`transcribeWithWindowing`. Four seconds of synthetic PCM produce two explicit
two-second windows without overlap: local times 0.25..1.25 become
0.25..1.25 and 2.25..3.25 in the final canonical result. The model-local
result objects retain their original times. Segment-only detail does not
invent words. No production mathematics, API or model code changed.

Verification: 16 windowing/stage tests, scoped ESLint, diff check and strict
standalone NodeNext test compilation with skipLibCheck disabled pass.
Still open: overlap-retained timestamp checks, token-detail behavior and
real supported-model/browser/companion-demo acceptance. This bounded
simulation does not complete P08 or certify timestamp quality of any model.

Overlap regression (2026-10-03): a two-window case initially failed with
`alpha shared alpha shared omega` and a timestamp rollback. `windowWords`
already included pending words, but both accumulation sites concatenated that
history again. Reused the existing interval-aware merge at those two sites;
the unchanged acceptance now retains the longer shared word at 1.6..2.0 s,
and a three-window extension retains earlier history and final tail times.
Source typecheck, strict test typecheck, scoped lint and 18 targeted tests pass.
The full suite before the third-window extension passes 1,378 / 33 SKIP,
including a repeat with inherited TLS-disable removed only in that child.
Build/browser/real-model/companion acceptance is not claimed; dist is untouched
while the live DFT diagnostic binds compiled asset hashes.

Browser control (2026-10-03): Chrome 154.0.8037.93 passed all four synthetic
public-root cases (two/three overlap windows and word/segment-only offsets).
`tests/smoke/window-timestamps-browser.py` bounds the asynchronous probe to
60 seconds and checks served JS bytes against the requested local build,
with build/tool hashes before and after execution. Wrong-build and existing
receipt negative controls fail without writing a success receipt. Evidence:
`tools/data/results/harness/window-timestamps-browser-bundled-2026-10-03.json`.
The isolated tsc output was bundled with installed esbuild for browser package
resolution; raw ESM correctly fails to resolve bare `pako`. Main dist remains
untouched. A directly owned, quiet loopback HTTP server was used after the
external shell-based helper and route.fetch interception proved unsuitable;
response hashing now observes original browser requests, not proxy requests.
This certifies synthetic windowing in a real browser only, not model accuracy,
GPU execution, token detail or sibling consumers. Those gates remain open.

## C13 — Compact public-root consumer bundles

Status: PARTIAL PASS — source consumer bundle closure, not inference acceptance.

Run `node tests/smoke/compact-browser-bundle.mjs` for a stdout-only receipt and
`node --test tests/smoke/compact-browser-bundle.test.mjs` for four regressions.
Installed esbuild (through the existing development toolchain) bundles two
public-root imports in memory: createSpeechRuntime is 22,074 bytes and
transcribeWithWindowing is 17,971 bytes, minified browser ESM. Each has a
32,768-byte regression budget; this is not a universal application size target.
Both retain no model/preset, ORT engine or native backend code and no external
imports. Metafile visited inputs are not confused with retained bytes.

The tool hashes JS/TS bytes supplied to esbuild and rechecks them after both
builds, plus root package metadata and its own source. Receipts include actual
esbuild version and bundle hashes. This is not exhaustive dependency-resolution
metadata provenance. Nothing is emitted to dist, and no dependency is installed.
Negative tests reject family/engine/native retention, Windows path variants,
external imports, exceeded budgets and missing/malformed output evidence.
Syntax, four tests and formatting pass. These tests use Node's test runner;
the existing Vitest glob does not include them.

Still open: published-package consumer bundles, startup cost and application
integration. No production optimization, model inference or WebGPU claim.

## C14 — Sibling consumer build acceptance

Status: PARTIAL PASS — actual in-memory Vite builds; browser workflows remain open.

`node tests/smoke/companion-build.mjs N:\github\asrjs\browser-demo` and the
same command for `streaming-demo` pass with their installed Vite 7.3.1 and
current public aliases into this checkout's src. Browser-demo transforms 307
modules, streaming-demo 344; transcription and VAD worker bundles are included.
Browser-demo CSS is 29,900 bytes with the correct consumer working directory,
not the earlier 5,852-byte wrong-cwd diagnostic. Streaming-demo CSS is 30,915
bytes. These are build measurements, not rendering or performance acceptance.

The normal Vite config bundler writes a temporary module into the sibling's
node_modules; its diskless runner instead fails on these configs' \_\_dirname.
The new probe compiles captured config bytes into in-memory ESM, supplies the
original absolute dirname, resolves packages from the consumer, and disables
Vite config reloading/build writes. Its process working directory is restored.
Only object configs with a vite.config.js entry are covered. Arbitrary custom
config/plugins are executable code: review them before using this probe; the
write:false option is not a security sandbox for plugins.

Stdout receipts bind config/package/tool hashes, actual Vite version and all
in-memory output byte counts/hashes, not exhaustive application source identity.
No dependency installation, sibling source edit or dist rewrite was performed.
Current checkout includes preserved paused Nemotron changes; this is not a
clean-commit release build. Browser-demo's existing config hash stays
05b797e775018c124b38ca310802453e5bb72854bcc7aebf68d525ffbd0dbbcb;
streaming-demo's stays ec887bf5b4c056e8a4181dd37b771ee422ebbde6cbb3df551087df414d4744de.
Browser-demo retains its classic coi-serviceworker script warning, and both
report stale Browserslist data; no package update was made to silence warnings.

Syntax, formatting and three prerequisite negatives pass using
`node --test tests/smoke/companion-build.test.mjs`: missing root, surplus
arguments and requested missing config fail without a PASS receipt.
These Node tests are separate from the existing Vitest glob.
Next: real browser model management/capture/hybrid workflows and other affected
companions. These builds do not complete those integration gates.

Additional consumer builds (2026-10-03): the same reviewed-config in-memory
harness passes for playground (304 modules, main JS720,714 bytes, CSS33,392),
vad-demo (344 modules, main JS936,946 bytes, CSS30,915), and benchmark-demo
(288 modules, main JS692,208 bytes, CSS4,428). Installed Vite7.3.1, same tool
SHA25649635ee0d807c15573bf8cae968739cba1ef1e4fab9c29624a56d90fb4173357.
Actual config SHA256s respectively:
`518e81c022fb9d320ad8e9683a0bb998292a919c10d3cda38e176b2efe4396d8`,
`a044208026c577751638462eb785194ba25366023d2f0d2a9bace8a3fd2d0f89`,
`9a4242d6740dcaec1f2154f0e5b86bdf87d6eb22605a02a8589537ab5838f197`.
Each command exits0 and binds config/package/tool/output identities in stdout;
no sibling output files or source edits. Stale Browserslist warnings persist
where reported, benchmark retains the classic coi-serviceworker warning.
These broaden source consumer build coverage, not browser app acceptance.

Independent actual-Chrome manual-controller check (2026-10-03): existing
`capture-lifecycle-smoke.py --scenario microphone-controller` passes against
the current untouched dist with the pinned ItsLifeJim WAV fake device.
Receipt: `tools/data/results/harness/manual-controller-current-dist-2026-10-03.json`.
Two stop/flush sessions each accept4 chunks; the active-disposal session accepts0.
All three close their AudioContexts and end owned tracks, with one stop call
and one worklet load each. The runner also checks delayed-handle disposal.
This is REAL_BROWSER_FAKE_DEVICE_MANUAL_CONTROLLER, not a sibling app, physical
microphone, ASR/VAD/WebGPU, or fresh source-to-build equivalence certification.
Dist was deliberately not rebuilt while the live DFT run binds compiled bytes.
Actual sibling model-management/capture/hybrid workflows remain open.

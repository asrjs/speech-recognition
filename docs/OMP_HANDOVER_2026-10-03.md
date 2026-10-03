# Full handover — OMP / next coding harness

Snapshot date: 2026-10-03. Implementation base: `279aab2`, branch `main`.
Documentation-only handover changes follow that base. Refresh Git before resuming.
This handover is not a claim of finished Orukeet support or a completed library.

## Start here

Read [AGENT_ENTRYPOINT.md](AGENT_ENTRYPOINT.md) and all required documents in its
order. The durable goal is [GOAL_PROMPT_RECOMMENDED.md](GOAL_PROMPT_RECOMMENDED.md).
The [PROJECT_CHARTER.md](PROJECT_CHARTER.md) is already stored locally; it need
not be pasted separately. Existing task files, not this document, hold task status.
Old Whisper handoffs and archived goals retain history, not current instructions.

The user requested a harness migration, so no new code development or model
inference was started during this handover. No push, upload, publication, global
installation, weight download, tool configuration change or sibling edit occurred.
The previous long-running whole-product goal is unfinished. Its intent is now
portable in the repository; OMP need not depend on Codex chat IDs, attachments,
agent handles, hidden memory or scheduling state.

## Product intent and confirmed decisions

- Build a compact, capable, easy multi-model speech library, API and examples
  for browser WebGPU and separately verified native Node.js/Bun. Model porting
  is one lane; API, runtime, VAD, streaming, long audio, assets, examples and
  engineering tools must also improve. Browser porting is our differentiator.
- The user owns `parakeet.js`. Preserve our Parakeet TDT v2/v3 and Whisper
  lessons without copying Transformers.js architecture/API/folder structure.
- Model order: Orukeet first, then genuine smaller Parakeet student/distilled
  models, then ranked non-NVIDIA ASR/alignment/diarization candidates. Smaller
  fp16/int8 files are not smaller student models. Orukeet is about 627M parameters.
- No new NVIDIA model work beyond existing Parakeet v2/v3. Nemotron remains
  paused, not declared bad or completed. Preserve its inherited dirty work.
- Reuse original official weights, original-engine captures and own fp32 ONNX
  exports. Numerical stages plus full token/text/state checks are required.
  Human quality labels are distinct from original-model implementation oracles.
  fp16 must follow fp32 acceptance and re-pass; int4/int8 is not the default.
- Improve the existing harness, tools, smokes, simulations and browser loops
  while shipping bounded product increments. Keep neutral orchestration shared
  and family-specific math/decode/tests separate. Never hide a model fix in runtime.
- Browser WASM thread cap: `floor(hardwareConcurrency / 2) + 2`, preserving lower
  explicit requests and single-thread fallback. Native thread options are separate.
- Scoped validated local commits are authorized. Push/publish/HF mutation,
  new large downloads, global repair/install/settings and out-of-scope sibling
  writes require new authority. Parallel writers require isolated writable
  checkouts; do not fall back to concurrent edits in the shared checkout.

## Architecture and workspace

Root: `N:\github\asrjs\speech-recognition`. Windows 10 / PowerShell.
Package `@asrjs/speech-recognition` 0.2.0, ESM, declared Node >=22.
Live host versions checked during handover: Node 26.2.0, Bun 1.3.14.
ORT Web 1.29.0 is a regular dependency; native ORT is an optional peer and a
development dependency (pinned `1.25.0-dev.20260228-6e72d31970`). Do not inherit
the archived claim that both engines are regular consumer dependencies. Check
the existing offline install-footprint/consumer receipts before redoing packaging.
One headless, framework-neutral package with coherent root API and intentional
subpaths. Inspect package exports and actual source, not old architecture sketches.

Actual source folders: `alignment`, `audio`, `chunking`, `ctc`, `inference`, `io`,
`models`, `pipeline`, `post-processing`, `presets`, `quality`, `runtime`,
`tokenizers`, `types`. There is no `src/processors`. Shared contracts/ownership
primitives feed family implementations; presets and apps consume them. Family
tensor wiring, decoding and caches stay in models. Brand configuration stays
in presets. Framework UI stays outside core. Preserve clone-safe canonical results.
Read architecture for exact dependencies; a source folder does not prove model support.

First-class consumers live in parent `N:\github\asrjs`: `streaming-demo`,
`browser-demo`, `benchmark-demo`, `playground`, `vad-demo`, `firered-vad-web`,
`webgpu-agent-test`, `parakeet-ui-webgpu`. Inspect their actual aliases and builds.
They exercise realtime, model management, API usage and Parakeet hybrid WebGPU
encoder/WASM decoder. They are not disposable unrelated demos. A FireRed VAD
app does not establish integrated FireRed ASR. Parent/sibling writes are outside
the current writable root; seek authority when an affected app needs edits.

## Git and protected local state

`git worktree list` showed one checkout only:
`N:/github/asrjs/speech-recognition 279aab2 [main]`. No alternate worktree needs
to be resumed. The index was empty before handover edits.

These twelve inherited tracked modifications are protected. Do not stage,
revert, reformat, clean or reinterpret them as owned handover changes:

```text
src/models/nemotron-rnnt/config.ts
src/models/nemotron-rnnt/executor.ts
src/models/nemotron-rnnt/ort.ts
src/models/nemotron-rnnt/types.ts
src/presets/descriptors.ts
src/presets/nemotron/factory.ts
tests/nemotron-rnnt.test.ts
tools/model-debugging/reference/nemotron-3.5-asr-streaming/smoke_export.mjs
tools/nemotron-onnx-export/convert_fp16.py
tools/nemotron-onnx-export/export_nemo_fp32.py
tools/nemotron-onnx-export/validate_nemo_fp32.py
tools/scripts/benchmark-nemotron-wasm.mjs
```

Many local JSON/gzip receipts, diagnostic scripts/graphs and candidate reports
are untracked; models may be ignored. They are present in this workspace but
not guaranteed in a fresh clone. Never `git add .`, clean them, commit weights
or assume Git alone transfers the evidence. For another machine copy needed
local artifacts independently, verify hashes and obtain download authority if
missing. The same-directory OMP switch preserves them. Root `AGENTS.md` is
locally ignored; it was updated locally but the tracked entrypoint is portable.
No owned execution handle remains active at handover. This does not assert that
every user-owned system process is stopped; never kill unknown browsers/servers.

## Recent committed product work

| Commit    | Scope and verified boundary                                                                                                                                                                          |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `279aab2` | C21: direct public TDT session leases drain accepted transcriptions, including canonical mapping, before executor disposal. Eight session tests; 75 related tests.                                   |
| `7e696dc` | C20: preprocessor registers accepted runs synchronously, drains through tensor cleanup, rejects new calls, prevents a warm acquired session starting after disposal. 31 processor tests; 56 related. |
| `1061a38` | C19: deduplicate tensor disposal by identity, attempt all releases despite failures, retain original inference errors and aggregate cleanup errors. 25 processor tests; 50 related.                  |
| `930c53c` | C18: release partial preprocessor allocations when a later tensor constructor fails. 16 processor tests; 41 related.                                                                                 |
| `bf5887c` | Shared browser WASM thread cap with family wiring; native intra/inter-op settings unaffected.                                                                                                        |

C18-C21 have unit/deferred-operation evidence, not certification of real ORT
shutdown at current HEAD, raw executor callers or all families. A hung provider
still delays disposal; no unsafe forced release was added. Runtime disposal
reaches model/session teardown asynchronously after inflight loads; do not claim
all model leases close instantly at the runtime call. Direct session disposal
closes its own lease synchronously. Existing output/decode contracts are unchanged.

Latest full-suite receipt was parsed during handover:
`tools/data/results/harness/nemo-tdt-public-session-disposal-full-suite-2026-10-03.json`:
**1,455 PASS / 33 SKIP / 0 FAIL**, success true. The recorded run exited zero.
TypeScript, scoped lint/format and focused checks passed at implementation time.
The handover did not rerun the model or full suite. These tests do not prove
numerical Orukeet parity, real GPU execution, quality or measured performance.
Recent lifecycle edits require refreshed actual-inference receipts/build bindings.

Other useful completed slices, with limits in PRODUCT_TASKS:

- Pinned Hugging Face revision cache isolation and no accidental main fallback;
  real Chrome IndexedDB isolation used intercepted tiny responses, not full HF ASR.
- Stock Parakeet v3 public browser hybrid cancel/reuse/disposal loop; cancellation
  stops the decode loop, not an already-submitted GPU kernel. Not Orukeet acceptance.
- Public native consumer `examples/nemo-tdt-native-local.mjs` and CLI smoke adapter;
  reuse them, do not create a second executor or decoder in another runner.
- Long-audio canonical token preservation and browser window controls. Overlap
  work retains warnings: a tested overlap still has four unreferenced context
  tokens. Do not claim universal deduplication or silently discard them.
- Compact bundle/package/consumer and companion build baselines exist; inspect
  current task receipts rather than inferring installed cost from lazy imports.

## Orukeet local assets and evidence

Original `.nemo`: `models/orukeet/source/orukeet-v0.1.0.nemo`, 2,509,342,720 bytes.
Pinned revision: `b59c13a733fce5cf193230d0fa317ee7f145a108`.
Recorded SHA-256:
`031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`.
This is the original checkpoint size, not runtime memory or a small student.
Config/tokenizer audit matches all 8,193 Parakeet v3 token IDs.

Own fp32 external-data pack: `models/orukeet/onnx/fp32-packed-2026-10-01/`.
Recorded identities (verify before using; no multi-GB rehash during handover):

| File                     |         Bytes | SHA-256                                                            |
| ------------------------ | ------------: | ------------------------------------------------------------------ |
| encoder-model.onnx       |     2,151,530 | `9924d5c30da5d54454dd3de6dbfc177aa771205eaac5da55d8f05ec813036f17` |
| encoder-model.onnx.data  | 2,475,088,896 | `f7a19051cc2913d99635d70a856bfe82795cab1f4af3670433e9fc1eb101e00f` |
| decoder_joint-model.onnx |    72,520,893 | `89fb566d2f0cd339387582dec9753959c84d54a90272581e7f51ceecc5a8eb90` |
| source-config.json       |       260,178 | `344297eaf9e28d7f849e099c06535fbb0df978926e1b5f753b6aa836d05acdb3` |

A separate persistent single-step head is under
`tools/data/results/orukeet/persistent-single-step-head-2026-10-02/`.
Inspect its contract/receipt before replacing any graph. Do not mix old and new
head sidecars, hide artifact identity or promote a public preset/fp16 variant.

Results below use `tools/data/results/orukeet/` basenames. Preserve failures.

| Gate                                              | Current evidence / limitation                                                                                                                                                                   |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Official original engine                          | JFK, speech/silence, pure silence and synthetic long captures exist. Multilingual held-out human quality is not closed.                                                                         |
| Production features                               | **FAIL**: JFK 313/140,928 mismatches, max `3.423690795898e-4`; speech/silence 74/519,808, max `8.1732869148e-5`; silence 12,800 mismatches. Existing feature tolerances `atol=1e-5, rtol=1e-4`. |
| Conditional encoder/head replays                  | Some official-feature/frozen-feed component controls PASS; not production-PCM end-to-end acceptance.                                                                                            |
| Real public Orukeet browser hybrid                | **FAIL** on normalized scores despite exact 38-token text/frame/duration and passing final recurrent state; see below.                                                                          |
| Node/Bun public native                            | Recorded JFK 38-token and same-captured-PCM speech/silence 69-token outputs match: **PARTIAL_PASS_OUTPUT_ONLY**. Refresh after C18-C21.                                                         |
| Long audio                                        | Official/native synthetic 8x clip control exists (~149.7s, 722 tokens), not real public-browser long-audio certification.                                                                       |
| Quality, normal source resampling, fp16 promotion | Open, not implied by matching output on selected controls.                                                                                                                                      |

### Avoid rediscovering completed diagnostics

Production processor math has not been replaced by diagnostic controls. No
width-8 normalization default, silence magic value or fixture hardcode is accepted.
Actual windowed float32 frames match the official engine byte-for-byte in aligned
controls; Torch/MKL RFFT on those frames reproduces the native spectrum. This
does not make a browser MKL implementation available or certify a portable FFT.

Already-tested FFT alternatives include fp32 direct DFT (78 downstream failures
on speech/silence), fp64 DFT (32), PocketFFT (56), PFFFT SSE (65)/scalar (64),
packed/full-complex/radix/rounding controls and ORT DFT/STFT. ORT DFT equals ORT
STFT, but differs from Torch and downstream acceptance fails. Do not repeat
these unchanged experiments or sweep variants without a discriminating hypothesis.

Latest attribution receipts, not production acceptance:

- `light-dft-library-power-speech-silence-2026-10-03.json`: injecting official
  complex values through library power/mel/log/width-8 normalization has zero
  feature mismatches, max `2.3543834686e-6`; candidate direct DFT still has 78.
  The official-complex control is already done, not the next novel experiment.
- `identical-power-mel-speech-silence-2026-10-03.json` (`026a384`): same native
  power plus actual filterbank gives byte-exact linear mel/guard values and
  passing features; only two log/feature bit differences remain. Means/stds
  match exactly. This isolates downstream arithmetic, not an accepted FFT fix.
- `identical-log-input-speech-silence-2026-10-03.json` (`9a88a32`) and exact
  same-log width-8 normalization control (`60b9d06`) are also done.
- Production-raw + width-8 speech/silence control (`acc07dd`) still FAILS with
  73 mismatches, versus ordinary production 74. Do not report 73 as production.
- Actual filterbank binary transport (`f05bea9`) preserves float32 signed zero;
  regenerated/JSON coefficients can lose bits. That diagnostic fix does not
  close full feature parity. Use actual captured coefficient bytes/config.

Probe checks at latest numerical control: 81 Python and five JS tests PASS.
Do not parse giant gzip traces indiscriminately (long trace ~257MB compressed).
Use scoped/streamed stage readers and summaries. Actual STFT capture filenames
include `official-speech-silence-stft-stages-2026-10-02.json.gz`.

### Real browser and input-contract failures remain visible

`browser-real-public-hybrid-jfk-2026-10-03.json`: actual fp32 WebGPU encoder plus
WASM persistent head, two public transcriptions, terminal exit 1. All 38 tokens,
frames, durations, text and final h/c pass; 46 head calls traced. Call 20, frame
69 has three failing normalized-score coordinates, max about `0.00057888` at
the unchanged score gate. Frozen-head controls do not certify accumulated feeds.
Native CPU versus WASM on identical captured call-20 head input passes; tiny
real encoder-feed differences (~`1.4566e-6` max) can yield those score failures.
Do not change normalization/tolerances or call exact text complete parity.

Original 22.05kHz speech/silence WAV produced 69 official versus 76 library tokens
with different resamplers. It is an input-contract failure, not proven decoder
fault. Canonical captured 16kHz mono float32 PCM (649,600 samples, no padding)
fixes the controlled output comparison but does not accept the normal source
resampler. Its recorded f32le hash is
`8f4293517daf07f433b307a4a971f7c17dc9ed348b5f72999945b217331ec3df`.
JFK legacy effective PCM is UNVERIFIED. Keep source-audio and captured-input
controls distinct; matching WAV hashes alone does not prove matching model input.

## Tools to reuse, not duplicate

- `tools/nemo-tdt-onnx-export/` and `tools/whisper-onnx-export/` own export recipes.
- `tools/model-debugging/reference/hf-parakeet-port/python-nemo-inspect.py`
  original-engine capture; `compare_js_feature_boundaries.py`,
  `probe_nemo_processor.py`, `processor_export_stages.py`,
  `replay_packed_fft.py` and nearby tests/recipes.
- `tools/model-debugging/scripts/node-asrjs-nemo-inspect.mjs` public-library
  inspection and explicit captured-input mode; `node-compare-stage-captures.mjs`,
  `tools/tensor_parity.py`, `tests/support/fixture-gates.ts` comparators/gates.
- `examples/nemo-tdt-native-local.mjs`,
  `tests/smoke/nemo-tdt-native-example-smoke.mjs` real consumer CLI + oracle adapter.
- `tests/smoke/transcribe-fixture.mjs`, compact bundle/companion build smokes,
  `npm run test:browser-local-lifecycle` and existing sibling browser loops.
  Read each CLI/help/source before constructing arguments; stock-v3 smoke is
  not an arbitrary-model runner. Browser loops must call the actual library.
- `tools/artifact_outputs.py`: fresh exclusive output receipts. Never overwrite
  a canonical capture or accept a stale success file after a failed command.

Existing NeMo Python environment is in
`tools/model-debugging/reference/nemotron-3.5-asr-streaming/.venv/Scripts/python.exe`.
Reusing its pinned environment for Orukeet is not resuming Nemotron. It has
Torch 2.6/cu118 and NeMo 3.0; a different default Python/Torch is not an equivalent
oracle. Inspect exact environment/provider before using it; do not auto-install
or change the official engine to make a CUDA warning disappear.

## Dependency-ordered resume priorities

These are the next slices to reconcile into the existing backlog, not a second queue:

1. **G0/G6:** verify Git/worktree/protected files, local artifacts and test baseline.
   Read the latest appended P-task evidence: earlier paragraphs can be historical
   and may still say a browser path or diagnostic is missing when it has since run.
2. **G3/G5, C18-C21 follow-up:** refresh actual Node/Bun public inference and real
   browser/hybrid lifecycle with a freshly source-bound build. Exercise accepted
   work during disposal, failures, reuse and awaited cleanup through existing
   consumers. Simulations already passed; real provider races are not yet closed.
3. **G1/G6:** choose one novel bounded earliest-feature FFT/power contract
   experiment after auditing completed controls. Record budget/exit criteria and
   why it distinguishes causes. Reuse immutable official inputs; no blind FFT
   rewrite or another identical-complex/power/log injection labeled new progress.
   Keep browser score failure independently recorded, without downstream hacks.
4. **G2-G5:** continue a useful independent public API/example/long-audio or
   compactness gap with an affected-consumer acceptance target. Do not let either
   unlimited numerics research or generic lifecycle refactoring consume the product.
5. Complete required fp32 original-engine/public native/browser/input/quality
   gates before fp16 optimization or model promotion. Then audit true smaller
   Parakeet students, and rank current official non-NVIDIA speech candidates.
   Ask before materially deferring Orukeet, changing model order or finish line.

Use G0-G6/P01-P10/C records. For each slice: user-visible outcome, dependencies,
owned files, baseline, acceptance checks, commands/identities, result paths,
honest status and next bounded step. No gate is waived because another lane passes.

## Baseline and launch commands

Read-only startup, from project cwd:

```powershell
git status --short
git diff --name-only
git diff --cached --name-only
git worktree list
git log -10 --oneline
node --version
bun --version
node node_modules/typescript/bin/tsc --noEmit
node node_modules/vitest/vitest.mjs run tests/nemo-preprocessor-contract.test.ts tests/nemo-tdt-session.test.ts tests/nemo-tdt-lifecycle.test.ts tests/nemo-tdt-native-ort.test.ts tests/model-session-initialization.test.ts tests/high-level-pipeline-lifecycle.test.ts tests/streaming-owner-cleanup.test.ts
```

Full suite is a separate regression lane. An inherited parent environment may
disable TLS; do not persist or propagate that workaround. Enable verification
only in the test child. Do not change global settings:

```powershell
powershell -NoProfile -Command '$env:NODE_TLS_REJECT_UNAUTHORIZED="1"; $env:NODE_USE_SYSTEM_CA="1"; node node_modules/vitest/vitest.mjs run'
```

These are invocation recipes, not a claim of fresh handover-run tests. Installed
dependencies already exist. Before a real-artifact run inspect package build
behavior: the normal build removes/recreates `dist`. Do not trust older dist
for later source fixes or destroy needed local build evidence. Prefer the
existing isolated/source-bound harness path where available, record code and
actual executable build identity, and obtain a clean build safely before claiming
current acceptance. Choose new receipt filenames and retain terminal exit status.

OMP interactive launch (CLI source inspected, not successfully executed here):

```powershell
omp --cwd 'N:\github\asrjs\speech-recognition'
```

Paste the startup prompt from AGENT_ENTRYPOINT in that interactive session.
`omp -p "<prompt>" --cwd 'N:\github\asrjs\speech-recognition'` is one-shot print
mode, not the requested interactive workflow. No model/provider override is needed.

**Known OMP environment issue:** previous `omp --help` and direct Bun CLI startup
both exited 1 with `Cannot find module '@babel/parser'`. The parser package file
exists; module resolution, not a proven missing package, is the issue. Installed
OMP is 18.4.12. No repair was attempted after two failures, and no permission to
repair the global installation has been received. This handover does not certify
OMP is runnable. Fix/approve that environment separately; do not mutate ASR code
or global packages as a workaround, and do not repeat identical failed launches.

## Acceptance of this handover

This change is documentation and routing only. Its validation is diff scope,
local links/paths, preserved goal requirements/history and untouched protected
files, not a new numerical or real-browser run. Reconcile any later commits and
task receipts at startup. The library mission and Orukeet acceptance remain open.

Handover checks: new goal/entrypoint/handover and local AGENTS formatting PASS;
relative link targets PASS; archived goal text matches the prior document;
all twelve protected file SHA-256 values unchanged; parsed suite summary and
small config hash agree with this snapshot. Independent read-only verification
caught a nonexistent tool filename, which was corrected to existing tools.
The existing model task file already had formatting differences before this
change; only its startup header was edited, not its historical body. Staged
diff whitespace checks PASS. No full suite, model export or inference rerun was
required or claimed for these documentation changes.

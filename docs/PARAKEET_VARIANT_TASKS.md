# Parakeet variants: ordered implementation tasks

Updated: 2026-10-01. Owner: current speech-recognition development chat.
Active candidate: `oruk/orukeet`, a Parakeet TDT v3 fine-tune. Smaller distilled
models and locale fine-tunes enter the same queue after their exact architecture
and tokenizer are audited. The user owns `parakeet.js`; reuse its verified
export, preprocessing and decoding experience as well as this library's tools.

User-confirmed scope: Orukeet first, then genuine Parakeet distillations and
new non-NVIDIA ASR candidates. No new NVIDIA model work beyond existing TDT
v2/v3 for now. TDT-CTC 110M and Nemotron 3 Diarization are not active tasks.
Keep parameter count separate from checkpoint/export/quantized file size:
Orukeet retains 627M parameters; it is not a small student model.

## Current checkpoint

Latest public-library browser control (2026-10-02): the reproduced single-step
head now passes live recurrence with frozen official encoder outputs for JFK,
pure silence and speech/silence on both WASM and WebGPU. Each fixture/backend
runs twice on one public model handle; tokens/frames/durations/text are exact
and equivalent final committed h/c states pass unchanged numerical tolerances.
This is CONDITIONAL_RECURRENCE_PASS, not production feature, real-encoder or
full-ASR acceptance. Intermediate live logits/states, grid execution and matched
memory/performance remain open. Next: trustworthy live-call observation and
intermediate numerical comparison, alongside whole-product lifecycle work.

Latest 2026-10-02: the opt-in single-step fp32 head transformation is now a
persistent tested family tool, not an inline-only recipe. The original
production graph still fails its WebGPU speech/silence cell-state gate; the
separate reproducible candidate passes all recorded JFK/pure-silence/speech-
silence inputs on native CPU (240 calls), WASM (480) and WebGPU (480). This is
not a promoted model or live recurrent ASR acceptance. See the latest section
below and `persistent-single-step-head-handoff-2026-10-02.json`. Next return to
the earlier production-feature divergence; retain candidate-fed library
recurrence, real browser grid execution and matched memory/performance gates.

2026-10-02 all-recorded browser head follow-up: `--all-steps` now selects every
validated joint binding, with selection/count metadata and missing-call tests.
JFK (46), pure silence (12) and speech/silence (182) ran twice per backend:
960 real graph calls total. WASM passes all three fixtures; WebGPU passes JFK
and pure silence, but speech/silence FAILS on cell-state output at joint 139
and 173 in both repeats. All token/duration argmax decisions are exact; scores,
hidden-h and lengths pass. No fetch or cleanup errors. Tolerances remain
`atol=1e-5, rtol=1e-4`; do not replace numerical acceptance with exact decisions.
This is recorded-input component evidence, not a live recurrent decode or
full-model port. Existing feature, encoder, long-reference and application gates
remain open. Small receipt: `browser-head-all-handoff-2026-10-02.json`.
Next isolate those two predictor bindings on a fresh WebGPU session, then
compare actual graph arithmetic/provider partitioning with the passing WASM
control. Keep shared-feature work and compact-runtime tasks in the queue.

Fresh-session isolation on the same date: explicit bindings 139 and 173 ran
twice on each backend (eight calls). WASM still PASS; WebGPU still FAIL on the
same two cell-state coordinates. All outputs on both backends are bit-identical
to their earlier full-matrix outputs. No fetch/cleanup errors; WebGPU records
29 dispatches per call. Thus preceding-call count/session accumulation is not
supported as the cause for these recorded inputs. The source graph concatenates
two LSTM cell-state outputs; failing flat indices 1102/1137 lie in layer 2.
Next localize the LSTM provider arithmetic, including the first-layer hidden
input to layer 2. This does not prove the underlying shader/operator cause.
Receipt: `browser-head-isolated-cells-handoff-2026-10-02.json`.

Provider audit: the original head graph contains two LSTM nodes, but installed
ORT Web 1.29.0's WebGPU resolver has no direct LSTM registration. Do not call
the state mismatch a proven GPU LSTM shader defect. The optimized runtime
partition remains UNVERIFIED. An opt-in kernel-profile experiment timed out
waiting for callback completeness; no profiling result was certified. Its
source edits were fully reverted and 11 original focused tests passed.
Async callback lookup versus kernel release is a source-level lifecycle risk,
not the proven cause of this timeout. Next capture layer-2 input/state taps on
both backends, or drain profiler callbacks before release in a bounded diagnostic.
Receipt: `browser-head-provider-audit-2026-10-02.json`. Feature parity remains
the earlier production failure; this independent provider investigation does
not replace its gate or the compact-runtime product queue.

Task P01 is complete; P02 and P03 are partial. Existing Nemotron changes on `main` are preserved.
Orukeet remains an experimental candidate until P08 passes. Resume the first
unfinished task and record commands, result files, and blockers here.
The recommended whole-product goal is `GOAL_PROMPT_RECOMMENDED.md`; this task
list is one workstream, not the complete web/Node/Bun library backlog.

Latest continuation (2026-10-01): ready docs, native runtime, harness and official
export tooling are committed as `1d636a2`, `32722da`, `01e7dd2` and `c3eba39`.
Paused Nemotron edits and large local artifacts remain deliberately unstaged.
The refreshed suite has 1,303 PASS / 27 SKIP / 0 FAIL; actual native Node/Bun
speech/silence consumers each match all 69 tokens on the captured official PCM.
Real native identity-graph acquisition and partial-failure cleanup also pass,
but these are output-only and resource-only controls, not model acceptance.

The independent direct-DFT diagnostic covers all 1,003 unique frame/bin pairs
supporting the full-FFT controls' 129/120 failed coordinates (mel bands 0–127).
On identical canonical float32 frame bytes, the Float64 JS control's maximum
DFT component error is `1.91e-15`; official complex64 and JS Float32 controls
are `8.22e-7` and `5.14e-7`. This weakens an incorrect full-Float64 FFT algorithm
hypothesis, not every possible production processor defect. Production still
has 313 feature mismatches at unchanged tolerances. See the small
`tools/data/results/orukeet/selected-dft-and-commit-handoff-2026-10-01.json`.
The follow-up packed-real control is now captured in
`tools/data/results/orukeet/library-feature-packed-dft-jfk-2026-10-01.json`.
Its raw complex components agree with independent DFT within `2.47e-15` at
all 1,003 selected frame/bin pairs; all rounded components match. This control
uses official-aligned float32 preemphasis/window products and rounded power,
not the unmodified production pipeline. Its 120 feature mismatches persist;
production still has 313. Forty-four weight-free probe tests pass.
The unmodified production capture is now recorded in
`tools/data/results/orukeet/library-feature-production-frames-final-jfk-2026-10-01.json`.
It preserves all normalized features, valid raw mel and original power exactly.
Production's Float64 frame products differ from official Float32 frame values
at 435,823 / 563,712 positions (max `9.44e-8`, mean `1.54e-9`); an explicitly
cast f32 control still has 365,586 differing bit patterns. On the exact original
production f64 frames, packed FFT versus independent NumPy FFT has max component
errors `7.11e-15` / `6.22e-15`. This is diagnostic evidence of input arithmetic
differences, not proof that these alone cause all 313 feature mismatches.
Forty-nine probe tests pass; the f64 reader rejects coerced mirrors, empty
frames and boolean/float metadata dimensions. Full suite was not rerun for this
tool-only slice. Small receipt: `production-frames-handoff-2026-10-01.json`.
The cumulative preemphasis/window controls and pure-silence capture are now
recorded below. The contiguous f32 cascade diagnostic's width-8 candidate
matches normalization of the same official raw mel on JFK and pure silence;
width 16 fails silence. This does not yet change the production JS processor.
The temporary JS control now matches Python across 58 size/valid-prefix/width
cases and nine real raw-mel boundaries. All silence controls are exact; JFK
still fails for production/aligned FFT raw inputs, while official complex
injections pass. Next bounded P03 experiment: localize the remaining valid-prefix
raw-mel/complex64 rounding differences on actual failure support; distinguish
FFT rounding from mel/log accumulation before any production processor change.
Keep the aligned control's residual reference complex64 differences and
mel/log/statistics amplification separate. Existing P02–P08 gates remain open.

Selected-power causal control: `cascade-power-override-jfk-final-2026-10-01.json.gz`
replaces only bin 1/frame 346 and bins 244–255/frame 551 with official float32
powers, after the original power snapshot and before unchanged JS mel/log.
FFT input bytes, pre-injection powers and all unselected power bits are exact
to the baseline. Four raw-mel cells change (shared filters affect adjacent
mel bands); fresh cascade statistics are independently checked. Mel 0/frame
346 raw error falls from `2.46048e-4` to zero. Mel 127/frame 551 retains only
`9.53674e-7`; its normalized error falls from `9.06587e-5` to `1.31130e-6`.
The diagnostic control's feature mismatches fall from 121 to 118, not zero.
This proves causal power sensitivity at selected points, not production parity
or a usable implementation: reference power injection must remain diagnostic.
65 probe tests, 6 Python reduction tests and 8 Node tests pass; the real run
exits 1 because the unchanged production gates still fail. Small receipt:
`selected-power-intervention-handoff-2026-10-01.json`.
Next bounded P03 step: isolate actual 12-bin mel accumulation and final log
rounding at mel 127/frame 551 using identical official power, while keeping
the still-open FFT complex64 rounding mismatch separate. No production change,
fp16 promotion or tolerance relaxation is justified by this control.

Global official-power mel/log control (2026-10-01):
`reference-power-mel-log-jfk-2026-10-01.json.gz` verifies all injected power bits
and exact valid-prefix reconstruction of the actual NeMo matmul/add/log boundary.
Observation hooks preserve the existing reference-power features exactly.
Raw mel/log has zero tolerance failures but 7,201 nonidentical valid cells
(max `1.90735e-6`); rounding only the JS log input reduces nonidentical cells
to 3,801, not zero. At mel 127/frame 551, both rounded linear mel and guarded
input already differ from NeMo, so this does not isolate a log-only defect.
The official-power feature control passes; unchanged production still has
313 feature failures. 67 weight-free probe tests pass; the real run exits 1
at the unchanged production gates. Small receipt: `reference-power-mel-log-handoff-2026-10-01.json`.
Next: evaluate JS log on identical official guarded inputs, separately from
the float32 matmul accumulation and still-open FFT rounding mismatch.

Identical official log-input control: `identical-log-input-jfk-2026-10-01.json.gz`
confirms all guarded input bits match NeMo and compares JS `Math.log` followed
by f32 storage, without FFT or mel accumulation. Only 6 / 140,800 valid raw
cells differ exactly (max `9.53674e-7`); zero fail the original tolerances.
Mel 0/frame 346 and mel 127/frame 551 raw logs are exact on identical inputs.
The separately checked diagnostic width-8 normalization also has zero tolerance
failures (max `6.85453e-7`); this is not production normalization or a port PASS.
69 probe tests pass, and the real run exits 1 with unchanged production's
313 feature failures. Receipt: `identical-log-input-handoff-2026-10-01.json`.
Next discriminate mel product/add/reduction arithmetic on identical official
powers and filterbank before revisiting the upstream FFT rounding mismatch.
Do not hardcode corrections for the six log cells or infer active GEMM kernel
dispatch from installed MKL availability.

Mel product/add discrimination (2026-10-01): on byte-identical official powers,
official filterbank coefficients and validated cached support, the diagnostic
`f64-product-f32-sum` reproduces all 140,800 valid float32 linear-mel and rounded
guarded-input cells exactly. The `f32-product-f64-sum` / `f32-product-f32-sum`
controls have 33,329 / 27,171 nonidentical linear-mel cells. The matching linear
control still has 741 nonidentical raw-log cells because its unchanged JS log
receives an unrounded double guard addition; this is separate from the previous
identical-f32-log-input control's six differences. All three raw-log controls
pass the unchanged tolerances. This is scalar mixed-precision evidence, not
proof of FMA emulation, active GEMM dispatch or production/model parity.
Production remains at 313 failed feature coordinates. 73 probe, 6 Python
reduction and 8 Node tests pass; the real run exits 1 at unchanged production
gates. Receipt: `mel-arithmetic-handoff-2026-10-01.json`.
Next validate the same arithmetic on silence, then isolate guard rounding and
normalization with actual library spectra before any shared processor change.

Post-commit continuation (2026-10-02): all three mel arithmetic controls match
all 12,800 valid silence linear/input/raw-log cells exactly, with official
support coverage verified. Production still fails 12,800 normalized-feature
cells. On JFK, the matching mixed-precision control's existing f32-guard-log
counterfactual is byte-identical to the independent official-input JS-log
control (six exact differences, zero tolerance failures). Receipt:
`mel-arithmetic-continuation-2026-10-02.json`. Next combine these isolated
arithmetic/guard controls with diagnostic normalization on actual library
spectra, retaining separate upstream FFT and production acceptance gates.

Actual-spectrum follow-up (2026-10-02): selected mixed-precision mel plus
optional f32 guard is now isolated on untouched production spectra and on the
existing official-complex-rounded path (JS computes power; no official-power
injection). Every candidate preserves its own baseline power/filterbank/bounds
bytes; width-8 normalization independently matches Python bytes. On JFK,
both production-spectrum variants still have 315 normalized-feature failures,
like the baseline width-8 production-raw control. Official-complex-rounded
variants have zero failures; f32 guard reduces maximum feature error from
`2.86102e-6` to `1.43051e-6`, without establishing an actual FFT implementation.
All four silence controls match raw and normalized values exactly. Unmodified
production remains FAIL: JFK 313, silence 12,800. Mel/guard rounding alone is
not a demonstrated production fix. 77 probe, 6 Python reduction and 8 Node tests
pass; both real runs exit 1 at unchanged production gates. Full suite and
companion/browser/model inference were NOT_RUN for this tooling-only change.

The expanded JFK probe exposed `RangeError: Invalid string length` when one
aggregate JSON string exceeded V8's limit. The slice was reverted before retry;
awaited per-field JSON output preserves all evidence and passes a simulated
aggregate-limit regression plus both real fixtures. Individual top-level fields
still have a string-size bound; this is not an unbounded streaming serializer.
Receipt: `mel-spectrum-handoff-2026-10-02.json`.
Next apply the selected downstream arithmetic to actual official-aligned JS FFT
spectra, preserving each prior FFT baseline and separating frame/complex/power
errors. Do not promote the rejected packed-f32 FFT or hardcode mel corrections.

Aligned JS packed-real FFT follow-up (2026-10-02): the named `aligned-packed`
control retains the existing aligned preemphasis/window/filterbank and f32
complex-power conversion, without injecting reference powers or complex values.
Both selected mel/guard variants preserve baseline spectra and still fail the
same 121 JFK feature coordinates as `complexPowerCapture` with width-8
normalization (zero added/removed coordinates; max `2.52336e-4`). Raw log values
pass their own tolerances, but this does not override normalized-feature FAIL.
Mel 0/frame 346 and mel 127/frame 551 retain raw deltas `2.46048e-4` and
`6.10352e-5` against the separately verified official-input log control.
Silence raw/normalized values match exactly; unmodified production remains
JFK 313 / silence 12,800 failures. This closes mel/guard-only fixes as a
demonstrated solution for the current support. Continue at FFT/complex/power
rounding on exact aligned frames, not another mel-specific correction.

79 probe, 6 Python reduction and 8 Node tests pass. A mutation test proves
reference spectrum changes do not affect the aligned-packed controls. Slim
mel captures omit duplicate FFT/frame payloads while baselines retain them.
Both real runs exit 1 at unchanged production gates; full suite and browser
inference were NOT_RUN. Receipt: `aligned-packed-mel-handoff-2026-10-02.json`.

## Ordered tasks

P06/P09 browser head checkpoint (2026-10-02): `browser_head_replay.py` now
replays first/middle/last official joint bindings twice through the compiled
NeMo TDT ORT bridge, with no duplicated decoder. One initial Chrome 154 /
ORT Web 1.29 WebGPU run on NVIDIA Blackwell passed scores/states/lengths and
exact token/duration decisions (29 compute dispatches per call). Its receipt
predates full tool-hash/raw-output binding. Subsequent WASM runs failed ESM or
WASM fetch; the hash-bound WebGPU repeat failed ONNX fetch. All had
`ERR_CONNECTION_RESET`, zero graph calls, no cleanup errors. Standalone Chrome
fetch and ESM import controls succeeded, so the delivery failure is not yet
localized. **INCONSISTENT / NOT ACCEPTED**, not model-op failure or full port
PASS. No active run remains. Next instrument/isolate loopback asset delivery
or reuse the existing Vite delivery with repo-local output; no precision,
tolerance or backend fallback workaround. Strict production features remain
313 JFK failures. Seven new browser-tool tests and all 23 shared + 22 family
Python tool tests pass. Protobuf 7 reflection incompatibility was reproduced
and fixed in shared `onnx_artifacts.py`; 10 tests pass on Protobuf 5 and 7.
Small receipt: `browser-head-handoff-2026-10-02.json`. P05/P06 remain open.

HTTP delivery follow-up (2026-10-02): the instrumented HTTP/1.0 control
reproduced a standalone WASM fetch reset with no ORT inference. The server
wrote its full 27,797,172 bytes with the correct hash; browser delivery failed.
Changing only the child-process handler protocol to HTTP/1.1 made nine ESM /
WASM / ONNX fetches in three clean Chrome cycles byte/hash-exact. The tool now
uses HTTP/1.1 with existing Content-Length framing and records per-request
status, written bytes/hash, elapsed time and errors. This bounds the local
delivery remedy; the underlying Windows TCP/filter cause is not proven.

Four hash-bound actual head runs now PASS: WASM and WebGPU each in two fresh
Chrome instances, three official JFK bindings replayed twice per instance.
All normalized scores, predictor h/c states and lengths pass existing
`atol=1e-5, rtol=1e-4`; all token/duration decisions are exact. Max score error
WASM `2.67029e-5`, WebGPU `6.10352e-5`; WebGPU has 29 compute dispatches per
call on NVIDIA Blackwell. No fetch failures or cleanup errors. This supersedes
the prior delivery-blocked selected-head checkpoint, **not** full browser
model acceptance. Features, encoder, all recorded head calls, accumulated
sequence, public API/application and performance gates remain open.
Nine tool tests cover HTTP framing/reuse and delivery receipts; all 24 family
and 23 shared Python tool tests pass. No runtime, model, precision, tolerance,
security setting or backend fallback changed. Next expand browser head replay
to the complete recorded fixture matrix, then continue feature/encoder and
public API rungs. Receipt: `browser-head-http11-handoff-2026-10-02.json`.

- [x] **P01 — Audit and acquire official assets.** Pin source revision and
      SHA-256, download original `.nemo` weights into ignored local model storage,
      retain license/attribution and compare config/tokenizer with Parakeet v3.
      Reuse the existing NeMo environment and export tools. Evidence belongs in
      `tools/data/results/orukeet/`.
- [ ] **P02 — Capture the official inference oracle.** Run official NeMo on fixed
      speech, silence, speech/silence and long-audio fixtures. Capture numerical
      features, encoder, predictor/state, joiner logits, token/duration decisions,
      timestamps and final text. Record environment and audio/weight hashes.
      Bind the oracle to exact effective float32 PCM, sample rate/channels,
      valid length and decoding/resampling/padding policy; verify that boundary
      before comparing model stages. A matching source-WAV hash is insufficient.
- [ ] **P03 — Export our fp32 graphs.** Reuse or extend the Parakeet combined
      encoder plus decoder/joint contract; validate I/O, external data and native
      ORT numerical parity using the existing stage comparator. Find the earliest
      divergence before proceeding.
- [ ] **P04 — Integrate the variant.** Add thin preset/source configuration
      above `nemo-tdt`, coherent public API/model selection and accurate language
      metadata; cover custom source/config and lifecycle with existing simulations.
      Separately exercise real public-library inference in native Node and Bun,
      recording actual host/engine/provider versions, full-output comparisons
      and awaited cleanup. Python ORT replay, import smoke and simulated sessions
      cannot certify either native consumer path.
- [ ] **P05 — Validate WASM.** Run the public library fixture smoke with JS
      features and exact token/text oracle; include repeated calls and disposal.
- [ ] **P06 — Run browser WebGPU loops.** Use `../webgpu-agent-test` and the
      existing browser harness: fp32 encoder, hybrid WASM decoder/joint, same audio
      and oracle, repeated load/transcribe/dispose. Require actual adapter use;
      collect session, numerical, token/text and memory failures as JSON.
- [ ] **P07 — Optimize and revalidate fp16.** Preserve fp32 oracle, compare
      graph variants and backend placement; measure warmed RTFx, loading, CPU/GPU
      transfers and peak memory. Integer quantization requires separate evidence.
- [ ] **P08 — Integrate examples and assess quality.** Exercise model
      management, streaming/windowing and timestamps in sibling demos. Compare
      with v2/v3 on fresh held-out audio with consistent scoring. Document license,
      language coverage, artifact provenance and limits before promotion.
- [ ] **P09 — Improve the porting machinery.** As each task reveals a gap,
      improve existing smoke/parity/simulation/WebGPU automation and add a focused
      regression check; record it with the task, rather than postponing all tooling
      work until the model is complete.
- [ ] **P10 — Choose the next Parakeet derivative.** Rank true smaller
      distillations and locale fine-tunes by independent quality, browser speed,
      memory and license. Keep Moonshine/diarization and broader product work in
      `docs/GOAL_PROMPT_RECOMMENDED.md`; Nemotron 3.5 ASR remains paused.

## Existing automation to reuse

- `tools/model-debugging/reference/hf-parakeet-port/python-nemo-inspect.py`
  provides the original NeMo inference entry point.
- `tools/model-debugging/scripts/node-asrjs-nemo-inspect.mjs` and
  `tests/smoke/transcribe-fixture.mjs` exercise the library on real artifacts.
- `tools/model-debugging/scripts/node-compare-stage-captures.mjs` compares
  stage captures by sample/audio identity and reports numerical errors.
- `tests/nemo-tdt-*.test.ts`, `tests/pipeline-windowing-stage.test.ts` and
  `reference/hf-parakeet-port/window-overlap-sim-test.mjs` cover simulated
  execution and transcript/window assembly.
- `../webgpu-agent-test` is the existing real browser test loop; discover its
  current CLI and local API before adding model-specific plumbing.

## Verified work and harness improvements

- G2/G3/G5/P04/P09 native consumer example slice (2026-10-01):
  `examples/nemo-tdt-native-local.mjs` uses only public imports, explicit native
  CPU composition, audited local candidate config/artifacts and awaited
  runtime-owned cleanup. Pure `audio/wav.ts` decoding is shared with browser
  media and exported through opt-in media/Node entrypoints; no decoder pipeline
  or model math was copied. The example rejects mismatched sample rates before
  engine creation, never downloads/resamples/falls back, and retains full
  canonical/native output. Focused checks: 75 passed; core/example typechecks
  and strict changed-test compilation passed. Full suite: 1,283 passed /
  27 skipped / 0 failed in
  `tools/data/results/harness/native-local-example-verified-full-suite-2026-10-01.json`.
  Actual public CLI controls in Node 26.2.0 and Bun 1.3.14 match all 38 JFK and
  69 speech/silence tokens, frame indices, TDT durations and full text, with
  awaited cleanup PASS. Receipts under `tools/data/results/orukeet/`:
  `public-example-node-jfk-verified-2026-10-01.json`,
  `public-example-bun-jfk-verified-2026-10-01.json`,
  `public-example-node-speech-silence-verified-2026-10-01.json`, and
  `public-example-bun-speech-silence-verified-2026-10-01.json`.
  All remain **PARTIAL_PASS_OUTPUT_ONLY**. JFK effective PCM is UNVERIFIED;
  speech/silence uses a derived IEEE-float WAV containing the exact captured
  official PCM, not acceptance of source resampling. The smoke wrapper reuses
  the existing oracle/input/output adapter, records either graph's sidecar,
  and fails on inconsistent rate/length/padding even with matching PCM hashes.
  The complete actual compiled build is hashed sequentially in every receipt;
  hashes identify executable code but do not alone prove source/build equivalence.
  Progress labels use the actual loading/transcription event contracts. IEEE
  float32/64 signed zero is preserved by the shared decoder after two RED/GREEN
  regressions; no resampling or feature-math replacement was introduced.
  Ten receipt regressions and 29 CLI/lifecycle simulations are not real
  numerical acceptance. Production feature math is unchanged; P02-P08 and
  actual failed partial engine-resource cleanup remain open. Package dry-run
  passed with workspace-local cache and child-only TLS validation enabled;
  no global settings or dependency installs were changed.
  The package's own dry-run baseline is 1,092,517 tarball bytes / 5,149,036
  unpacked bytes / 1,401 files. This excludes model weights and installed engine
  dependencies and is not an optimization claim. Earlier preliminary and
  build-bound controls remain preserved; no failed evidence was overwritten.
- P03/P09 raw/statistics attribution (2026-10-01): instance-local raw capture,
  per-mel statistics, all failing coordinates and four-term arithmetic residuals
  are now tested (28 focused probe tests PASS). Fresh weight-free receipt:
  `library-feature-normalization-decomposition-jfk-2026-10-01.json` under
  `tools/data/results/orukeet/`, **DIAGNOSTIC_NOT_PORT_ACCEPTANCE**. Production
  remains 313 mismatches; counterfactual official-statistics normalization gives 314. Full-FFT controls remain 129/120 versus counterfactual 131/121; official-
  complex controls remain zero. Closure is below `8.51e-16`; reference/candidate
  reconstruction passes. Production processor and tolerances are unchanged.
  Follow-up weighted-power attribution is complete in
  `library-feature-weighted-power-jfk-2026-10-01.json`: all four controls
  reconstruct every valid raw-mel value exactly; power/log deltas dominate
  reference projection rounding at failed coordinates in low and high bands.
  Full-FFT controls remain 129/120; production remains 313. Next: an independent
  direct DFT on identical canonical windowed bytes at these coordinates,
  compared with official float32 and both JS FFT controls. No production FFT
  replacement is justified yet. Existing P02-P08 gates remain open.
- G2/G3/P09 production lifecycle slice (2026-10-01): NeMo TDT and Whisper
  now reject creation after model disposal and disposal-raced initialization
  with `model-disposed`; session/model repeated disposal awaits the ongoing
  cleanup. Four new controlled async integration cases failed before the fix.
  The shared internal ownership helper remains model-neutral; executors and
  decode rules stay family-owned. Focused checks: 45 passed; separate strict
  test typing and core typechecking passed. Fresh full suite: 1,217 passed /
  27 skipped / 0 failed in
  `tools/data/results/harness/owned-session-race-full-suite-2026-10-01.json`.
  Browser import guard and package dry-run smoke passed. Rebuilt real Node
  26.2.0 and Bun 1.3.14 native CPU JFK runs match all 38 tokens, frames,
  durations and text with awaited cleanup PASS. Receipts under
  `tools/data/results/orukeet/`:
  `library-node-native-fp32-jfk-owned-session-race-2026-10-01.json` and
  `library-bun-native-fp32-jfk-owned-session-race-2026-10-01.json`.
  Both remain **PARTIAL_PASS_OUTPUT_ONLY**; the legacy JFK oracle lacks effective
  PCM capture. Mocked lifecycle coverage does not prove actual failed partial
  ORT acquisition, all-family disposal or browser graph release. See
  `owned-session-race-handoff-2026-10-01.json` for source/compiled receipt bindings.
- P03/P09 official-complex injection control (2026-10-01): canonical official
  STFT real/imag components supplied to either isolated JS power formula yield
  normalized features with zero mismatches, max error `3.5762786865234375e-6`,
  at unchanged `atol=1e-5, rtol=1e-4`. Power allclose also passes, but exact
  equality does not (153,688 / 1,862 values differ). Production features still
  FAIL at 313 mismatches; full-FFT controls remain 129/120. Receipt:
  `tools/data/results/orukeet/library-feature-official-complex-injection-jfk-2026-10-01.json`,
  **DIAGNOSTIC_NOT_PORT_ACCEPTANCE**. Production processor hashes are unchanged.
  Next: capture raw mel arrays and mismatch coordinates, compare per-mel mean/
  sample-std and counterfactual normalization with official statistics. This
  separates pointwise raw perturbation from propagated statistics error;
  low-frequency FFT cancellation remains a hypothesis, not an established cause.
- P03/P09 exact-input complex FFT control (2026-10-01): the diagnostic captures
  canonical f32le bytes before each full FFT. A new bit-identity regression
  exposed JSON numeric capture losing `-0`; the canonical-byte fix passes.
  All 563,712 input values now match the reconstructed official windowed frames
  bit-for-bit in both controls, SHA-256
  `e6a1dc984525ae456e0adc92085442b147ff62427581b30012ade46f4f8aaf30`.
  Official framed `torch.fft.rfft` matches NeMo's actual complex STFT exactly.
  JS real/imag components pass existing allclose tolerances against Torch on
  those same bytes and against official STFT, but normalized features still
  FAIL at 129/120 mismatches. Receipt:
  `tools/data/results/orukeet/library-feature-complex-fft-f32le-jfk-2026-10-01.json`.
  Fourteen diagnostic tests pass; production source/compiled processor hashes
  are unchanged. Preserve the earlier JSON-only
  `library-feature-complex-fft-boundary-jfk-2026-10-01.json` as evidence of a
  diagnostic serialization issue, not bit-identical input acceptance. Next:
  attribute residual differences at failing mel-bin/frame coordinates using
  the completed official complex-output injection control recorded above.
- P09 generic session ownership (2026-10-01):
  `tests/support/fixture-sessions.ts` provides test-only acquired-session rollback
  and awaited/idempotent teardown without model branches. Nine simulated unit
  cases cover creation/post-validation failure, all-release attempts, duplicate
  registration and retained setup/cleanup errors. Two actual Whisper harness
  setup regressions failed before wiring and now pass: partial resources release
  and a rejected cached setup can retry. Model assertions/thresholds are unchanged.
  Strict changed-test compilation passes; focused tests 31 passed / 3 skipped;
  full suite 1,203 passed / 27 skipped / 0 failed. Receipt:
  `tools/data/results/harness/fixture-session-cleanup-full-suite-2026-10-01.json`.
  Real Whisper inference was not requested; these are harness/lifecycle controls.
- P03/P09 discriminating FFT control (2026-10-01): the weight-free feature
  diagnostic now compares full 512-point FFT with Float32 and Float64 scratch
  on identical float32 windowed input, using the existing compiled FFT helper.
  Independent direct-DFT, complex/power rounding, shape/masking, source/cache
  immutability and contract-guard tests pass (10 tests). The real JFK control
  reproduces the official processor exactly and keeps the existing tolerances.
  Full-FFT Float32 features have 129 mismatches (max `3.0434131622314453e-4`);
  Float64 has 120 (max `2.5162100791931152e-4`), like the prior final-complex
  rounding control. Both spectra pass their own allclose checks, but features
  FAIL. This rejects a full-FFT replacement as a demonstrated fix; production
  processor arithmetic is unchanged. Receipt:
  `tools/data/results/orukeet/library-feature-full-fft-control-jfk-2026-10-01.json`,
  status **DIAGNOSTIC_NOT_PORT_ACCEPTANCE**. Next: inspect the official complex
  FFT boundary on the same framed input before another production hypothesis.
- P09 cross-family harness correctness (2026-10-01): the Whisper reproducibility
  harness's strict test-file compilation now passes after actual tensor-metadata
  narrowing and tensor feed typing. Five runner regressions verify disabled
  inference is SKIP and requested missing/empty/malformed reference or missing
  mel/audio is FAIL before native sessions. The first three failed before gate
  migration. Existing 100% feature-input and >=80% WAV-input acceptance remains
  unchanged. Focused run: 20 passed / 3 skipped. Full suite: 1,192 passed / 27
  skipped / 0 failed; two prior false PASS cases are now genuine SKIP. Receipt:
  `tools/data/results/harness/fixture-gates-full-suite-2026-10-01-repro.json`.
  This is test/harness acceptance, not fresh artifact-enabled Whisper inference.
  Partial-session-setup teardown and real-artifact replay remain separate work.
- P02/P03/P09 effective-input control (2026-10-01): official NeMo capture adds
  waveform/valid-length stages, producing 1,122 speech/silence stages with all
  six unchanged-output controls. Captured PCM is 649,600 mono float32 samples at
  16 kHz, no right padding, SHA-256
  `8f4293517daf07f433b307a4a971f7c17dc9ed348b5f72999945b217331ec3df`.
  `official-speech-silence-waveform-summary-2026-10-01.json` records the contract.
  Public-library native Node and Bun runs with `--input-mode oracle-pcm` both
  match all 69 tokens, frames, TDT durations and text; cleanup passes. Receipts:
  `library-node-native-fp32-speech-silence-same-pcm-2026-10-01.json` and
  `library-bun-native-fp32-speech-silence-same-pcm-2026-10-01.json`, both
  **PARTIAL_PASS_OUTPUT_ONLY** under `tools/data/results/orukeet/`. Keep the old
  linear-resampling 69-versus-76 failure; same raw audio identity did not prove
  same model-input PCM. Source decoding/resampling acceptance, numerical feature
  parity, pure silence/long audio, browser and fp16 promotion remain open.
- P03/P09 continuation (2026-10-01): initializer-only export auditing missed
  the encoder's 40,955,904-byte external `Constant` attribute. The true total
  is 295 tensor files / 2,475,088,896 bytes. A failing attribute regression now
  passes; model-neutral `tools/onnx_artifacts.py` traverses initializer,
  attribute, nested/sparse/function tensors. `tools/repack_onnx_external_data.py`
  safely streams ranges in 8 MiB blocks into a fresh candidate, preserves
  source files/inline weights, verifies every source/destination range hash,
  and checks the resulting graph. Packed assets:
  `models/orukeet/onnx/fp32-packed-2026-10-01/`. Independent metadata checking
  confirms the graphs differ only in external-data references.
- P03 partial numerical PASS (2026-10-01): native Python ORT 1.23.2 CPU replay
  on JFK and speech/silence passed all 916 numerical comparisons at
  `atol=1e-5, rtol=1e-4` plus exact token/duration argmax for 228 joint calls.
  Encoder max/mean errors: JFK `1.5721e-6 / 2.2502e-8`; speech/silence
  `1.1921e-6 / 4.0792e-8`. Max normalized-score error `5.3406e-5`; max
  predictor-state error `2.4319e-5` (all explicit allclose checks pass).
  `tools/nemo-tdt-onnx-export/validate_stage_replay.py` replays the real saved
  labels/input states/raw frames; no search loop. Normalize raw export scores
  across all 8,198 channels to the original CPU boundary. Evidence:
  `tools/data/results/orukeet/native-fp32-stage-replay-2026-10-01.json`.
  Predictor raw output is not exposed; this is not accumulated decoding,
  public-library text parity, Node/Bun native support or browser acceptance.
  Pure silence/long audio and P04-P08 remain open. Tool checks: 8 exporter,
  9 generic artifact, 5 numerical metric and 7 replay-contract tests passed.
  Next: complete the oracle matrix and test explicit-family public inference
  with packed fp32 assets and JS features before fp16 or promotion.
- P02/P03 partial (2026-10-01): official speech/silence capture produced 1,120
  stages (71 predictor / 182 joint calls), with exact baseline/captured/after
  controls, audio/source hashes and tokens/durations/frames. Pure silence and
  long-audio remain open. The new family-owned `tools/nemo-tdt-onnx-export/`
  exporter restored the original checkpoint, used NeMo's actual no-extra-SOS /
  raw-joint export preparation and produced two fp32 opset-17 graphs. Names,
  dtypes/ranks, ONNX checker, external-data ranges and hashes passed audit.
  Initial status: **EXPORTED_UNVALIDATED**. Encoder graph: 2,138,699 bytes;
  the initial audit's 294-file external count was incomplete (corrected above).
  Combined decoder/joint: 72,520,893 bytes. Original export remains the control
  under `models/orukeet/onnx/fp32-nemo-2026-10-01/`; repacking and partial
  native numerical replay are recorded above. Do not promote or run fp16 yet.
  Small tracked receipt: `tools/data/results/orukeet/fp32-export-2026-10-01.json`.
  Seven initial exporter safety tests passed; the new attribute regression
  brings this to eight. Safety tests are not model parity evidence.
- P09 (2026-10-01): the NeMo inspector no longer requires a processor ONNX
  when `preprocessorBackend=js`; ONNX-mode prerequisites are unchanged. The
  regression failed before the fix, then all three artifact-contract tests and
  their strict TypeScript check passed. CLI import no longer starts inference,
  allowing the existing resolver to be tested without a duplicate pipeline.
  Direct-family config selection, portable defaults, general native ORT and
  Bun model acceptance are still open; this small fix does not certify them.
- P09 (2026-10-01): shared, model-neutral Vitest prerequisites now serve
  Whisper merged/split-graph, timestamped decoder, mel reference and GigaAM CTC
  tests. Disabled cases are real SKIP; requested missing/empty files or malformed
  references FAIL. Machine-specific roots are configurable/local. Eight generic
  unit checks and six real-runner regression cases pass; one deliberate skip
  checks Vitest reporting. Model graph contracts and parity thresholds stay in
  their own tests. Whisper session teardown is registered immediately, including
  partial setup failures. Actual ONNX teardown still needs artifact-enabled runs.
  Standing boundaries and ordered remaining work are recorded in
  `docs/MODEL_HARNESS_CONTRACT.md`; this is not an all-harness rewrite.
  Receipt: `tools/data/results/harness/fixture-gates-2026-10-01.json`. Final
  full suite: 1,106 passed / 25 skipped; changed-test strict typing plus
  core/examples typechecks passed. Skips and unmigrated historical early-return
  cases are not numerical inference evidence.
- P01: `source-provenance-2026-09-30.json` records official revision
  `b59c13a733fce5cf193230d0fa317ee7f145a108`, original 2,509,342,720-byte
  `.nemo` and verified SHA-256. `source-contract-2026-09-30.json` checks the
  original checkpoint configuration and all 8,193 indexed tokens against v3:
  zero token differences. This proves contract compatibility, not inference parity.
- P02 partial: `official-jfk-2026-09-30.json` contains the official NeMo CPU
  transcript, 38 token IDs and frame timestamps. The compressed
  `official-jfk-stages-2026-09-30.json.gz` and its `official-jfk-stage-summary`
  contain 464 numerical stages, 40 predictor calls, 46 joint calls, token
  durations, audio/source hashes and exact baseline/captured/after token-text
  controls. The official greedy-batch decoder is retained, not reimplemented.
  Pure silence is now captured separately (see the checkpoint below). Long-audio
  fixtures remain to be captured; the speech/silence continuation is recorded above.
- P09: the stage comparator now rejects empty captures and supports required
  `stage_order`; earliest divergence follows processing order rather than
  alphabetical stage names. Compressed JSON is supported; duration/frame
  divergence and malformed metadata now fail too. Empty stages/tensors and
  missing audio identity cannot PASS. Eight regression cases were
  added; all 40 tests in the comparator, TDT grid/executor and windowing subset
  passed, plus 9 Python instrumentation/simulation tests (no pytest install needed).
- P09: the official NeMo inspector uses a local Numba cache, explicitly reloads
  Numba's already-imported configuration, limits CPU threads, and emits progress
  plus timeout stack diagnostics. This resolved the Windows Librosa/JIT cache
  stall without changing the installed environment or model.
- P06 control only: `parakeet-v3-browser-control-2026-09-30.json` records the
  existing real-browser v3 fp16 WebGPU encoder/WASM decoder loop: one warmup,
  two exact-transcript passes (611.795 / 617.830 ms for 18.714 s audio), actual
  NVIDIA adapter and clean teardown. This is **stock v3**, not an Orukeet result;
  JS memory samples are not VRAM measurements.
- P09: the full-suite GigaAM stage-localization test no longer writes candidate
  features into the external source-model directory. It writes its diagnostic
  `.npy` under `tools/data/results/gigaam/`; numerical acceptance is unchanged.
- Final validation: full Vitest reported 1,098 passed / 18 skipped; 9 Python
  simulations, core/examples typechecks and `git diff --check` passed. Some
  opt-in real-model/reference tests remain unexecuted without fixtures, so this
  is not all-model parity certification. Small machine-readable handoff:
  `tools/data/results/orukeet/harness-validation-2026-10-01.json`.

### Frame arithmetic and pure-silence checkpoint (2026-10-01)

The cumulative JFK diagnostic preserves original packed power expressions:
production / f32 preemphasis / official window / rounded window products fail
at 313 / 200 / 139 / 121 normalized-feature coordinates respectively. The last
control has exactly the official FFT input values; this is still not feature
acceptance. These are temporary diagnostic modules, not production changes.

An exact-zero 1-second, mono 16 kHz PCM16 fixture now has an original-weight
official NeMo capture: empty text/tokens, 60 stages and all six instrumentation
controls PASS. Own fp32 native ORT recorded-boundary replay passes 50 numerical
comparisons and 12 token/duration decisions. This is PARTIAL PASS, not an
end-to-end library, WASM or Orukeet browser certification.

The same silence feature diagnostic exposes a separate earlier-boundary issue:
frames and raw mel match exactly, but all 12,800 valid normalized values fail
(`atol=1e-5`, `rtol=1e-4`, max error `0.1600533276796341`). Official float32
masked mean/std reductions differ from the JS double reductions even for a
constant row. Next inspect and test reference-equivalent reduction semantics
over varying valid and physical lengths; do not hardcode a silence offset or
relax acceptance. Re-test speech and silence before production edits.

Lossless gzip reports and opt-in `--summary-only` keep full numerical evidence
while reducing output overhead. 53 focused probe/output tests and eight output
safety tests pass; core typecheck passes. Full suite was not rerun for this
checkpoint. Receipt: `tools/data/results/orukeet/frame-arithmetic-and-silence-handoff-2026-10-01.json`.

### Contiguous cascade normalization diagnostic (2026-10-01)

The generic `Reduce.h` four-by-sixteen hypothesis fails the key 101/100 silence
case. The actual floating sum algorithm in upstream PyTorch v2.6.0
`SumKernel.cpp` uses cascade partials and accumulates scalar tails before vector
lanes. An independent width-8 model matches all 20 constant masked-prefix sum
controls; width 16 matches 12. CPU AVX512 capability alone does not prove the
compiled sum dispatch. Width is an explicit diagnostic parameter, not a
model-ID branch or automatic host selection.

On identical official raw mel, width-8 normalization has zero silence error,
including exact mean/std, and JFK max error `2.384185791015625e-7` with zero
failed feature coordinates at unchanged `1e-5/1e-4` tolerances. JFK mean matches
exactly; three std values differ at exact comparison (max `2.384185791015625e-7`).
Width 16 fails all 12,800 valid silence coordinates. Production feature failures
remain unchanged: these are Python numeric diagnostics, not a JS fix or accepted
port. Five new weight-free reduction tests, 53 existing probe tests and eight
output safety tests pass. Full suite/native/browser acceptance was not rerun.
Small receipt: `normalization-cascade-handoff-2026-10-01.json`.

### Temporary JS normalization and raw-boundary discrimination (2026-10-01)

The diagnostic-only `normalization-reduction.mjs` has five small Node tests;
Python has six reduction tests (including byte-exact cross-language feature,
mean and std across 58 layouts) plus 56 probe tests. Eight output safety tests
pass. Neither the public package nor shared processor changed.

On nine named real raw-mel boundaries, JS/Python features/mean/std match exactly
for JFK and pure silence. With width 8, all silence normalized outputs match the
official capture exactly. JFK official-raw normalization max error is `2.38e-7`;
official-complex library/rounded-power controls pass at max `2.74e-6` / `2.98e-6`.
However production raw still fails at 315 coordinates; aligned window-products,
complex-power, magnitude round-trip, full-f32/full-f64 FFT controls fail at
122 / 121 / 122 / 129 / 121. Thus fixing normalization alone is insufficient.

Raw-mel comparisons must use equivalent valid prefixes: official feature
extraction masks its invalid physical tail, whereas captured JS raw buffers
remain unmasked. Initial full-buffer raw comparisons produced a misleading
128-coordinate failure and are superseded by fresh `*-prefix` receipts; no
earlier evidence is overwritten. A regression now proves invalid tails are
excluded without hiding valid-prefix errors. Raw input hashes bind complete
buffers; prefix metrics explicitly report both exact and original tolerances.
All corrected JFK valid raw prefixes pass the original tolerance, but exact
differences remain (production max `3.55e-4`; aligned complex-power `2.71e-4`;
official-complex controls `1.91e-6`). Normalization amplifies some small raw
differences beyond normalized-feature acceptance; raw tolerant PASS alone must
not promote this port. Silence raw valid prefixes are exact on every control.
The overall diagnostic still exits 1. Browser/WASM and complete model acceptance
remain open. Receipt: `js-normalization-cascade-handoff-2026-10-01.json`.

## Upstream recipes to inspect during export

### Observed cascade statistics attribution (2026-10-01)

The new separate cascade report injects the actual JS candidate mean/std into
the existing diagnostic; its legacy scalar-f64 mode remains available. For
JFK aligned complex-power, every one of 121 failed feature coordinates is
dominated by pointwise raw-mel error, not mean/std or final f32 arithmetic.
Maximum pointwise contribution is `2.521e-4`; mean/std contributions are at
most `7.984e-7` / `4.043e-6`. Official-statistics counterfactual still fails at
121 coordinates. Closure is below `9e-16` across all ten JFK controls; all ten
silence controls match with zero closure. This does not change production FAIL.

Next inspect weighted power/log terms at those exact coordinates, beginning
with mel 0/frame 346 (raw delta `2.46048e-4`), then high mel edge bands. Do not
attempt another normalizer or precision-only FFT patch on this evidence.
Receipt: `cascade-statistics-decomposition-handoff-2026-10-01.json`.
The parallel compact-install work item is tracked in [PRODUCT_TASKS.md](./PRODUCT_TASKS.md).

### Packed f32 FFT precision hypothesis — rejected (2026-10-01)

The diagnostic preserves the packed 256-point algorithm, exact official f32
windowed input bytes and aligned complex-power/mel arithmetic. It rounds
twiddles, butterfly products/stores and real-spectrum recombination, without
touching production. The candidate's JFK normalized-feature failures increase
from 121 to 215 at the original tolerances. Raw/power/complex tolerant PASS
does not outweigh this feature FAIL. Pure silence matches the temporary
normalization control; unchanged production still fails 12,800 coordinates.
Do not promote this candidate or infer that f32 alone reproduces MKL.

Continue with a cascade-aware normalization decomposition at the actual failed
coordinates: separate pointwise raw differences from raw-dependent mean/std
differences. The old decomposition uses double JS statistics and cannot isolate
those terms for the new cascade. Keep the official output oracle and all gates.
Small receipt: `packed-f32-stage-handoff-2026-10-01.json`.

- [Orukeet sherpa export](https://github.com/Oruk-AI/orukeet/blob/main/export/onnx/README.md):
  rebuildable source-to-ONNX graphs, boundary metadata and validation receipts.
  Study its fp32 path for our own encoder + combined decoder/joint export.
  Its INT8 CPU deployment/operator decisions do not establish WebGPU support.
- [Depthwise optimization evidence](https://github.com/Oruk-AI/orukeet/blob/main/evidence/speed20260910/README.md):
  useful isolated-operator/dynamic-shape/batch parity methodology; measure our
  fp32/fp16 graph variants on our browser before adopting any transformation.
- [transcribe.cpp recipe](https://huggingface.co/oruk/orukeet/blob/main/transcribe-cpp/README.md):
  a distinct GGUF layout, conversion audit and silence/cancellation/reuse checks.
  Its quantized native outputs are secondary controls, not the original-weight
  fp32 oracle and not a browser acceptance result.

### Harness work accompanies every task

Reuse the existing Whisper and Parakeet test machinery. Extend data-driven
candidate/artifact/oracle selection instead of cloning an inference pipeline.
Run cheap unit/simulation checks before native, WASM and real-browser rungs.
Tests with mocked sessions verify scheduling, shapes, cache/grid behavior,
window assembly and disposal only; they cannot certify numerical model parity
or WebGPU support. Add regression cases for every discovered tooling/runtime
failure and preserve machine-readable evidence with explicit timeouts, process
cleanup, backend identity, fixture hashes and first failed stage.

The acceptance criteria from `docs/AGENT_SAFETY_RULES.md` apply to every task.
Record PASS / FAIL / BLOCKED / UNTESTED per parity rung; task completion does
not imply full-model promotion.

### Recorded LSTM tap localization (2026-10-02)

The reusable `tools/onnx_graph_taps.py` appends inferred intermediate outputs
without changing any original graph fields, rejects external weights, pins
the source hash and refuses existing destinations. Six tests include real CPU
ORT output equality. This diagnostic transformation is not a production export.

On recorded bindings 139 and 173, twice each, native CPU ORT and real browser
WASM/WebGPU tap runs preserve all four original outputs exactly against their
untapped controls. WASM passes all eight native tap comparisons. WebGPU fails
only the second LSTM cell output at local coordinates 462/497 (the original
concatenated cell coordinates 1102/1137). Embedding transpose and layer-2
initial hidden/cell match native exactly; layer-1 outputs and layer-2 X pass
the unchanged tolerances but are not exact. Thus layer-2 arithmetic versus
amplified upstream error remains unresolved. Actual per-node provider is not
proven by the 29 observed compute dispatches.

Next use matched-input layer-2 controls to separate those hypotheses; do not
change tolerances or promote the port. The browser experiment was an inline
probe reusing the existing session bridge, not a new persistent CLI. Raw local
receipts retain all tap values; the small tracked evidence pointer is
`browser-lstm-taps-handoff-2026-10-02.json`. Production feature and full WebGPU
head failures remain open independently.

### Matched-input second-LSTM control (2026-10-02)

A checker-valid extracted diagnostic preserves the second LSTM node and its
three embedded weight tensors byte-for-byte. CPU on native captured inputs
exactly reproduces full native taps. Six input cases (two bindings, each with
native/WASM/WebGPU upstream X and initial h/c) run twice per browser backend.
All 24 browser calls pass same-input CPU comparisons at unchanged tolerances;
each backend's own captured-input case reproduces full-graph hidden/cell outputs
exactly, and repeat outputs are exact. Initial h/c were already bit-identical,
so the input-origin difference here is X from the first LSTM.

CPU on the recorded WebGPU X reproduces the cell failures. At cell coordinates
462/497 its upstream-induced deltas are `1.14143e-5` / `1.37575e-5`, versus
remaining full-WebGPU deltas of `6.25849e-7` / `-3.35276e-8`. This supports
upstream-error amplification, not a second-LSTM-only arithmetic fix. Investigate
first-LSTM output arithmetic next. The isolated WebGPU session submits five
compute dispatches/call; per-node provider still is not proven. This is selected
diagnostic evidence, not whole-model parity or performance acceptance.

Small receipt: `layer2-matched-input-handoff-2026-10-02.json`; raw local cases
and graph: `tools/data/results/orukeet/layer2-matched-input-2026-10-02/`.

### First-LSTM matched-input control and reusable extraction (2026-10-02)

On the same two recorded bindings, extracted first-LSTM native CPU outputs
match full native taps exactly. Both browser backends run each input twice;
standalone outputs reproduce their own full-browser taps exactly and repeats
are exact. Same-input CPU tolerance checks pass, but WebGPU hidden differences
are `7.21216e-6` / `6.79493e-6`, versus WASM `7.60891e-7` / `1.01328e-6`.
The small first-layer arithmetic difference is therefore independently
reproduced, and its downstream amplification was measured by the preceding
second-layer control. Matrix reduction versus nonlinear arithmetic is still
unproven; inspect the actual LSTM contract before any production graph change.

`tools/onnx_subgraph.py` now provides model-neutral named-boundary extraction:
pin the source with `--expected-sha256`, repeat `--input` / `--output`, and use
a fresh `--output-dir`. It rejects external weights and unknown/duplicate
boundaries, checks retained node/weight bytes and emits an explicitly unvalidated
diagnostic receipt. Six tests include a real matched-input CPU control and CLI
hash/overwrite failures. Its CLI reproduced the first-layer diagnostic graph
hash exactly. Shared tools: 35 PASS; family tools: 26 PASS. Browser probes remain
inline experiments using the existing session bridge, not a persistent browser
diagnostic CLI. Small receipt: `layer1-matched-input-handoff-2026-10-02.json`.

### Explicit-equation fp32 head candidate (2026-10-02)

The one-step first-LSTM equation diagnostic matches fused CPU hidden/cell
outputs within `1.94e-7` / `3.58e-7`. Explicit WebGPU matrix taps differ from
CPU by up to `3.05e-5`, while fixed-preactivation hidden differences are at
most `2.38e-7`. Both experiments repeat exactly with clean resource controls.
The expansion is not bit-identical to fused WebGPU, so it is not proof of
the fused kernel's internal cause or provider attribution.

A separate own fp32 head candidate replaces both default forward LSTMs with
explicit IOFC equations and fixes only predictor target sequence width to one,
matching current executor calls. Batch/encoder-frame-width contracts stay
unchanged. Source non-LSTM node bytes and transposed/split weight reversibility
were checked; no quantization or production source change was made.

On all 182 official speech/silence recorded bindings, native CPU passes. Both
WASM and WebGPU run the matrix twice (364 calls each) and pass the original
`atol=1e-5, rtol=1e-4`, exact length and token/duration argmax gates. Raw-array
re-evaluation with the existing family evaluator passes, repeats are exact,
and the fixture hash matches the original failing matrix. This improves the
candidate head evidence but does not change the original production graph's
FAIL status. The first generation had duplicate node names and was rejected
by ORT before inference; its invalid artifact was preserved and a fresh
candidate was rebuilt from the original with unique-name validation.

Next persist/test the opt-in family transformation, validate JFK/pure-silence
matrices, candidate-fed recurrent decoding and encoder-grid widths, then
measure matched performance/memory. Forty-seven WebGPU dispatches/call are
observed, not a speedup claim. Production features, encoder and live public
API acceptance remain open. Large candidate/traces stay local. Small receipt:
`linear-gate-head-candidate-handoff-2026-10-02.json`.

### Persistent single-step candidate and full head matrix (2026-10-02)

`tools/nemo-tdt-onnx-export/expand_single_step_lstm.py` now hash-pins source
graph/config buffers, derives dimensions from the actual family contract,
requires fresh output and emits an explicitly unvalidated transformation
receipt. It rejects unsupported LSTM options/inputs, external weights, name
collisions, nonshared-contract shapes and mismatched config. It proves the
one-token target constraint propagates to both LSTM sequence axes, preserves
retained fields and byte-reversible weights, and does not touch default export
or runtime/preset behavior. Nine tests cover CPU math over two small widths,
candidate-fed synthetic recurrence, batch/grid controls, hash/overwrite and
forged candidate receipts. Family/shared Python suites each have 35 PASS.

`browser_head_replay.py --candidate-dir` retains all original export/repack/
config audits and reproduces the transformation in memory before accepting
the candidate graph hash. It reuses the existing library session bridge and
evaluator, not a new decoder. The persisted candidate has graph SHA
`fbdf8b7c141d0a999aa71c4edb0ed9cfac1aa762d68b8d1bf7b006acb71f8718`.

Actual native recorded inputs pass 46 JFK + 12 pure-silence + 182 speech/
silence calls. Each browser backend passes two repeats of all 240 calls:
480 WASM and 480 WebGPU, with original `atol=1e-5, rtol=1e-4`, exact length
and token/duration decisions. Independent raw-array re-evaluation passes;
fixtures hash-match the original matrices, all repeats are bit-exact, no
browser/cleanup errors exist and graph HTTP delivery hashes match. WebGPU
records 47 dispatches/call, not proof of every node's provider or a speedup.

Native source-graph controls also pass encoder-grid widths 1/2/8/32 for JFK
and speech/silence, and 1/2/8/13 (available length) for silence. These are
source-graph controls, not a new official multi-frame oracle or browser-grid
acceptance. Synthetic candidate-fed recurrence is not actual model token/text
acceptance. Production features remain FAIL; full encoder/live library and
long-reference gates remain open. Do not promote or quantize the candidate.
No sibling/runtime API changes occurred. Large graphs/traces stay local;
small receipt: `persistent-single-step-head-handoff-2026-10-02.json`.

### Reference FFT layout control (2026-10-02)

A bounded JFK CPU control reproduces the current NeMo STFT contract with
float32 preemphasis, symmetric Hann(400), centered constant padding and
FFT512/hop160; it does not rerun original model weights or replace the frozen
oracle. Torch 2.6.0+cu118 reports Intel oneAPI MKL 2025.0.1 and AVX512 support.
On identical windowed frame values, contiguous rFFT matches STFT bit-exactly
over all 1,101 frames. A sample-stride-2 view changes 487,607 real/imag component
values, max absolute difference `1.90735e-6`. At frames 346/551, the stride-2
views differ at 469/457 components (max `1.90735e-6` / `4.76837e-7`); their
contiguous clones match STFT exactly, as do contiguous single-frame calls.

This demonstrates layout-sensitive reference FFT arithmetic, not that layout
causes our JS feature failure, nor which MKL plan/kernel executes. Source/capture
tensor clones currently erase stride information; retain actual input layout
provenance in a future reproducible probe before choosing an FFT implementation
change. Production arithmetic, oracle and tolerances are untouched. Existing
feature FAIL and candidate/live encoder/token/text gates remain open.

### Actual-model STFT binding (2026-10-02)

The opt-in NeMo capture now retains pre-snapshot source layout and records
the actual instance-local featurizer STFT input/window/real/imag boundaries.
Original-weight JFK capture has all six before/captured/after controls true;
464 common stages, tokens and text match the older frozen capture exactly.
The method returns the original output object and restores hooks on failure.

`compare_js_feature_boundaries.py --require-actual-stft` now requires exact
float32 bits for all four captured boundaries and matching FFT parameters.
Partial, malformed or contradictory required captures fail; legacy captures
without this evidence remain explicitly UNTESTED. Source strides are provenance,
not a physical layout to apply to flattened JSON data. Production is unchanged.
Eight new weight-free gate tests, 79 existing boundary tests, six normalization
tests and 20 capture tests pass.

The real JFK run passes all four reproduction boundaries at zero max/mean
error. Production features still FAIL with 313 mismatches at `1e-5/1e-4`, max
absolute error `0.00034236907958984375`; the diagnostic exits 1 as expected.
This binds the diagnostic spectrum to the original model, not library parity,
an FFT kernel identity, or completion. The large local receipt is
`actual-stft-bound-feature-jfk-final-2026-10-02.json.gz`, SHA-256
`6ecf1fab598fd0fc5f5058d05f775d1e90a908a4798a9543d0ecd873c53bf4dc`.
Capture SHA-256 is `4328840597d82b6942e640c2d9a4f810ca6c589588fb833009d02d8b14178508`;
diagnostic source SHA-256 is `67c4e19288c1f7c8b9e3b6ed4caf98122bc9c5bca8972c4511230e683a23ad78`.
After that full run, review added an exact-true capture-marker guard and made
the STFT status visible in summary-only output. Eight tests and a fresh direct
Torch replay from WAV/preemphasis/symmetric Hann pass the current guard with
all four boundaries bit-exact; the full receipt above identifies the earlier
source hash, not this later metadata-only tightening.
Next use this verified spectrum to discriminate JS FFT/mel rounding rather
than rerunning unbound reference reconstructions. Full fixture/live model gates
remain open; no preset promotion or precision change is justified.

### Full FFT complex-product rounding discriminator (2026-10-02)

A bounded CPU/Node control uses exact captured STFT input/window values and
float32 frame products over all 1,101 JFK frames. It runs the actual compiled
library's 512-point `fft()` with guarded temporary source transforms, not the
production 256-point packed/recombination path. Across 565,914 real/imag values,
f64-final-f32 has 489,809 bit mismatches and mean error `2.096496679017805e-8`;
separate-f32-products has 496,113 / `2.631569987115098e-8`. Rounding only the
right product, only the left product, or neither product before final f32
rounding gives 494,427 / `2.548069699147843e-8`, 494,517 /
`2.546677179117999e-8`, and 492,784 / `2.457823876104226e-8`, respectively.
Max component error is `1.9073486328125e-6` except rounded-left-product control
at `2.86102294921875e-6`. None closes the reference boundary or improves mean
error over f64. Binary64 product-sum controls are not hardware-FMA proof.

Reject these full-FFT variants as production fixes. This does not rule out
different packed/radix/recombination arithmetic and does not measure feature,
token/text, browser GPU or speed acceptance. Production stays unchanged; do
not repeat the already completed selected-bin DFT to rediscover this seam.

### Long-sequence original reference and native replay (2026-10-02)

Added a separately identified synthetic stress fixture: original mono 16 kHz
PCM16 LibriVox payload concatenated eight times without resampling/gain/gaps,
2,395,400 samples / 149.7125 seconds. It is not natural long-speech quality gold.
The source remains immutable; WAV and effective float32 PCM hashes are recorded
and independently verified against the actual model-input boundary.

Original pinned Orukeet weights with official NeMo CPU yield 8,222 stages,
724 predictor calls, 806 joint calls and 722 tokens. All six before/captured/
after token/text/frame/duration controls pass. Actual STFT input/window/complex
components and source layouts are captured. The 257 MB compressed trace stays
local; do not commit the WAV, original weights or large stage artifacts.

Existing own fp32 graphs pass the unchanged native CPU recorded-boundary
validator: 3,226 numerical comparisons at `atol=1e-5, rtol=1e-4` and all 806
exact token/duration decisions. Encoder shape is `[1,1024,1872]`, max/mean
absolute error `4.410743713378906e-6` / `1.9678874604167998e-8`.
Seven weight-free replay-contract tests also pass. No runtime math or export
changes were made. Hashes, fixture recipe and result pointers are in
`long-repeat8-reference-handoff-2026-10-02.json`.

P02 now has a real original-engine synthetic long-sequence reference, not a
completed natural long-recording/quality/soak matrix. Long library Node/Bun,
browser WASM/WebGPU, candidate-fed recurrence and accumulated tokens/text remain
UNTESTED; production features remain FAIL. Next reuse this reference with the
existing head/browser/library adapters, preserving those independent gates.
The long replay also exposed repeated full encoder-list conversion inside
each joint binding; hoist only after a focused feed-identity/count regression,
and measure before claiming a speedup. Do not add another decoder.

### Replay encoder materialization reuse (2026-10-02)

The family-owned recorded-boundary validator now materializes the captured raw
encoder once after chronology validation and reuses it for comparison and
contiguous frame slices. Chronology retains its independent validation; no
decoder, score normalization, tolerance or runtime source changed.

A two-frame fake-session regression failed before the change (four full encoder
conversions instead of two), then passed with all head feed keys, values,
dtypes and contiguous frame slices preserved. All eight replay-contract tests
and Python compilation pass. This is synthetic harness evidence, not ASR parity.

Actual native CPU fp32 JFK replay also passes: 186 numerical comparisons and
46 exact decisions. Its entire case result equals the prior receipt exactly.
Receipt: `native-fp32-jfk-encoder-reuse-2026-10-02.json`, SHA-256
`7cc3b6cf37b7172251cf1d0123b014d6993fcce531e7358d9a133992ca0f2878`.
The modified tool's real 149.7125-second synthetic long-sequence replay also
passes all 3,226 numerical comparisons and 806 exact decisions. Its entire case
equals the pre-change receipt exactly. New local receipt:
`native-fp32-long-repeat8-encoder-reuse-2026-10-02.json`, SHA-256
`311c8b967c6eeb0b5e5342ba3c587af87b76640d623486962c5a3f90d38da309`.
Wall-time/RSS A/B measurements remain UNTESTED; conversion-count reduction is
not a measured end-to-end speedup.
Production feature and accumulated library/browser acceptance gates remain open.

Next earliest-divergence experiment: split the existing diagnostic-only packed
256-point f32 transform into butterfly-only, real-recombination-only, combined
single-round product sums and unchanged packed-f32 control. Use actual captured
STFT input/window/complex bytes and fixed downstream power/mel/normalization.
Guard source drift and cancellation/half-ULP cases first; retain original feature
tolerances. JavaScript binary64 products with one final f32 rounding are only
FMA-like controls, not hardware FMA evidence. Do not repeat full-512 FFT or DFT.

### Factorized packed-product rounding diagnostic (2026-10-02)

`packed-f32-source.mjs` now exposes independently guarded butterfly and real
recombination product modes (`separate` / `single-round`). No options preserves
the prior source transform. Six focused Node tests pass, including source-drift,
mode isolation, cancellation/ties-to-even and actual generated FFT versus an
independent negative-exponent DFT. Eleven combined normalization/transform tests
pass. A mistaken half-ULP test expectation was removed before correcting the
fixture; no runtime or oracle was changed to satisfy a test.

Weight-free spectrum-only run: use the actual captured pre-STFT input and Hann
window, zero-center pad, stride 160, f32 frame products, then the transformed
production 256-point complex FFT and real recombination source. All modes use
the same 1,101 x 512 canonical frame bytes, SHA-256
`e6a1dc984525ae456e0adc92085442b147ff62427581b30012ade46f4f8aaf30`.
Compare f32 real/imag components against the frozen actual model STFT, not DFT.

| Butterfly / recombination | Mean component abs error | Bit mismatches |
| --- | ---: | ---: |
| separate / separate | 2.6667971262710457e-8 | 494,612 |
| separate / single-round | 2.6484949320305885e-8 | 493,956 |
| single-round / separate | 2.522761948865524e-8 | 491,337 |
| single-round / single-round | 2.5031335696156946e-8 | 490,278 |

Neither branch reproduces actual complex bytes. Lower mean error alone is not
feature or ASR acceptance. Receipt `packed-product-spectrum-jfk-2026-10-02.json`
binds capture/source/transform/component hashes. This run does not re-execute
Torch or identify hardware FMA/MKL dispatch. An earlier full-feature attempt
was interrupted before writing a receipt; its results are UNTESTED, not PASS.
Next integrate these options into the existing feature probe and compare the
same downstream mel/log/width8 normalization; production features remain FAIL.

### Packed-product feature rejection and reproducible probe options (2026-10-02)

Completed the frozen-input feature matrix without full NeMo import: rebuild
the filterbank with the exact installed NeMo-source librosa Slaney recipe,
use captured waveform/window, and require every captured f32 FFT frame byte to
match the prior spectrum run. Diagnostic power, production mel accumulation
and width8 normalization are identical across branches. Against actual captured
features at unchanged `atol=1e-5, rtol=1e-4`, mismatch counts are:

| Butterfly / recombination | Feature mismatches | Max abs error |
| --- | ---: | ---: |
| separate / separate | 215 | 3.078877925872803e-4 |
| separate / single-round | 220 | 3.078877925872803e-4 |
| single-round / separate | 179 | 3.626048564910889e-4 |
| single-round / single-round | 187 | 3.626048564910889e-4 |

All FAIL. The best branch is still worse than the earlier 121-mismatch aligned
control and increases peak error. Reject these as production fixes; lower mean
spectrum error did not establish feature parity. No new original-engine run,
token/text, browser or performance acceptance is implied. Small receipt:
`packed-product-features-jfk-2026-10-02.json`, SHA-256
`c9ee5badbf22cc6e10baf65cde80b3eab59102a3afe2ed7ae9c98ad136cd2d5b`.

The existing full feature probe now accepts `--packed-butterfly-products` and
`--packed-recombination-products` (`separate` default / `single-round`) and
records both in its receipt. Only its named packed-f32 control changes; all
other branches and frame bytes are preserved. Two actual-probe regressions
failed before wiring and pass after; all 81 boundary tests and 11 Node tests
pass, with Python compilation and CLI help verified. Full original-engine CLI
execution with these flags remains UNTESTED. Do not repeat these modes seeking
another mean-error gain; next isolate algorithm/order rather than weaken gates.

### Independent PocketFFT rejection (2026-10-02)

Weight-free frozen-input control used SciPy 1.15.3 `scipy.fft.rfft`, workers=1,
on the same canonical 1,101 x 512 f32 frames (hash above). Repeated PocketFFT
output is byte exact. Torch 2.6.0+cu118 framed `rfft` re-execution reproduces
both captured actual STFT components byte exactly; no NeMo model was rerun.
Complex arrays must be populated through separate `.real` / `.imag` assignments:
the initial `real + 1j * imag` reconstruction erased some signed-zero bits and
invalidated that attempted control. Direct component checks resolved this;
neither the capture nor production code was changed.

Inject each spectrum through the existing canonical complex-byte decoder and
the same diagnostic rounded-power, mixed mel accumulation, f32 log guard and
width8 normalization. Frozen PCM and the installed NeMo-source Slaney
filterbank recipe are unchanged. The official-spectrum downstream control has
0 feature mismatches (max abs 1.430511474609375e-6), while PocketFFT has 177
(max abs 3.260374069213867e-4, mean abs 8.493264467229297e-7), at unchanged
`atol=1e-5, rtol=1e-4`. Its 565,914 spectrum components pass that numerical
tolerance, but 493,469 differ in bits; mean spectrum abs error is
2.522016293103963e-8. Spectrum allclose is therefore not feature acceptance.

Receipt: `pocketfft-frozen-jfk-2026-10-02.json`, SHA-256
`9002aa4976df4f7aed1ce64670d2df2074336b95f2c0877c1150881ddce8fcb4`.
This is a one-off frozen-input diagnostic, not a reusable export recipe,
production fix, original-model rerun, token/text or browser acceptance.
Reject PocketFFT as a parity solution; do not port its algorithm merely to
obtain a closer intermediate spectrum. The result supports investigating the
original FFT arithmetic/dispatch boundary, but does not identify an MKL plan
or instruction. Next distinguish a reproducible original FFT operation order
from unsupported backend-specific assumptions before changing library math.

### Conditional packed256 factorization feature PASS (2026-10-02)

Discriminating experiment: pack the exact original f32 frame's even samples
into complex real components and odd samples into imaginary components, execute
Torch/MKL complex256 `fft`, then independently reconstruct the real512 spectrum
using the library's even/odd factorization. The original real512 framed Torch
control remains byte exact against the actual STFT. No weights or model were
rerun. Three explicit recombination modes share the same complex256 output;
none identifies MKL's internal implementation.

| Recombination | Spectrum mean abs error | Feature mismatches |
| --- | ---: | ---: |
| f64 intermediates, final f32 | 4.414092749510645e-9 | 3 |
| f32, separately rounded products | 4.259244081299885e-9 | 1 |
| f32, single-round product sums | 3.6958557846953095e-9 | 0 |

All feature comparisons retain `atol=1e-5, rtol=1e-4`, diagnostic rounded
power, mixed mel accumulation, f32 log guard and width8 normalization. The
official-spectrum control has 0 mismatches. The single-round branch passes
all 140,928 features (max abs 5.340576171875e-5, mean abs
1.1230153506610368e-7); the relative tolerance remains relevant. Its spectrum
still differs in 304,569 component bits. Single-round means binary64 product
intermediates followed by f32 rounding, not hardware FMA emulation.

Receipts: `mkl-packed256-factorization-jfk-2026-10-02.json`, SHA-256
`04ca352a604fc0c18197f7d7ed3268bed64c26b354dbe7fc5c57aa6ba1c9ad8c`,
and `mkl-packed256-features-jfk-2026-10-02.json`, SHA-256
`9009394dad032491a6b3fef4527a675683ea3b3e5361770904858c3caf1dacf7`.
These are one-off diagnostic controls, not a deployable processor or reusable
export recipe. **CONDITIONAL PASS ONLY:** the complex256 FFT comes from Torch,
not JS/browser/WASM. Production feature parity, additional fixtures, live ASR,
token/text and promotion remain open. This supports isolating the packed256
FFT arithmetic/order rather than replacing preprocessing with ONNX or changing
tolerances. Next reproduce that complex256 boundary in portable arithmetic,
then re-run the same downstream control and fixture matrix before production
changes. Do not claim that MKL necessarily implements real512 this way.

### Portable radix4 DIT candidate rejected (2026-10-02)

Added a diagnostic-only `radix4-f32.mjs` under the existing reference tooling,
not `src/`: base4 digit reversal, negative-exponent radix4 DIT stages, explicit
f32 butterfly stores/twiddles, and separately rounded versus single-round
binary64 product sums. Five small tests pass: exact four-point signs/bin order,
impulse/output ownership, complex-tone direction, independent complex DFT at
4/16/64/256, and malformed input/options. DFT sanity uses its own stated error
bound; it does not replace model tolerances or certify MKL arithmetic.

Frozen JFK: replace only the diagnostic packed256 FFT with this candidate,
retain the prior single-round f32 real recombination and exact frame hash, then
inject its complex spectrum through the existing rounded-power/mixed-mel/f32-
guard/width8 path. The official-spectrum control still has 0 mismatches.
Separate products give 214 feature mismatches (max abs
3.4669041633605957e-4); single-round gives 174 (max abs
2.02178955078125e-4). Both FAIL unchanged `atol=1e-5, rtol=1e-4`, although
their spectrum components pass those tolerances. Receipt
`radix4-packed-features-jfk-2026-10-02.json`, SHA-256
`ff7a29cf41079ff343ef97cde4d40a704b8f2aa5efb470187fa025586cd492ba`,
binds actual capture, compiled processor, existing probe and candidate hashes.

Reject this basic radix4 DIT as a production solution. The earlier conditional
Torch-complex256 feature PASS remains valid, but neither ordinary radix2 nor
this radix4 order reproduces it portably. No MKL-plan identity, browser/model
acceptance or speedup is claimed. This run used a one-off guarded probe
composition; the standalone FFT and tests are reusable, not a complete saved
end-to-end replay CLI. Next investigate original complex256 sub-boundaries or
a genuinely discriminating operation-order control; do not keep sweeping
rounding modes based only on improved mean spectrum error.

### Radix4 diagnostic cardinal-root correction (2026-10-02)

Review exposed nonzero trig residue at exact quarter-turn roots: a size16
impulse at index2 produced real bin2 `6.123234262925839e-17` instead of zero.
A new regression failed before the fix, then passed for both product modes
and amplitudes 1 and f32(1e38). Cardinal roots now use exact sign/swap rotations;
other twiddles and arithmetic are unchanged. All six radix4 tests pass.
This is a diagnostic-only correction, not production feature parity.
The preceding JFK receipt remains bound to the old candidate source hash;
its numbers must not be attributed to the corrected module. A fresh full
feature replay is still required before judging the corrected candidate.

### Corrected radix4 replay remains rejected (2026-10-02)

Fresh frozen-JFK replay after the cardinal-root fix completed successfully.
The official-spectrum downstream control again has zero mismatches across
140,928 features. Separate and single-round products still give 214 and 174
mismatches respectively, with the same feature metrics as the prior source.
Both FAIL unchanged `atol=1e-5, rtol=1e-4`; the canonical windowed-frame hash
is unchanged. Thus cardinal trig residue was a real diagnostic defect but
does not explain this feature divergence. Do not promote the corrected FFT.

Receipt: `radix4-cardinal-packed-features-jfk-2026-10-02.json`, SHA-256
`31e9439014e3d821817639e63d3519090d6ef47c145f357128ff2566777330d4`;
candidate source SHA-256
`f742a645cf8eb4192683d01700f10815481b51391500eda7726133717c50c06c`.
The guarded one-off composition also records its own hash; this is not a
saved replay CLI, original-engine rerun, browser execution or ASR acceptance.
Two initial composition attempts failed before writing any receipt (newline
escaping, then incorrect option names); the actual source contract was
re-inspected before the successful run. No production files were changed.

The second speech/silence waveform capture has 649,600 samples and final
features `[1,128,4061]` with valid length 4060, but no captured STFT/window/
complex boundary. It cannot independently certify the original FFT order.
A bounded read-only Torch recreation did not finish and was terminated;
no reconstruction result is claimed. Next capture a real second STFT boundary
through the existing original-engine instrumentation before generalizing the
conditional Torch-complex256 result; avoid another unbounded reconstruction.

### Second actual STFT capture and conditional feature failure (2026-10-02)

Completed the existing original-engine CLI on the local pinned Orukeet `.nemo`
with `librivox-blankgaps-synthetic.wav`, CPU, four threads, and
`--capture-feature-stft`. The process exited 0 and saved 1,126 stages. All six
before/after instrumentation controls pass. PCM remains byte-identical to the
prior speech/silence capture (649,600 samples, hash `8f4293517daf07f4...`);
features have 4061 frames with valid length 4060. The actual STFT complex output
is CPU noncontiguous with stride `[1043677,1,257]`. This replaces the missing
second STFT boundary, not the old capture or its evidence.

Capture: `official-speech-silence-stft-stages-2026-10-02.json.gz` (43,509,936
bytes, kept out of Git), SHA-256
`07ae6f00dcab69eb3425566af7b5690a85dc2bee91e7510ddf27cca7b4370809`.
Its small provenance/result summary is
`official-speech-silence-stft-summary-2026-10-02.json`. The original weights
and audio hashes match the prior capture; no downloads or production edits.

Weight-free downstream control using those actual frames: Torch real512 is
byte-exact with the captured STFT. Official complex injection has zero feature
mismatches across 519,808 values, max abs `9.5367431640625e-7`. The prior
Torch-complex256/f32-single-round recombination recipe instead has **one
mismatch**, first flattened index 9149 (mel2/frame1027), max abs
`1.4750286936759949e-5`, mean abs `2.3716362878875184e-8`. It FAILS unchanged
`atol=1e-5, rtol=1e-4`. The conditional JFK PASS does not generalize to this
fixture. No portable FFT, processor, ASR or browser acceptance is claimed.

Receipt: `mkl-packed256-features-speech-silence-2026-10-02.json`, SHA-256
`bd0b00ea0ca50a454657874849c62c214a282c2db8ab4ddf58e4fe4d68d01abc`.
The first diagnostic attempt failed on a noncontiguous complex dtype view
before writing a receipt; the successful run compares canonical component
bytes without altering original values. Four-thread child-local Torch setup
keeps this recreation bounded; both successful processes are terminal.
Next isolate the single divergent feature's complex/power/mel/log/normalization
dependencies rather than promoting a one-fixture PASS or sweeping FFT modes.

### Single failure localized to bin2/frame1027 (2026-10-02)

Replay of the same actual second STFT reproduces the one feature failure.
Mel2 has exactly one nonzero filterbank bin, FFT bin2. Original-complex
downstream control versus Torch-packed256 candidate at frame1027:

- mean and std are identical: `-16.046541213989258`, `1.1639012098312378`;
- raw log mel: `-16.08138084411621` versus `-16.081398010253906`;
- power: `1.5336382830355433e-6` versus `1.5335759826484718e-6`;
- normalized feature: `-0.029933493584394455` versus `-0.029948243871331215`.

Replacing only candidate complex bin2 at frame1027 with the original captured
complex value removes all feature mismatches (0/519808), while full-prefix
normalization still uses valid4060 within physical4061. This is a causal oracle
injection diagnostic, never a production correction or accepted FFT. Raw mel
here is the injected-spectrum diagnostic output, not a captured NeMo raw stage.
Receipt `packed256-single-feature-attribution-2026-10-02.json`, SHA-256
`2a799a3a88db43c99fe0c06713f6f09129e4ca6895367674528107cb20d04bf5`.
The one-off terminal code copies the candidate spectrum and assigns only that
coordinate, but the receipt does not save composed probe bytes or explicit
outside-mask identity assertions. Independent verification from the receipt
alone is therefore limited; add those guards to the reusable replay tool.

Single-frame subboundary control reveals cancellation: real output combines
`er=-0.3727142810821533` with `tr=0.37148404121398926` to produce about
`-0.00123024`. The candidate real error is `2.60770320892334e-8`.
Using f64 recombination of the *same f32 complex256 outputs* does not repair
that real error (`2.7241185307502747e-8`); it is not enough to simply switch
the final recombination to double precision. Secondary f64/`math.fsum` DFT is
only a mathematical sanity control, not the original implementation oracle.
Receipt `packed256-bin2-subboundary-2026-10-02.json`, SHA-256
`88675e921a351b4284df99fd91b2251876565a4ae4e8b2807ffea2853269cfed`.
Both diagnostic processes exited 0; no original-model rerun, portable FFT,
production edit, browser acceptance or tolerance change is claimed.

Next: turn this frozen-stage replay into a bounded reusable tool with explicit
physical/valid length and source/fixture checks, then compare packed256 bins2
and254 before recombination on this cancellation frame. Do not add a special
case for the failing model/bin/frame or repeat broad FFT rounding sweeps.

### Replay mask guard foundation (2026-10-02)

Added `complex_intervention.py` beside the family diagnostics. It validates
nonempty finite batch-one complex64 spectra, matching shapes, explicit unique
bin/frame coordinates inside the valid prefix, exact oracle components inside
the mask and unchanged component bits everywhere outside it (including padding
and signed zero). Canonical hashes bind the mask and all three input spectra;
noncontiguous STFT layouts are supported without mutating inputs.

Seven weight-free tests pass, including negative controls for one-ULP selected
errors, out-of-mask/padding edits, signed zero, malformed coordinates, dtype,
shape and nonfinite input. This is only the tested guard foundation: it has not
yet been connected to the saved JFK/speech-silence replay or composed Node probe.
It does not retrospectively certify the older receipt, reproduce model
inference, fix the FFT, or close any production/browser parity gate.
Next wire this guard into the bounded replay CLI and save the exact composed
probe bytes with capture/source hashes before rerunning both frozen fixtures.

### Frozen packed256 replay CLI verified on both captures (2026-10-02)

`tools/model-debugging/reference/hf-parakeet-port/replay_packed_fft.py` now
connects the mask guard to the saved original-engine STFT and existing JS
diagnostic probe. It requires capture SHA-256 and an exact sample ID, validates
instrumentation/PCM/tensor contracts, recreates centered f32 frames, and rejects
a Torch real512 control unless its complex bytes reproduce the capture exactly.
It uses one fixed packed256/f32-single-round recipe, not a mode sweep or new
production FFT. Librosa regenerates the Slaney filterbank; its bytes are hashed.

Explicit physical/valid lengths are checked by both Python and Node. Optional
`--coordinate MEL FRAME` replaces only that mel row's active spectrum bins at
the requested valid frame, with exact selected/oracle and outside-mask guards.
Full-prefix width8 diagnostic normalization is retained. The exact composed
Node probe is saved separately, hashed in the small receipt, and the original
complex downstream control must pass before interpreting the candidate.
Output paths are distinct, protected and exclusively created; an existing
output is rejected before capture or heavy imports. `--timeout` bounds the
Node subprocess, not the entire Python invocation.

Invocation (existing Python environment, built `dist/audio/js-mel.js`):

```powershell
& tools/model-debugging/reference/nemotron-3.5-asr-streaming/.venv/Scripts/python.exe -u tools/model-debugging/reference/hf-parakeet-port/replay_packed_fft.py --capture tools/data/results/orukeet/official-speech-silence-stft-stages-2026-10-02.json.gz --expected-capture-sha256 07ae6f00dcab69eb3425566af7b5690a85dc2bee91e7510ddf27cca7b4370809 --sample-id speech-silence --coordinate 2 1027 --output tools/data/results/orukeet/NEW-replay.json --probe-output tools/data/results/orukeet/NEW-replay.mjs --cpu-threads 4 --timeout 60
```

For JFK, use `official-jfk-stft-layout-stages-2026-10-02.json.gz`, SHA-256
`4328840597d82b6942e640c2d9a4f810ca6c589588fb833009d02d8b14178508`, sample
`jfk-short`, no coordinate, and fresh output paths.

Actual CLI receipts `packed256-replay-cli-jfk-final-2026-10-02.json` and
`packed256-replay-cli-speech-silence-final-2026-10-02.json`, each with the matching
`.mjs` sidecar, reproduce the prior results at unchanged `1e-5/1e-4`:

- JFK physical1101/valid1100: official control 0 mismatches, packed256 0;
  candidate max abs `5.340576171875e-5` (mixed absolute/relative tolerance).
- Speech/silence physical4061/valid4060: official control 0, candidate 1 at
  flattened9149 (mel2/frame1027), max abs `1.4750286936759949e-5`.
- Selected bin2/frame1027 intervention: 0 feature mismatches, exact oracle
  components selected and every unselected component bit unchanged. This is
  causal injection evidence, not a deployable correction.

Twelve focused unit/negative tests pass. Review found that incomplete Node
results could write a control-only receipt before a later missing-candidate
error. The CLI now requires exactly every expected result, strict finite f32
feature/statistic counts, positive standard deviations and complete finite
point fields consistent with full-prefix output. Negative regressions cover
missing/extra results, boolean/lossy values, counts/statistics and points.
All imported helper/source hashes are captured before execution and checked
again before the receipt, which also binds packed256 and each spectrum hash.
Final disk verification also caught Windows LF-to-CRLF translation by the text
writer: preliminary/reviewed probe files did not match the executed probe hash.
A binary exclusive UTF-8 probe writer and byte-exact/overwrite regression now
preserve execution bytes. Final receipts use fresh paths; all nine source hashes,
actual probe file hashes and both known canonical frame hashes were independently
checked against disk. Prior preliminary/reviewed files remain local diagnostics,
not accepted provenance or committed evidence, and were not overwritten.
The initial wrong JFK sample ID was
rejected before writing outputs. Two first actual replays were stopped after
unexpected CPU-consuming delay with no artifacts. A capture-only control took
1.28s; real512/complex256/recombination took milliseconds. A bounded stack
probe localized delay to Numba `ensure_cache_path`/`NamedTemporaryFile` while
Librosa lazily imports filters. Using the existing child-local
`.cache/nemo-reference/numba` produced the exact expected filterbank hash in
1.95s; the CLI now sets that child-only cache before import. Corrected JFK
finished in 9.36s (final version: 9.21s) and both final replay processes exited0.
The complete nearby Python diagnostic suite passes 134 tests. No global
environment settings, model weights, production processor or sibling examples
were changed. The failed/diagnostic probe processes are terminal.

Next compare the candidate complex256 bins2/254 against a discriminating
factorization control on the cancellation frame. The full portable feature,
ASR and browser/WebGPU gates remain open; do not promote this Torch-only recipe
or repeat already-rejected broad FFT arithmetic sweeps.

### Fixed split128 cancellation-frame control (2026-10-02)

Added `probe_fft_factorization.py`: a bounded single-frame diagnostic accepting
an explicit capture/hash/sample/frame/bin and new receipt path. It reuses the
frozen capture validator, verifies selected Torch real512 output byte-exactly
against the original STFT, then compares complex256 with a fixed even/odd
complex128 factorization. Both use the existing unchanged real512 recombination.
Twiddles are f32 and complex products have one final f32 rounding. There is no
model/bin correction, factorization sweep, weight loading or production edit.

Actual speech-silence frame1027/bin2 run:
`tools/data/results/orukeet/split128-cancellation-frame-2026-10-02.json`.
Frame SHA-256 matches `d25846d1b1a3764b4cd67e36ef448b1168920b16b20a7eb21f131a7fabd2d2a7`.
Splitting changes packed bins2/254 real components by one f32 ULP in opposite
directions, but recombined real stays `-0.0012302398681640625`: error remains
`2.60770320892334e-8`. Imaginary error grows from `7.101334631443024e-9` to
`3.6903657019138336e-8`. This excludes this specific split128 recipe as a repair
for the selected cancellation, not all FFT128 factorizations or MKL internals.

Syntax, imports, help, actual frozen-frame control and exclusive-output negative
check pass; attempted overwrite rejects before reading a deliberately missing
capture and leaves the receipt unchanged. Existing replay's 12 unittest checks
pass. An initial pytest invocation failed because pytest is not installed in
the existing environment; no dependency was installed. This new diagnostic has
not run full-feature replay and does not close the known one-feature mismatch.
The receipt binds the direct script, replay helper and output helper; imported
transitive helper hashes are not independently enumerated by this small probe.

Next: inspect a distinct real even/odd FFT256 factorization control rather than
extending this rejected complex split or changing feature tolerances. Keep
portable feature, full ASR and actual-browser recurrent gates open.

### Real even/odd FFT256 cancellation control (2026-10-02)

Extended the same single-frame probe with independent real FFT256 transforms
of even and odd input samples. It compares their terms with packed-complex
deinterleaving, using identical f32 twiddles/single-round products and final
addition. Both selected bin2 and reciprocal bin254 are checked; half-spectrum
conjugation applies to the FFT256 halves, not to the final FFT512 bin254.
A fail-closed guard requires this component recombination to reproduce the
existing packed recipe at both coordinates before writing the receipt.

Actual evidence: `tools/data/results/orukeet/real-half256-cancellation-control-2026-10-02.json`.
Original selected-frame FFT512 bytes and frame hash still match. At bin2,
the direct-half real error grows from `2.60770320892334e-8` to
`5.587935447692871e-8`; imaginary error grows from `7.101334631443024e-9` to
`5.180481821298599e-8`. The packed method matches captured bin254 exactly;
direct halves introduce `2.9802322387695312e-8` imaginary error there. This
specific factorization therefore does not repair the selected cancellation.
It does not establish which internal MKL plan produced the reference.

Syntax, actual frozen-frame/recipe guards and existing replay's 12 tests pass.
The receipt binds the updated direct script/helper hashes; the prior split128
receipt remains bound to commit `18e3f6f`, not the updated script. No production
processor, weights, tolerances or public model status changed. Full-feature
replay and browser/model acceptance are not claimed by this single-frame tool.

Next: use the now-compared even/odd terms to distinguish twiddle/product and
final-add contributions with exact per-operation residuals, not another broad
factorization or precision sweep. Keep the one-feature failure open until a
portable implementation passes both complete frozen fixtures unchanged.

### Exact operation residual boundary (2026-10-02)

Added dependency-light `probe_fft_residuals.py`, using Python Fraction and f32
struct conversion on the pinned real-half receipt, without rerunning FFT/model
inference. It verifies exact finite f32 inputs, recreates both recorded outputs
and writes exact rational product/add residuals. New outputs are exclusive;
input/source bytes are checked before/after. A canonical JSON hash separately
binds content across Git LF/CRLF conversion; the byte hash still identifies the
actual local input used. Receipts do not assert transitive original-model parity.

Evidence: `tools/data/results/orukeet/exact-fft-operation-residuals-final-2026-10-02.json`.
For bin2, both packed and real-half paths have **zero final-add rounding error**
in both components. Packed real twiddle-product rounding contributes exactly
`31891649/2251799813685248` (about `1.416273720522554e-8`), while the required
rotated real term for the oracle with fixed packed even term is
`99719481/268435456`, not representable in f32. The cancellation add of these
near-equal opposite-sign f32 terms is exact; changing only that final add cannot
repair this decomposition. This does not attribute all oracle difference to
the product: transform terms and f32 twiddles can already differ upstream.
Bin254 is explicitly separate: its packed imaginary add has nonzero rounding
error despite the final recorded value matching the oracle. Zero final-add
error is therefore a bin2 observation, not a universal FFT claim.

Syntax, help, actual recorded-output guards and five malformed-pair checks pass
(boolean, non-f32, nonfinite, wrong count, missing pair), with a valid control.
The earlier preliminary residual receipt remains local and binds the earlier
script; only the final version is the checkpoint. No runtime/model/tolerance
changes or completed feature acceptance are claimed.

Next: stop final-add/rounding sweeps. Any new FFT implementation must avoid the
demonstrated early rounded cancellation or identify the reference plan, then
pass full JFK and speech/silence features at unchanged thresholds. Keep this
model gate open while continuing independent whole-product tasks.

### Public native cancellation/reuse and silence controls (2026-10-02)

Extended the existing public-library NeMo inspector with opt-in `--cancel-reuse`,
not another ASR decoder. It requires an official oracle, aborts only at positive
unfinished decode progress, requires PipelineAbortedError, then transcribes twice
with fresh signals through the same public model/session handles. Both repeats
must match all official token IDs, frames, TDT durations and text. Missing decode
progress, ignored cancellation, wrong errors or divergent recovery fail closed.
The existing awaited cleanup still owns releases. The opt-in receipt pins the
inspector's exact source bytes before/after; it is not a full compiled-build hash.

Fresh source compilation/runtime-asset copy preceded real native CPU runs using
the existing own packed fp32 graphs and original-weight-bound official PCM.
Node 26.2.0 and Bun each aborted speech/silence at encoder frame 1/508, then
matched all 69 tokens/frames/durations and text in both recovered transcriptions.
Final source-bound receipts (preliminary pre-lint versions remain local only):

- `tools/data/results/orukeet/library-node-native-cancel-reuse-reviewed-2026-10-02.json`
- `tools/data/results/orukeet/library-bun-native-cancel-reuse-reviewed-2026-10-02.json`

The same public inspector also ran the existing original pure-silence oracle on
both hosts, without the lifecycle option: empty tokens/frames/durations/text and
effective captured PCM match. Receipts:

- `tools/data/results/orukeet/library-node-native-pure-silence-reviewed-2026-10-02.json`
- `tools/data/results/orukeet/library-bun-native-pure-silence-reviewed-2026-10-02.json`

Full receipts stay local. The committed compact checkpoint is
`tools/data/results/orukeet/native-cancel-silence-handoff-2026-10-02.json`,
binding all four receipt byte hashes and 11 distinct rechecked provenance files.
For --cancel-reuse, transcribeMs covers the aborted call plus both repeats;
it is not a single-transcription latency measurement.

All four runs exit0 and remain PARTIAL_PASS_OUTPUT_ONLY. There are no new weights,
downloads, production-math edits or tolerance changes. Public handle identity is
not a measured engine-allocation/memory bound, and these native CPU controls do
not prove WASM/WebGPU, intermediate-state parity or complete Orukeet acceptance.
Eight simulated failure/recovery tests plus related inspector/family lifecycle
and native-bridge regressions pass (67 total). Strict new-test compilation uses
allowJs with checkJs false for the existing imported mjs harness; syntax, scoped
lint, formatting and diff checks pass. The JS lint host-global gap was fixed
locally with explicit Buffer import and globalThis.AbortController, not by
changing global lint policy. Sibling execution and live browser recurrence are
still untested for this slice. Next: retain the earlier feature gate and develop
the candidate-head live recurrent browser control without claiming feature parity.

### Frozen encoder fixture prerequisite (2026-10-02)

Prepared a family-scoped diagnostic tool, `frozen_encoder_replay.py`, with no
Orukeet-specific runtime branching or production changes. It preserves the own
encoder ONNX boundary but deliberately ignores feature/length inputs and returns
the frozen official encoder tensor and length. This avoids extracting or copying
the library's decoder loop solely to inject an oracle boundary. It is not a new
model export, an encoder optimization or a completed recurrence test.

The original .nemo, detailed export, packed graph, config, official capture,
summary and audio identities are checked before artifact creation. The initial
command mistakenly used the compact export handoff; it failed closed before
writing. Using the detailed export receipt passed. The existing speech/silence
capture produced a 2,081,417-byte diagnostic graph (SHA-256
`34c622506781fd57660fbe9b6601add2030b7e7397f2729b68837baf3a52c7ab`).
Native ORT 1.29.0 delivered the [1,1024,508] fp32 tensor and int64 length bitwise
exactly with both captured and zero features. Nine new weight-free tests and all
45 family export/replay tests pass. No tolerances were changed.

Receipt: `tools/data/results/orukeet/diagnostic-frozen-speech-silence-2026-10-02/frozen-boundary-receipt.json`.
The binary remains local and reproducible with the README command. This receipt
proves only frozen native delivery, not source external-weight validation,
features, real encoder, candidate decoder, browser or complete model acceptance.

Next: public-library conditional browser recurrence with exact captured PCM,
this frozen encoder on WASM, reproduced candidate head on WASM then WebGPU,
grid batching disabled, token/frame/duration/text checks and numerical state
checks. JS preprocessing will execute but its output will be ignored; explicitly
label it unaccepted and the real encoder bypassed. Retain full production feature
and real-encoder gates. Do not add the diagnostic graph to presets or distribute
it as production weights.

### Conditional browser recurrence checkpoint (2026-10-02)

The public-library loop now passes the frozen speech/silence encoder control in
actual Chrome 154 on both WASM and WebGPU, twice on the same loaded handle.
All 69 token IDs, frame indices, TDT durations and canonical text match exactly.
The exposed committed h/c cache has zero tolerance mismatches at the unchanged
atol=1e-5, rtol=1e-4. Maximum h/c absolute errors are 2.3842e-6/2.2888e-5 on
WASM and 2.5928e-6/1.9073e-5 on WebGPU. Each WebGPU repetition observes 8,554
compute dispatches; this is not every-node placement or a VRAM/performance claim.

This cache boundary is before consuming the last emitted token, not NeMo's later
retained post-last-label state. Tokenless runs instead expose pre-SOS zero state.
Intermediate live states/logits and continuation remain untested. Features run
but are ignored by the frozen graph; production features, real encoder and full
ASR acceptance remain open. No candidate promotion or runtime math change occurs.

The reusable harness pins tools, compiled source, input and ORT artifacts before
execution, rechecks them afterward and verifies served bytes. Exact PCM uses
base64 f32le transport: the installed Playwright numeric serializer changed
positive zero to negative zero, caught by the PCM hash before inference. A
preliminary run also caught concurrent harness formatting and was rejected.
Eight new synthetic guard tests and all 53 family Python tests pass.

Compact evidence: `tools/data/results/orukeet/browser-recurrence-handoff-2026-10-02.json`.
Full hash-bound receipts and binary graphs remain local. Next: extend this
conditional control to pure silence/JFK, then intermediate live-state coverage;
keep the earlier production feature divergence visible throughout.

### Conditional browser pure-silence control (2026-10-02)

After the speech/silence checkpoint, reused the unchanged harness for the
official one-second pure-silence fixture. WASM and WebGPU each pass two
same-handle transcriptions: zero tokens, empty text, exact zero exposed pre-SOS
h/c cache, 244 exact served artifacts and no browser/cleanup errors. WebGPU
observes 564 dispatches per repetition. This does not compare NeMo's different
retained SOS state or accept production features/real encoder/full ASR.
An initial WebGPU command had an export-path typo and failed before writing;
the corrected command passed without source edits or tolerance changes.
Compact receipt: `tools/data/results/orukeet/browser-silence-recurrence-handoff-2026-10-02.json`.
Next: JFK conditional browser control, then intermediate live-state coverage.

### Conditional browser JFK control (2026-10-02)

Reused the unchanged harness and candidate for the official JFK fixture. WASM
and WebGPU pass two same-handle repetitions with exact 38 tokens, frames,
durations and canonical text. Equivalent final committed h/c cache has zero
tolerance mismatches; maximum h/c absolute errors are 5.4240e-6/1.7166e-5 on
WASM and 3.6620e-6/1.5259e-5 on WebGPU. Each WebGPU repetition observes 2,162
dispatches. Each backend delivers 244 artifacts exactly, with no browser or
cleanup errors. No downloads, source math edits or tolerance changes occurred.

Compact receipt: `tools/data/results/orukeet/browser-jfk-recurrence-handoff-2026-10-02.json`.
This completes the three-fixture conditional recurrence matrix only. Features,
real encoder, all intermediate states/logits, stateful continuation, grid
execution, performance/memory and real companion applications remain unproven.
Next bounded model step: observe live head calls without another decoder and
compare equivalent inputs/outputs against official bindings, with an observer
non-perturbation control. Whole-product lifecycle/API work remains in scope.

### Generic streaming-owner cleanup recovery (2026-10-02)

The whole-product pass reproduced early dispose rejection on both loaded and
built-in public handles when one tracked transcriber failed while another was
still releasing. The owning model cleanup was skipped. Reused the existing
all-settled resource helper and added a small internal ordered cleanup boundary:
drain every transcriber, then attempt/await the owner even after failure.
Single errors retain identity; combined errors retain both stages. No public
API, model-ID branch, decoder or backend policy changed.

Five new regressions cover both handle paths, pending sibling/owner cleanup,
single and combined errors, synchronous/asynchronous failures, closed handles
and repeated disposal. Fresh clean-child-environment full suite: 1,351 PASS /
33 SKIP / 0 FAIL; production and strict new-test types plus browser-root import
smoke pass. Scoped lint has zero errors and an existing load.ts size warning.
The first ad-hoc test compile mixed source and dist runtime identities; using
consistent public entrypoints fixed test typing without altering production
families. TLS-disable inheritance was removed only in the rerun child process.
These are lifecycle simulations, not fresh real engine or sibling application
acceptance. The implicit owned-runtime branch remains untested by the new cases.
Receipt: `tools/data/results/harness/streaming-owner-cleanup-handoff-2026-10-02.json`.

### Conditional live head tensor matrix (2026-10-02)

The actual public-library decoder now has opt-in bounded session observation,
not another decoder or a production hook. Pure silence (12 calls), JFK (46)
and speech/silence (182) pass on both actual Chrome WASM and WebGPU: 480 live
calls in total. Every ordered call matches the official binding, exact frozen
frame/target/length and token/duration decision, and numerical incoming h/c,
prospective outgoing h/c and normalized scores at unchanged atol=1e-5,
rtol=1e-4, with zero tolerance mismatches. Blank-call prospective state is
separate from committed cache. Each observer is removed before run two; token,
frame, duration, text and final cache remain bitwise unchanged. No source
model mathematics, precision, download or sibling application edits occurred.

Seven comparator guard tests and four observer tests cover malformed/missing/
extra observations, exact storage, output changes, limits, restoration and
concurrency. All 60 family Python tests, scoped JS lint, TypeScript build and
browser-root import guard pass. Full raw tensors stay local; compact evidence:
`tools/data/results/orukeet/browser-live-head-handoff-2026-10-02.json`.

This closes intermediate live-head coverage only for these frozen-encoder
controls. The production feature divergence, real encoder, stateful
continuation, grid execution, fp16, performance/memory and companion app gates
remain open; the candidate is not promoted and full ASR is NOT_PROVEN.
Next model step: return to the earliest production feature divergence using
one source-backed discriminating experiment, not downstream head changes or
another speculative FFT rewrite. Balance with the explicit whole-product
implicit-owned-runtime cleanup coverage gap before further model-only work.

### Implicit runtime streaming-owner control (2026-10-03)

Closed the simulated ownership coverage gap without changing production code.
Two public built-in loads omit `runtime`, then attach controlled transcribers
to the existing weight-free TDT scaffold. The ownership route must drain both
transcribers before calling the actual owned runtime/model cleanup, await a
delayed model release even after transcriber failure, preserve single or
combined errors, and memoize repeated handle disposal. Spies forward original
cleanup; the test seam does not advertise scaffold streaming capability.

Both new cases, all 24 targeted tests, strict test types and scoped lint pass.
Full clean-child suite: 1,353 PASS / 33 SKIP / 0 FAIL; browser-root guard passes.
The initial test edit failed strict typing by directly calling an optional
method; it was reverted and re-added using the existing capability guard.
No API, model math, package dependency, precision or sibling app change occurs.
These are lifecycle simulations, not native/WebGPU streaming acceptance.
Receipt: `tools/data/results/harness/implicit-runtime-streaming-cleanup-handoff-2026-10-03.json`.
Next: the earliest Orukeet production feature divergence, with existing source
captures and one discriminating experiment; retain all full-model gates.

### FFT thread-count discrimination (2026-10-03)

The existing guarded packed-FFT replay ran against the identical official
speech/silence capture with Torch CPU threads 1 and 8, compared with the recorded
4-thread control. All three preserve the exact captured real FFT, identical
windowed-frame, packed-complex and recombined-candidate hashes, and the same
single final-feature mismatch at mel 2 / frame 1027 (flat index 9149; max absolute
error 1.4750286936759949e-5). Official complex injection and the diagnostic
selected-bin patch each retain zero mismatches. Both fresh commands exit zero.

Thread count does not explain this residual for this fixed build and fixture.
This is DIAGNOSTIC_NOT_PORT_ACCEPTANCE: it neither fixes production features
nor proves browser performance. The independently committed WASM worker cap
is runtime policy, not a numerical workaround. No tolerance or production
mathematics changed; the Torch-dependent candidate remains unpromoted.
Receipt: `tools/data/results/orukeet/fft-thread-control-handoff-2026-10-03.json`.
Next: inspect the actual FFT operation boundary for a source-backed distinction;
retain production-feature, real-encoder and full-model acceptance gates.

### Speech/silence FFT operation control (2026-10-03)

Ran `probe_fft_factorization.py` on the captured frame 1027 / bin 2 with
real512 byte identity required, then the hash-pinned exact-rational operation
probe. Complex256, split128 and real even/odd halves do not reproduce both
oracle components. The recorded final bin-2 additions are exact, while the
required oracle rotated terms are not representable as f32 for these fixed
even terms. This rejects a simple final-add-rounding explanation for this
recorded arithmetic; it does not identify MKL internals or validate features.
Local receipts: `fft-factorization-speech-silence-2026-10-03.json` and
`fft-operation-residuals-speech-silence-2026-10-03.json` under the Orukeet results.

The proposed current-production boundary command was also executed, but
`compare_js_feature_boundaries.py` rejected the original canonical audio before
computation because its input loader requires same-rate mono PCM16. No output
receipt or model PASS was produced. Next: extend that diagnostic to consume
the exact captured PCM with original audio/capture identity checks, not hidden
resampling, then compare oracle-complex injection through production reduction.

### Exact captured-PCM production boundary (2026-10-03)

Added explicit `--captured-pcm` to the boundary diagnostic: source audio SHA
is still checked, and exact finite float32 waveform, shape/rate/count/padding
and model-input SHA are required. Receipt records both identities. Three new
guard tests plus 89 existing boundary/STFT tests pass. Independent inspection
found missing semantic boundary validation; after the live process completed,
added `preprocessor_input_signal` validation and wrong/missing-boundary cases.

The actual 40.6s run completed with diagnostic exit 1 and wrote
`feature-boundary-captured-pcm-speech-silence-2026-10-03.json.gz`. Official
processor and actual STFT reproduction pass; effective PCM SHA matches the
capture. Current production features have 74 tolerance mismatches. Normalizing
the identical official raw mel has zero; oracle-complex library-power and
rounded-power feature controls each have zero. This localizes the remaining
production issue upstream of normalization for this fixture; power allclose
alone still does not imply feature parity. No production arithmetic or
tolerance changed and full-model acceptance remains open.

The raw receipt remains bound to the source before the semantic-boundary guard
addition; it is not mislabeled as executing the later helper version. Next:
record a compact hash-bound metrics handoff and commit the validated harness,
then investigate the FFT/power seam without coordinate-specific corrections.

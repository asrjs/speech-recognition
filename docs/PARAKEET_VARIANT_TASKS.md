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

Next integration gap (audited 2026-10-03): no receipt proves Orukeet's public
end-to-end browser path with the real encoder and live recurrent head together.
The current browser encoder probe uses official features; public recurrence
uses frozen encoder output. Stock-v3 local lifecycle smoke hardcodes its preset
ID and file-role selection, so it cannot certify arbitrary compatible weights.
Do not relabel that smoke as Orukeet. Next bounded implementation: an opt-in
NeMo-TDT family browser diagnostic with explicit candidate identity/config,
byte-bound real encoder plus external data, validated head and tokenizer, exact
captured PCM, repeated public transcriptions and awaited cleanup. Preserve
separate feature/encoder/head/token/text gates; a failed feature gate must stay
FAIL even if full tokens/text match. No preset promotion or tolerance waiver.
Pure-silence width-8 normalization is already measured and regression-tested;
do not repeat it as a new hypothesis. Six current reduction tests pass.

Implemented follow-up (2026-10-03): `browser_recurrent_replay.py` now accepts
`--real-encoder-backend wasm|webgpu`, retaining the frozen default and existing
frozen-native delivery control. Real mode binds the packed encoder/data to the
repack receipt, checks exact HTTP delivery and actual per-run component metrics,
and uses the public technical-family API without registering/promoting a brand.
It never reports full feature/ASR acceptance. Thirty-three encoder/head/recurrence
harness tests pass, including real-mode/provider/dispatch negative controls.

First actual JFK hybrid run (fp32 WebGPU encoder + WASM persistent head,
`--trace-head`) completes two transcriptions with all 38 tokens, frames,
durations and text exact. Final committed h/c states pass unchanged tolerances.
All 247 delivered artifacts verify, with no browser/cleanup errors. The trace
observer preserves outputs and final state exactly versus the unobserved repeat.
Overall status remains `REAL_ENCODER_OUTPUT_DIAGNOSTIC_FAIL`: at head call 20,
frame 69, normalized scores have three failing coordinates (first 8195), max
error `0.0005788803100585938`, mean `9.097685666164557e-5` at `1e-5/1e-4`.
All recorded head inputs/states and other score calls pass their respective
gates. This does not waive the earlier feature failure or establish full ASR
parity. Receipt: `browser-real-public-hybrid-jfk-2026-10-03.json` under Orukeet
results. Next isolate the failing score boundary using the already captured
actual head feeds; do not change score normalization or acceptance tolerances.

Actual filterbank diagnostic (2026-10-03): opt-in STFT capture now snapshots
the instantiated NeMo `model.preprocessor.featurizer.fb` float32 buffer, its
shape/hash and preprocessor config rather than regenerating reference weights.
Fresh official JFK execution captured 471 stages with unchanged before/after
instrumentation controls. `probe_captured_filterbank.py` compares the actual
`[1,128,257]` coefficients with built JS at `atol=0, rtol=0`: 157 numerical
mismatches, max error `3.725290298461914e-9`, no support differences.
Librosa stores the triangle in float32 before its in-place Slaney scaling;
JS currently rounds only after scaling. An opt-in, in-memory triangle-rounding
intervention gives zero numerical coefficient mismatches. One serialized-bit
difference remains because Node JSON loses negative-zero signs; this is not
byte-exact acceptance. Production code is unchanged, and feature/ASR gates
remain unresolved (production feature mismatches still 313 / 74).
Receipts under `tools/data/results/orukeet/`:
`captured-filterbank-js-jfk-final-2026-10-03.json` and
`captured-filterbank-js-triangle-jfk-final-2026-10-03.json`.
Twenty-six capture/probe regression tests pass, with independent read-only
review. Follow-up `probe_filterbank_features.py` changes only the matrix on
identical byte-hashed JS production power (not captured original NeMo power).
An unmodified production-output control passes. Actual-buffer and triangle
conditions produce numerically identical raw log-mel and normalized features;
their change from baseline is at most `1.907349e-6` / `1.549721e-6` respectively.
All three JFK conditions still fail the unchanged official feature gate at
313 coordinates, with identical max error `0.00034236907958984375`.
Receipt: `filterbank-feature-jfk-2026-10-03.json`. Four new result/Node synthetic
guards pass; independent read-only review of this follow-up passed.
This rejects matrix rounding as a solution to the observed JFK feature failure;
no runtime fix or tolerance change is justified by this diagnostic alone.
Next: inspect the earliest remaining power/log/normalization boundary against
the existing negative controls before selecting another source-grounded test.
Do not repeat rejected FFT/reduction variants or change tolerances.

Pure-silence actual-boundary control (2026-10-03): fresh official execution
captures 65 stages including the actual STFT/filterbank buffers, with all six
before/after instrumentation checks true. Original empty tokens/text remain
unchanged. Real/imaginary STFT components are entirely zero; actual filterbank
hash matches JFK (`f5608e00763a20a191022fbb00a43346db5f1ec06828a97a6f4157f7d02b76c8`).
The matrix-feature probe's three conditions have identical all-zero production
power (verified against the expected zero-buffer hash), raw log-mel and features.
All still fail the official feature gate at 12,800 valid coordinates, max error
`0.1600533276796341`: official valid values are `-0.1600533276796341`, whereas
production gives zero. This directly excludes FFT/filterbank differences as
the remedy for this silence fixture; normalization arithmetic remains separate
from the nonzero-spectrum speech failure. Do not invent a silence correction
or hardcode this value. New receipts: `official-pure-silence-filterbank-summary-2026-10-03.json`,
`official-pure-silence-filterbank-stages-2026-10-03.json.gz` and
`filterbank-feature-pure-silence-2026-10-03.json` under Orukeet results.
No runtime, weights or acceptance tolerances changed; full ASR remains open.

Alternate official-backend control (2026-10-03): config-only NeMo
`FilterbankFeatures.eval()` on the captured JFK float32 PCM, CPU threads 4,
without restoring weights or running ASR. Config matches the pinned export's
preprocessor; eval disables dithering on both devices. CPU features reproduce
the captured original features exactly (max/mean error 0). CUDA features fail
the unchanged `atol=1e-5, rtol=1e-4` gate at 1,028 / 140,928 coordinates,
max error `0.0007370710372924805`, mean `2.513591626575393e-6`.
Preemphasis is exact across devices. STFT real/imaginary, power, linear mel,
guarded log input and raw log-mel pass their numerical gates, but are not
byte-exact; normalized features fail. Repeated CPU and CUDA features are each
exact within that device. This is alternate-backend evidence, not a new oracle
or a tolerance waiver, and does not isolate normalization from upstream error.
Environment: Torch `2.6.0+cu118`, RTX 5060 Ti (sm_120); Torch warns this GPU is
outside its compiled architecture list despite these operations completing.
Evidence is terminal output from owned session 72134 (exit 0), not a persisted
stage capture. Existing CPU oracle and full ASR acceptance remain unchanged.

Browser production-feature sensitivity (2026-10-03): opt-in
`browser_encoder_replay.py --production-features` processes byte-hashed captured
PCM through the actual built JS processor in Chrome and runs two official-feature
and two production-feature calls on one real fp32 WebGPU encoder session.
JFK and speech/silence official controls pass; production encoder outputs also
pass unchanged `atol=1e-5, rtol=1e-4` with zero mismatches, exact lengths and
max errors `3.427267e-7` / `5.595386e-6`. Every call records 1,347 dispatches.
Feature comparisons still FAIL at 313 / 74 coordinates, respectively. Status
is deliberately `BROWSER_ENCODER_SENSITIVITY_MEASURED`, not model acceptance.
PCM identity, graph/data body hashes, unchanged sources and owned cleanup
passed; no blocked requests or browser/cleanup errors. Receipts under
`tools/data/results/orukeet/`:
`browser-production-feature-encoder-jfk-2026-10-03.json` and
`browser-production-feature-encoder-speech-silence-2026-10-03.json`.
Thirteen synthetic/preflight regression checks include failed feature gates,
failed official/production outputs, requested missing data and PCM/length identity.
Full decoder/token/text and public Orukeet browser acceptance remain open;
continue source-grounded preprocessing work without waiving its numerical gate.

Real browser encoder component (2026-10-03): the new family-owned
`tools/nemo-tdt-onnx-export/browser_encoder_replay.py` loads the actual packed
fp32 encoder plus 2,475,088,896-byte external data through the library session
bridge on Chrome 154 / NVIDIA Blackwell WebGPU. Official captured features
feed two runs on one real session, not a constant-output graph. JFK max error
`1.005828e-6` and speech/silence `4.611909e-6`; zero mismatches at unchanged
`atol=1e-5, rtol=1e-4`, exact encoded lengths, 1,347 dispatches per call.
Both graph and data HTTP bodies match pinned SHA-256 and byte counts; sources
unchanged, no blocked requests/page errors/cleanup errors, owned server stopped.
Receipts: `browser-real-encoder-jfk-2026-10-03.json` and
`browser-real-encoder-speech-silence-2026-10-03.json` in the Orukeet results.
This is official-feature encoder component acceptance, not production-feature
browser acceptance, every-node GPU placement or complete ASR. Feature gates
remain failed. The paired JS-feature diagnostic is now recorded above and
preserves this official-feature control.

Production-feature sensitivity (2026-10-03): the new family-owned
`probe_feature_encoder_sensitivity.py` runs the unchanged built JS processor
on exact captured PCM and feeds its actual features through the own packed
fp32 encoder. Both JFK and speech/silence native CPU controls have a passing
official-feature baseline and passing production-feature encoder outputs at
unchanged `atol=1e-5, rtol=1e-4`; lengths are exact. Production-feature encoder
max error versus official outputs: JFK `4.917383e-7`, speech/silence
`3.226101e-6`. Feature gates still FAIL (313 and 74 coordinates respectively).
This is sensitivity evidence, not permission to waive preprocessing parity or
promote Orukeet. Native ORT 1.29.0, four CPU threads; graphs, external data,
capture and processor bytes were hashed before/after. Receipts:
`production-feature-encoder-sensitivity-jfk-2026-10-03.json` and
`production-feature-encoder-sensitivity-speech-silence-2026-10-03.json`
under `tools/data/results/orukeet/`. Decoder, browser and full-ASR gates remain
open. Next source-grounded preprocessing work must preserve the feature gate;
this control can quantify downstream effects without substituting for it.

Public lifecycle control (2026-10-03): the owned local browser smoke now offers
`--verify-cancellation`. Stock v3 fp16 WebGPU encoder / fp32 WASM decoder passes
baseline transcription, progress-triggered in-flight decode cancellation,
same-handle transcription reuse with exact repeated text, and awaited disposal.
Receipt: `tools/data/results/harness/local-public-hybrid-v3-cancel-2026-10-03.json`.
Chrome 154 used the pinned local ORT 1.29.0 engine assets; no blocked requests
or page errors, inputs unchanged, owned preview stopped and port closed.
This proves cooperative cancellation between decoder iterations, not GPU kernel
interruption, worker cancellation, long-session memory or Orukeet parity.

Latest model/harness checkpoint (2026-10-03): identical windowed-frame controls
isolate the observed native/ONNX spectrum difference to FFT/backend arithmetic.
Preemphasis, frame gathering and window multiplication are byte-exact for the
speech/silence fixture; full pairwise ONNX features still FAIL at76 coordinates.
Do not repeat fp64 DFT, zero-imaginary complex FFT, PocketFFT or PFFFT variants
as new untested fixes: their failed downstream controls are recorded below.
The general processor diagnostic now supports exporter-pruned optional inputs
without dropping required inputs; this is harness reliability, not model parity.
Next model experiment must have a materially new, source-grounded hypothesis;
whole-product API, lifecycle, example and native acceptance work stays active.

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

Post-commit real-input guard control: loaded the hash-pinned actual speech/silence
capture through `replay_packed_fft.read_capture`, then the committed
`captured_pcm` helper including semantic boundary validation. All 649,600 samples
are byte-identical; computed SHA is
`8f4293517daf07f433b307a4a971f7c17dc9ed348b5f72999945b217331ec3df`.
Command exits zero. This validates the later input guard, not a rerun of the
full feature receipt or a new model acceptance result.

### Own ONNX DFT component feasibility (2026-10-03)

Ran a new tiny own fp32 opset-17 DFT graph on all exact captured speech/silence
windowed frames via native ORT 1.29 CPU. Graph checker and output shape/dtype/
finite guards pass. Spectrum has zero tolerance mismatches at unchanged
atol=1e-5/rtol=1e-4, max absolute error 2.086162567138672e-6, but 1,050,800
component bit differences. This is DIAGNOSTIC_DFT_COMPONENT_ONLY, not feature
or model parity. No production export, browser DFT support, speed or memory
claim. Receipt: `tools/data/results/orukeet/ort-dft-speech-silence-2026-10-03.json`.
Next: feed this candidate spectrum through the existing downstream feature
control; component allclose cannot promote it. Tool syntax and actual run pass;
focused helper/negative tests and review are still needed before committing it.

Downstream preparation control: a standalone NumPy-only reconstruction of the
Slaney 16 kHz / FFT512 / 128-bin float32 filterbank produces SHA-256
`f5608e00763a20a191022fbb00a43346db5f1ec06828a97a6f4157f7d02b76c8`, exactly
matching the recorded librosa bank in `packed256-thread1-speech-silence-2026-10-03.json`.
The command exits zero without importing librosa. This permits a separately
tested dependency-light diagnostic helper after the live downstream run ends;
it does not establish the cause of that run's delay or feature/model parity.
Session 18933 remains live and has not been restarted or modified.

Implemented the dependency-light bank and four weight-free regression tests in
commit `b95b52c` (pinned recipe identity, wrong identity, invalid recipe inputs,
independent allocations). It remains diagnostic-only and is not integrated
into the live DFT run. A read-only Windows module inventory for PID 14664
confirms SciPy, llvmlite and Numba DLLs are loaded; this supports that the broad
dependency path has been entered, but is not a stack trace or proof of the
current instruction/cause of delay. Final downstream metrics remain pending.

### Dependency-light frozen DFT feature result (2026-10-03)

An independent control in `probe_light_dft_features.py` reuses the captured-input,
probe-composition and numerical validation helpers plus the hash-bound NumPy
bank. It does not modify/restart live session 18933 or import librosa/Torch.
The new final-source run exits zero and records a negative candidate result in
`tools/data/results/orukeet/light-dft-features-speech-silence-bound-2026-10-03.json`.
At unchanged atol=1e-5/rtol=1e-4: official-complex downstream control has zero
mismatches (max abs 9.5367431640625e-7); ORT DFT spectrum has zero tolerance
mismatches, but its downstream features have 78 (max abs 0.0001283884048461914,
first flattened index 98). Thus component allclose does not qualify this DFT
path for feature parity or production promotion. This is not a speed comparison
against the packed-FFT candidate or browser DFT feasibility evidence.

Receipt binds PCM, window, oracle components, empirical reconstructed frames,
bank recipe/identity, graph, executed JS probe and full source closure; records
4061 physical/4060 valid frames and actual CPU provider/engine versions.
Complex components are assigned separately to preserve signed zeros. Frame
padding is explicitly empirical, not claimed as independently captured FFT
input. Three CLI guard tests (four cases), four bank tests, syntax and actual
full diagnostic runs pass. Guards preserve existing outputs, reject input
aliases and reject a wrong bank before capture/probe execution. No model
weights, production code, acceptance thresholds or dist changed.

Next: investigate the remaining FFT/power arithmetic seam with equivalent
captured inputs and final-feature controls; retain this negative DFT candidate.
Do not escalate to fp16/preset promotion or use a coordinate-specific patch.
The older broad-dependency run remains independently pending while live.

Power-boundary A/B (2026-10-03): `--power library` changes only the globally
injected complex-to-power expression to production-style JS `re*re + im*im`
followed by the float32 store. The named probe guard changes with that boundary;
a regression proves reversing these named substitutions restores the baseline
probe bytes exactly, and invalid options fail. Four light CLI/probe tests pass.
Actual run exits zero with an explicitly diagnostic receipt:
`tools/data/results/orukeet/light-dft-library-power-speech-silence-2026-10-03.json`.
PCM, window, bank, graph and oracle hashes match the rounded control. Official
complex features still have zero tolerance mismatches, while DFT features still
have 78, the same max absolute error and first divergent index. Thus changing
this global power rounding boundary does not resolve the DFT feature failure.
Retain both controls; next inspect spectrum/power contributions before selecting
another FFT algorithm. No production code or acceptance threshold changed.

Read-only coordinate trace (2026-10-03): optional `--coordinate MEL FRAME`
uses the existing verified JS point capture, checks the valid prefix, and
records active bank weights and the unchanged oracle/candidate components.
It performs no injection override or coordinate correction. Five CLI/probe
guard tests pass, including bounds, booleans/nonintegers and independent lists.
Actual library-power trace at mel0/frame98 is recorded in
`tools/data/results/orukeet/light-dft-first-point-speech-silence-2026-10-03.json`.
It retains the same 78 mismatches and graph/input/probe identities.

This mel row has only bin1, weight 0.02837754227221012. Oracle real/imag are
(-0.00010862957424251363, -0.0006376661476679146), versus DFT
(-0.00010868882236536592, -0.0006375877419486642). Observed linear mel changes
1.1873688698926799e-8 -> 1.1871216898384773e-8; raw log mel changes
-16.45387077331543 -> -16.453907012939453. Mean -16.235523223876953 and std
0.8171498775482178 are identical. Thus at this first coordinate the candidate
spectrum difference reaches the feature through power/mel/log; mean/std do not
cause the local difference. This point does not prove the cause of all 78
mismatches or certify an FFT replacement. Next compare valid-frame spectral
error/energy distributions before choosing a global, portable arithmetic fix.

Native ORT dtype control (2026-10-03): `--dft-precision fp64` exactly widens the
same reconstructed fp32 frame values on the host, uses a checked DOUBLE DFT
graph, and casts real/imag components to the same complex64 boundary before
unchanged library-power/mel/log/normalization. Input dtype-specific bytes and
graph identity are recorded separately. This is a distinct ORT kernel/dtype
control, not a new Fourier algorithm or proposed fp64 browser production path.
Both native precisions pass exact impulse/silence controls; six tool tests pass,
including a hash assertion that the default fp32 graph remains unchanged.

Actual full speech/silence result:
`tools/data/results/orukeet/light-dft-fp64-boundary-speech-silence-2026-10-03.json`.
The official-complex control still passes. Candidate spectrum max abs improves
to 4.76837158203125e-7 and features have 32 mismatches versus 78 in fp32
(max abs 5.131959915161133e-5, first index still 98). At that first point the
fp64-to-fp32 candidate normalized value is -0.26725757122039795 versus reference
-0.26720625162124634, so higher FFT precision does not reproduce the official
fp32 kernel there. It remains a failed feature-parity candidate, with no
production, browser support, speed/memory or model-completion claim.

Direct complex512 kernel control (2026-10-03): extended the existing frozen
factorization probe, without modifying its packed256/split128 comparisons.
Real512 must still reproduce the captured frame bytes before any comparison.
The new route assigns captured fp32 values and zero imaginary components
separately, runs Torch complex512, and reports component counts/errors on all
257 positive bins. It does not replace the production FFT.

Both final-source commands exit zero with diagnostic receipts:
`tools/data/results/orukeet/fft-direct-complex512-frame98-2026-10-03.json`
and `fft-direct-complex512-frame1027-2026-10-03.json`. Default Python's actual
Torch is 2.7.1+cu128, four CPU threads, not the prior 2.6.0 environment.
Real512 reproduces both captured frames byte-for-byte. Direct complex512 has
435/437 numerically unequal components, max abs 3.5762786865234375e-7 /
2.384185791015625e-7. These are exact-inequality counts, not allclose failures.
Frame98/bin1 is (-0.00010865181684494019, -0.0006376001983880997), versus
oracle (-0.00010862957424251363, -0.0006376661476679146).

This rules out assuming Torch real512 and zero-imaginary complex512 have the
same fp32 arithmetic at these frames. It does not rule out every full-complex
algorithm or establish final-feature parity. Existing portable JS full-complex
controls remain negative; no new production algorithm is justified by this
control. Syntax passes; the existing-output negative control fails before
capture/FFT and preserves the receipt hash. Source/capture/frame identities
are recorded. No thresholds, weights, dist or live DFT sources changed.

Next: investigate a materially different, source-grounded portable real-FFT
factorization before another full-feature run. Do not repeat higher-precision
or zero-imaginary full-FFT variants as if they were untested fixes.

Portable real-FFT research (2026-10-03): next bounded candidate is PFFFT's
FFTPACK-derived real-transform flow, not another arbitrary radix2 rounding
variant. Upstream documents fp32 real transforms and WASM SIMD support:
[PFFFT README](https://github.com/marton78/pffft/blob/master/README.md).
Its [implementation](https://github.com/marton78/pffft/blob/master/src/pffft_priv_impl.h)
contains distinct real passes/preprocessing/finalization; this suggests a
different operation order, not guaranteed MKL equivalence. Length512 still
uses power-of-two factors; do not call it a new prime-factor family.
Record the exact revision/compiler/flags before executing any future control;
compare the same captured frame bytes and all downstream feature values.
Do not use a wrapper that inserts another window or changes scaling/layout.

PocketFFT is a secondary real-codelet comparison:
[upstream README](https://github.com/mreineck/pocketfft/blob/cpp/README.md).
At that research checkpoint neither candidate had been compiled, measured or
promoted here; the installed SciPy control below now measures PocketFFT only.
Read and preserve upstream license notices before incorporating source.
The local PATH check finds cmake but not emcc/clang; this is not an exhaustive
toolchain inventory or authorization to install one. First check reusable local
assets/toolchains and a dependency-light native real-FFT control; browser build
and feature gates remain separately open. No new model weights are needed.

PocketFFT native control (2026-10-03): the light feature tool now accepts
`--fft-engine scipy-pocketfft --dft-precision fp32 --power library --coordinate 0 98`.
Use the same speech/silence capture, sample ID and pinned capture/bank SHA256
as the preceding light DFT controls, with new exclusive output paths.
The final-source run is recorded in
`tools/data/results/orukeet/pocketfft-features-bound-speech-silence-2026-10-03.json`
and its adjacent `.mjs` probe. SciPy 1.11.3's private `basic.rfft` is called
directly with axis=-1, backward normalization and one worker, bypassing backend
dispatch. The receipt hashes its Python wrappers and installed Windows binding;
ORT provider and graph fields are null because this candidate is not ORT.

Same captured fp32 frame SHA256, same downstream JS probe, unchanged atol=1e-5
and rtol=1e-4: all 2,087,354 spectrum components pass (max abs 4.76837158203125e-7),
but final features FAIL with 56 mismatches / 519,808 values, max abs
0.00010156631469726562, first index98 (mel0/frame98). Official-complex downstream
control passes with zero mismatches. This rules out promoting this installed
PocketFFT control as the feature fix, not every possible portable FFT build.
No production FFT, tolerances, weights or live broad-DFT dependencies changed.
Nine lightweight tests pass, including native impulse/silence, input immutability,
shape/dtype/nonfinite rejection and fp64 option rejection before output creation.
PFFFT's independently compiled real flow remains the next algorithm candidate;
browser feature/encoder/token/text acceptance remains open.

PFFFT native preparation (2026-10-03): acquired the small upstream source repo
at exact revision `aa16fd3db58de4ba5dae8b0438440bb9da46b6fa` under
`tools/data/results/orukeet/pffft-native-control-source-2026-10-03`.
Source license notices remain intact; no upstream source modifications or
toolchain installation. Existing MinGW GCC13.1 builds `pffft.c` together with
`pffft_common.c`, include directory `include/pffft`, flags
`-shared -O2 -ffp-contract=off -DPFFFT_EXPORTS`, and `-lm`, producing the local
`pffft-sse-control.dll`. Initial missing-export/common-source linker failures
were resolved by inspecting the header and upstream CMake target, not by
patching source. Actual binding reports SSE1, SIMD width4.

An aligned-buffer ctypes FFT512 REAL/FORWARD ordered-output sanity control
passes exact impulse and silence checks. DC and Nyquist are unpacked from
ordered values0/1; middle bins are interleaved. This is preparation only:
full captured-spectrum and downstream-feature comparisons are UNTESTED,
and browser/WASM equivalence, performance and model acceptance are UNTESTED.
Next: add a hash-bound optional adapter with resource cleanup and malformed
input guards, then run the same pinned frames through the existing JS feature
probe. Do not promote on this native sanity result.

PFFFT SSE feature control (2026-10-03): implemented opt-in `pffft-native`
in the existing light feature tool, with explicit trusted library path and
`--expected-pffft-sha256`. Native code loading is not a sandbox; only locally
trusted builds may be requested. The adapter validates finite fp32 FFT512
frames, verifies binary SHA256 before loading and after execution, uses aligned
buffers, cleans buffers/setup in `finally`, and unpacks DC/Nyquist explicitly.
Its source enters the same pre/post diagnostic hash closure. Default ORT
graphs and candidate behavior remain unchanged; no binary is committed.

Final-source receipt:
`tools/data/results/orukeet/pffft-sse-features-bound-speech-silence-2026-10-03.json`
with adjacent `.mjs` probe; binary SHA256
`5209587996e8edf345491c84bc322b7a452e1ff21dfa4542a49f07acd294d97e`.
Use the same pinned capture/bank/sample and library-power/coordinate options
as PocketFFT, adding `--fft-engine pffft-native --pffft-library <local-dll>`
and the explicit binary hash. Spectrum passes allclose with zero mismatches,
max abs 5.140900611877441e-7; features FAIL with 65/519,808 mismatches,
max abs 8.402019739151001e-5, first index98. Official-complex control passes.
Same frame/probe hashes and unchanged atol=1e-5/rtol=1e-4; no promotion.

Twelve tests pass with the real local DLL, including shifted-impulse phase,
silence, input immutability and identity/input rejection. Without a requested
DLL, only the native execution test SKIPs; requested missing/hash-mismatched
inputs do not silently skip. Syntax and diff checks pass. Native compiler/source
provenance remains the preceding preparation record, separate from binary-bound
execution receipt. Browser/WASM and end-to-end gates remain UNTESTED.
Next discriminating step: scalar PFFFT build of the same revision with the same
non-contracted arithmetic flags; then reassess real-kernel versus official
engine rounding rather than relaxing feature tolerance.

PFFFT scalar control (2026-10-03): same pinned revision and GCC13.1 flags,
adding only `-DPFFFT_SIMD_DISABLE`, produces `pffft-scalar-control.dll` with
SHA256 `7044f871b33946a428c544c2daf78224651bfc86186657a1985975f0cea62ce8`.
Actual adapter reports Scalar / width1; all three real-binary/input/identity
tests pass. No upstream source or diagnostic-source changes. Same full capture,
bank, reconstructed frames, probe and library-power contract are used in
`tools/data/results/orukeet/pffft-scalar-features-bound-speech-silence-2026-10-03.json`
and its `.mjs` probe, via the existing explicit DLL/hash CLI options.

Spectrum allclose passes, zero mismatches, max abs 4.76837158203125e-7.
Features FAIL: 64/519,808 mismatches, max abs 9.34600830078125e-5, mean abs
1.1415412756377583e-7, first index98. Official-complex control passes unchanged.
Compared with SSE's 65 mismatches, scalar operation order is different but
disabling SIMD does not solve the feature gate. Native controls do not prove
browser equivalence, performance or end-to-end model acceptance.

Bound the algorithm search here: PocketFFT, PFFFT SSE and scalar are measured
negative candidates, not production fixes. Do not rerun arbitrary compiler,
precision or rounding permutations without a new discriminating hypothesis.
Next model work should inspect the official real-FFT kernel/factorization and
its actual arithmetic boundary (captured-frame reproduction already passes),
or a reproducible family-owned export path for that processor. Keep the same
feature acceptance and do not skip encoder/token/text gates. Whole-product
API/example acceptance work may proceed independently while this gate is open.

Full native oracle reproduction (2026-10-03): opt-in `--fft-engine torch-rfft`
now runs fp32 real FFT512 on every reconstructed frozen frame, with backward
normalization, four CPU threads, installed Torch version/build configuration
and explicit native-only scope. Torch is imported only on this selected route;
default ORT/PocketFFT/PFFFT branches remain unchanged. Receipts now independently
report spectrum byte equality, not merely allclose. fp64 is rejected before
capture/output creation; ten light diagnostic tests and syntax/diff checks pass.

Final-source receipt and adjacent probe:
`tools/data/results/orukeet/torch-rfft-features-bound-speech-silence-2026-10-03.json`.
Same pinned capture/bank/frame/probe hashes as the previous controls, library
power and coordinate0/98. Torch2.7.1+cu128 reports MKL2025.1 support. All
2,087,354 spectrum components are bit-exact to capture (zero numerical error),
not only the previously selected frames. The same diagnostic JS downstream
features pass with zero mismatches / 519,808 values, max abs
2.3543834686279297e-6, identical to official-complex injection. Tolerances remain
atol=1e-5/rtol=1e-4. This strengthens the frame reconstruction/control chain,
not a production/browser implementation or end-to-end port.

Primary route inspection:
[PyTorch2.7.1 CPU spectral implementation](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/mkl/SpectralOps.cpp).
Its MKL branch uses separate REAL versus COMPLEX descriptors, batched strides,
out-of-place execution and complex-complex conjugate-even storage for real
transforms. Source plus installed build configuration supports that route;
neither exposes MKL's internal radix/twiddle instruction order. Do not invent
a portable MKL-equivalent butterfly from the wrapper. Next: inspect feasible
family-owned processor graph/export controls and retain this complete native
oracle reproduction as the numerical gate. Binary provenance is version/build
metadata, not exhaustive hashes of Torch's loaded DLLs.

Processor artifact boundary hardening (2026-10-03): the family-owned
`OnnxNemoPreprocessor` now rejects missing outputs, wrong feature dtype/rank,
batch sizes other than one, invalid dimensions/counts, nonfinite features,
and malformed or out-of-range integer lengths before returning features.
Mel-bin count remains artifact-derived, not a Parakeet/Orukeet hardcode.
Valid int64/int32 lengths and owned feature copies retain their behavior;
input/output tensors are released on these contract failures.

`tests/nemo-preprocessor-contract.test.ts`: 14 controlled tests pass (12
negative cases reproduced the missing guards before the fix). Source typecheck,
focused lint and diff checks pass, as do 40 existing NeMo TDT/RNNT/AED tests
and nine WASM bridge tests. This is mocked contract/lifecycle evidence, not
numerical processor parity or real ONNX/browser inference. The original
processor graph/export and feature/encoder/token/text gates remain open.
The live broad DFT session18933 was polled and remains running without new
output; its source/dist closure was preserved.

STFT export feasibility (2026-10-03): inspected installed NeMo
`FilterbankFeatures.stft/forward`: constant centered padding, nonperiodic Hann,
complex STFT followed by real-pair magnitude are the actual source boundary.
Own TDT exports currently contain no processor graph. A weight-free Torch2.7.1
opset17 legacy-export control rejects `return_complex=True` with
`SymbolicValueError: STFT does not currently support complex types`, zero graph
bytes written. This is exporter/operator scope, not failure of the model.

Changing only the bounded control's output representation to
`return_complex=False` preserves native spectrum bytes at 1600/3200 samples
and exports a 6014-byte ONNX graph containing STFT. Native ORT1.29 CPU runs both
dynamic lengths: spectrum allclose passes at unchanged atol=1e-5/rtol=1e-4,
max abs 1.564621925354004e-7 / 1.825392246246338e-7 in the initial seeded run;
ORT spectra are not bit-exact. No graph was saved/promoted and no NeMo source
was monkeypatched. The representation is deprecated in installed Torch.

Repeatable weight-free controls:
`python tools/nemo-tdt-onnx-export/test_processor_stft_export.py -v`.
Two tests pass; syntax/diff checks pass. These fixtures intentionally describe
FFT512/window400/hop160, not a generic processor implementation. The Python
tests preserve process thread count and use a local seeded generator. They
prove operator feasibility only: preemphasis, mel/log/normalization, official
full PCM feature parity, browser STFT support and complete ASR remain open.
Next: reconstruct the official family processor from pinned source config,
verify its frozen features first, then use a local export-only real-pair
adapter and compare the full processor output. Do not assume that STFT export
transfers MKL arithmetic or repairs the known near-silence feature failures.

Config-only processor reconstruction checkpoint (2026-10-03): streamed
SHA256 of the original `.nemo` again matches
`031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`.
Read only its regular `./model_config.yaml` archive member (bounded below
1 MiB), without extraction or weight loading. Its full `preprocessor` mapping
equals the packed `source-config.json` mapping exactly; packed config SHA256
remains `344297eaf9e28d7f849e099c06535fbb0df978926e1b5f753b6aa836d05acdb3`.
This is verified configuration identity, not numerical processor acceptance.

Started session74451 using the existing NeMo venv: official
`AudioToMelSpectrogramPreprocessor`, known target checked then removed from a
copied mapping for the inspected direct constructor, `.eval()`, four Torch CPU
threads, and the existing hash-bound `read_capture`/tensor comparator. It uses
the full speech/silence frozen PCM and unchanged atol=1e-5/rtol=1e-4; exact
valid length is required. No full model restore, source monkeypatch, installs
or graph writes. Official processor import completed; construction/comparison
has not reported a result. Subsequent handle polls confirm it still running.
Do not duplicate/restart on observation timeout or call the gate passed.
The older session18933 is also still live and its dependency closure remains
protected. Current reconstruction output is terminal-only; a successful result
will still need a reproducible source-bound receipt before export promotion.

Follow-up: both stalled runs are now terminal (2026-10-03). `py-spy dump`
identified both main threads in Librosa import -> Numba cache writability probe
-> Windows `tempfile._mkstemp_inner`. The inspected implementation retries
PermissionError when the directory-level W_OK check succeeds; this is a bounded
retry loop, not evidence of model computation. Owned sessions74451/18933 were
interrupted after this diagnosis, not merely because an observation expired.
Setting NUMBA_CACHE_DIR only in each child launch to a writable results directory
made the minimal Librosa filterbank control finish in 5.6 seconds. No global
environment, installed dependency or production processor was changed.

`probe_nemo_processor.py` now persists this config-only native control with
config/capture hashes, diagnostic source pre/post hashes, installed NeMo source
hashes after import/pre/post execution, Torch version/thread count, length and
numerical metrics. Four CLI preflight tests pass without optional imports.
Existing outputs and source/config/capture aliases fail before inference;
receipt writes are exclusive. Run in the existing NeMo venv with process-local
NUMBA_CACHE_DIR and explicit --config/--expected-config-sha256,
--capture/--expected-capture-sha256, --sample-id and a new --output path.

Actual `nemo-processor-reconstruction-speech-silence-2026-10-03.json`:
Torch2.6.0+cu118 CPU/four threads, all 519,808 feature values bit-exact,
max/mean error zero, valid length4060 (physical4061), unchanged atol1e-5/rtol1e-4.
This proves native processor reconstruction from the verified configuration,
not an exported processor, browser feature parity or full Orukeet acceptance.
The recovered ORT DFT run also completed: spectrum allclose, downstream features
still FAIL at78 cells, max0.0001283884048461914; official-complex control passes.
Receipt: `ort-dft-features-speech-silence-2026-10-03.json`.
Next use the reproducible official processor as the full-output control for a
real-pair STFT export; compare its full processor output without relaxing gates.
Portable DFT feasibility alone does not repair the known near-silence mismatch.

### Full real-pair processor export control (2026-10-03)

`probe_nemo_processor.py --onnx-probe` now compares the official config-only
processor, an explicit eval-only family adapter, and its in-memory fp32 ONNX
graph against the same captured PCM/features. No installed source monkeypatch,
weight restore, production change, graph publication or browser claim. The
adapter reuses official length/normalization helpers and config-derived window,
filterbank, preemphasis and log settings. Unsupported exact padding, training,
gradient mode, splicing, frame padding and normalization modes fail closed.
Five preflight tests, two adapter contract tests (eight rejected variants), and
the two existing actual STFT export tests pass. These tests do not prove ASR parity.

Speech/silence native adapter: all519,808 values bit-exact to the captured
official features; valid length4060. Full ONNX graph:150,416 bytes, SHA256
`66184fc8b4dd381d2de6f4bb939bd3c3c8c177ee8a6bdd127f1951555caf735b`;
dynamic sample axis traced at1600 samples and executed at649600. Explicit CPU
ORT1.23.2 (existing NeMo venv), Torch2.6.0+cu118, ONNX1.20.1. Length is exact,
but features FAIL unchanged atol1e-5/rtol1e-4:10,575 mismatches, first index0,
max0.00013090670108795166, mean0.00001838071101380855. Native byte identity
shows the real-pair adapter itself preserves this fixture; it does not prove
equivalence after ONNX export/operator execution. No optimized graph is promoted.

Receipt: `nemo-processor-real-pair-export-final-speech-silence-2026-10-03.json`.
Overall export status is FAIL and CLI exits1, separately retaining native PASS.
Earlier receipts identify earlier diagnostic hashes; they are not rewritten.
Next add bounded intermediate STFT/power/mel/log/normalization comparisons,
including identical-input downstream controls, to locate the earliest ONNX
arithmetic divergence. Do not treat graph feasibility as numerical acceptance
or adjust tolerances to accept the candidate. Browser/WASM gates remain open.

### Processor taps and identical-input normalization (2026-10-03)

`--onnx-probe --stage-probe` adds a separate eight-output graph through the
explicit family adapter: features/length, spectrum, power, mel, log-mel, mean,
std. Default two-output graph is unchanged (same SHA256 above). Captured native
features and all2,087,354 native spectrum components remain bit-exact. The
tapped ONNX graph's final features/length are bit-exact to the untapped graph:
DIAGNOSTIC_TAPS_UNCHANGED, not numerical port acceptance. Tap graph SHA256
`205e89268b329870ee8b59638024cf96a86780a34b9d371caea99f5a11b102e1`.

Every measured intermediate stage passes unchanged atol1e-5/rtol1e-4 but is
not bit-identical. First measured nonidentical stage is spectrum; first measured
tolerance failure is final features (10,575 cells, unchanged). This does not
prove a specific STFT kernel defect: preemphasis is upstream of this first tap.
Log-mel max error0.00010395050048828125, mean max error0.000064849853515625,
std max error0.000008344650268554688. Relative acceptance at these boundaries
does not imply acceptance after subtracting/dividing near-zero feature values.

A second in-memory graph reuses the official normalization helper on exactly
the native log-mel/length input, excluding FFT/mel/log differences. Its native
features/mean/std match the full native adapter bit-exactly. CPU ORT normalization
still FAILS:10,615 final feature mismatches, max0.00007557868957519531,
mean0.000018362965904706375, first index0. Mean/std pass their own tolerances,
but mean differs up to0.0000629425048828125. Normalization-path arithmetic is
therefore independently insufficient; this is not attribution of every full-
processor mismatch to reduction alone. Graph8270 bytes SHA256
`40a5174860b03f726dc9b77070dabc8e1976848a57f60d1a0ddfb1a4821dc399`;
input log-mel f32le SHA256
`abc571bd5da089c75d5c2ab07cb607eb02a4f2c29d27d26849a9defb57a518b3`.

Receipt: `nemo-processor-stage-normalization-speech-silence-2026-10-03.json`.
Seven CLI preflight, five stage comparator and two adapter contract tests pass;
the existing two actual STFT export tests remain independent feasibility checks.
Next inspect the proven native reduction-order controls and ONNX reduction
boundary on identical input, then validate any explicit arithmetic candidate
against the full oracle. No tolerance, production processor or preset changes.

### Dynamic fp32 pairwise normalization control (2026-10-03)

`--onnx-probe --stage-probe --pairwise-normalization` now tests an opt-in
family diagnostic using a scripted dynamic ONNX Loop, preserving odd tails,
valid-prefix masking, sample variance and the normalization epsilon. No fp64,
quantization, installed-source patch or production runtime change. Three
weight-free tests cover odd/even lengths 1 through 4061, single-frame handling,
masked padding and explicit CPU ONNX execution; nine CLI preflight tests and
five comparator tests pass. These are not model/browser acceptance tests.

Same native log-mel input as the prior control: all 519,808 candidate ONNX
features pass unchanged atol1e-5/rtol1e-4 against official normalization.
Mismatch count falls from baseline 10,615 to zero; max error
0.000003337860107421875, mean 0.0000005418963173514237. Mean/std also pass.
Candidate native versus ONNX has zero numerical error for all three outputs.
Graph 17,086 bytes, SHA256
`c92caa8c639db6947bc3801a0981a801d6d5246864632f87bb7e9a729d59b19a`.
Receipt: `nemo-processor-pairwise-normalization-speech-silence-2026-10-03.json`.

The unmodified full processor still has 10,575 mismatches and the receipt/CLI
correctly retain PROCESSOR_EXPORT_FAIL/exit1. This is PARTIAL PASS for isolated
normalization only, not proof of a particular native/ORT kernel algorithm.
Next test this explicit candidate inside a separate full-processor diagnostic
against the same oracle, then expand fixtures and WASM/browser checks before
any promotion. No backend, speed or whole-port claim follows from this result.

### Full processor with explicit pairwise diagnostic (2026-10-03)

The existing opt-in pairwise probe now also exports a separate full eight-output
processor graph. `RealPairProcessor` accepts an explicit registered diagnostic
normalizer module; its default remains the official helper. Four constructor
tests cover default preservation, module registration, invalid injection and
the existing unsupported processor contracts. This changes tools only.

On the same pinned speech/silence PCM, the native pairwise full processor passes
all stages against the native oracle; spectrum/power/mel/log-mel remain byte
identical. Its features have zero tolerance mismatches, max3.337860107e-6.
The full pairwise ONNX graph reduces official-oracle feature failures from
10,575 to76, but remains FAIL at unchanged atol1e-5/rtol1e-4. Max error
0.00012484192848205566, mean0.0000005949233135658844, first flattened index98;
valid length4060 is exact. Candidate-native versus candidate-ONNX features
also FAIL (80 mismatches). Spectrum/power/mel/log boundaries retain the same
small errors as the baseline; improved normalization does not repair them.
No particular FFT kernel or preemphasis cause is established by these taps.

Graph165,766 bytes SHA256
`59634de7379cba818dbb373594cae3ebab9d34c98cff276da2e044273bcc3d95`.
Receipt: `nemo-processor-full-pairwise-speech-silence-2026-10-03.json`.
Baseline graph hash and DIAGNOSTIC_TAPS_UNCHANGED are preserved; overall CLI
still exits1 with PROCESSOR_EXPORT_FAIL. No production preset, processor,
precision policy or tolerance change; WASM/WebGPU and ASR acceptance stay open.
Next isolate the earliest remaining waveform/preemphasis/STFT boundary with
identical-input controls, rather than tuning normalization to hide upstream
errors. Expand the official fixture matrix after a candidate passes this gate.

### Captured STFT input and preemphasis controls (2026-10-03)

The stage probe now adds separate eval preemphasis and identical-input STFT
graphs. Both use config-derived settings and captured inputs; neither changes
the full processor graph. Native preemphasis reproduces all649,600 captured
STFT-input values byte-for-byte. CPU ONNX preemphasis is also byte-exact.
The separate native real-pair STFT reproduces all2,087,354 captured spectrum
components byte-for-byte, while ONNX on the exact same captured STFT input
retains max2.086162567138672e-6/mean6.686923942797314e-9 error. Spectrum itself
passes atol1e-5/rtol1e-4, but is not identical; downstream full pairwise features
still FAIL with76 mismatches. This isolates an arithmetic difference within
the STFT execution boundary for this fixture, not preemphasis. It does not
identify a particular internal FFT, windowing or framing kernel as the cause.

Preemphasis graph2862 bytes SHA256
`9616e160d88d0cf27ca6318b76332c305bc375a59afb26f09120de71b5300109`;
STFT graph6048 bytes SHA256
`5188032dddc8226eff219ff7effa765cfe01e1555cfa09d84b60b882ead5b865`.
Captured STFT-input f32le SHA256
`47bb4059b5531cfe79b5daa77a17d78473cff6754716ed8fe037dc5ad0eb331f`.
Receipt: `nemo-processor-stft-input-control-speech-silence-2026-10-03.json`.
Five comparator and nine CLI preflight tests pass; real graph execution is
separate measured evidence. Baseline/full-candidate hashes remain unchanged,
overall export FAIL/exit1 preserved. Next inspect windowed frame/FFT execution
with this identical input; do not add per-model numerical correction constants.
Also keep the companion browser acceptance runner on the whole-product queue;
this model diagnostic does not certify examples, WASM or WebGPU execution.

### Owned browser-demo no-download UI smoke (2026-10-03)

`tests/smoke/browser-demo-smoke.py` builds the explicit sibling consumer in
memory, starts its own loopback preview and opens a fresh headless Chrome
context. It verifies served HTML/JS/CSS/public asset hashes against the build
receipt and protects the diagnostic and selected consumer inputs against
changes. It never writes to the sibling checkout or installs dependencies.

Run with existing Python Playwright, Chrome and Node:

```powershell
C:/Python311/python.exe tests/smoke/browser-demo-smoke.py --consumer-root N:/github/asrjs/browser-demo --output tools/data/results/harness/browser-demo-owned-ui-smoke-2026-10-03.json
```

Use a new output path on each run; existing outputs are refused. Four independent
preflight/parser tests pass via `test_browser_demo_smoke.py`. The real Chrome
154.0.8037.93 run passes five UI checks: boot, stock v3 hybrid selection
(WebGPU encoder/WASM decoder), WASM selection, local source panel, and actionable
missing-folder validation with Load Model becoming enabled again. Seven remote
font/badge/model-metadata requests were aborted; no model-weight request or
page error occurred. The owned preview process stopped and its port was no
longer reachable after cleanup.

Receipt: `browser-demo-owned-ui-smoke-2026-10-03.json` under the harness result
directory. This is PASS_COMPANION_UI_SMOKE only: selection is not proof of GPU
execution, and missing-folder validation is not model loading or inference.
Actual local artifact load/transcribe/cancel/dispose and Orukeet browser parity
remain open. Next add real local-model acceptance without remote downloads or
duplicating the library decoder in the harness.

### Stock-v3 real local companion gate (2026-10-03)

The same runner now has an explicit `--local-model-root` opt-in for stock-v3
local hybrid loading and the demo's existing warm-up transcription. It selects
the real directory input, requires a WebGPU adapter, records bounded application
console output, and requires both Model Loaded and Model ready. The encoder
precision is explicit (`--encoder-precision fp32|fp16`, default fp32); decoder
remains fp32/WASM. Existing fp16 stock-v3 artifacts are a companion control, not
permission to quantize or promote Orukeet before its fp32 parity gate passes.

Selected graph/tokenizer and colocated external-data inputs are required,
nonempty and SHA-bound before/after. Hashing uses one-MiB chunks rather than a
complete multi-gigabyte file copy. Six Python preflight/parser/hash tests and
eight local browser-loader unit tests pass; neither suite runs actual inference.
The first opt-in browser attempt failed in the harness before model load:
Runtime Backend's visible WebGPU label has value `webgpu-hybrid`, not `webgpu`.
That selector was corrected after inspecting App.jsx; the failed receipt is
preserved as `browser-demo-real-local-v3-2026-10-03.json`.

The corrected real run detects all86 local files and a physical NVIDIA
Blackwell adapter (fallback=false), but FAILS before inference: NeMo ORT Web
initialization requests `onnxruntime-web@1.29.0/dist/ort-wasm-simd-threaded.jsep.mjs`
from jsDelivr. The no-network policy correctly aborts that dependency. The
in-memory bundle's emitted WASM assets alone are not a self-contained runtime;
`src/models/nemo-tdt/ort.ts` defaults to a CDN prefix, and the local-browser
options/Parakeet adapter do not currently forward an explicit engine-asset path.
Receipt: `browser-demo-real-local-v3-network-2026-10-03.json`; CLI exit1, owned
preview stopped and port closed. Do not infer that the model or WebGPU kernel
is incompatible from this engine-bootstrap failure. Failure receipts now retain
blocked URLs, and the runner detects the visible terminal app error immediately
instead of waiting three minutes for a success that will not arrive.

Next implement a supported caller-supplied local ORT asset configuration through
the existing public loader boundary, package/serve the pinned matching engine
assets, then rerun this same local control without a CDN interception workaround.
This gate does not implement app cancellation or certify session disposal;
closing the owned browser is process cleanup, not proof of awaited library
disposal. Orukeet feature/encoder/token/text parity remains independently open.

### Explicit local ORT engine prefix API (2026-10-03)

`LoadSpeechModelFromLocalEntriesOptions`, `ResolveParakeetLocalEntriesOptions`
and `ParakeetFromUrlsConfig` now expose the existing family `wasmPaths` string
prefix. Real local resolution preserves it, and both the generic adapter and
legacy direct loader forward it into NeMo TDT source options. No model math,
decoder, backend selection, default engine URL or native settings changed.
Applications must serve matching ORT engine `.mjs`/`.wasm` files themselves;
the setting is ORT module-wide and should be chosen before session creation.

Six focused simulations cover v2/v3 through real File-to-blob resolution,
both legacy local-folder entry points and legacy direct explicit/omitted
prefixes, with disposal. Together with existing browser-local and Parakeet-helper
tests: 41 passed (63 including WASM thread regressions). TypeScript source compilation
passes; focused lint has zero errors and the existing compat.ts max-lines warning.
The current-source browser-demo in-memory build passes (307 modules); main
bundle752181 bytes SHA256
`ac815e9837f91bfc6b6d3d2b7f15520b19894bd8699ca36c9b73f75df1b83162`.
This is API forwarding/build evidence, not fresh engine or model inference.
The sibling demo still does not pass this new option; its source was preserved.
Next package/serve matching engine assets and exercise this public API in the
owned browser harness, then integrate the companion option within its writable
authority. The earlier real offline model-loading failure is not yet resolved
end to end, and no Orukeet promotion follows from this additive API change.

### Owned local engine serving checkpoint (2026-10-03)

`node tests/smoke/companion-build.mjs <consumer-root> --serve-local-ort`
now snapshots only the installed ORT Web JSEP `.mjs`/`.wasm` pair under
`/engines/ort-<installed-version>/`. Readiness reports `engine.wasmPaths`,
version, sizes and SHA-256; no arbitrary filesystem/node_modules route,
weight copy, CDN interception or sibling write is introduced. The ordinary
`--serve` path remains unchanged. This opt-in is for the current hybrid JSEP
path, not a promise to serve asyncify/JSPI or every ORT build variant.

Six Node tests pass (three CLI guards and three snapshot/identity guards).
The actual browser-demo current-source in-memory build passes with307 modules.
On the owned preview, Node HTTP reads both engine bodies successfully;
Chrome imports the actual module in a cross-origin-isolated context and
validates the WASM binary. Browser WASM SHA256
`db816fadbab47a755170c08f933961e231412ac17f5981f9a62e519708a44dea`
matches the installed bytes, with no blocked requests or page errors.
An earlier Python urllib body read timed out; Node and browser transfer controls
passed without changing the server. Its underlying transient cause is unknown.
The owned preview session was interrupted and its listener on15993 verified gone.

Scope is engine serving/import/binary validity only: no engine session, model
inference, numerical parity or performance claim. The companion still does not
pass `wasmPaths`. Next connect the public local-loader API to this prefix in an
owned harness page and run stock-v3 real hybrid inference, preserving the
separate Orukeet fp32 feature/encoder/token/text gates.

### Public local hybrid inference checkpoint (2026-10-03)

The owned `tests/smoke/local-parakeet.html` page now uses only public browser
local-loader and transcription APIs, not a duplicate decoder. The opt-in
companion preview bundles it in memory with separate worker asset names.
NeMo TDT preserves initialized encoder/decoder/joint session providers through
the canonical transcript metrics. These describe selected session providers,
not per-node GPU placement; WebGPU shape nodes can still run on CPU.

Reproduce with existing local stock-v3 assets (no downloads):

```powershell
C:/Python311/python.exe tests/smoke/local-parakeet-smoke.py --consumer-root N:/github/asrjs/browser-demo --model-root N:/models/onnx/nemo/parakeet-tdt-0.6b-v3-onnx --audio N:/github/asrjs/browser-demo/src/assets/Harvard-L2-1.ogg --expected-phrase 'The boy was there when the sun rose' --encoder-precision fp16 --output tools/data/results/harness/local-public-hybrid-v3-final-2026-10-03.json
```

Choose a fresh output filename for every rerun: existing receipts are never
overwritten. Expected text must match the whole normalized transcript, not a
substring. Normalization ignores case/punctuation only; this is a smoke check,
not WER normalization or original-engine numerical parity.

Final receipt `local-public-hybrid-v3-final-2026-10-03.json`: PASS in Chrome
154.0.8037.93; two identical transcripts, WebGPU encoder + WASM decoder/joint,
awaited disposal, no blocked requests/page errors, inputs unchanged, owned
preview process stopped and listener refused connections. Observed engine and
bundle paths were re-fetched from the immutable server snapshot and checked
against byte counts/SHA-256 (not CDP capture of every original response body).
Model identity is caller-selected with recorded file hashes, not certification
against an upstream stock-v3 manifest. Load7482ms; first2068ms/second261ms wall
times are observations, not matched performance optimization evidence.

TypeScript check,69 targeted simulated/component tests,7 Node harness guards
and5 Python preflight/probe tests pass. Earlier negative receipts are retained:
missing canonical provider forwarding, unavailable pthread CDP body IDs and an
inconclusive Python socket closure probe were harness/API issues, not evidence
of model parity failure. Cleanup now requires Node ECONNREFUSED; unknown probe
errors fail closed.

Scope remains stock-v3 control with existing fp16 assets: no Orukeet fp16
promotion, numerical parity, cancellation, memory/VRAM or soak acceptance.
The sibling app source remains untouched and still needs authorized integration
of its local `wasmPaths` option. Next model task remains the earliest Orukeet
fp32 feature divergence, followed by encoder/token/text and browser gates.

### Identical windowed-frame FFT control (2026-10-03)

`processor_export_stages.py` now separates centered constant padding, frame
gathering and window multiplication from FFT execution. Native frames agree
byte-for-byte with independent NumPy reconstruction; native `torch.fft.rfft`
on these exact frames reproduces the captured official STFT bytes. The ONNX
framing graph reproduces all2,079,232 windowed-frame values byte-for-byte.
ONNX DFT on those identical frames reproduces ONNX STFT bytes exactly, but
differs from native FFT: max2.086162567138672e-6, mean6.686923942797317e-9.
This isolates the observed difference to FFT/backend arithmetic for this
fixture, rather than preemphasis, frame gathering or window multiplication.
It does not identify a particular FFT implementation instruction as defective.

Receipt: `tools/data/results/orukeet/nemo-processor-windowed-frame-control-2026-10-03.json`.
Frame f32le SHA256 remains
`f6d9996ff874003c84efd270f8558681a5c16303a913c185ea68a6553d3e1810`;
framing graph9255 bytes SHA256
`642ab73f89eeec902c75f41a7f72ca5ee0ea93b7dad0785b64735ce0e8acd536`;
DFT graph204 bytes SHA256
`8708de7e907aca3e811bb1b7f440dd0efbe4aa90c0d78d13e62c71cd5f079cba`.
Run the existing `probe_nemo_processor.py` with pinned packed config/capture
hashes, `--onnx-probe --stage-probe --pairwise-normalization` and a fresh output.
No model weights are loaded by this config-only diagnostic.

Two actual CPU synthetic operator tests pass (aligned1600 and uneven1773
samples; changed oracle rejected), plus5 stage-comparator regressions.
The first synthetic fixture used no preemphasis, which pruned ONNX's `valid`
input and failed before the new control. That fixture was reverted and rebuilt
with the actual preemphasis contract; it was not a production numerical defect.

Full pairwise processor remains FAIL/CLI exit1 with76 feature mismatches,
max0.00012484192848205566 at unchanged atol1e-5/rtol1e-4. No production processor,
preset, precision or tolerance changed. Next evaluate a portable FFT arithmetic
candidate on these same frames, retaining native-oracle and full-feature gates;
do not compensate with per-model correction constants. Encoder/token/text and
WASM/WebGPU Orukeet acceptance remain open.

### Exporter-pruned processor input fix (2026-10-03)

Disabled preemphasis exports an identity graph without the unused declared
`valid` input. The previous diagnostic sent `valid` anyway and failed with
`Invalid input name: valid`. A regression reproduced that error before the fix.
The session now receives only actual graph inputs, after verifying that caller
feeds exactly match declared input names and contain every required graph input.
Receipts list the actual retained names. This is not a missing-input fallback.

Three actual synthetic operator tests now pass, including disabled preemphasis
(`inputs=['pcm']`) and enabled preemphasis (`inputs=['pcm','valid']`), aligned
and uneven lengths, and corrupted-oracle rejection. Five comparator and9 CLI
preflight tests also pass. The fresh full fixture receipt
`nemo-processor-pruned-input-control-2026-10-03.json` retains native byte-exact
PASS and full pairwise ONNX FAIL:76 mismatches, max0.00012484192848205566;
baseline and pairwise graph hashes are unchanged. Production math and numerical
tolerances were not edited.

Experiment audit also found that the proposed fp64 DFT control already exists
in `probe_light_dft_features.py` and has32 failed downstream feature cells.
The redundant uncommitted probe/test additions were reverted rather than
rerunning them or adding another inference path. Preserve those earlier
negative receipts and the bounded algorithm-search rule above.

### Real browser call 20 same-input head control (2026-10-03)

Replayed only captured call 20 through the existing persistent fp32 head with
native ORT 1.29.0 CPUExecutionProvider; no alternate decoder, model restore,
browser rerun or acceptance change. Inputs came from
`browser-real-public-hybrid-jfk-2026-10-03.json` and the frozen control
`browser-live-head-wasm-jfk-2026-10-02.json` under
`tools/data/results/orukeet/`. The graph was
`persistent-single-step-head-2026-10-02/decoder_joint-model.onnx`.

For each capture independently, native versus WASM scores pass unchanged
atol1e-5/rtol1e-4: real-input raw and normalized max1.52587890625e-5;
frozen-input raw max1.9073486328125e-5, normalized max1.52587890625e-5;
all four comparisons have zero mismatches across8198 values. Comparing the
two WASM normalized outputs instead fails three coordinates, first8195,
max0.0005776882171630859. Their targets, lengths and both input recurrent
states are byte-identical; only encoder frame differs (max1.4565885066986084e-6,
mean3.285514083017915e-7; its numerical input gate passes). This isolates this
control's score sensitivity to the changed encoder input, not evidence of a
same-input WASM head defect. It does not prove all head calls or full ASR.

Separately, `compare_live_head` still requires frozen encoder bytes even in
real-encoder mode: all46 real frames pass numerical gates but fail this exact
check. Keep frozen byte-exact acceptance intact; add an explicit real-input
comparison scope with regressions before using that diagnostic again. Removing
that inappropriate frozen requirement will not remove the independently failed
three score coordinates. Overall real-path and feature acceptance remain FAIL.
Next return to the earliest unresolved feature/encoder boundary, preserving
these same-input head controls and all existing tolerances.

### Explicit live encoder comparison scope (2026-10-03)

Implemented the boundary correction above: `compare_live_head` defaults to
`encoder_boundary='frozen'`; the real-encoder browser mode explicitly selects
`'real'`. Both record frame byte equality and require unchanged numerical
atol1e-5/rtol1e-4. Frozen mode additionally requires frame byte equality;
targets/lengths and traced-versus-unobserved state controls stay exact in both.
Unknown scopes fail closed. Three new regressions failed before implementation
and now pass;43 live-head/browser replay tests and Python syntax checks pass.
This is a comparator repair, not new browser inference or model acceptance.
The prior trace's score failure and unresolved feature gate are not waived.
An attempted offline full-trace re-evaluation did not yield observable command
output, so its result is UNVERIFIED and no old receipt was overwritten.

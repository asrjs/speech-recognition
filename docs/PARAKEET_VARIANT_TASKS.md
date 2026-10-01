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

Task P01 is complete; P02 and P03 are partial. Existing Nemotron changes on `main` are preserved.
Orukeet remains an experimental candidate until P08 passes. Resume the first
unfinished task and record commands, result files, and blockers here.
The recommended whole-product goal is `GOAL_PROMPT_RECOMMENDED.md`; this task
list is one workstream, not the complete web/Node/Bun library backlog.

## Ordered tasks

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
  Next: correlate signed filterbank-weighted power error and active FFT bins at
  failing coordinates in both low and high mel bands, not another stats-only
  or unproven production replacement. Existing P02-P08 gates remain open.
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
  Pure silence and long-audio fixtures remain to be captured; the speech/silence
  continuation is recorded above.
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

## Upstream recipes to inspect during export

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

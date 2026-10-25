# Model Debugging Skill

This is a local `@asrjs/speech-recognition` engineering playbook, not a globally installed Codex
skill.

Use it when working on:

- model-family implementation bugs
- transcript quality regressions
- parity mismatches against native/original stacks
- resampler, tokenizer, preprocessor, or decoder-loop issues
- long-form and realtime transcript-merging regressions

## Discovery prerequisite

Start at the [task-to-tool map](../README.md#task-to-tool-map), then read the
matching [family playbook](playbooks/README.md) and [reference guide](reference/README.md).
This file is available in the repository even when the harness lists no matching
global skill. Do not install another skill to discover these tools.

Before a new port or a blocker claim, inspect existing family implementations,
local original weights, ONNX exports, reference environments and receipts.
Qwen3-ASR, X-ASR, GigaAM and NeMo already have family-specific porting helpers.
Read the official inference source and compare existing onnx-asr/sherpa ports.
Use those ports as references, not replacements for the official oracle.

For Whisper self-exports, inspect the actual decoder graph input names before
reusing a reference runner or runtime feed builder. Some four-graph exports
declare rank-one INT64 `cache_position`; bind it only when declared, starting
at the decoder self-attention KV sequence length and advancing once per input
token. Decoder initialization and alignment graphs have separate contracts.
Preserve failures caused by missing runner inputs as invalid-harness evidence,
then compare full native and ONNX token/text output with the corrected runner.
An ONNX path check or successful session creation is structural acceptance only.

Checkpoint generation metadata can override model-config defaults. The pinned
original Whisper Large v3 checkpoint has model-config PAD 50256 but tokenizer
and generation-config PAD 50257. Preserve both values in provenance and use the
effective generation value in exported generation metadata. For a metadata-only
repair, retain the original manifest and independently verify graph/sidecar
hashes unchanged; do not claim that metadata equality alone proves ASR parity.

On Windows, a restore that appears stuck may be retrying a denied Numba cache
write. Capture periodic Python stacks before calling it a model incompatibility.
The observed chain is `librosa` -> `numba.core.caching.ensure_cache_path` ->
`tempfile._mkstemp_inner`; Windows can repeatedly retry when `os.access` reports
writability but the actual file open is denied. Pass a process-local
`NUMBA_CACHE_DIR` under writable task temporary storage before importing
Torch/NeMo/librosa. Verify a temporary-file write there, preserve the failed
attempt's logs, and retry with a bounded watchdog and periodic
`faulthandler.dump_traceback_later` output. Do not change global environment
settings or installed package files. This resolved the 2026-10-24 Canary export
restore retry; successful restore alone is not export, inference or parity PASS.

Check captured tensor shapes and layouts before reshaping arrays. Keep raw
log-mel separate from normalized features. A failed import in the default
Python interpreter does not prove that a model's local environment is absent.
Check the [Python environment inventory](../../docs/WORKSPACE_CONTEXT.md#python-reference-environment-inventory)
and [environment-selection rules](../README.md#python-reference-environments).
Inspect Conda and uv as well as project venvs. Record the imported module path:
the project NeMo venv overlays NeMo 3.0.0 on Conda's NeMo 2.4.0 stack, and a
`.venv` directory does not establish isolation or dependency consistency.

## Goals

Keep debugging disciplined and reproducible:

1. reproduce with a stable audio fixture
2. compare against a known-good native/reference stack
3. reduce the problem to the earliest mismatching stage
4. capture the result in JSON so the issue can be replayed without rerunning
   the whole stack

## Working Rules

### 1. Start with deterministic inputs

Prefer:

- WAV fixtures from `tools/data/fixtures/audio`
- pre-resampled shared audio when comparing runtimes
- saved JSON traces when validating mergers or transcript assembly

### 2. Separate audio prep bugs from model bugs

Before touching decoder logic:

- compare original audio vs pre-resampled audio
- compare browser audio prep vs Node deterministic prep
- compare `AudioContext`-based decode against WAV-parser + linear resample

If Node deterministic prep is correct and browser output is not, the issue is
probably in audio preparation, not the model implementation.
Confirm the effective PCM before drawing that conclusion: matching source WAV
hashes or a deterministic resampler do not establish equal model inputs.

### 2.5. Match the scoring pipeline before calling it a regression

When transcript quality looks worse:

- first compare transcript text against the known-good stack directly
- then confirm both sides use the same text normalizer for WER/CER
- only treat the issue as a recognition regression after transcript parity and metric parity disagree

Different normalizers can create false alarms even when the underlying transcripts are the same.

### 3. Compare stage by stage

Use the reference scripts and `@asrjs/speech-recognition` scripts to compare:

- normalized / resampled PCM
- preprocessor outputs
  - include valid-length mode and normalization mode, not just tensor shape
- encoder outputs or shapes
- decoder/joint outputs
- token ids and token pieces
- timestamps
- final transcript text

When the browser session creates but tokens are corrupted or unexpectedly
slow, verify the ORT package entry point before changing model code. Keep
`onnxruntime-web/webgpu` mapped to the WebGPU bundle and use the all bundle only
for the plain package import. See
[ort-webgpu-entrypoint.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\ort-webgpu-entrypoint.md)
for the reproducible Qwen 1.29 case.

For graph-capture experiments, keep the request opt-in and model-specific.
Record the exact partitioning error, dynamic dimensions, cold session-create
time, warmed inference time, token parity, and disposal result. Retry without
capture only for a graph-capture partitioning error; never hide unrelated
session failures or promote a fallback as a capture speedup.

For greedy decoder graph surgery, first measure the output-transfer share. An
`ArgMax` output such as `next_token_id` can avoid downloading a full logits
row, but it may introduce an expensive provider reduction or inhibit graph
fusion. Generate candidates with
`reference/qwen3-asr-0.6b/append_argmax_output.py` without overwriting the
original graph or external data, then compare exact token parity, provider
partitioning, warmed `session.run()` time, output handling, load time, and
disposal. Keep scalar-output support as an optional compatibility path and
reject it when total latency regresses, even if the readback phase improves.

### 4. Save artifacts

Write results into `tools/data/results/...` whenever the run is useful for:

- documenting a bug
- confirming a fix
- comparing multiple pipelines

### 5. Prefer Node first for deterministic model debugging

For Parakeet/NeMo quality work:

- reproduce in Node first
- use local direct artifacts when possible
- freeze and hash the official engine's effective float32 PCM when isolating
  model parity; do not assume linear resampling matches the official resampler
- record rate, channels, sample/valid lengths, padding and f32le identity; avoid
  a PCM16 roundtrip that changes the captured input
- validate normal source decoding/resampling independently of frozen-PCM runs
- do not assume a shared `nemo128` processor is correct without checking the
  saved native preprocessor outputs first
- treat the shared pure-JS NeMo frontend as the expected implementation path
  for new ports
- extend the JS frontend when the model contract differs; do not plan around
  exporting `nemo80.onnx` or `nemo128.onnx` preprocessors

Then move the same case into the browser demo.
See the Orukeet same-input control in `playbooks/librivox-domain-parity.md`:
69 versus 76 tokens became 69 exact tokens in both native Node and Bun when
given the unchanged official PCM. This is output-only evidence, not feature
numerical parity or an accepted replacement resampler.

### 6. Treat recurrent-state placement as a model-specific experiment

For Parakeet TDT WebGPU work, the library's `decoderStateOutputLocation` option
is deliberately opt-in. Keep decoder logits on CPU while testing only
`output_states_1` and `output_states_2` on `gpu-buffer`; this isolates the
state-transfer hypothesis and keeps vocabulary/duration post-processing
observable. A state-placement candidate is promotion-ready only when the
library entry point shows exact token parity, stable repeated transcriptions,
safe replacement/disposal, and a measured latency/RTFx win against the same
artifact and audio control. If a host lacks a second WebGPU adapter, record the
typed `WEBGPU_NO_ADAPTER` boundary instead of treating it as a model failure.

When probing ORT WebGPU buffer-cache modes, include the runtime default as an
explicit control and compare `bucket`, `simple`, `disabled`, and `lazyRelease`
on the same model artifact, audio fixture, browser/adapter, backend placement,
and warmed-repeat schedule. Capture latency/RTFx, load time, memory, and exact
transcript parity. Do not promote a cache mode from one faster run; preserve
the default or keep the candidate opt-in until the result is repeatable and
model-specific.

For lifecycle soaks, make teardown observable: record model/runtime disposal
errors instead of swallowing them, and distinguish a browser heap sample from
actual GPU resource reclamation. Report load time separately from warmed
transcription latency, and require exact parity on every repeated run before
considering a state-placement change for promotion.

### 7. Prove incremental frontend work before optimizing the encoder

Streaming frontends must not recompute the complete accumulated waveform on
every chunk. First establish the frame contract against a full-buffer
reference, including `snip_edges`, left/right reflection, frame count, feature
layout, and finalization behavior. For a frontend whose right-edge padding
depends on future samples (for example Kaldi `snip_edges=false`), emit only
fully sample-backed frames during ordinary pushes, retain the smallest bounded
raw-audio tail that covers the next frame, and flush the reflected boundary
frames exactly once when `final=true`.

The incremental path must be tested with uneven chunk sizes, tiny initial
chunks, a final empty push, and a deterministic waveform. Compare every
feature value with one full-buffer run (`maxAbs` must be within the documented
floating-point tolerance) before measuring speed. The benchmark must include
the same residual costs on both sides—such as cumulative audio-copy work—when
isolating frontend speed. If the storage strategy is also being optimized,
label the result as a combined frontend/storage comparison and keep a separate
frontend-only control; never report either microbenchmark as end-to-end RTFx.
Use the X-ASR reference harness as the template:

```powershell
npm run benchmark:x-asr-frontend -- --runs=3 --durations=2,10 --json
```

Record baseline/candidate medians for both the frontend-only control and any
combined storage comparison, along with the chunk schedule and parity error,
in `docs/reports/`. Only after this CPU contract
is proven should the same chunk schedule be run through the real browser
encoder-cache path; report that browser run as a separate artifact-parity
checkpoint unless a pre-change browser control exists.

## Recommended Entry Points

- [README.md](N:\github\asrjs\speech-recognition\tools\model-debugging\README.md)
- [librivox-domain-parity.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\librivox-domain-parity.md)
- [node-asrjs-nemo-inspect.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-asrjs-nemo-inspect.mjs)
- [node-compare-transcript-jsons.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-compare-transcript-jsons.mjs)
- [reference/medasrjs/upstream-tests/README.md](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\medasrjs\upstream-tests\README.md)

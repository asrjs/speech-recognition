# Debugging Scripts

These are the active `@asrjs/speech-recognition`-native debugging scripts we want to keep using
and improving.

Unlike `reference/`, this folder should trend toward:

- stable CLI arguments
- direct `@asrjs/speech-recognition` imports
- reproducible output JSON
- clear troubleshooting purpose

## Current Scripts

- [port_manifest.py](./port_manifest.py)
  - one-command local artifact/fixture/oracle digest preflight and declared runs
  - `runner: browser` is companion warm-up only (`comparesOracle: false`)
  - explicit `runner: browser-oracle` delegates to a model-owned adapter; current
    Orukeet fp32 adapter invokes the existing automatic Chrome runner, compares
    two full token/text outputs, PCM/artifacts, pinned delivered HTML/JS/WASM
    and awaited cleanup; it does not certify stage numerics or native streaming
  - requires an already-running demo and installed Chrome/Playwright; no installs
  - run from the repository root with a fresh output filename:

    ```powershell
    python tools/model-debugging/scripts/port_manifest.py --manifest tools/model-debugging/ports/oruk-orukeet.port.json --run chrome-oracle-jfk-webgpu-hybrid --output tools/data/results/orukeet/manifest-chrome-fresh-run.json
    ```

  - [demo startup](../../../examples/demo/README.md#automatic-chrome-model-test-and-json-capture)
    and [model-owned contract](../reference/orukeet/README.md#manifest-driven-chrome-oracle-smoke)

- [node-asrjs-nemo-inspect.mjs](./node-asrjs-nemo-inspect.mjs)
  - runs Parakeet/NeMo TDT through `@asrjs/speech-recognition` in Node
  - accepts a local model directory and audio fixture
  - writes detailed JSON for parity/debugging
  - useful for transcript, token, timing, and audio-prep investigations
  - opt-in `--cancel-reuse` requires an official oracle: cancel at an unfinished
    decode progress event, require `PipelineAbortedError`, then compare two
    fresh-signal transcriptions on the same public model/session handles
  - use `--input-mode oracle-pcm` to isolate model lifecycle on exact captured
    PCM; Node/Bun native CPU and WASM are separate runs, not interchangeable
  - receipts label output-only acceptance; this does not certify intermediate
    features/states, memory stability or browser execution
- [qwen3-asr-node-wasm-benchmark.mjs](../../../tests/smoke/qwen3-asr-node-wasm-benchmark.mjs)
  - existing offline Qwen library CLI under `tests/smoke/`, not a second decoder
  - accepts `--model-dir`, `--audio`, `--backend wasm|webgpu`, `--encoder dynamic|static-t1100`,
    `--dtype fp16|fp32`, `--warmup`, `--runs`, `--language` and `--reference`
  - starts from local graphs/tokenizer and calls the library; no weight download
  - use `--help` to inspect the current CLI, then follow the
    [Qwen reference guide](../reference/qwen3-asr-0.6b/README.md#reuse-the-existing-offline-library-runner)
  - historical output receipts and CLI help are not current model acceptance
- [node-asrjs-parakeet-realtime-parity.mjs](./node-asrjs-parakeet-realtime-parity.mjs)
  - runs the Parakeet realtime RNNT preset through `@asrjs/speech-recognition` in Node/WASM
  - compares the runtime output against `parakeet-realtime-eou-120m-v1-reference.json`
  - automatically falls back to `parakeet-realtime-eou-120m-v1-reference.json.gz` if the raw JSON has been packed out of the repo
  - uses `PARAKEET_MODEL_DIR` or `--model-dir` for the local ONNX directory; the bundled fallback path is only a placeholder
  - reports visible text, raw text with `<EOU>`, token ids, and EOU/EOB signal parity
- [node-asrjs-medasr-compare.mjs](./node-asrjs-medasr-compare.mjs)
  - compares `@asrjs/speech-recognition` MedASR output against the original `medasrjs` Node wrapper
  - uses the same local ONNX/tokens assets for both stacks
  - runs against labeled WAV fixtures and writes machine-readable JSON
  - useful for transcript-quality regression checks before stage-level parity work
  - reports heuristic WER/CER only; do not compare those numbers directly with Python leaderboard-style benchmarks
- [node-canary-js-frontend-parity.mjs](./node-canary-js-frontend-parity.mjs)
  - compares a JS mel frontend against `tools/data/results/canary/canary-180m-flash-reference.json`
  - automatically falls back to `canary-180m-flash-reference.json.gz` if the raw JSON has been packed out of the repo
  - auto-selects the in-repo `asrjs` valid-length and normalization contract from the reference model id
  - pass `--frontend asrjs` explicitly; the CLI defaults to the sibling `meljs` implementation
  - read the parser rather than invoking `--help`, which this legacy CLI does not implement
  - consumes RNNT/Canary `audio.waveform` and `preprocessor.features` references,
    not the generic TDT inspector's `samples/stages` capture envelope
  - use this when extending the shared pure-JS NeMo frontend for a new model port
  - supports `asrjs`, `meljs`, and `parakeet.js`
  - accepts `--valid-length-mode onnx|centered` and `--normalization per_feature|none` when you want to override the auto-detected `asrjs` frontend contract explicitly
  - reports feature-length deltas, max/mean absolute difference, RMSE, and the largest mismatching bins/frames
  - useful before extending the shared pure-JS NeMo frontend for a future NeMo-family port
- [node-reference-artifacts.mjs](./node-reference-artifacts.mjs)
  - packs large debugging/reference JSON files into `.json.gz`
  - restores them on demand for deep parity work
  - supports `status`, `pack`, and `unpack` commands for the current heavy Canary, MedASR, and Parakeet artifact sets
- [node-compare-transcript-jsons.mjs](./node-compare-transcript-jsons.mjs)
  - compares two transcript JSON outputs by shared `file` key
  - supports `*.rows[]` debug JSON or plain benchmark arrays
  - useful when two stacks appear to disagree on WER, and you want to prove whether the transcript text actually changed
- [node-compare-stage-captures.mjs](./node-compare-stage-captures.mjs)
  - compares reference and implementation captures by stable `sample_id`,
    requiring matching audio identity on both sides
  - reports shape/length/dtype metadata, max/mean absolute error, RMSE,
    cosine similarity, argmax agreement, top-k values, token/EOS/text parity,
    and the first mismatch
  - exposes the earliest failed stage without aligning rows by order or text
- [score-transcripts-python.py](./score-transcripts-python.py)
  - scores a transcript JSON with the copied MedASR Python normalizer stack
  - accepts dotted `--prediction-path` and `--reference-path`
  - useful when you want Node-produced results to be directly comparable with Python MedASR benchmark summaries
  - if auto-detection misses the leaderboard normalizer, pass `--open-asr-leaderboard-dir`

### Stage-capture contract

The generic comparator expects a small JSON envelope:

```json
{
  "schema_version": 1,
  "stage_order": ["features", "logits"],
  "samples": [
    {
      "sample_id": "stable-clip-id",
      "audio": { "sha256": "..." },
      "tokens": [101, 102],
      "transcript": "reference text",
      "eos": 2,
      "stages": {
        "features": { "data": [0.1], "shape": [1, 1], "dtype": "float32" },
        "logits": { "data": [0.2], "shape": [1, 1], "dtype": "float32" }
      }
    }
  ]
}
```

Run it after the native/reference and library captures exist:

```powershell
node node-compare-stage-captures.mjs `
  --reference reference.json `
  --candidate asrjs.json `
  --output stage-parity.json
```

This is a diagnostic report, not a quality score. The upstream/reference
capture remains an implementation oracle, while human or benchmark labels
remain separate quality evidence.

Plain JSON and `.json.gz` inputs are supported. `samples` and numerical stage
data must be nonempty; a sample with no stages is rejected. Both captures must
declare matching audio identity (`audio.sha256` is preferred). Missing identity
fails, even when both captures omit it. Optional reference `stage_order`
declares unique required stage names in processing order: a stage absent from
both captures still fails. Additional stages follow reference insertion order,
then candidate-only stages; the earliest failure is not selected alphabetically.
Use this to report the real first divergence (features before encoder before
decoder/joint), with explicit numerical tolerances. Unit tests and emulated
sessions cannot replace native/WASM/real-browser numerical and token/text parity.

When supplied, `token_duration` and `token_frames` are exact integer arrays
with one value per token; changed or one-sided missing metadata fails. A valid
silence hypothesis may have empty token/duration/frame arrays, but still needs
nonempty captured features/encoder stages and the same audio identity.

## Candidate artifact audit

- node-audit-onnx-artifact.mjs audits a local ONNX bundle without model-hub
  access.
- It hashes every file, checks likely external-data sidecars, and creates each
  graph through the native CPU execution provider.
- Add `--x-asr-contract` for a local X-ASR-zh-en deployment. This validates all
  four chunk directories, matching encoder/decoder/joiner graphs, token files,
  and native CPU graph loading without downloading anything:

  `node node-audit-onnx-artifact.mjs --model-dir <deployment/models> --recursive --x-asr-contract`

- It records graph input/output names plus ORT-discovered dtypes/shapes, and
  checks the minimum encoder/decoder/joiner boundary. It fails by default when
  a graph cannot be loaded.
- Pass `--operator-inventory` to add ONNX IR/opset versions, operator/domain
  counts, and conservative provider-risk hints from the local Python `onnx`
  module. Use `--require-operator-inventory` when the audit environment must
  have that parser; otherwise an unavailable parser is reported without
  invalidating the native graph audit.
- Pass `--whisper-contract` to classify splitgraph causal alignment and merged
  `cross_attentions.*` capability. Use `--require-causal-alignment` or
  `--require-merged-cross-attention` as publish gates; legacy artifacts report
  a warning instead of being treated as attention-aligned.

## Metric Caveat

Before treating WER/CER deltas as a model-quality regression, make sure both sides use the
same normalization pipeline. A simple lowercase-and-punctuation-strip score is useful for quick
local checks, but it is not directly comparable to `open_asr_leaderboard`-style scoring.

## Expected Direction

New scripts in this folder should ideally:

- work from repo-relative fixtures by default
- support explicit `--model-dir` / `--output`
- write machine-readable JSON
- be useful for debugging, not just one-time experiments

# Debugging Scripts

These are the active `@asrjs/speech-recognition`-native debugging scripts we want to keep using
and improving.

Unlike `reference/`, this folder should trend toward:

- stable CLI arguments
- direct `@asrjs/speech-recognition` imports
- reproducible output JSON
- clear troubleshooting purpose

## Current Scripts

- [node-asrjs-nemo-inspect.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-asrjs-nemo-inspect.mjs)
  - runs Parakeet/NeMo TDT through `@asrjs/speech-recognition` in Node
  - accepts a local model directory and audio fixture
  - writes detailed JSON for parity/debugging
  - useful for transcript, token, timing, and audio-prep investigations
- [node-asrjs-parakeet-realtime-parity.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-asrjs-parakeet-realtime-parity.mjs)
  - runs the Parakeet realtime RNNT preset through `@asrjs/speech-recognition` in Node/WASM
  - compares the runtime output against `parakeet-realtime-eou-120m-v1-reference.json`
  - defaults to the in-repo JS frontend path used by the preset
  - reports visible text, raw text with `<EOU>`, token ids, and EOU/EOB signal parity
- [node-asrjs-medasr-compare.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-asrjs-medasr-compare.mjs)
  - compares `@asrjs/speech-recognition` MedASR output against the original `medasrjs` Node wrapper
  - uses the same local ONNX/tokens assets for both stacks
  - runs against labeled WAV fixtures and writes machine-readable JSON
  - useful for transcript-quality regression checks before stage-level parity work
  - reports heuristic WER/CER only; do not compare those numbers directly with Python leaderboard-style benchmarks
- [node-canary-js-frontend-parity.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-canary-js-frontend-parity.mjs)
  - compares a JS mel frontend against `tools/data/results/canary/canary-180m-flash-reference.json`
  - automatically falls back to `canary-180m-flash-reference.json.gz` if the raw JSON has been packed out of the repo
  - auto-selects the in-repo `asrjs` valid-length and normalization contract from the reference model id
  - use this when extending the shared pure-JS NeMo frontend for a new model port
- [node-reference-artifacts.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-reference-artifacts.mjs)
  - packs large debugging/reference JSON files into `.json.gz`
  - restores them on demand for deep parity work
  - supports `status`, `pack`, and `unpack` commands for the current heavy Canary and MedASR artifact sets
  - supports `asrjs`, `meljs`, and `parakeet.js`
  - accepts `--valid-length-mode onnx|centered` and `--normalization per_feature|none` when you want to override the auto-detected `asrjs` frontend contract explicitly
  - reports feature-length deltas, max/mean absolute difference, RMSE, and the largest mismatching bins/frames
  - useful before extending the shared pure-JS NeMo frontend for a future NeMo-family port
- [node-compare-transcript-jsons.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-compare-transcript-jsons.mjs)
  - compares two transcript JSON outputs by shared `file` key
  - supports `*.rows[]` debug JSON or plain benchmark arrays
  - useful when two stacks appear to disagree on WER, and you want to prove whether the transcript text actually changed
- [score-transcripts-python.py](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\score-transcripts-python.py)
  - scores a transcript JSON with the copied MedASR Python normalizer stack
  - accepts dotted `--prediction-path` and `--reference-path`
  - useful when you want Node-produced results to be directly comparable with Python MedASR benchmark summaries
  - if auto-detection misses the leaderboard normalizer, pass `--open-asr-leaderboard-dir`

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

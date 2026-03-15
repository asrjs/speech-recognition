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
- [node-asrjs-medasr-compare.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-asrjs-medasr-compare.mjs)
  - compares `@asrjs/speech-recognition` MedASR output against the original `medasrjs` Node wrapper
  - uses the same local ONNX/tokens assets for both stacks
  - runs against labeled WAV fixtures and writes machine-readable JSON
  - useful for transcript-quality regression checks before stage-level parity work
  - reports heuristic WER/CER only; do not compare those numbers directly with Python leaderboard-style benchmarks
- [node-canary-js-frontend-parity.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-canary-js-frontend-parity.mjs)
  - compares a JS mel frontend against `tools/data/results/canary/canary-180m-flash-reference.json`
  - supports `asrjs`, `meljs`, and `parakeet.js`
  - auto-selects the in-repo `asrjs` valid-length contract from the reference model id
  - accepts `--valid-length-mode onnx|centered` when you want to override that default explicitly
  - reports feature-length deltas, max/mean absolute difference, RMSE, and the largest mismatching bins/frames
  - useful before switching Canary or future NeMo AED ports away from `nemo128.onnx`
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

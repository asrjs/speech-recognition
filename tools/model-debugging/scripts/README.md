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

## Expected Direction

New scripts in this folder should ideally:

- work from repo-relative fixtures by default
- support explicit `--model-dir` / `--output`
- write machine-readable JSON
- be useful for debugging, not just one-time experiments

# NeMo RNNT Porting

Use this playbook when porting a NeMo RNNT model into
`@asrjs/speech-recognition`, especially when encoder/decoder export looks right
but the end-to-end transcript still differs from the original stack.

## Core Rule

Do not assume every 128-bin NeMo model uses the same frontend contract.

Also, do not treat `nemo80.onnx` or `nemo128.onnx` export as a required
deliverable for a successful port in this repo. The shared pure-JS NeMo
frontend is the expected implementation path, and we extend it when a model
needs a different contract.

Confirm all of these before debugging the RNNT loop:

- whether the model expects raw log-mel features or per-feature normalized mels
- whether valid feature length should be the centered frame count or the
  shorter ONNX-style length
- whether a shared `nemo128.onnx` artifact is actually valid for this model

`nvidia/parakeet_realtime_eou_120m-v1` is the current example that forced this
rule:

- encoder and decoder ONNX exports were correct
- the RNNT loop was correct
- the remaining bug was the frontend contract
- the model expects centered raw log-mel features, not the normalized
  `nemo128` contract reused by earlier ports

## Recommended Flow

### 1. Capture a native reference first

Save waveform, preprocessor features, encoder outputs, manual greedy token ids,
and final text in one JSON artifact.

Current reference entry point:

- [generate_parakeet_realtime_reference.py](../reference/parakeet-realtime-eou-120m-v1/generate_parakeet_realtime_reference.py)

### 2. Export encoder and decoder artifacts before frontend work

Get encoder and decoder/joint parity stable first.

Current reference entry points:

- [export_parakeet_realtime_onnx.py](../reference/parakeet-realtime-eou-120m-v1/export_parakeet_realtime_onnx.py)
- [verify_parakeet_realtime_onnx.py](../reference/parakeet-realtime-eou-120m-v1/verify_parakeet_realtime_onnx.py)

If verification only passes when you feed saved reference features into the
encoder, the remaining bug is probably frontend parity, not RNNT decoding.

### 3. Compare the frontend directly against the saved reference

Use a JS frontend parity helper before touching runtime decode logic.

Current helper:

- [node-canary-js-frontend-parity.mjs](../scripts/node-canary-js-frontend-parity.mjs)

This script now auto-selects the in-repo `asrjs` frontend contract from the
reference model id, including both valid-length mode and normalization mode.

### 4. Only then run end-to-end Node parity

Once frontend parity is credible, run the full local Node/WASM stack against
the same reference JSON.

Current helper:

- [node-asrjs-parakeet-realtime-parity.mjs](../scripts/node-asrjs-parakeet-realtime-parity.mjs)

## Current RNNT Lessons

- A shared NeMo mel frontend is not automatically portable across RNNT and TDT
  families.
- The PyTorch ONNX exporter still fails on some NeMo STFT preprocessor paths,
  so a JS frontend can be the correct runtime path even when encoder and
  decoder ship as ONNX.
- Put frontend behavior in shared model config, not one-off scripts.
- For new NeMo RNNT ports here, plan to extend the shared JS frontend rather
  than exporting `nemo80.onnx` or `nemo128.onnx`.

## Good End State

Leave behind:

- a saved native reference JSON under `tools/data/results/...`
- stable ONNX export and verify scripts
- a Node parity script with machine-readable JSON output
- model-family config that captures the real frontend contract
- at least one focused regression test for the shared frontend behavior you had
  to add

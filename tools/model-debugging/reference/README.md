# Reference Script Sets

`reference/` keeps copied or legacy script sets that are still useful for
debugging, parity checks, and troubleshooting.

They are intentionally preserved as references rather than fully normalized
tooling.

## Sets

### `asr-merger-test`

Useful for:

- offline JSON replay
- long-form window simulation
- transcript merger debugging

### `hf-parakeet-port`

Useful for:

- Python/Node/native parity checks
- NeMo/ONNX inspection
- resampling comparisons
- sentence-level regressions

### `legacy-nemo-tdt-tests`

Useful for:

- older executor and transcript-behavior expectations
- porting assertions into modern `@asrjs/speech-recognition` tests

### `medasrjs`

Useful for:

- ONNX export checks
- runtime compatibility diagnostics
- benchmark helpers
- FP16 investigation
- copied reference tests and fixtures for stage-by-stage MedASR parity troubleshooting

Start with:

- [upstream-tests/README.md](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\medasrjs\upstream-tests\README.md)

### `canary-180m-flash`

Useful for:

- NeMo-side stage capture for Canary AED
- ONNX export wrappers for encoder/decoder and optional preprocessor
- tokenizer metadata extraction
- FP16 conversion
- INT8 conversion
- ONNX-vs-PyTorch parity checks for prompt ids, token ids, and final text
- documenting a modern NeMo AED porting workflow

Start with:

- [canary-aed-porting.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\canary-aed-porting.md)
- [canary-180m-flash/README.md](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\canary-180m-flash\README.md)

Important current rule:

- do not treat `nemo128.onnx` as a required artifact for Canary-style ports
- a JS frontend or feature extractor can be the preferred path when it matches
  the saved NeMo reference
- for new NeMo-family work, prefer extending the shared pure-JS frontend over
  trying to export `nemo80.onnx` or `nemo128.onnx`

### `parakeet-realtime-eou-120m-v1`

Useful for:

- NeMo RNNT stage capture
- encoder and decoder/joint ONNX export
- raw-text vs visible-text `<EOU>` parity checks
- JS frontend validation for centered raw log-mel features
- documenting the current NeMo RNNT porting workflow

Start with:

- [nemo-rnnt-porting.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\nemo-rnnt-porting.md)
- [parakeet-realtime-eou-120m-v1/README.md](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\parakeet-realtime-eou-120m-v1\README.md)

Important current rule:

- do not reuse a generic normalized `nemo128.onnx` artifact for this model
- the current runtime path should use the shared JS frontend with centered
  raw-log NeMo settings
- more generally, do not treat `nemo80.onnx` or `nemo128.onnx` export as a
  porting goal for new NeMo models in this repo

## Rule

Keep copied scripts here if they still teach us something.

Move or rewrite them into `tools/model-debugging/scripts/` only when they have
been normalized around the current `@asrjs/speech-recognition` architecture.

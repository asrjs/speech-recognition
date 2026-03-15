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

## Rule

Keep copied scripts here if they still teach us something.

Move or rewrite them into `tools/model-debugging/scripts/` only when they have
been normalized around the current `@asrjs/speech-recognition` architecture.

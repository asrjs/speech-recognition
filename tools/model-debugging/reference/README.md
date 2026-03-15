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

## Rule

Keep copied scripts here if they still teach us something.

Move or rewrite them into `tools/model-debugging/scripts/` only when they have
been normalized around the current `@asrjs/speech-recognition` architecture.

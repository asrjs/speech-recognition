# MedASR Reference Tests

This folder is a copied reference suite from:

- `N:\github\ysdede\medasr.js\medasrjs\tests`

Its purpose is troubleshooting and parity investigation during model porting.
These files are intentionally preserved close to original form.

## What Is Here

- Original test files:
  - `sanity_fixture.test.mjs`
  - `ctc_timestamps.smoke.test.mjs`
  - `preprocessor.parity.test.mjs`
  - `encoder.parity.test.mjs`
  - `pipeline.parity.test.mjs`
  - `pipeline.smoke.test.mjs`
  - `medasr_node_wrapper.parity.test.mjs`
- Copied fixture assets:
  - `fixtures/sanity_sample.wav`
  - `fixtures/sanity_sample.label.json`
- Copied reference artifacts:
  - `reference_medasr/metadata.json`
  - `reference_medasr/features.json`
  - `reference_medasr/attention_mask.json`
  - `reference_medasr/logits.json`

Large copied tensors may be stored as compressed `.json.gz` files in git:

- `reference_medasr/features.json.gz`
- `reference_medasr/logits.json.gz`

The copied parity tests now read either raw `.json` or `.json.gz`.

## How To Use

- Treat these as reference/debugging helpers, not asrjs runtime tests.
- Use them to understand expected MedASR behavior stage-by-stage when porting or troubleshooting.
- Prefer running our asrjs-native tests and scripts first, then use these when deeper parity detail is needed.

## asrjs-Native Counterparts

- Lightweight, CI-safe helper tests:
  - `tests/lasr-ctc-medasr-port-helpers.test.ts`
- Active debugging scripts:
  - `tools/model-debugging/scripts/`

## Notes

- Some original parity tests require local ONNX artifacts and environment setup (`onnxruntime-node`, model files, and valid audio paths).
- `reference_medasr/metadata.json` may contain environment-specific absolute paths from the original workspace; update locally when replaying full parity checks.
- Use the shared archive helper when you need to restore or repack the large
  MedASR tensor dumps:

```powershell
node tools/model-debugging/scripts/node-reference-artifacts.mjs status --target medasr
node tools/model-debugging/scripts/node-reference-artifacts.mjs unpack --target medasr
node tools/model-debugging/scripts/node-reference-artifacts.mjs pack --target medasr --delete-originals
```

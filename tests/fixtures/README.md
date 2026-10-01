# Fixture audio smoke tests

This directory is intentionally empty by default. Put short local WAV samples here when you want repeatable offline model smoke tests.

Run after building the package:

```bash
npm run build
ASRJS_FIXTURE_SMOKE=1 npm run test:fixture-smoke -- \
  --audio tests/fixtures/sample.wav \
  --model parakeet-tdt-0.6b-v2 \
  --expect "expected words"
```

Notes:

- The harness accepts RIFF/WAVE PCM fixtures: 16-bit, 24-bit, 32-bit integer PCM, or 32-bit float PCM.
- Audio is downmixed to mono in the smoke harness before calling the library.
- Without `ASRJS_FIXTURE_SMOKE=1` or `--force`, the command skips cleanly so normal CI does not try to run heavyweight local inference.
- If model assets are unavailable and `--force` is not set, the command also skips cleanly.
- Use `--force` or `ASRJS_FIXTURE_SMOKE_FORCE=1` when you want unavailable assets or failed expectations to fail CI.
- Keep fixtures short and do not commit large/private audio files unless explicitly intended.

## Vitest model fixtures

The CLI smoke behavior above is separate from Vitest integration tests. In
Vitest, a disabled integration test is a real SKIP; an explicitly requested
test with missing/empty/malformed required artifacts FAILS. Shared prerequisites
live in `tests/support/fixture-gates.ts`; graph names and assertions stay in the
model-specific tests. See `docs/MODEL_HARNESS_CONTRACT.md`.

- Merged Whisper: set `ASRJS_FIXTURE_SMOKE=1` (or
  `ASRJS_FIXTURE_SMOKE_FORCE=1`) and `WHISPER_MERGED_FIXTURE_DIR`. Optional
  `WHISPER_MERGED_ENCODER`, `WHISPER_MERGED_DECODER` and
  `WHISPER_MERGED_TOKENIZER` override the existing relative filenames.
- Timestamped Whisper tiny: use the same opt-in and
  `WHISPER_TIMESTAMPED_FIXTURE_DIR`; `WHISPER_TIMESTAMPED_DECODER` overrides
  the existing `decoder_model_merged_int8.onnx` filename. This legacy tiny
  contract does not require integer quantization for other model ports.
- Split-graph Whisper: setting `WHISPER_SPLITGRAPH_FIXTURE_DIR` requests its
  tests. An unset value skips; an empty or invalid configured value fails.
- Whisper mel oracle: set `WHISPER_MEL_REFERENCE_JSON`, or supply
  `tests/fixtures/whisper_mel_ref_440hz_1s.json`. Missing configured files and
  JSON parsing failures must not be swallowed as skips.
- GigaAM CTC: set `GIGAAM_CTC_ONNX_SMOKE=1`. `GIGAAM_CTC_ONNX_DIR` defaults
  to `models/gigaam/multilingual-ctc`; `GIGAAM_CTC_FP16_ONNX_DIR` defaults to
  that directory plus `-fp16`. `GIGAAM_CTC_REFERENCE_JSON` overrides
  `tools/data/results/gigaam/multilingual-ctc-jfk-short-reference.json`.

- Whisper reproducibility: a non-empty `WHISPER_REFERENCE_JSON` requests the
  reference-backed tests; an unset or empty value skips them. Once requested,
  missing/empty/malformed reference JSON, missing audio, and invalid required
  mel values fail. `WHISPER_REFERENCE_AUDIO` overrides the reference audio path.
  A prerequisite failure or skip is not executed numerical evidence.

## Native resource lifecycle (Node/Bun)

Build first with `npm run build`. The resource smoke imports the built library.
`tests/native-resource-smoke.test.ts` keeps Node prerequisite checks always on;
actual native session checks are opt-in. A portable, repository-owned 125-byte
Identity ONNX graph and synthetic vocabulary are materialized in a unique
temporary directory and removed after each case. No model download, local
Python environment, or original vocabulary is needed. This graph certifies
acquisition/cleanup only, never ASR inference or numerical parity.

PowerShell, with an already installed native optional peer and Bun:

```powershell
$env:ASRJS_NATIVE_RESOURCE_NODE = '1'
$env:ASRJS_NATIVE_RESOURCE_BUN = '1'
$env:ASRJS_BUN_EXECUTABLE = 'C:\path\to\bun.exe'
node node_modules/vitest/vitest.mjs run tests/native-resource-smoke.test.ts tests/native-resource-fixture.test.ts
```

Leave the opt-ins unset for genuine SKIP of native integration. Requested Bun
tests require an explicit executable; missing/invalid executables fail, not
skip or fall back to Node. Existing native sessions run through the family
executor using CPU and verify success plus partial-acquisition rollback and
repeated disposal. The fixture integrity/overwrite unit test runs without
native bindings. Test environment settings above apply only to that shell;
do not set global environment variables or install a missing engine silently.

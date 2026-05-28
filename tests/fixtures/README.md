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

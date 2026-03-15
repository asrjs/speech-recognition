# Audio Prep Parity Playbook

Use this playbook when transcript quality changes depending on:

- browser vs Node runtime
- original audio vs pre-resampled audio
- `AudioContext` decode vs deterministic WAV parsing
- browser resampler vs simple linear resampler

## Goal

Decide whether the bug belongs to:

- audio preparation
- model execution
- transcript reconstruction

## Fast Triage

If any of these are true, suspect audio prep first:

- a pre-resampled 16 kHz WAV gives a better transcript than the original WAV
- Node gives the right text but browser does not
- a domain token or punctuation cluster changes around obvious audio boundaries
- model logits/tokens look reasonable after deterministic prep

## Recommended Workflow

1. Start with the original WAV fixture.
2. Generate or reuse a 16 kHz shared fixture.
3. Run deterministic Node inference with the same model artifacts.
4. Compare browser output using:
   - native-rate decode
   - linear resampling
   - any browser target-rate path still under test
5. Save the results into `tools/data/results/...`.

## Useful Fixtures

- [librivox.org.wav](N:\github\asrjs\speech-recognition\tools\data\fixtures\audio\librivox.org.wav)
- [librivox.org.16k.wav](N:\github\asrjs\speech-recognition\tools\data\fixtures\audio\librivox.org.16k.wav)
- [librivox.org.16k.pcm16.wav](N:\github\asrjs\speech-recognition\tools\data\fixtures\audio\librivox.org.16k.pcm16.wav)

## Useful Scripts

- [node-asrjs-nemo-inspect.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-asrjs-nemo-inspect.mjs)
- [node-nemo-shared-audio-parity.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\hf-parakeet-port\node-nemo-shared-audio-parity.mjs)
- [python-resample-wav.py](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\hf-parakeet-port\python-resample-wav.py)

## Rule

If deterministic Node + simple linear resampling gives the correct text,
do not start by changing decoder logic.

First check:

- WAV parsing
- channel downmixing
- sample-rate conversion
- browser-specific decode behavior

# Tools

`tools/` is the engineering workspace for `@asrjs/speech-recognition`.

It is intentionally separate from `src/` because these files are not part of
the published runtime. They exist to help us:

- debug transcription quality regressions
- compare `@asrjs/speech-recognition` against native/original model stacks
- validate audio preprocessing and resampling behavior
- inspect intermediate outputs and final transcripts
- benchmark exported models and runtime configurations
- troubleshoot long-form and realtime transcript assembly

## Layout

```text
tools/
  README.md
  data/
    fixtures/
    results/
  browser-validation/
  model-debugging/
    README.md
    SKILL.md
    playbooks/
    scripts/
    reference/
```

## Main Areas

### `tools/data`

Stable fixtures and captured results used across scripts and regression work.

Use this for:

- known-good audio inputs
- saved comparison outputs
- parity snapshots
- reusable debugging artifacts

### `tools/model-debugging`

The active debugging and parity toolbox for model-family work.

This is broader than “porting”:

- model implementation debugging
- parity validation
- quality troubleshooting
- audio pipeline investigation
- runtime regression checks
- model-porting helper suites (including copied MedASR reference tests)

Start with:

- [README.md](N:\github\asrjs\speech-recognition\tools\model-debugging\README.md)
- [SKILL.md](N:\github\asrjs\speech-recognition\tools\model-debugging\SKILL.md)
- [playbooks/README.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\README.md)
- [scripts/README.md](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\README.md)

### `tools/browser-validation`

Contains browser-level acceptance probes for sibling applications. The
`streaming-demo-mic-smoke.py` probe launches Chromium with a deterministic fake
microphone, then drives the demo's normal `getUserMedia` capture, model worker,
segmenter, transcription queue, and latency HUD. It can run without a model to
isolate capture/controller behavior or with a local Parakeet artifact directory
to exercise the real browser inference path.

The probe is intentionally outside the published runtime and does not replace
physical-device testing. Fake-device results are reproducible integration
evidence; they are not a claim about microphone hardware or broad ASR quality.

## Recommended Workflow

For most debugging or parity tasks:

1. start from a known fixture in `tools/data/fixtures/audio`
2. check the nearest playbook in `tools/model-debugging/playbooks`
3. run an `@asrjs/speech-recognition`-native script from `tools/model-debugging/scripts`
4. compare against a reference script set only if needed
5. save useful output under `tools/data/results`

Before opening a new model-port branch, run the dated candidate survey and
read [candidate-discovery.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\candidate-discovery.md).
It separates Hugging Face popularity from verified ONNX/WebGPU evidence and
helps avoid duplicating an existing browser implementation.

## Current Lesson From LibriVox

The recent `LibriVox.org` debugging work reinforced an important rule for
Parakeet/NeMo troubleshooting:

- when a browser result looks wrong, reproduce it in Node with deterministic
  WAV loading and simple linear resampling first
- if Node produces the correct text, the remaining suspect is usually the audio
  preparation path rather than the decode loop itself

That exact workflow is documented in:

- [audio-prep-parity.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\audio-prep-parity.md)
- [librivox-domain-parity.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\librivox-domain-parity.md)

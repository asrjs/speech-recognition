# Model Debugging Toolkit

`tools/model-debugging/` is the active engineering workspace for model
debugging, parity validation, and transcription-quality troubleshooting in
`@asrjs/speech-recognition`.

This folder is intentionally broader than "model porting". We use it to:

- debug incorrect transcripts and punctuation/domain splits
- compare `@asrjs/speech-recognition` against native/reference runtimes
- inspect audio preparation, resampling, tokenizer, and decoder behavior
- capture reproducible JSON artifacts for regressions
- validate long-form, overlap, and realtime transcript assembly

These files are not published runtime code. They exist to help us understand
why a model works, fails, regresses, or behaves differently across runtimes.

## Layout

```text
tools/model-debugging/
  README.md
  SKILL.md
  playbooks/
    README.md
    audio-prep-parity.md
    librivox-domain-parity.md
  scripts/
    README.md
    node-asrjs-nemo-inspect.mjs
  reference/
    README.md
    asr-merger-test/
    hf-parakeet-port/
    legacy-nemo-tdt-tests/
    medasrjs/
```

## Start Here

For day-to-day debugging, open these in order:

1. [SKILL.md](N:\github\asrjs\speech-recognition\tools\model-debugging\SKILL.md)
2. [playbooks/README.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\README.md)
3. [scripts/README.md](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\README.md)

If the problem is related to the `LibriVox.org` split or browser audio prep,
start directly with:

- [audio-prep-parity.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\audio-prep-parity.md)
- [librivox-domain-parity.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\librivox-domain-parity.md)

## Main Workflows

### 1. Transcript-quality debugging

Use this when:

- a domain token splits incorrectly
- punctuation looks suspicious
- browser and Node disagree
- pre-resampled audio fixes a bug

Recommended sequence:

1. reproduce with a stable WAV fixture from `tools/data/fixtures/audio`
2. rerun with a pre-resampled 16 kHz fixture
3. compare deterministic Node prep against browser prep
4. save a JSON artifact under `tools/data/results`
5. only then inspect tokenizer, decoder, or timestamp logic

### 2. Native parity debugging

Use this when:

- a new model family is being added
- `@asrjs/speech-recognition` text differs from the original stack
- intermediate outputs need to be compared stage by stage

Recommended sequence:

1. generate native/reference output first
2. run `@asrjs/speech-recognition` on the same shared audio
3. compare:
   - normalized PCM
   - frontend output
   - encoder output or shape slices
   - decoder/joint output
   - token ids and token text
   - timestamps
   - final transcript
4. fix the earliest mismatching stage, not the whole pipeline at once

### 3. Long-form and realtime debugging

Use this when:

- overlap windows drift
- transcript merging regresses
- window simulation is easier than rerunning full ASR

Start with the curated reference set in:

- [reference/asr-merger-test](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\asr-merger-test)

## High-Value Assets

### Playbooks

- [audio-prep-parity.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\audio-prep-parity.md)
- [librivox-domain-parity.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\librivox-domain-parity.md)

### Native `@asrjs/speech-recognition` scripts

- [node-asrjs-nemo-inspect.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-asrjs-nemo-inspect.mjs)

### Reference script sets

- [reference/README.md](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\README.md)

## Current Lessons

### Node-first audio prep debugging is worth it

The recent `LibriVox.org` issue confirmed a strong rule for Parakeet/NeMo work:

- if deterministic Node audio prep gives the correct text and browser does not,
  the first suspect is audio preparation, not the TDT loop

### JSON should be the source of truth

For transcript quality and merger debugging:

- save detailed JSON
- replay JSON when possible
- use human-readable text or VTT only as a convenience view

## Writing New Tools

When adding a new debugging script or notebook-like helper:

- keep it under `tools/model-debugging/`
- prefer a plain CLI script over one-off shell snippets
- write output to `tools/data/results/...`
- document the purpose in the nearest README or playbook
- prefer `@asrjs/speech-recognition`-native scripts when possible, keep older copied scripts under
  `reference/`

## Important Boundary

This folder is intentionally allowed to be pragmatic:

- hardcoded local model paths are acceptable in reference scripts
- Python helpers are acceptable
- copied legacy scripts are acceptable

But active scripts should gradually move toward:

- consistent CLI flags
- `@asrjs/speech-recognition` imports
- reproducible result JSON
- documented troubleshooting playbooks

# Model Debugging Skill

This is a local `@asrjs/speech-recognition` engineering playbook, not a globally installed Codex
skill.

Use it when working on:

- model-family implementation bugs
- transcript quality regressions
- parity mismatches against native/original stacks
- resampler, tokenizer, preprocessor, or decoder-loop issues
- long-form and realtime transcript-merging regressions

## Goals

Keep debugging disciplined and reproducible:

1. reproduce with a stable audio fixture
2. compare against a known-good native/reference stack
3. reduce the problem to the earliest mismatching stage
4. capture the result in JSON so the issue can be replayed without rerunning
   the whole stack

## Working Rules

### 1. Start with deterministic inputs

Prefer:

- WAV fixtures from `tools/data/fixtures/audio`
- pre-resampled shared audio when comparing runtimes
- saved JSON traces when validating mergers or transcript assembly

### 2. Separate audio prep bugs from model bugs

Before touching decoder logic:

- compare original audio vs pre-resampled audio
- compare browser audio prep vs Node deterministic prep
- compare `AudioContext`-based decode against WAV-parser + linear resample

If Node deterministic prep is correct and browser output is not, the issue is
probably in audio preparation, not the model implementation.

### 2.5. Match the scoring pipeline before calling it a regression

When transcript quality looks worse:

- first compare transcript text against the known-good stack directly
- then confirm both sides use the same text normalizer for WER/CER
- only treat the issue as a recognition regression after transcript parity and metric parity disagree

Different normalizers can create false alarms even when the underlying transcripts are the same.

### 3. Compare stage by stage

Use the reference scripts and `@asrjs/speech-recognition` scripts to compare:

- normalized / resampled PCM
- preprocessor outputs
- encoder outputs or shapes
- decoder/joint outputs
- token ids and token pieces
- timestamps
- final transcript text

### 4. Save artifacts

Write results into `tools/data/results/...` whenever the run is useful for:

- documenting a bug
- confirming a fix
- comparing multiple pipelines

### 5. Prefer Node first for deterministic model debugging

For Parakeet/NeMo quality work:

- reproduce in Node first
- use local direct artifacts when possible
- use simple linear resampling when validating parity

Then move the same case into the browser demo.

## Recommended Entry Points

- [README.md](N:\github\asrjs\speech-recognition\tools\model-debugging\README.md)
- [librivox-domain-parity.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\librivox-domain-parity.md)
- [node-asrjs-nemo-inspect.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-asrjs-nemo-inspect.mjs)
- [node-compare-transcript-jsons.mjs](N:\github\asrjs\speech-recognition\tools\model-debugging\scripts\node-compare-transcript-jsons.mjs)
- [reference/medasrjs/upstream-tests/README.md](N:\github\asrjs\speech-recognition\tools\model-debugging\reference\medasrjs\upstream-tests\README.md)

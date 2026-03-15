# Debugging Playbooks

These playbooks capture debugging workflows that were important enough to keep.

Use them when a bug has a repeatable shape and we want the next person to start
from a proven path instead of rediscovering the workflow.

## Current Playbooks

- [audio-prep-parity.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\audio-prep-parity.md)
  - use when browser and Node disagree, or when resampling seems to change text quality
- [canary-aed-porting.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\canary-aed-porting.md)
  - step-by-step NeMo AED porting flow for Canary-style models, including reference generation, ONNX export, FP16/INT8 variants, and JS frontend guidance
- [librivox-domain-parity.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\librivox-domain-parity.md)
  - the concrete `LibriVox.org` case that drove recent WAV and Node-path fixes
- [model-porting-parity.md](N:\github\asrjs\speech-recognition\tools\model-debugging\playbooks\model-porting-parity.md)
  - workflow for merging reference test suites and keeping CI-safe parity helpers

## When To Create A New Playbook

Create one when:

- the bug took multiple comparison steps to isolate
- more than one script was needed
- the investigation produced a reusable rule
- future model-family work is likely to hit the same class of issue

Good playbooks usually include:

- the symptom
- the environment assumptions
- the exact fixtures used
- the scripts to run
- the expected outputs
- the conclusion
- any library fixes that resulted from the investigation

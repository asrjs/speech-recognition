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

## Task-to-tool map

This is the discovery entrypoint for humans and coding agents. Paths below are
relative to this checkout. The [Model Debugging Skill](model-debugging/SKILL.md)
is a project-local playbook; it does not need global installation or harness
registration. Read it before model-porting or parity work.

| Task or symptom                                                         | Read first                                                                                                                                             | Existing tool or reference                                                                                                                                                  |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| New model port, or an apparently missing port                           | [End-to-end parity workflow](model-debugging/playbooks/model-porting-parity.md), then the [family playbook index](model-debugging/playbooks/README.md) | [Reference sets](model-debugging/reference/README.md); inspect `src/models`, `models` and local receipts before writing code or downloading weights                         |
| Missing Python import, wrong reference version or environment selection | [Python reference environments](#python-reference-environments)                                                                                        | [Verified workstation inventory](../docs/WORKSPACE_CONTEXT.md#python-reference-environment-inventory); check venv provenance, Conda and uv before installing                |
| Verify a ported model end to end on one command                         | [Port manifests](model-debugging/ports/)                                                                                                               | [Manifest runner](model-debugging/scripts/port_manifest.py); binds artifacts, fixtures and oracles by SHA-256, then runs the declared Node, Bun and browser cases           |
| NeMo/Parakeet raw log-mel, normalization or valid-length mismatch       | [NeMo feature-contract workflow](model-debugging/playbooks/nemo-rnnt-porting.md)                                                                       | [JS feature parity](model-debugging/scripts/node-canary-js-frontend-parity.mjs), [official NeMo capture](model-debugging/reference/hf-parakeet-port/python-nemo-inspect.py) |
| Orukeet original weights, export or numerical replay                    | [Orukeet reference guide](model-debugging/reference/orukeet/README.md), [TDT export/replay guide](nemo-tdt-onnx-export/README.md)                      | [Source audit](model-debugging/reference/orukeet/audit_source.py), existing `hf-parakeet-port` probes and `nemo-tdt-onnx-export` runners                                    |
| Qwen3-ASR encoder, explicit KV decoder, WASM or WebGPU                  | [Existing Qwen port](model-debugging/playbooks/qwen3-asr-porting.md)                                                                                   | [Official capture/export helpers and local artifact paths](model-debugging/reference/qwen3-asr-0.6b/README.md); do not assume this port or its weights are absent           |
| Kaldi fbank or streaming encoder caches                                 | [X-ASR/sherpa workflow](model-debugging/playbooks/x-asr-porting.md)                                                                                    | [Official sherpa capture and fbank dump](model-debugging/reference/x-asr/README.md)                                                                                         |
| Browser and Node disagree on input preparation                          | [Audio preparation parity](model-debugging/playbooks/audio-prep-parity.md)                                                                             | [Effective-PCM control](model-debugging/playbooks/librivox-domain-parity.md), [NeMo library inspector](model-debugging/scripts/node-asrjs-nemo-inspect.mjs)                 |
| Find the first divergent captured stage                                 | [Stage-capture CLI contract](model-debugging/scripts/README.md#stage-capture-contract)                                                                 | [Stage comparator](model-debugging/scripts/node-compare-stage-captures.mjs), [numerical metrics](tensor_parity.py)                                                          |
| Attribute a failing stage to one input                                  | [Shared harness boundary](../docs/MODEL_HARNESS_CONTRACT.md)                                                                                           | [Fail-closed attribution API](stage_attribution.py); reuse it through a family adapter, not a second inference loop                                                         |
| ONNX graph, external-data or packaging failure                          | [Artifact audit CLI](model-debugging/scripts/README.md#candidate-artifact-audit)                                                                       | [Local audit](model-debugging/scripts/node-audit-onnx-artifact.mjs), [external-data inspector](onnx_artifacts.py), [bounded repacker](repack_onnx_external_data.py)         |
| ORT WebGPU module, provider or numerical discrepancy                    | [ORT entrypoint control](model-debugging/playbooks/ort-webgpu-entrypoint.md)                                                                           | TDT browser runners record selected-module identity and fetched-artifact hashes; inspect actual receipt fields, not only package versions                                   |
| Real browser capture/controller behavior                                | [Browser validation guide](browser-validation/README.md)                                                                                               | Existing browser probes; model inference and fake-device capture are separate acceptance scopes                                                                             |
| Whisper export or alignment graphs                                      | [Whisper exporter](whisper-onnx-export/README.md), [export workflow](../docs/whisper-export-workflow.md)                                               | Existing four-graph export tools and model-owned parity checks                                                                                                              |

### Before adding tools or reporting BLOCKED

1. Open the matching family playbook and reference README. Read the runner's
   argument parser before choosing a command. Use `--help` where supported.
2. Inspect existing family code, original weights, converted graphs, tokenizer,
   local Python environments and result receipts. A default-Python import
   failure does not establish that the reference environment is unavailable.
3. Read official inference source and the pinned model configuration. Compare
   onnx-asr, sherpa-onnx and prior working ports through the
   [workspace reference map](../docs/WORKSPACE_CONTEXT.md#reference-repos).
   These implementations inform the port; the official engine remains the oracle.
4. Verify effective PCM identity, tensor shape/layout, valid length, normalization
   and runtime artifact identity before attributing numerical differences.
5. Reuse the nearest existing runner and shared comparison/control. Report
   missing prerequisites with the exact paths and environments checked.
   Old PASS receipts are leads, not proof for the current build.

Tool discovery grants no extra authority. Preserve unrelated changes and failed
receipts. New large downloads, installs, uploads and sibling edits still need
approval. Do not search site-packages or backup copies when the active toolkit
already supplies the route.

## Python reference environments

Use a project-owned environment per **reference dependency stack**, not per
model and not one global Python environment for every tool. NeMo, Qwen and
MedASR have incompatible version requirements; NeMo 2.x and 3.x also need
distinct identities when reproducing existing captures. Compatible model
families can share one verified stack.

The current checkout is only partly isolated. See the
[workstation inventory](../docs/WORKSPACE_CONTEXT.md#python-reference-environment-inventory)
for actual interpreters, package provenance and dependency-check results.
Those existing environments are discovery leads, not reproducible installation
manifests. Do not replace an environment during a numerical-parity investigation.

Before choosing an interpreter:

1. Inspect repository `pyvenv.cfg`, dependency manifests and lockfiles. A
   `.venv` with `include-system-site-packages = true` inherits packages from its
   base interpreter and can also expose the user site.
2. Check Conda registrations, even if `conda` is absent from `PATH`. On Windows,
   `$HOME/.conda/environments.txt` supplies registered paths; invoke the installed
   Conda executable explicitly for `env list --json`.
3. Run `uv python list --only-installed` and `uv tool list` if uv is available.
   The Python list includes system interpreters as well as uv-managed ones.
   A downloaded interpreter or cache is not a project environment.
4. Invoke the selected Python directly. Record `sys.executable`, `sys.prefix`,
   `sys.base_prefix`, package versions and imported module `__file__` paths.
   Run the needed imports and `-m pip check`. Successful imports do not imply
   dependency consistency or model parity.
5. Record actual ORT providers, not merely installed CPU/GPU distribution names.
   Provider enumeration is not proof that a CUDA session runs.

When provisioning is approved, keep each stack's dependency specification and
lockfile in the repository and its installed environment out of version control.
Use isolated environments without system/user-site inheritance, explicit
interpreter routes in family guides, and an import/model smoke before changing
those routes. Pin reference-engine versions; do not silently upgrade an oracle.
This inventory does not install, migrate or delete environments.

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

- [README.md](model-debugging/README.md)
- [SKILL.md](model-debugging/SKILL.md)
- [playbooks/README.md](model-debugging/playbooks/README.md)
- [scripts/README.md](model-debugging/scripts/README.md)

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
read [candidate-discovery.md](model-debugging/playbooks/candidate-discovery.md).
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

- [audio-prep-parity.md](model-debugging/playbooks/audio-prep-parity.md)
- [librivox-domain-parity.md](model-debugging/playbooks/librivox-domain-parity.md)

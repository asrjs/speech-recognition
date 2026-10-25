# Workspace Context

Start with [AGENT_ENTRYPOINT.md](AGENT_ENTRYPOINT.md) for current goal/handover
routing. App descriptions below are a workspace map, not fresh acceptance
claims; inspect each sibling's current wiring and Git state before using it.

This document is a short memory map for the local ASR workspace around `@asrjs/speech-recognition`. Use it when returning to the project after a break or when an agent needs to understand which sibling repo/app is the right place to inspect.

## Core Project

### `speech-recognition`

This is the `asr.js` core library: a single-package, speech-first TypeScript runtime for browser and native Node.js/Bun inference where independently verified.

The source of truth is here. Keep it headless, framework-neutral, ESM-first, and runtime-oriented. The public API is exported from `src/index.ts` with intentional narrow subpaths; internal boundaries use the actual `runtime`, `audio`, `io`, `inference`, `models`, `presets`, `tokenizers` and `types` folders, plus existing speech-composition modules. There is no `src/processors` folder to revive.

The core package may own reusable browser/runtime primitives such as capture, resampling, ring buffers, chunking, VAD, rough gate, detector/controller logic, waveform renderers, canvas utilities, and snapshot-style monitor APIs. React/Vue/Svelte/Solid wrappers should stay in thin sibling packages if they are needed.

## Maintained companion applications (2026-10-25)

The maintained suite is the sibling `demos` repository: `apps/showcase`,
`apps/playground`, `apps/realtime`, and `apps/benchmark`. They share
`packages/demo-runtime` and `packages/ui`, consume the speech package, and
use one theme/navigation system. Realtime is the current microphone,
TEN-VAD/FireRed VAD and waveform integration surface.

The standalone applications listed below document the evolution of the
project. Their old resume points do not supersede the consolidated suite.
See [the suite README](https://github.com/asrjs/demos) for current commands.

## Historical implementation apps

These apps exist because browser ASR behavior crosses many layers: microphone capture, chunking, model artifact loading, VAD, visualization, benchmarking, and realtime transcription. The library should remain reusable; the apps provide focused integration surfaces.

### `streaming-demo`

Former standalone realtime integration app; maintained work now lives in `demos/apps/realtime`.

Use this to test microphone streaming, waveform canvas rendering, rough gate behavior, TEN-VAD/FireRed VAD diagnostics, streaming ASR flows, local model folders, and Hugging Face model loading.

Historical resume point:

- waveform canvas, VAD markers, and energy triggering are the active realtime surface
- rough-gate-first segmentation is the stable fallback path
- TEN-VAD can still be used as the lighter VAD path
- FireRed VAD is experimental and may be diagnostics-only depending on the branch/app wiring
- TEN-VAD historically used roughly 16 ms chunks, while FireRed VAD expects 10 ms frames; when visualizing or comparing them, align results to the shared timeline instead of assuming identical frame sizes

### `browser-demo`

General browser ASR demo and earlier integration surface.

Use it for upload/sample-file workflows, model loading UX, local-vs-remote artifact behavior, and regression checks that do not need the full realtime streaming UI.

### `benchmark-demo`

Performance and benchmark surface.

Use it for comparing model/runtime variants, measuring latency or throughput, and checking whether changes to loaders, executors, or processors affect runtime cost.

### `playground`

Scratch app for API and UI experiments.

Use it when exploring ideas before deciding whether they belong in the core library, a demo, or a thin framework adapter.

### `vad-demo`

VAD-focused implementation app.

Use it for isolated VAD behavior, timeline rendering, segment boundaries, and comparisons between rough gate, TEN-VAD, and FireRed-style signals.

### `firered-vad-web`

Experimental dedicated FireRed VAD browser port/library.

Use it to inspect FireRed-specific model loading, frame sizing, inference behavior, and diagnostics before moving stable, reusable primitives into `speech-recognition`.

## Reference Repos

These repos are useful for archaeology and comparison. They are not the architecture to copy directly into `speech-recognition`.

### `parakeet.js`

Older focused Parakeet TDT runtime.

Use it as a working reference for ONNX artifact loading, browser demo patterns, JavaScript audio preprocessing, decoder loops, and model-specific Parakeet behavior. It is narrower than `speech-recognition`, so copy lessons carefully rather than copying its structure wholesale.

### `transformers-v4-parakeet-demo`

Transformers.js v4 Parakeet TDT fork/demo.

Use it as a reference for Transformers.js-compatible model formatting, Hugging Face model management, and how Parakeet TDT can be represented inside a Transformers.js-style stack. Do not bring the generic Transformers.js architecture or public API philosophy into this repo.

### `medasr.js`

Older/sibling MedASR experiments.

Use it for MedASR-specific history, model artifact clues, and comparison against the newer preset/model-family organization in this library.

### `NeMo`

Upstream or research reference for NVIDIA NeMo model behavior.

Use it when checking architecture details, export assumptions, tokenizer behavior, or TDT model semantics.

### `onnx-asr`

ONNX ASR reference/conversion work.

Use it when checking ONNX file layout, external data conventions, browser-compatible exports, and older artifact naming patterns.

### Local reference locations

These are discovery leads for this workstation, not required paths in a fresh
clone. Verify that each path exists and record its actual version before use.
Reading reference code does not authorize sibling edits or environment installs.

- Original NeMo checkout:
  `N:\github\ysdede\NeMo\nemo\collections\asr\parts\preprocessing\features.py`.
- The installed source used by recent Orukeet captures is under
  `tools/model-debugging/reference/nemotron-3.5-asr-streaming/.venv/Lib/site-packages/nemo/`.
  Its interpreter is `.venv/Scripts/python.exe` in that same reference folder.
  Reusing this environment for Orukeet does not resume paused Nemotron model work.
  Compare installed source with the checkout; they can differ.
- Established onnx-asr references:
  `N:\github\ysdede\onnx-asr\preprocessors\nemo.py`,
  `N:\github\ysdede\onnx-asr\src\onnx_asr\preprocessors\numpy_preprocessor.py`,
  and `N:\github\ysdede\onnx-asr\tests\preprocessors\test_nemo.py`.
- Its sherpa adapter is `N:\github\ysdede\onnx-asr\wrappers\sherpa.py`.
  The project-local [X-ASR reference helpers](../tools/model-debugging/reference/x-asr/README.md)
  capture sherpa streaming output and Kaldi fbank. That feature contract is not
  interchangeable with Orukeet's NeMo contract.

Use the [tool map](../tools/README.md#task-to-tool-map) for the family workflow.

### Python reference environment inventory

Read-only audit on 2026-10-03: eleven interpreters inspected; six reference
stacks passed their required imports and a small Torch CPU arithmetic check.
Receipt: `tools/data/results/python-runtime-environments-2026-10-03.json`.
No model or CUDA inference was run for this inventory. No environment was
installed, moved, upgraded or deleted.

| Environment / interpreter                                                                            | Observed stack                                                                                               | Isolation and verification                                                                                                                                                                               |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Project `tools/model-debugging/reference/nemotron-3.5-asr-streaming/.venv/Scripts/python.exe`        | Python 3.10.12; local NeMo 3.0.0; Torch 2.6.0+cu118; Transformers 4.51.3                                     | **Overlay**, not standalone: inherits Torch, ONNX, ORT and NumPy from Conda `nemo`; user site enabled. ASR imports pass; `pip check` fails with seven missing/incompatible requirements.                 |
| Project `tools/model-debugging/reference/qwen3-asr-0.6b/.venv/Scripts/python.exe`                    | Python 3.11.4; qwen-asr 0.0.6; Torch 2.13.0+cpu; Transformers 4.57.6; NumPy 2.4.6                            | Isolated from system/user sites. Qwen imports and `pip check` pass.                                                                                                                                      |
| `C:\Users\steam\anaconda3\envs\nemo\python.exe`                                                      | Python 3.10.12; NeMo **2.4.0**; Torch 2.6.0+cu118                                                            | Different oracle from the project NeMo 3.0.0 overlay. ASR imports pass; `pip check` fails with three missing/incompatible requirements.                                                                  |
| `C:\Users\steam\anaconda3\envs\medasr_env\python.exe`                                                | Python 3.11.14; Torch 2.10.0+cpu; Transformers 5.3.0.dev0                                                    | Core imports pass; `pip check` fails on missing notebook dependencies. Do not merge this development Transformers stack into Qwen's stack.                                                               |
| `C:\Users\steam\anaconda3\envs\asrds_py311\python.exe`                                               | Python 3.11.14; Torch 2.9.1+cu126; editable onnx-asr 0.10.3.dev8+g55d59284c from `N:\github\ysdede\onnx-asr` | Core imports pass; ORT 1.24.2 advertises CUDA/TensorRT/CPU. GPU inference untested. `pip check` reports a google-genai/websockets conflict.                                                              |
| `N:\github\salute-developers\GigaAM\.venv\Scripts\python.exe`                                        | Python 3.11.4; local sherpa-onnx 1.13.6; FunASR 1.4.4 and Torch 2.7.1+cu128 from `C:\Python311`              | **Sibling overlay**, not project-owned or isolated. Sherpa/FunASR imports pass; `pip check` fails on inherited dependency conflicts. Existing GigaAM, SenseVoice and X-ASR recipes use this interpreter. |
| uv-managed `C:\Users\steam\AppData\Roaming\uv\python\cpython-3.13.15-windows-x86_64-none\python.exe` | Python 3.13.15; NumPy 2.3.2 and SoundFile supplied by the user site                                          | Interpreter only, not a project reference stack. No uv tool environments, project `uv.lock` or project dependency `pyproject.toml` found.                                                                |

Other metadata-only checks: Conda `vgtts` has Torch 2.10.0+cu126 and
Transformers 4.57.6; Conda base is Python 3.13.5; system Python 3.11.4 has
Torch 2.7.1+cu128 and FunASR 1.4.4; system Python 3.12.7 has none of the
surveyed ASR/runtime packages. These were not import-smoked as reference stacks.

Conda is installed at `C:\Users\steam\anaconda3\Scripts\conda.exe` but was
not found by `Get-Command conda`. Its registry contains duplicate path-case
entries, not additional environments. uv is installed and lists both its
managed Python and system interpreters.

CPU and GPU ORT distributions coexist in several environments. The imported
ORT in NeMo and the sibling GigaAM overlay exposes only Azure/CPU providers;
installed GPU distribution metadata alone does not prove GPU availability.

For future provisioning, follow
[Python reference environment rules](../tools/README.md#python-reference-environments):
own an isolated, version-pinned stack per reference library ecosystem, with a
separate stack where reference versions conflict. Keep the current captures
bound to their original interpreter; do not relabel NeMo 2.4 output as NeMo 3.0.

### `keet`

Experimental or older Parakeet-related scratch work.

Treat it as reference material until its current contents are rechecked.

## Current Direction

`speech-recognition` is meant to be the platform and model agnostic layer above the earlier focused experiments. The old projects are valuable because they contain working solutions and failure history, but this repo should stay:

- speech-first instead of model-zoo-first
- single-package instead of workspace-package-heavy
- headless and framework-neutral
- model-family based under `src/models`
- branded only at the preset layer under `src/presets`
- careful about keeping backend differences from changing transcript semantics

## Where To Look First

- Runtime/library bug: start in `speech-recognition`.
- Realtime microphone, waveform, VAD, and streaming behavior: start in `streaming-demo`.
- Remote Hugging Face or local-folder model loading behavior: compare `speech-recognition`, `streaming-demo`, `browser-demo`, `parakeet.js`, and `transformers-v4-parakeet-demo`.
- VAD frame-size or timeline alignment issue: compare `streaming-demo`, `vad-demo`, and `firered-vad-web`.
- ONNX external data, artifact naming, or Parakeet TDT loader issue: compare `speech-recognition`, `parakeet.js`, `onnx-asr`, and `transformers-v4-parakeet-demo`.

Remote model loading resilience is documented in [HUGGINGFACE_DOWNLOAD_RESILIENCE.md](./HUGGINGFACE_DOWNLOAD_RESILIENCE.md). FireRed diagnostic fallback behavior is documented in [FIRERED_VAD_DEGRADED_TROUBLESHOOTING.md](./FIRERED_VAD_DEGRADED_TROUBLESHOOTING.md).

The Windows-to-WSL2 maintenance handoff is documented in [WSL2_PROJECT_HANDOFF.md](./WSL2_PROJECT_HANDOFF.md).

## Delegating to OMP subagents in this workspace

The default subagent model resolved to `deepseek/deepseek-v4-flash` and failed
immediately with HTTP 402 `Insufficient Balance`. This is a provider balance
limit, not a missing package, a repository problem or a code defect. Delegation
works by pinning the spawn to the parent's live model with the per-task
`model: "@default"` selector; a scout then completed normally. When a subagent
fails with 402, diagnose the provider first and retry with `@default` or another
explicitly available model rather than editing repository code or repeating the
same failing spawn.

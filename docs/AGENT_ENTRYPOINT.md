# Agent entrypoint — all coding harnesses

Updated: 2026-10-28. This is the startup router, not another task queue.
Use it with Codex, OMP or another repository-aware harness.

Current checkpoint: [2026-10-28 handover](reports/HANDOVER-2026-10-28.md).
It separates merged work from the uncommitted approved-profile/Blob changes,
preserves real-model receipts and records current verification limits.

## Read order and authority

1. Read the project-root `AGENTS.md` when present, and
   [AGENT_SAFETY_RULES.md](AGENT_SAFETY_RULES.md) completely before editing.
2. Read [PROJECT_CHARTER.md](PROJECT_CHARTER.md) and the
   [documentation index](README.md) first. The product is an easy-to-integrate
   offline and streaming speech library for browser and Node/Bun developers.
   [ACTIVE_SPRINT.md](archive/ACTIVE_SPRINT.md) and the dated review are historical
   work records; verify task status before resuming them.
3. Read the [engineering tool map](../tools/README.md#task-to-tool-map).
   For model ports, feature extraction, inference or parity work, read the
   project-local [Model Debugging Skill](../tools/model-debugging/SKILL.md).
   These resources live in this checkout, not in the harness's global skill list.
4. Read [architecture.md](architecture.md),
   [PROJECT_CHARTER.md](PROJECT_CHARTER.md),
   [MODEL_HARNESS_CONTRACT.md](MODEL_HARNESS_CONTRACT.md) and
   [WORKSPACE_CONTEXT.md](WORKSPACE_CONTEXT.md).
5. For historical task logs (>7,000 lines of past porting/parity records), refer
   to the summaries in [PRODUCT_TASKS.md](PRODUCT_TASKS.md) and
   [PARAKEET_VARIANT_TASKS.md](PARAKEET_VARIANT_TASKS.md), or the archives under
   `docs/archive/` ([PARAKEET_VARIANT_TASKS-2026-10-17.md](archive/PARAKEET_VARIANT_TASKS-2026-10-17.md)
   and [PRODUCT_TASKS-2026-10-17.md](archive/PRODUCT_TASKS-2026-10-17.md)).
   Do not load archived logs into context during standard startup.

Latest direct user instructions define the scope. Repository safety and current
architecture govern implementation. The goal defines durable outcomes; existing
task files hold the work records; the dated handover is evidence, not permanent
certification. Resolve contradictory records against actual source/artifacts and
ask before materially changing the model order or required acceptance gates.

## Before declaring a model or tooling blocker

Use the [task-to-tool map](../tools/README.md#task-to-tool-map) to select an
existing playbook and runner. Check the family source, local original weights,
exports, reference environments and receipts before requesting a download or
writing a new exporter, feature extractor or inference loop.

Read the original model's inference code. Compare established ports such as
onnx-asr and sherpa-onnx as implementation references; they do not replace the
official oracle. [WORKSPACE_CONTEXT.md](WORKSPACE_CONTEXT.md#reference-repos)
maps sibling references. Inspect their actual paths and versions before use;
sibling writes and new large downloads still require authority.

Missing global skill registration, one failed interpreter import, or no new
numerical hypothesis is not evidence that project tooling is absent. Record
the paths, environments and existing controls checked before reporting BLOCKED.

## Architecture boundary

This is one speech-first, ESM-first, headless TypeScript package, not a
Transformers.js clone. Public contracts are coherent and root-exported with
intentional narrow subpaths. Actual folders include `src/audio` and `src/io`,
not `src/processors`. Family math, decoder/cache rules and model-specific tests
stay in `src/models`; brand configurations stay in `src/presets`. Framework
components/state stay outside core. Never move a one-model workaround into shared
runtime merely to get a transcript. Preserve canonical output semantics.

## Historical records are not startup instructions

[`docs/archive/`](archive/) holds `GOAL_PROMPT.md`, `HANDOVER.md`, `HANDOFF.md`,
`SESSION_HANDOVER.md`, `AGENT_TASKS.md`, `AGENT_STATUS_REVIEW.md` and
`WSL2_PROJECT_HANDOFF.md`. They and the archived recommended goal retain useful history. Their old branches, agent
names, resume prompts, queue entries and PASS claims do not direct current work.
Follow them only when the user explicitly resumes that historical scope, after
checking source/build/artifact freshness. Do not check out an old branch because
a historical handoff says to do so. The October 8 model order was Orukeet, then
Parakeet TDT students/distills and quantized/faster derivatives, then ranked
non-NVIDIA candidates. Full-size community TDT variants are eligible for measured
speed/memory/quality benefit; Parakeet CTC is on hold and Nemotron remains paused.

## Portable startup

Start OMP from the project directory. Installed OMP 18.4.12 discovers standalone
`AGENTS.md` up from cwd to the repository boundary. The local root `AGENTS.md`
is ignored by this checkout's Git rules, so it is not guaranteed in a new clone.
This tracked entrypoint is the explicit portable bootstrap. Do not invent OMP
configuration files or change global provider settings.

Give the harness this prompt:

```text
Read docs/AGENT_ENTRYPOINT.md and its required documents completely. Start
tool discovery at tools/README.md; for model work read
tools/model-debugging/SKILL.md and the matching porting playbook. Use
docs/GOAL_PROMPT_RECOMMENDED.md as the whole-product goal and reconcile
docs/archive/OMP_HANDOVER_2026-10-03.md with live Git/source/build/artifacts. Preserve
protected dirty work and failed evidence. Resume the first safely actionable
bounded slice in the existing task files, validate it, then commit only owned
changes. Do not resume Nemotron, relax parity gates, repeat completed controls,
push/publish/upload, install globally or download large weights without authority.
```

## Current product focus (2026-10-25)

The user reaffirmed library design, public API, developer documentation and
project organization as the priority. Preserve established Parakeet TDT and
Whisper custom-graph paths. Shared VAD, capture, waveform and streaming
components support this product; a model test matrix is supporting evidence,
not the product roadmap. Start from the README and charter, then select a
bounded task matching the latest user request.

## Historical focus overlays

Historical sprint (2026-10-17): read
[ACTIVE_SPRINT.md](archive/ACTIVE_SPRINT.md) and
[reports/deep-review-2026-10-17.md](reports/deep-review-2026-10-17.md)
for the active 7 task cards (T1–T7), 3-tier verification gates, and strategic roadmap.
Also refer to [HANDOFF-subagents-2026-10-17.md](reports/HANDOFF-subagents-2026-10-17.md).

Historical user scope (2026-10-08): **do not work on Parakeet CTC now**. Resume
Parakeet TDT distills, quantized/faster derivatives and the recorded 2026
non-NVIDIA speech candidates. Read the active P10 checklist at the top of
[PARAKEET_VARIANT_TASKS.md](PARAKEET_VARIANT_TASKS.md) and C52 in
[PRODUCT_TASKS.md](PRODUCT_TASKS.md). Same-size TDT derivatives can be useful;
do not revive old exclusions solely for full parameter count or v3 teacher
lineage. Preserve earlier failed controls and numerical gates. The dated list
is [light-asr-diarization-hunt-2026-09-30.md](reports/light-asr-diarization-hunt-2026-09-30.md).

[GOAL_MODEL_PORT_HARNESS.md](GOAL_MODEL_PORT_HARNESS.md) is derived from that
goal and is the current emphasis: making the next model port cheaper and
compounding harness value. It narrows focus; it does not replace the
whole-product goal, waive any gate or change the confirmed model order.

For the recovered model queue and source/provenance corrections, read the
[2026-10-08 status report and working guide](reports/model-port-status-and-guide-2026-10-08.md).
It distinguishes existing Qwen dynamic/fp16 support from the newer export's
failure, and model parameter count/origin from packaging or repo ownership.
Use it with the current goal and existing task files, not as a replacement queue.

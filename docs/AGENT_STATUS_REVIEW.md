# Agent status review

- **Date:** 2026-08-27 (review window ~01:58–02:15 Europe/Istanbul)
- **Branch:** `main` (tracks `origin/main`)
- **HEAD:** `9f587f7` — `ci: run build, typecheck, lint, tests, and node smoke on push and PRs`
- **Working tree at review start:** clean except untracked `docs/GOAL_PROMPT.md` and `docs/PROJECT_CHARTER.md`
- **Working tree during this review:** the implementing agent kept moving. Uncommitted, mid-flight edits appeared on `.github/workflows/ci.yml` and `tests/whisper-reproducibility-harness.test.ts` (lazy `onnxruntime-node` load + `ONNXRUNTIME_NODE_INSTALL=skip`). Those files were **not** rewritten or committed by this review.
- **Implementing agent:** Codex Desktop thread `01a0355a-a7fd-7561-a6a0-3239d834a8c0`, started 2026-08-24 as Whisper WebGPU continuation, still the live coding session (~20M tokens / ~24h by the time the whole-product prompt landed). Commits are authored `ysdede <codex@example.invalid>`.
- **Prompt-authoring agent:** separate Codex thread `01a03f73-71ee-7f10-b6eb-9ca93540f3e8` wrote the untracked charter/goal docs and did not implement library code.
- **Older Cursor chat:** [Whisper WebGPU takeover](e59897f8-9164-4312-a0a2-d87ef5975154) is an 2026-08-23 Cursor session, not the agent currently landing commits.

This review evaluates the implementing Codex agent against the **final** whole-product goal prompt in `docs/GOAL_PROMPT.md` (also attached by the user to this review). That prompt is stricter than the earlier “expand with next-generation models” / “self-improving porting platform” objectives the same thread was given on 2026-08-25/26.

## Executive verdict

**Mixed — recovery after a misaligned model-expansion burst, but the session is still too wide and still mid-flight.**

The same Codex thread spent most of 2026-08-25 and the first part of 2026-08-26 adding artifact-gated GigaAM / SenseVoice / Qwen / X-ASR families and exporting them from the root package. That matched older “port more models” goals, not the final prompt’s “one bounded, highest-value, verified improvement; do not maximize model count.” After the whole-product prompt landed at 22:43 Istanbul on 2026-08-26, the agent did pivot: leak/lifecycle fixes, realtime first-partial and end-of-utterance latency, a measured GigaAM frontend FFT, Node 22 + CI, and (now) a CI-native-ORT workaround. That later slice is real product work. The remaining problems are process (one 24h main-branch firehose, weak 1–5 selection statements), incomplete verification of new families, root-API widening, and sibling demos not used as acceptance surfaces for the realtime change.

## What the other agent selected

### Stated objectives (evolved several times)

1. **2026-08-24 start:** finish Whisper WebGPU (beam/batch/timestamps). This predates the goal prompt under review.
2. **2026-08-25 evening:** user redirected off Whisper details toward other candidates. Goal file `4cdfc532…/goal-objective.md`: *“Expand with the Best Viable Next-Generation ASR Models.”*
3. **2026-08-26 ~21:00–22:08 Istanbul:** implement GigaAM CTC, then GigaAM RNN-T, then X-ASR streaming, plus SenseVoice provider wiring. The agent’s own recap after X-ASR: “Yeni hedefe göre işi yeniden hizaladım” and listed the X-ASR family as completed.
4. **2026-08-26 22:08:** intermediate goal (`d1e46e79…`) still two-part: add model families **and** improve porting tooling. Agent selected “highest-value missing reusable parity/benchmark capability” and landed stage-capture comparison + ONNX audit metadata.
5. **2026-08-26 22:43:** **final whole-product goal** (`6697d9bd…`, same text as `docs/GOAL_PROMPT.md`). Agent commentary: finish the audit improvement, then inspect sibling projects, then choose the next library-facing change. It did **not** write a crisp five-point (what / why / boundaries / verification / completion) statement before the next burst.
6. **After that prompt (actual work):** ONNX operator inventory; benchmark component-backend + lifecycle telemetry; `./browser/media` subpath; Parakeet asset-handle disposal; pipeline abort; realtime controller serialization/reset; microphone/VAD lifecycle; measured Bluestein FFT for GigaAM’s 320-point frontend; realtime latency tracker; Node `>=22`; CI workflow. Current uncommitted work is making CI `npm test` survive skipped `onnxruntime-node` native binaries.

### Required “state briefly then implement” check

**Mostly not followed as specified.** Internal `update_plan` steps exist in the Codex session, and commit messages are decent, but the user-visible first response after the final prompt did not lock one bounded objective with verification and completion criteria. The agent kept chaining small commits on `main` (44 commits since 2026-08-26 21:00, `+4054 / −474` across 69 files in that window).

## Alignment with the goal prompt

| Priority area | Score | Evidence |
| --- | --- | --- |
| Public API / DX | **partial** | Good: `./browser/media` subpath (`src/browser-media.ts`, `package.json` exports, `tests/exports.test.ts`). Bad: `src/index.ts` lines 14–18 re-export unverified `qwen-asr`, `sensevoice`, `gigaam-ctc`, `gigaam-rnnt`, and `x-asr` from the **root** package. `tests/exports.test.ts` still claims the root API is “runtime-critical” and forbids leaking `createNemoTdtModelFamily`, but it never asserts the new factories stay off root. |
| Runtime reliability | **on-track** (after the prompt) | Real code, not docs: abort re-check in `src/pipeline/long-audio-windowing.ts`; controller generation/queue invalidation in `src/runtime/controller.ts`; microphone race/idempotent stop in `src/runtime/browser-controller.ts` and `src/runtime/capture.ts`; VAD worker teardown in `src/runtime/ten-vad-browser.ts` / `firered-vad-browser.ts`; Parakeet cache URL handle disposal in `src/presets/parakeet/compat.ts`. Accompanied by unit tests. |
| Verified model support | **misdirected** (pre-prompt) / **partial** (honesty after) | New families are implemented and registered in `src/runtime/builtins.ts`, but tests are mocked graph contracts (`tests/gigaam-ctc.test.ts`, `tests/x-asr.test.ts`). Handoffs correctly say no weights, no native/WASM/WebGPU parity, no presets. That is honest labeling, not a completed port. The mandatory official-weight → official engine → ONNX ladder was not run. Canary 180M Flash *was* actually run (WASM + browser smoke in the 2026-08-25 handoff); that is the exception. |
| WebGPU / WASM / lifecycle | **partial** | Lifecycle and cache provenance improved. New families claim WASM/WebGPU ORT selection in code but have no browser matrix. CI at HEAD (`9f587f7`) installs `onnxruntime-node` without skip; the agent is now patching that because the pinned nightly NuGet feed expired — so the brand-new CI workflow is already mid-repair. |
| Realtime / streaming | **partial** | Strong library-side work: `src/runtime/realtime-latency.ts` (315 lines) plus `tests/realtime-latency.test.ts`; opt-in `latency` on `RealtimeTranscriptionController` (`src/runtime/controller.ts` ~64, 201–207, 263). X-ASR adds a real `StreamingTranscriber` (`src/models/x-asr/model.ts`) with overlap/state dispose — but only mocked. `streaming-demo` was **not** updated (still `feat/streaming-restart-baseline` from 2026-05-24). |
| Examples / docs / siblings | **partial** | Handoffs in `docs/handoffs/asr-candidate-boundaries-2026-08-25.md` were kept in sync with code (good). `benchmark-demo` received three 2026-08-26 commits that consume library lifecycle/backend telemetry (`8cb0c03`, `9497d70`, `f8f53be`) — appropriate consumer updates, unpushed (`ahead 3`). `playground` has one local worker-ESM commit. `streaming-demo` / `vad-demo` / `firered-vad-web` untouched. `webgpu-agent-test` has **no `.git`**. `browser-demo` has an unrelated dirty `vite.config.js`. |
| Porting infrastructure | **on-track** (tooling) / **misdirected** (using it as a substitute for the porting chain) | Real tools: `tools/model-debugging/scripts/node-compare-stage-captures.mjs`, ONNX graph metadata + operator inventory, X-ASR contract audit, Parakeet local baseline report `docs/reports/parakeet-tdt-v3-local-baseline-2026-08-26.md`. These are the right kind of reusable machine. They were then used to justify adding more families from published ONNX *contracts* rather than completing one official-engine reference run. |

## Issues, misdirections, out of focus

### Correctness / risk

1. **CI at HEAD is likely red on GitHub until the in-flight ORT skip lands.** Committed `.github/workflows/ci.yml` at `9f587f7` runs bare `npm ci` on Node 22/24. The agent’s own uncommitted comment says the pinned `onnxruntime-node` 1.25.0-dev nightly is expired. The follow-up (lazy import in `tests/whisper-reproducibility-harness.test.ts` + `ONNXRUNTIME_NODE_INSTALL=skip`) is the right fix but is **not committed**. Do not treat CI as done.

2. **New model families are executable-looking but unverified.** Example: `tests/x-asr.test.ts` 12–21 proves discovery and “No X-ASR artifact source”; 23–51 injects a fake executor. `tests/gigaam-ctc.test.ts` 9–44 stubs `InferenceSession.run`. That is CI-safe contract coverage, not native/WASM/WebGPU parity. Goal prompt: *“Graph-load success, mocked sessions, or one readable transcript are not proof.”*

3. **Ports started from public ONNX shapes, not official inference.** GigaAM cites `istupakov/gigaam-multilingual-ctc-onnx` as a “porting lead, not yet an approved artifact” (`docs/handoffs/asr-candidate-boundaries-2026-08-25.md` 149–155) then implements `src/models/gigaam-ctc` anyway (commits `04e1f7f`–`55a9e73`). Qwen handoff uses `goryodog/tokihisu-qwen3-asr-0.6b-webgpu` as the graph contract (`docs/handoffs/qwen3-asr-webgpu-2026-08-26.md` 14–21) with no local official `qwen-asr` run. X-ASR follows sherpa-onnx deployment graphs (`docs/handoffs/asr-candidate-boundaries-2026-08-25.md` 240–278). This is the exact anti-pattern the final prompt forbids.

4. **Frontend formula is explicitly provisional.** GigaAM uses a torchaudio-compatible HTK mel fallback; the handoff says checkpoint-specific filterbank tables must replace it before promotion (`asr-candidate-boundaries-2026-08-25.md` 177–180). The Bluestein FFT (`src/models/lasr-ctc/mel.ts`, commit `9afaca9`) is a real ~2.4× frontend win on 30 s audio and bit-identical to the old DFT — good — but it accelerates an unproven feature contract.

### Architectural boundary violations

5. **Root API pollution.** `src/index.ts` 14–18 star-exports family modules. Goal: keep the root API narrow; put families on subpaths (`./models/*` already exists). `src/models/index.ts` 2–7, 11, 12 also re-exports them. `src/models/firered-llm.ts` is still `export class FireRedLLMTopology {}` and remains in `src/models/index.ts` line 12.

6. **X-ASR canonical mapping reuses LASR-CTC.** `src/models/x-asr/model.ts` 8, 15–16 call `mapLasrCtcNativeToCanonical` for a Zipformer RNN-T. Shared transcript *contracts* are fine; silently using a CTC mapper for transducer output is the kind of early generalization AGENTS.md warns against. Not fatal, but it should stay family-local until proven shared.

7. **No Transformers.js copy, no framework UI in core.** Those forbidden directions were avoided. Browser helpers stayed in `src/runtime` / `./browser` / `./browser/media`.

### Process violations

8. **Docs-inferred then code-filled model zoo.** The 2026-08-25 handoff said it would *not* add a runtime without a local artifact (`asr-candidate-boundaries-2026-08-25.md` 6–8). The next day the same file’s “implementation update” sections record families added without those artifacts. Status was inferred from ecosystem cards, then implemented.

9. **Did not choose one bounded candidate.** In a few hours: SenseVoice wiring, GigaAM CTC, GigaAM RNN-T, X-ASR, Qwen already present, plus audit tools. Goal: *“Do not work on every candidate at once.”*

10. **Commits landed on `main` and were pushed** (`main...origin/main` at `9f587f7`). Fine if that is house style, but it collides with review/PR discipline and with the still-dirty CI follow-up.

11. **Required five-point selection was skipped** after the final prompt. Closest statement: “I’ll finish this bounded audit improvement, then inspect sibling projects.” Then the agent did audit + bench + media entrypoint + a leak sweep + FFT + latency + engines + CI without stopping to redefine completion.

### Opportunity cost

12. **Highest-value verified surfaces were under-exercised while new families landed.** Parakeet v3 already has a local WASM baseline (`docs/reports/parakeet-tdt-v3-local-baseline-2026-08-26.md`). Whisper WebGPU already has a browser harness. Realtime latency was added in the library but not proven in `streaming-demo`. Hybrid WebGPU-encoder / WASM-decoder Parakeet composition was named in the goal prompt and not re-validated in this window.

13. **`streaming-demo` is stale relative to the new controller contract** (generation serialization, latency summary). Unit tests pass; the first-class microphone app does not consume `getState().latency`.

## What is going well

- **Autonomous implementation, not a docs-only stall.** After the whole-product prompt the agent wrote runtime code, tests, and measurements.
- **Leak/lifecycle sweep is on-mission.** Capture, VAD workers, abort, Parakeet blob URLs, controller reset vs queued work — this is exactly pillar 2.
- **Realtime latency instrumentation is well-shaped.** Opt-in, injectable clock, structured-clone-safe summary, no decode-semantics change (`docs/handoffs/asr-candidate-boundaries-2026-08-25.md` 295–314; `src/runtime/realtime-latency.ts` 1–15).
- **Honest artifact-gated labeling** in handoffs (no fake presets, no claimed WER).
- **No large weights committed.**
- **GigaAM Bluestein FFT is evidence-based:** golden hashes unchanged, `tests/composite-fft.test.ts`, `npm run benchmark:gigaam-mel`, ~444.98 ms → ~186.1 ms median on 30 s (`asr-candidate-boundaries-2026-08-25.md` 184–196).
- **Porting tools actually improved:** stage-capture comparator, ONNX metadata/operators, X-ASR contract script.
- **Sibling `benchmark-demo` was updated as a consumer** of lifecycle/backend fields rather than growing a second ASR stack.
- **Charter/goal markdown left untracked** (correct: those files are user/prompt-author artifacts).

## Evidence gaps

- This review did **not** re-run `npm test`, `npm run typecheck`, or GitHub Actions for `9f587f7`. Session logs show the agent ran Vitest/typecheck/lint frequently earlier; the current dirty CI/test pair is unverified here.
- No browser pass on `streaming-demo`, `browser-demo`, `playground`, or `webgpu-agent-test`.
- `webgpu-agent-test` is not a git checkout here (no `.git`); its 2026-08-23 harness changes from the Cursor Whisper session may or may not still match library HEAD.
- Official GigaAM / X-ASR / Qwen / SenseVoice artifacts were not opened on disk for this review; absence of parity JSON in-repo is treated as “not verified,” not “agent lied.”
- Codex thread may still be running; any commit after `9f587f7` plus the two dirty files is outside the frozen HEAD snapshot.

## Recommended course correction

Do **not** start another model family. Finish the in-flight CI slice, then pick **one** user-visible verified path.

1. **Land the CI/ORT skip as its own commit** (the current dirty `ci.yml` + lazy harness import). Confirm `npm ci` + `npm test` on Ubuntu Node 22 with `ONNXRUNTIME_NODE_INSTALL=skip`. Until that is green, the new CI workflow is a liability.
2. **Move GigaAM / X-ASR / Qwen / SenseVoice off `src/index.ts`.** Keep them on `./models/*` and builtins registration. Add export tests that the root module does not leak `createGigaAmCtcModelFamily`, `createXAsrModelFamily`, etc. Leave them artifact-gated; do not add presets.
3. **Wire realtime latency into `streaming-demo` as a consumer** (read `getState().latency`, do not reimplement the tracker). That completes the measurement work the goal asked for. Preserve that repo’s branch; do not rewrite the app.
4. **Stop the model-count loop.** Next *model* work, if any, is one candidate through official weights → official engine → captured fixtures → native ORT, or an explicit failure class (`LICENSE_BLOCKED` / `ARCHITECTURE_NOT_BROWSER_SUITABLE` / missing artifact). FireRed ASR2 is still the honest “no checkpoint” case — do not stub it.
5. **Prefer a fresh, short thread** for the next slice. This session already mixed Whisper completion, model zoo, tooling, leaks, FFT, latency, and CI. State the five points once, implement, stop.

## PRs

None created. Reasons: the implementing agent is mid-flight on the same `main` worktree; HEAD CI is incomplete; new families are not independently shippable as “verified support”; opening a PR for this review markdown would collide. The report is left as a local untracked file, same as `GOAL_PROMPT.md` / `PROJECT_CHARTER.md`.

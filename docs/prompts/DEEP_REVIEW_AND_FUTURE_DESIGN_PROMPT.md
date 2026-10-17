# Deep review and future design prompt

Give everything below the line to a strong, independent reviewing agent. It is
self-contained. Last updated 2026-10-17.

---

You are a principal engineer and technical strategist. You are reviewing
`asr.js` (`@asrjs/speech-recognition`), a headless, speech-first, ESM-first
TypeScript speech runtime. Its primary backend is browser WebGPU, with WASM as a
fallback. Many agents have built it over several months, and its owner wants an
honest outside view. Understand the project deeply, find what is missing or
wrong, and design its future. Previous agents have produced a lot of
self-reported progress. Treat their reports as claims to verify.

## Repository and context

- Main checkout: `N:/github/asrjs/speech-recognition` (Windows 10, PowerShell;
  heredocs do not work, so write files with tools). Node v26, `onnxruntime-node` 1.30.0.
- Linked worktree with Qwen fp16 artifacts:
  `U:/parallel-agents/qwen-fp16-wt`. Do not modify or remove it.
- Sibling repos the owner maintains, used as references only: `N:/github/ysdede/parakeet.js`,
  `N:/github/ysdede/transformers.js`, `N:/github/ysdede/onnx-asr`, `N:/github/ysdede/keet`,
  `N:/github/ysdede/medasr.js`, `N:/github/ysdede/NeMo`, and demos under
  `N:/github/asrjs/` (`browser-demo`, `streaming-demo`, `benchmark-demo`,
  `playground`, `vad-demo`, `firered-vad-web`, `webgpu-agent-test`).
- The project must not become a Transformers.js clone in architecture, API
  philosophy or folder layout.

## Read first (in this order)

1. `AGENTS.md` (if present), `docs/AGENT_ENTRYPOINT.md`, `docs/AGENT_SAFETY_RULES.md`.
2. `docs/architecture.md`, `docs/PROJECT_CHARTER.md`, `docs/GOAL_PROMPT_RECOMMENDED.md`,
   `docs/GOAL_MODEL_PORT_HARNESS.md`.
3. `docs/reports/HANDOFF-subagents-2026-10-17.md`, the current snapshot and known risks.
4. `docs/reports/model-port-status-and-guide-2026-10-08.md`,
   `docs/reports/light-asr-diarization-hunt-2026-09-30.md`.
5. `tools/README.md` (task-to-tool map), `tools/model-debugging/SKILL.md`.
6. Skim `docs/PRODUCT_TASKS.md` and `docs/PARAKEET_VARIANT_TASKS.md`. Both are
   very large. Read the active sections, then sample older ones.
7. Read the actual source: `src/index.ts` and the public exports in `package.json`,
   then `src/runtime`, `src/inference`, `src/audio`, `src/io`, `src/models/*`,
   `src/presets/*` and `tests/`.

## Evidence rules

- Verify with live Git, source and commands. Run at least `npx vitest run` and
  `npm run typecheck`. Run `npm run build` and one package smoke test if time allows.
- Every finding needs evidence: a file path with line numbers, a command and its
  output, or a receipt path. Label each finding VERIFIED, INFERRED or UNVERIFIED.
- Parity means numerical parity (max/mean abs error, cosine, allclose with explicit
  tolerances). Shape and magnitude alone are not parity. A token match on one
  fixture is not model acceptance.
- Do not take PASS claims in task files at face value. Re-check a sample of at
  least five against their receipts and code.

## Review dimensions

Cover each one. Say plainly when a dimension is healthy.

1. **Product and positioning.** Who is this for? What can a user do with it
   today, end to end, in a browser? Compare it honestly with Transformers.js,
   sherpa-onnx (WASM), onnx-asr, whisper.cpp/WebGPU ports and the Web Speech API.
   Where is the real differentiation, and is it being delivered?
2. **Architecture.** Do the code and `docs/architecture.md` agree? Check folder
   boundaries (`src/models` vs `src/runtime` vs `src/presets`), dependency direction,
   model-specific logic leaking into shared runtime, premature generalization,
   and "util blob" modules. Is the canonical transcript contract actually stable
   across families and backends?
3. **Public API and packaging.** Coherence of root exports and subpaths, naming,
   error model, cancellation and disposal, versioning (currently 0.2.0), tree-shaking,
   bundle size, optional native peers, and whether a stranger could adopt it from
   the README and examples alone.
4. **Model coverage and quality.** List every family and preset with its real
   status per backend (Node CPU, WASM, WebGPU) and precision (fp32, fp16, other),
   backed by evidence. Families include NeMo TDT/Parakeet/Orukeet, HF CTC,
   Whisper, Moonshine, Qwen3-ASR, FireRed, GigaAM, MedASR, Nemotron (paused) and
   Parakeet CTC (on hold). Separate "exported" from "validated" from "shippable".
5. **Backend strategy.** Is WebGPU actually primary in practice? Check the hybrid
   per-component backends, fallback honesty, and fp16 handling. Known lesson: ORT
   CPU runs many fp16 ops in fp32, so native-fp16 parity is the wrong gate.
   Assess memory limits in the browser and the plan for models that do not fit.
6. **Streaming and real-time.** Capture, VAD (TEN, FireRed, rough gate), chunking,
   rolling windows, timestamps, latency and the controller/store APIs. Is the
   streaming story coherent across models?
7. **Porting machinery and harness.** `tools/` exporters, oracles, receipts,
   manifests and the automatic Chrome runner. Does each new port get cheaper?
   What is missing for a reliable "add a model in days" path?
8. **Testing and verification rigor.** Unit vs integration vs browser coverage,
   skipped tests, tests that depend on local-only assets, flaky or environment-bound
   tests, receipt reproducibility (absolute machine paths), and CI. Is there CI at all?
9. **Process and agent hygiene.** Uncommitted work, unpushed history, the size of
   task files and reports, duplicated or contradictory docs, historical handoffs,
   about 1,900 untracked receipts, and local-only environments (git-ignored venvs,
   artifacts on U:). How much context does a new agent need to load, and how can
   that shrink?
10. **Performance.** What is measured vs claimed: RTF, first-token latency, load
    time, memory, warm vs cold. Are benchmarks matched and reproducible?
11. **Risks.** Licensing of weights and exports, HF repo ownership, dependency on
    ORT dev builds, browser API churn, bus factor and maintainability.

## Deliverables

Write one report: `docs/reports/deep-review-<YYYY-MM-DD>.md`. Write it in
Turkish and keep code identifiers in English. Structure:

1. **Executive summary.** At most 10 lines: overall health, the three most important
   problems and the single most important next move.
2. **Verified current state.** A table of families and presets by backend and
   precision with an evidence column.
3. **Findings.** Order by severity (Critical / High / Medium / Low). Each finding
   gets evidence, impact, a recommendation and an effort estimate (S/M/L).
4. **Gaps.** What a production-ready browser speech runtime needs that is missing.
5. **Future design.**
   - Target architecture, if it differs from today, with a migration path that
     avoids a rewrite.
   - A product roadmap in three horizons: next 2 weeks, next quarter, next 12
     months. Include explicit non-goals.
   - The model strategy: which families to finish, cut or start, and why, by
     measured value.
   - A process redesign for multi-agent work: task file structure, gates,
     receipts, CI and commit/push policy.
6. **Decisions for the owner.** A short numbered list of choices only the owner
   can make. Give options and a recommendation for each.
7. **Ready-to-run task cards.** The first 5–8 cards, in the same format as
   `HANDOFF-subagents-2026-10-17.md` §5 (goal, inputs, steps, acceptance,
   prohibitions).

Be direct. Call out work that is overbuilt, unfinished or not worth continuing.
Do not pad the report. Specific evidence matters more than breadth.

## Limits

- Review is read-only by default. The only file you may create is the report
  above, plus scratch files outside the repo. Do not edit source, tests, task files
  or other docs.
- Do not push, publish, upload, install globally, download large weights,
  edit sibling repos or change acceptance gates.
- Do not resume Nemotron or Parakeet CTC work, and do not run long GPU jobs.
  Short browser smoke runs are allowed only if the existing harness already
  supports them.
- If something blocks you, record what you checked and continue with the rest.
  Do not stop at the first blocker.
- When finished, commit only the report with
  `docs(review): deep review and future design <date>` and report the commit hash.

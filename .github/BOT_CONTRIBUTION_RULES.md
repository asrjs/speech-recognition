# Rules for automated contributors (Jules, Bolt and similar)

Read this file completely before choosing a task.

## Before opening a PR

1. Search closed PRs for the same files and idea:
   `gh pr list --state closed --search "<file or keyword>"`.
   If a matching PR was closed without merging, do not reopen the idea.
2. Open at most one PR at a time. If an automated PR is open, stop.
3. One concern per PR, under ~150 changed lines, with a targeted test.

## Do not propose (previously rejected)

- Replacing `??` with `!` or other micro-edits in numeric loops.
- Map/Set caches for the model catalog or preset lookups.
- Pooling, LUTs or copy avoidance in mel/padding/concatenation code.
- Returning an input array instead of a copy ("single-chunk fast path").
  Callers do not own those buffers.
- Removing `async` from dispose/cleanup callbacks (PR 238 proved a regression).
- Adding Vitest path aliases unless a test currently fails without one.
- Retyping `ProductionTranscript` without first checking the real executor
  output (`WhisperNativeTranscript` uses `text/startTime/endTime/confidence`).
- Tests that only re-assert trivial pure functions already covered.

## Performance changes

Include before/after numbers from `npm run benchmark:hot-paths` or an
end-to-end transcription timing on a real fixture. Without a measured
improvement of at least 5% on a realistic workload, do not open the PR.

## Behavior changes

State which callers own the data you change and why aliasing, ordering or
error propagation stays the same. "Risk near zero" is not an argument.

## Good targets

Pick one item from `docs/reports/REVIEW-2026-10-28.md`, "Öncelikli aksiyon
listesi", that is not already done, and reference it in the PR title.

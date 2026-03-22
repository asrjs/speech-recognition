# Pull Request Rules

## GitHub Text Formatting

Apply these rules for all GitHub write actions:

- `gh pr create`
- `gh pr edit --body-file`
- `gh pr comment`
- `gh issue comment`
- review replies and thread resolutions

Rules:

1. Never post literal escaped newline sequences like `\n` in final text.
2. Prefer `--body-file <path>` for PR bodies and longer comments.
3. Verify the rendered Markdown before posting:
   - no `\n` artifacts,
   - no truncated lines,
   - no broken bullets or code fences.
4. Wrap commands, paths, package names, and commit hashes in backticks.
5. Keep write-ups short and structured: summary, verification, risk, follow-ups.

## PR Scope

1. One concern per PR. Split separate feature work, runtime fixes, and repo-maintenance changes.
2. Keep public API changes explicit in the PR summary.
3. If fragile runtime paths are touched, include targeted tests in the same PR.

Fragile paths include:

- public export surfaces and subpath entry points
- runtime/model loading and asset resolution
- browser local-folder loading
- realtime capture, segmentation, TEN-VAD, and waveform helpers
- ORT session or tensor lifecycle management
- preset/model adapter boundaries

## Verification Expectations

Before opening or updating a PR, run the smallest meaningful verification set for the changed area. Prefer exact commands in the PR body, for example:

```bash
npm run typecheck
npm run lint
npm test -- --run tests/streaming-detector.test.ts
npm run build
```

If a full command was not run, state that directly.

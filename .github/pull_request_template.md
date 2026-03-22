## Summary
- What changed:
- Why this change is needed:

## Scope
- [ ] This PR is scoped to one concern.
- [ ] I did not mix unrelated cleanup, formatting churn, or drive-by refactors.

## Areas Touched
- [ ] Public exports / package entry points (`src/index.ts`, `src/browser.ts`, `src/realtime.ts`, etc.)
- [ ] Model loading / runtime composition (`src/runtime/load.ts`, browser or node runtime helpers)
- [ ] Browser local-folder loading / asset resolution
- [ ] Realtime capture / detector / TEN-VAD / waveform helpers
- [ ] Preset or model-adapter logic (`src/presets/**`)
- [ ] ORT session / tensor lifecycle or disposal-sensitive paths
- [ ] Docs / templates / repo automation only
- [ ] None of the above

## Verification
- [ ] `npm run typecheck`
- [ ] `npm run lint`
- [ ] `npm test`
- [ ] `npm run build`
- [ ] Added or updated targeted tests for the changed behavior

### Test Evidence
Paste the exact commands and key output snippets here, or link CI runs.

## Risk
- Risk level: `low` / `medium` / `high`
- Main risk area:
- Rollback plan:

## Follow-ups
- Closes/Fixes:
- Follow-up work, if any:

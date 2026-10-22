# Root import budget — 23 Ekim 2026

Root static graph: **44 modules**, target **<45**. Threshold and smoke script unchanged. Evolution 78 ->70 ->66 ->56 ->45 ->44. No model implementation or native ORT loader is statically pulled into the root.

## Changes

- Type-only barrels emit no runtime edges.
- Synchronous catalog metadata is generated from the authoritative descriptor builder, language maps and artifact manifests. `npm run build` checks the snapshot for drift. Descriptor/source tests compare every field including explicit undefined properties. To update metadata: `npx tsc -p tsconfig.json`, `node scripts/generate-catalog-data.mjs --write`, then `npm run build`. Commit the regenerated file alongside its authoritative source change.
- Backend factories remain synchronous and share bindings with public backend subpaths. Probes and execution contexts load asynchronously when requested; native runtime detection stays outside the eager root.
- Asset handle implementations, transcript formatting and disposal helpers load inside existing asynchronous operations. Ownership snapshots are taken before awaiting helper imports.
- Explicit reexports and direct imports avoid unrelated barrel dependencies. Public exports, class identity and synchronous factory/list contracts are regression-tested against the 128-export baseline.

## Verification

- Full retry: 1850 tests passed,36 skipped;218 files passed,7 skipped.
- Initial full run:1849 passed,1 export test hit its existing15000ms timeout. The unchanged test passed in isolation in2.9seconds and the unchanged full rerun passed. Both logs preserved; no timeout increased.
- Typecheck/build/lint passed; lint retains13 warnings/0errors.
- Node import and package tarball smoke passed.
- Root graph smoke passed, exit0,44modules. Published package smoke found1645 files.
- Independent source review found no concrete regression; parent separately ran all checks.
- Browser physical microphone, EOU120 and broader model parity remain unverified; this report does not certify them.

Evidence: evidence/root-import-2026-10-23. GitHub CI and merge receipts will be recorded after publication; local success alone is not CI success.

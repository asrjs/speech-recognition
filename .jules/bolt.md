## 2024-04-24 - Model lookup optimization using reverse mapping
**Learning:** In model catalog presets (e.g., `src/presets/canary/catalog.ts` and `src/presets/parakeet/catalog.ts`), functions like `getModelKeyFromRepoId` and `getModelConfig` use O(N) linear searches via `Object.values()` or `Object.entries()` over the `MODELS` object to lookup by `repoId\'. This involves redundant array allocations on each invocation. While N is small, this operation occurs very frequently.
**Action:** Implement a static reverse mapping object (e.g., `REPO_ID_TO_KEY`) derived from the `MODELS` configuration to enable O(1) lookups by `repoId`. This replaces O(N) linear searches, eliminating redundant array allocations and significantly improving performance.

## 2024-04-24 - Using bun pm trust modifies package.json
**Learning:** Running `bun pm trust --all` as a fallback for dependency installation can automatically modify `package.json` by appending `trustedDependencies`. If modifying `package.json` is explicitly forbidden by user instructions (like the Bolt persona rules), this will result in a blocking failure during code review.
**Action:** If `bun pm trust --all` must be used and modifying `package.json` is forbidden, always revert the changes to `package.json` (e.g., using `git checkout package.json`) before submitting.

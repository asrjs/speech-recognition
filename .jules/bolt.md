## 2025-04-30 - Avoid repository-wide Prettier formats in focused PRs
**Learning:** Running `npm run format` applies Prettier across the entire codebase. While technically correct, this introduces significant noise to the git diff, hiding the actual functional changes. In strict code review environments, these mixed-concern PRs will be rejected because the core intent (performance improvement) is drowned out by formatting noise.
**Action:** Always format *only* the specific files modified during the task using a targeted command like `npx prettier --write <file>`. Never run global formatting scripts when aiming for a focused, small PR.

## 2025-04-30 - O(N) Array methods in hot lookups
**Learning:** In internal presets (like `canary/catalog.ts` and `parakeet/catalog.ts`), using `Object.entries(MODELS)` or `Object.values(MODELS)` to find model keys or configurations by `repoId` requires O(N) linear time and allocates arrays unnecessarily on every call.
**Action:** Replace `Object.entries()`/`Object.values()` iterations with a precomputed static reverse lookup map (e.g., `REPO_ID_TO_KEY`) to achieve O(1) performance and eliminate memory allocation overhead on hot paths.

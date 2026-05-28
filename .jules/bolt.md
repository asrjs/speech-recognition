## 2024-05-24 - Model Catalog O(N) Lookup Bottleneck
**Learning:** In model catalog presets (e.g., `src/presets/canary/catalog.ts`, `src/presets/parakeet/catalog.ts`), repeated `Object.entries()` or `Object.values()` calls for linear searches (`repoId` lookups or `array.includes()` language checks) cause redundant allocations and CPU overhead on hot paths.
**Action:** Implement static reverse mapping Maps (e.g., `REPO_ID_TO_KEY`) at initialization to enable O(1) lookups by `repoId`. Use `WeakMap<Config, Set<string>>` for caching language sets in membership checks, eliminating O(N) scans.

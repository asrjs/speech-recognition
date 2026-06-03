## 2024-05-24 - Model catalog lookups and Set caching
**Learning:** Model catalogs have functions like `getModelKeyFromRepoId` and `supportsLanguage` which iterate over `MODELS` or arrays. These cause unnecessary O(N) overhead during runtime and repetitive allocations, especially in a browser environment with high invocation counts.
**Action:** Use a static reverse mapping Map (`REPO_ID_TO_KEY`) for repo-id lookups. For language support, cache a `Set<string>` in a `WeakMap<ParakeetModelConfig, Set<string>>` for O(1) lookups.

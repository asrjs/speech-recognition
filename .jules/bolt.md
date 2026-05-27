## 2024-05-27 - O(N) Array Operations in Model Catalogs
**Learning:** Frequent model catalog lookups (`getModelConfig`, `getModelKeyFromRepoId`) and feature checks (`supportsLanguage`) were using linear O(N) searches like `Object.entries(MODELS)` and `Array.includes()`. In a hot path, this adds unnecessary overhead.
**Action:** Use static reverse mapping `Map`s generated once at startup for `O(1)` ID lookups. Use `WeakMap` with `Set` values to cache derived boolean structures like supported languages, turning O(N) `Array.includes()` into O(1) `Set.has()`.

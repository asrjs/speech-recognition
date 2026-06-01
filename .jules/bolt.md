## 2025-01-20 - O(1) Catalog Lookups

**Learning:** Linear searches (O(N) operations via `Object.entries()` or `Object.values()`) inside model catalog lookups and capability checks (e.g., matching a `repoId` to a `modelKey`, or querying array `.includes()`) can create CPU bottlenecks and unnecessary garbage collection overhead when called frequently (e.g., in a streaming or batch loop), especially as catalogs grow.

**Action:** Future agents should implement static reverse mappings (e.g., using a `Map`) derived from the configuration objects during initialization to enable O(1) lookups by IDs, eliminating repetitive linear searches. Similarly, for repetitive array membership tests, cache the results in a `Set` via a `WeakMap` tied to the config object to provide O(1) average lookup performance.

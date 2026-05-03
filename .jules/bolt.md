## 2025-05-03 - Avoid Array Iteration Chaining in Hot Mapping Loops
**Learning:** In audio post-processing pipelines (like mapping native transcripts to canonical ones), nested array chains like `.filter(...).map(...)` executed for every word create severe O(N*M) bottlenecks and extreme GC pressure from intermediate arrays.
**Action:** When extracting subset values (like `tokenIndices` for a word) from a parent list (`tokens`), always replace `.filter().map()` chains with a single-pass `for` loop that populates a pre-allocated or local array directly.

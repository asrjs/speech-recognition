## 2024-06-07 - Avoid nested array methods in audio post-processing mapping

**Learning:** In audio post-processing mapping loops (e.g., `lasr-ctc/mapping.ts`, `nemo-common/mapping.ts`), using nested `.filter(...).map(...)` chains to find matching `tokenIndices` for every word creates an O(N*M) performance bottleneck and severe GC pressure due to intermediate array allocations.

**Action:** Avoid executing nested `.filter(...).map(...)` chains to calculate `tokenIndices`. Instead, use a single-pass `for` loop over the tokens array to populate a local `tokenIndices` array directly, eliminating intermediate allocations and reducing the allocation overhead to O(N).

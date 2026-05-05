## 2025-05-05 - Avoid chained array methods inside execution loops
**Learning:** Using `.filter().map()` inside another `.map()` loop (e.g., when resolving `tokenIndices` for every word in transcription mapping) introduces severe O(N*M) time complexity and massive GC pressure due to multiple intermediate array allocations.
**Action:** Replace nested chained array methods (`.filter(...).map(...)`) with a single-pass `for` loop that populates a local array (e.g., `tokenIndices`) directly. This is a critical optimization pattern for DSP/mapping functions processing large segment arrays in V8.

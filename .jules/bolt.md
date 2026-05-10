## 2024-05-14 - Map/Filter Chain Anti-pattern
**Learning:** Found a memory mentioning that `.map().filter()` chains are an anti-pattern in audio post-processing mapping loops (like `lasr-ctc/mapping.ts` and `nemo-common/mapping.ts`) because they create O(N*M) performance bottlenecks and GC pressure when searching for token indices.
**Action:** Replace `.map().filter()` or nested `.filter().map()` chains with a single-pass `for` loop, especially when mapping tokens to utterances or words.

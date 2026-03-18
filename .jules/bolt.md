## 2026-03-16 - Native Array Copy Optimization
**Learning:** Replaced manual array copying with `TypedArray.prototype.set()` inside the audio processing loop. Native `set()` in V8 is implemented in C++ and handles nullish checks implicitly, making it significantly faster than iterating with a manual JavaScript loop and nullish-coalescing.
**Action:** Always prefer native `TypedArray.prototype.set()` for array copying in hot paths to avoid performance degradation.
## 2026-03-18 - Array Buffer Pooling Overlap and Memory Leaks
**Learning:** Although buffer pooling (reusing `Float32Array`, etc.) is an excellent optimization for reducing GC overhead, a naive implementation where the buffers ONLY grow and never shrink introduces a severe memory regression. Long-lived executors or those processing large one-off inputs will permanently retain the peak memory allocated, essentially causing a memory leak over the application's lifecycle.
**Action:** When implementing buffer pooling, always consider the lifecycle and memory retention policy of the pool. Ensure there is a mechanism to release or shrink buffers if they grow excessively large to prevent long-term memory bloat.

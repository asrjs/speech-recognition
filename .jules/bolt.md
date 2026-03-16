## 2026-03-16 - Native Array Copy Optimization
**Learning:** Replaced manual array copying with `TypedArray.prototype.set()` inside the audio processing loop. Native `set()` in V8 is implemented in C++ and handles nullish checks implicitly, making it significantly faster than iterating with a manual JavaScript loop and nullish-coalescing.
**Action:** Always prefer native `TypedArray.prototype.set()` for array copying in hot paths to avoid performance degradation.

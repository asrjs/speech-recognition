
## 2024-05-18 - TypedArray inner loops and nullish coalescing
**Learning:** In V8 (Node/Bun/Chrome), using the nullish coalescing operator `??` inside tight inner loops dealing with TypedArrays significantly degrades performance compared to using the non-null assertion operator `!`, even when `noUncheckedIndexedAccess` flags array indexing. The overhead of checking for `undefined` on every element access dominates the execution time.
**Action:** When iterating over TypedArrays in hot paths (like audio processing or downmixing) and you are completely certain that bounds are safe, use `!` instead of `?? 0` to preserve maximum performance.

## 2024-04-05 - V8 TypedArray optimization
**Learning:** Using the nullish coalescing operator (`??`) inside tight inner loops dealing with TypedArrays significantly degrades performance compared to using the non-null assertion operator (`!`) in V8 (Node/Bun/Chrome). Even though a Float32Array always returns a valid number or undefined out of bounds, the check introduces a deoptimization.
**Action:** When bounds are known to be safe, prefer `!` to preserve performance when accessing elements inside tight processing loops.

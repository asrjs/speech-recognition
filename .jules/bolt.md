## 2024-03-23 - TypedArray Inner Loop Performance in V8
**Learning:** In V8 (Node/Bun/Chrome), using the nullish coalescing operator (`??`) inside tight inner loops dealing with TypedArrays significantly degrades performance compared to using the non-null assertion operator (`!`). When bounds are known to be safe, `??` forces slower execution paths.
**Action:** Always prefer `!` over `??` when iterating safely over TypedArrays in hot calculation paths (like `argmax` and `confidenceFromLogits`).

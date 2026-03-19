## 2024-05-18 - [Optimize `nullish coalescing` array lookups in `MedAsrJsPreprocessor`]
**Learning:** In V8 (Node/Bun/Chrome), using the nullish coalescing operator (`??`) inside tight inner loops dealing with TypedArrays significantly degrades performance compared to using the non-null assertion operator (`!`). In hot paths doing a lot of array lookups, like mel spectrogram computation, `??` forces bounds-checking overhead and degrades fast path optimization in the engine.
**Action:** Replace `??` with `!` for TypedArray accesses in hot loops when bounds are known to be safe.

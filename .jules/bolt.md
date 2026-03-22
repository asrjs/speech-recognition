## 2024-03-22 - [Buffer pooling in hot audio paths]
**Learning:** `computeRawMel` in audio preprocessors (like `MedAsrJsPreprocessor`) is a hot path where frequent instantiation of TypedArrays causes measurable garbage collection overhead, particularly with large Float32/Float64 buffers.
**Action:** Always utilize internal buffer pools as class properties and dynamically resize them with a 1.2x growth factor for these DSP routines. Ensure you don't return un-copied sub-arrays from internal pools if they might escape scope.

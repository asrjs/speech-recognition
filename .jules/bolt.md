## 2024-04-01 - Float16 Decoding LUT
**Learning:** Bitwise float16-to-float32 conversions in hot loops (like normalizing ONNX logit arrays) create severe CPU bottlenecks in JS. Since float16 only has 65,536 possible bit patterns, they can be entirely precomputed at module load time into a Float32Array lookup table.
**Action:** When handling float16 tensors (`Uint16Array` in JS) from ONNX output, precompute a `Float32Array(65536)` LUT and perform index lookups (`LUT[uint16]`) rather than performing bitwise decoding per element.

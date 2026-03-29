## 2024-03-29 - [Memory Efficiency]
**Learning:** Audio models in this codebase expect 32-bit floating-point precision. Prefer using `Float32Array` over `Float64Array` for internal DSP buffers (like padded audio arrays) to reduce memory footprint by 50% and align with standard DSP practices.
**Action:** Review uses of `Float64Array` in audio/DSP processing blocks such as `src/models/lasr-ctc/mel.ts` and `src/audio/js-mel.ts` and change to `Float32Array`.

## 2025-02-28 - Optimize pre-emphasis DSP loop
**Learning:** Extracting class properties to local variables and replacing redundant previous-element array lookups with a local `prev` variable loop-state significantly improves V8 JIT performance for hot DSP operations over TypedArrays.
**Action:** When writing or optimizing DSP loops processing audio frames sequentially, always maintain loop state in local variables rather than reading `array[index - 1]` each iteration, and pull class properties (`this.preemphasis`) into function scope before the loop.

## 2024-11-05 - Avoid fast-path stream aliases

**Learning:** When converting accumulated stream chunks into a consolidated TypedArray (e.g., `streamToBytes`), returning `chunks[0]` when `chunks.length === 1` creates dangerous aliased array references that can be mutated downstream, leading to subtle and difficult-to-debug bugs, especially if the array buffer is transferred to a worker.
**Action:** Always allocate a new buffer and copy data instead of returning direct chunk references unless explicit architectural validation proves memory sharing is safe for the specific pathway.

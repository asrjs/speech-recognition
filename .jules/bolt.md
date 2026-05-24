## 2024-05-24 - [LayeredAudioBuffer sequence indexing optimization]
**Learning:** `LayeredAudioBuffer` pushes chunks that have a monotonic sequence counter starting from `0`. Trimming removes from the start, preserving contiguous sequence IDs.
**Action:** Instead of O(N) `.find(chunk => chunk.sequence === target)`, calculate O(1) direct index via `target - this.entries[0].chunk.sequence`.

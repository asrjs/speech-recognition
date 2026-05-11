## 2024-05-11 - Optimize FrameAlignedTokenMerger anchor finding
**Learning:** Nested loops in streaming token merging ($O(N \cdot M)$) become a severe bottleneck.
**Action:** Pre-grouping pending tokens by ID into a Map reduces lookup complexity to average $O(N)$.

## 2024-05-13 - Optimize token anchor lookup in streaming token merger
**Learning:** During streaming inference, calculating token overlap by searching an entire list of pending tokens inside a loop over overlapping tokens causes an O(N*M) performance bottleneck, creating severe CPU load.
**Action:** Pre-group the `pendingTokens` into a `Map<number, FrameAlignedToken[]>` keyed by `tokenId` before iterating over `overlapTokens`. This reduces the lookup complexity from O(N*M) to O(N) in the average case and improves streaming inference performance.

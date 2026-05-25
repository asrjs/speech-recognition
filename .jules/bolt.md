## 2024-05-24 - Unsafe Micro-optimizations in Hot Paths
**Learning:** Micro-optimizations (like replacing `??` with `!`) in critical audio, VAD, or math hot paths (e.g., `toMonoPcm`) are often rejected as unsafe or unverified by reviewers, despite showing positive micro-benchmark results. The risk of undefined behavior or lockfile churn outweighs marginal loop speedups without extensive end-to-end vetting.
**Action:** Avoid modifying verified critical audio/math hot paths for purely theoretical or micro-benchmark performance gains unless explicitly requested and rigorously vetted end-to-end.

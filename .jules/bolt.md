
## 2024-03-27 - [Performance of nullish coalescing vs non-null assertion with TypedArrays in V8]
**Learning:** In V8 (Node/Bun/Chrome), using the nullish coalescing operator (`??`) inside tight inner loops dealing with TypedArrays (e.g. `logits[index] ?? Number.NEGATIVE_INFINITY`) significantly degrades performance compared to using the non-null assertion operator (`!`). When bounds are known to be safe, preferring `!` preserves performance in hot paths like logit processing.
**Action:** When working on hot execution paths or inner loops with TypedArrays where array bounds are known and safe, use `!` instead of `?? fallback` to avoid performance regressions.

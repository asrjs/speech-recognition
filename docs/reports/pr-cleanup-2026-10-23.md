# Open PR cleanup — 23 Ekim 2026

Snapshot:64 open speech-recognition PRs. Each has an explicit accept/adapt/consolidate/reject/superseded decision below and an exact head SHA plus diff hash in the JSON companion.58 were closed after live SHA checks; no branch was deleted and no GitHub message/review comment was posted.6 selected source PRs remain open until this replacement passes CI and merges. Remote final states must be checked after merge.

## Adopted changes

- #229 chunk validation/boundary tests absorb #143/#144; omit bundled lint edits already on main.
- #146 sentence joining/punctuation/empty/missing-text tests; deliberate invalid input annotated.
- #191 assert the expected cache warning; existing afterEach restores spies even after failure.
- #147 preserve canonical manifest graph names during post-export FP16 conversion. Reject the PR's machine-specific fixture path. New stdlib tests execute the real organizer with a conversion test double: old code fails, fixed code passes3 tests. They verify naming/layout/options only; actual ONNX conversion and numerical FP16 parity remain unverified. Run these layout tests through a Vitest subprocess bridge in both existing CI jobs; workflow remains unchanged.

## Independent verification

Parent inspected source/diffs, reviewed all adopted changes and reran tests rather than trusting agent statements. Source-only secondary review found no concrete regression. Some earlier review claims were corrected: #183 has cache-write alias code, not only lock churn; #212 changes internal ownership and is not the cache-write bug; #218 changes own-key enumeration to inherited-key enumeration. Constructor accepts a fractional smoothing window, so #185 can produceNaN: preserved standalone receipt compares old/candidate arithmetic. Rejecting an optional optimization for lack of measured benefit is not a claim it is numerically wrong.

Local validation:1859 passed/36 skipped;3 Python layout tests passed; lint0 errors/13 existing warnings; typecheck/build/Node import/package smoke passed. Root graph remains44 (<45), threshold unchanged. Published artifacts1645 files. Failure-before exporter receipt preserved. No real-model/physical-microphone acceptance is inferred.

## Per-PR decisions

| PR | Decision | Rationale |
|---|---|---|
| [#233](https://github.com/asrjs/speech-recognition/pull/233) | REJECT | Comment/deprecation/catch-binding changes are cosmetic or remove useful guidance; bundled lint cleanup already landed. No demonstrated runtime defect. |
| [#232](https://github.com/asrjs/speech-recognition/pull/232) | REJECT | Optional loop/async/copy cleanup without demonstrated bug or measured benefit. Async-wrapper removal also changes exception collection semantics in general; no need to take mixed stale changes. |
| [#231](https://github.com/asrjs/speech-recognition/pull/231) | REJECT | Comment/deprecation/catch-binding changes are cosmetic or remove useful guidance; bundled lint cleanup already landed. No demonstrated runtime defect. |
| [#230](https://github.com/asrjs/speech-recognition/pull/230) | REJECT | Comment/deprecation/catch-binding changes are cosmetic or remove useful guidance; bundled lint cleanup already landed. No demonstrated runtime defect. |
| [#229](https://github.com/asrjs/speech-recognition/pull/229) | ADAPT | Take only chunk validation/boundary tests, not stale bundled lint edits. |
| [#228](https://github.com/asrjs/speech-recognition/pull/228) | INCORPORATED | Useful behavior or coverage is on main via PR235/5037c83a. TEN-VAD adaptation preserves pointer ABI and destruction failure checks; no claim of full init-path certification. |
| [#220](https://github.com/asrjs/speech-recognition/pull/220) | INCORPORATED | Useful behavior or coverage is on main via PR235/5037c83a. TEN-VAD adaptation preserves pointer ABI and destruction failure checks; no claim of full init-path certification. |
| [#219](https://github.com/asrjs/speech-recognition/pull/219) | REJECT | Switches non-security stream IDs to crypto.randomUUID without portable fallback; host/secure-context availability differs. |
| [#218](https://github.com/asrjs/speech-recognition/pull/218) | REJECT | for-in also sees inherited enumerable keys, unlike Object.keys; no semantic regression/ownership test or measured benefit. |
| [#217](https://github.com/asrjs/speech-recognition/pull/217) | INCORPORATED | Useful behavior or coverage is on main via PR235/5037c83a. TEN-VAD adaptation preserves pointer ABI and destruction failure checks; no claim of full init-path certification. |
| [#216](https://github.com/asrjs/speech-recognition/pull/216) | REJECT | Actually enables prompt-token injection using shared mutable contextBuilder across awaits. Needs per-call state/concurrency and real-model quality proof; not just unused-variable cleanup. |
| [#215](https://github.com/asrjs/speech-recognition/pull/215) | INCORPORATED | Useful behavior or coverage is on main via PR235/5037c83a. TEN-VAD adaptation preserves pointer ABI and destruction failure checks; no claim of full init-path certification. |
| [#214](https://github.com/asrjs/speech-recognition/pull/214) | INCORPORATED | Useful behavior or coverage is on main via PR235/5037c83a. TEN-VAD adaptation preserves pointer ABI and destruction failure checks; no claim of full init-path certification. |
| [#213](https://github.com/asrjs/speech-recognition/pull/213) | INCORPORATED | Useful behavior or coverage is on main via PR235/5037c83a. TEN-VAD adaptation preserves pointer ABI and destruction failure checks; no claim of full init-path certification. |
| [#212](https://github.com/asrjs/speech-recognition/pull/212) | REJECT | Single-chunk direct return changes internal ownership in browser/VAD helpers. NOT the stream-cache bug; ownership and performance acceptance unverified. |
| [#210](https://github.com/asrjs/speech-recognition/pull/210) | REJECT | Overlapping public output/option type rewrites narrow unknown executor contracts or reshape results without source-derived compatibility evidence. |
| [#208](https://github.com/asrjs/speech-recognition/pull/208) | INCORPORATED | Useful behavior or coverage is on main via PR235/5037c83a. TEN-VAD adaptation preserves pointer ABI and destruction failure checks; no claim of full init-path certification. |
| [#206](https://github.com/asrjs/speech-recognition/pull/206) | REJECT | Overlapping public output/option type rewrites narrow unknown executor contracts or reshape results without source-derived compatibility evidence. |
| [#205](https://github.com/asrjs/speech-recognition/pull/205) | INCORPORATED | Useful behavior or coverage is on main via PR235/5037c83a. TEN-VAD adaptation preserves pointer ABI and destruction failure checks; no claim of full init-path certification. |
| [#204](https://github.com/asrjs/speech-recognition/pull/204) | REJECT DUPLICATE OPTIMIZATION | Same TypedArray.set/subarray proposal in many PRs; no demonstrated correctness bug or measured benefit. Declined without declaring it unsafe or tested. |
| [#203](https://github.com/asrjs/speech-recognition/pull/203) | REJECT | FFT/reflection/inner-loop rewrite lacks representative numerical equivalence and benchmark acceptance. No speed or parity PASS. |
| [#201](https://github.com/asrjs/speech-recognition/pull/201) | REJECT | Overlapping public output/option type rewrites narrow unknown executor contracts or reshape results without source-derived compatibility evidence. |
| [#200](https://github.com/asrjs/speech-recognition/pull/200) | REJECT DUPLICATE OPTIMIZATION | Same TypedArray.set/subarray proposal in many PRs; no demonstrated correctness bug or measured benefit. Declined without declaring it unsafe or tested. |
| [#198](https://github.com/asrjs/speech-recognition/pull/198) | REJECT | Overlapping public output/option type rewrites narrow unknown executor contracts or reshape results without source-derived compatibility evidence. |
| [#197](https://github.com/asrjs/speech-recognition/pull/197) | REJECT DUPLICATE OPTIMIZATION | Same TypedArray.set/subarray proposal in many PRs; no demonstrated correctness bug or measured benefit. Declined without declaring it unsafe or tested. |
| [#196](https://github.com/asrjs/speech-recognition/pull/196) | REJECT DUPLICATE OPTIMIZATION | Same TypedArray.set/subarray proposal in many PRs; no demonstrated correctness bug or measured benefit. Declined without declaring it unsafe or tested. |
| [#195](https://github.com/asrjs/speech-recognition/pull/195) | REJECT | Optional loop/async/copy cleanup without demonstrated bug or measured benefit. Async-wrapper removal also changes exception collection semantics in general; no need to take mixed stale changes. |
| [#194](https://github.com/asrjs/speech-recognition/pull/194) | REJECT | Overlapping public output/option type rewrites narrow unknown executor contracts or reshape results without source-derived compatibility evidence. |
| [#192](https://github.com/asrjs/speech-recognition/pull/192) | REJECT DUPLICATE OPTIMIZATION | Same TypedArray.set/subarray proposal in many PRs; no demonstrated correctness bug or measured benefit. Declined without declaring it unsafe or tested. |
| [#191](https://github.com/asrjs/speech-recognition/pull/191) | ADAPT | Take expected-warning assertion; existing afterEach restoreAllMocks supplies failure-safe spy cleanup. |
| [#190](https://github.com/asrjs/speech-recognition/pull/190) | REJECT DUPLICATE OPTIMIZATION | Same TypedArray.set/subarray proposal in many PRs; no demonstrated correctness bug or measured benefit. Declined without declaring it unsafe or tested. |
| [#189](https://github.com/asrjs/speech-recognition/pull/189) | REJECT DUPLICATE OPTIMIZATION | Same TypedArray.set/subarray proposal in many PRs; no demonstrated correctness bug or measured benefit. Declined without declaring it unsafe or tested. |
| [#188](https://github.com/asrjs/speech-recognition/pull/188) | SUPERSEDED | Current streamToBytes has one-chunk path and X-ASR helper uses normalized path.posix.relative; no new fix needed. |
| [#187](https://github.com/asrjs/speech-recognition/pull/187) | REJECT | spawnSync uses executable/argv with no shell. No demonstrated shell injection; new broad guards and Windows path assumptions are not portable. |
| [#185](https://github.com/asrjs/speech-recognition/pull/185) | REJECT | Constructor only Math.max-clamps window size, permits2.5. Candidate probs[i-windowSize] can address fractional property ->NaN while old shift-based window stays finite. |
| [#184](https://github.com/asrjs/speech-recognition/pull/184) | SUPERSEDED | Current X-ASR relative helper already normalizes both paths independent of host; prefix workaround unnecessary. |
| [#183](https://github.com/asrjs/speech-recognition/pull/183) | REJECT | Actual diff includes stream-cache single-chunk alias, beyond lockfile/path edits: retained yielded buffer can mutate cache after completion. |
| [#181](https://github.com/asrjs/speech-recognition/pull/181) | REJECT | Current compact renderer escapes dynamic labels/values before innerHTML. PR does not demonstrate current XSS; broad DOM rewrite declined. |
| [#180](https://github.com/asrjs/speech-recognition/pull/180) | REJECT | Sequential pipeline/context behavior already implemented and covered. Duplicate benchmark-labelled test/comment is unnecessary. |
| [#167](https://github.com/asrjs/speech-recognition/pull/167) | SUPERSEDED | Current vitest.config.ts already contains legacy and @asrjs pipeline/chunking/quality/post-processing aliases; current public-subpath tests and CI passed. |
| [#166](https://github.com/asrjs/speech-recognition/pull/166) | SUPERSEDED | Current vitest.config.ts already contains legacy and @asrjs pipeline/chunking/quality/post-processing aliases; current public-subpath tests and CI passed. |
| [#165](https://github.com/asrjs/speech-recognition/pull/165) | SUPERSEDED | Current vitest.config.ts already contains legacy and @asrjs pipeline/chunking/quality/post-processing aliases; current public-subpath tests and CI passed. |
| [#164](https://github.com/asrjs/speech-recognition/pull/164) | SUPERSEDED | Current vitest.config.ts already contains legacy and @asrjs pipeline/chunking/quality/post-processing aliases; current public-subpath tests and CI passed. |
| [#163](https://github.com/asrjs/speech-recognition/pull/163) | REJECT DUPLICATE OPTIMIZATION | Same TypedArray.set/subarray proposal in many PRs; no demonstrated correctness bug or measured benefit. Declined without declaring it unsafe or tested. |
| [#162](https://github.com/asrjs/speech-recognition/pull/162) | SUPERSEDED | Current vitest.config.ts already contains legacy and @asrjs pipeline/chunking/quality/post-processing aliases; current public-subpath tests and CI passed. |
| [#161](https://github.com/asrjs/speech-recognition/pull/161) | SUPERSEDED | Current vitest.config.ts already contains legacy and @asrjs pipeline/chunking/quality/post-processing aliases; current public-subpath tests and CI passed. |
| [#160](https://github.com/asrjs/speech-recognition/pull/160) | REJECT DUPLICATE OPTIMIZATION | Same TypedArray.set/subarray proposal in many PRs; no demonstrated correctness bug or measured benefit. Declined without declaring it unsafe or tested. |
| [#159](https://github.com/asrjs/speech-recognition/pull/159) | REJECT DUPLICATE OPTIMIZATION | Same TypedArray.set/subarray proposal in many PRs; no demonstrated correctness bug or measured benefit. Declined without declaring it unsafe or tested. |
| [#158](https://github.com/asrjs/speech-recognition/pull/158) | REJECT DUPLICATE OPTIMIZATION | Same TypedArray.set/subarray proposal in many PRs; no demonstrated correctness bug or measured benefit. Declined without declaring it unsafe or tested. |
| [#157](https://github.com/asrjs/speech-recognition/pull/157) | REJECT | Optional loop/async/copy cleanup without demonstrated bug or measured benefit. Async-wrapper removal also changes exception collection semantics in general; no need to take mixed stale changes. |
| [#156](https://github.com/asrjs/speech-recognition/pull/156) | SUPERSEDED | Current vitest.config.ts already contains legacy and @asrjs pipeline/chunking/quality/post-processing aliases; current public-subpath tests and CI passed. |
| [#155](https://github.com/asrjs/speech-recognition/pull/155) | SUPERSEDED | Current vitest.config.ts already contains legacy and @asrjs pipeline/chunking/quality/post-processing aliases; current public-subpath tests and CI passed. |
| [#154](https://github.com/asrjs/speech-recognition/pull/154) | REJECT DUPLICATE OPTIMIZATION | Same TypedArray.set/subarray proposal in many PRs; no demonstrated correctness bug or measured benefit. Declined without declaring it unsafe or tested. |
| [#153](https://github.com/asrjs/speech-recognition/pull/153) | SUPERSEDED | Current vitest.config.ts already contains legacy and @asrjs pipeline/chunking/quality/post-processing aliases; current public-subpath tests and CI passed. |
| [#152](https://github.com/asrjs/speech-recognition/pull/152) | SUPERSEDED | Current vitest.config.ts already contains legacy and @asrjs pipeline/chunking/quality/post-processing aliases; current public-subpath tests and CI passed. |
| [#151](https://github.com/asrjs/speech-recognition/pull/151) | SUPERSEDED | Current vitest.config.ts already contains legacy and @asrjs pipeline/chunking/quality/post-processing aliases; current public-subpath tests and CI passed. |
| [#150](https://github.com/asrjs/speech-recognition/pull/150) | REJECT DUPLICATE OPTIMIZATION | Same TypedArray.set/subarray proposal in many PRs; no demonstrated correctness bug or measured benefit. Declined without declaring it unsafe or tested. |
| [#149](https://github.com/asrjs/speech-recognition/pull/149) | REJECT | FFT/reflection/inner-loop rewrite lacks representative numerical equivalence and benchmark acceptance. No speed or parity PASS. |
| [#148](https://github.com/asrjs/speech-recognition/pull/148) | REJECT | FFT/reflection/inner-loop rewrite lacks representative numerical equivalence and benchmark acceptance. No speed or parity PASS. |
| [#147](https://github.com/asrjs/speech-recognition/pull/147) | ADAPT | Canonical graph names must point to converted graphs. Adapt in-place naming only, reject /app fixture path; add stdlib orchestration tests and CI step. No numerical FP16 parity claim. |
| [#146](https://github.com/asrjs/speech-recognition/pull/146) | ADAPT | Take sentence join/punctuation/empty/missing-text tests; explicitly mark deliberate invalid runtime input. |
| [#144](https://github.com/asrjs/speech-recognition/pull/144) | CONSOLIDATE | Validation cases included in the selected #229 adaptation; close after its tested replacement merges. |
| [#143](https://github.com/asrjs/speech-recognition/pull/143) | CONSOLIDATE | Validation cases included in the selected #229 adaptation; close after its tested replacement merges. |
| [#142](https://github.com/asrjs/speech-recognition/pull/142) | REJECT | Optional loop/async/copy cleanup without demonstrated bug or measured benefit. Async-wrapper removal also changes exception collection semantics in general; no need to take mixed stale changes. |

Evidence is under evidence/pr-cleanup-2026-10-23. CI and merge are not declared successful by this pre-publication snapshot.

Publication adjustment: initial push rejected by GitHub because OAuth lacks workflow scope. Workflow edits removed; mandatory Python layout tests now run via tests/whisper-export-variant-layout.test.ts without skips. No permission change or gate weakening.

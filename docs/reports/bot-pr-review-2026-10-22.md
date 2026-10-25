# Bot PR review - 2026-10-22

Read-only review of open pull requests returned by GitHub. This report records workflow status separately from code review; CI success alone is not a code PASS. The live API returned 76 open PRs in asrjs/speech-recognition and 0 in asrjs/demos, versus the expected count of 78.

## Inventory

| PR | Title | Head SHA | Base SHA | Changed files | Latest CI workflow |
|---:|---|---|---|---:|---|
| [234](https://github.com/asrjs/speech-recognition/pull/234) | Bolt: Optimize element copying in padToFrames using native Subarray | 8a393f9f1efb | af8d147a6f09 | 7 | CI:success |
| [233](https://github.com/asrjs/speech-recognition/pull/233) | 🧹 reword comment in ctc-viterbi.ts to avoid false-positive dead code detection | 2968c206b536 | af8d147a6f09 | 6 | CI:failure |
| [232](https://github.com/asrjs/speech-recognition/pull/232) | ⚡ perf(tokenizer): replace findIndex with indexOf in MedAsrTextTokenizer candidate resolution | 6c90eaa5f17c | af8d147a6f09 | 6 | CI:failure |
| [231](https://github.com/asrjs/speech-recognition/pull/231) | 🧹 Remove deprecated comment from ctc.ts | 414cec85e769 | af8d147a6f09 | 6 | CI:failure |
| [230](https://github.com/asrjs/speech-recognition/pull/230) | 🧹 update empty catch block in x-asr executor | 514820b529e8 | af8d147a6f09 | 6 | CI:failure |
| [229](https://github.com/asrjs/speech-recognition/pull/229) | 🧪 add edge case tests for whisper chunk planning | 1d4d3bbd303f | af8d147a6f09 | 6 | CI:failure |
| [228](https://github.com/asrjs/speech-recognition/pull/228) | Bolt: fix ten-vad-worker memory leak on early destruction | 84a377ff9fa2 | af8d147a6f09 | 6 | CI:failure |
| [227](https://github.com/asrjs/speech-recognition/pull/227) | Bolt: Fix missing vitest path aliases for worker and directory-based models | 353066b898dc | af8d147a6f09 | 7 | CI:success |
| [226](https://github.com/asrjs/speech-recognition/pull/226) | Bolt: skip typed-array allocation and copy for single-chunk audio hot paths | 3067dc43b2a0 | af8d147a6f09 | 3 | CI:failure |
| [220](https://github.com/asrjs/speech-recognition/pull/220) | Bolt: fix memory leak in ten-vad worker when initialization fails | 79376151e025 | 609e53e22b7f | 1 | CI:success |
| [219](https://github.com/asrjs/speech-recognition/pull/219) | 🔒 Security Fix: Replace Math.random with crypto.randomUUID in streaming detector | 450f151a1e4a | 609e53e22b7f | 2 | CI:success |
| [218](https://github.com/asrjs/speech-recognition/pull/218) | ⚡ perf: replace redundant Object.keys().find for logits in Whisper executor | aa79558355d7 | 609e53e22b7f | 1 | CI:success |
| [217](https://github.com/asrjs/speech-recognition/pull/217) | test: add compressionRatioGate edge case coverage | 4bb7bb2e0722 | 609e53e22b7f | 1 | CI:success |
| [216](https://github.com/asrjs/speech-recognition/pull/216) | 🧹 resolve unused variable suppression in EnhancedWhisperExecutor | 03d139c5cc30 | 609e53e22b7f | 2 | CI:success |
| [215](https://github.com/asrjs/speech-recognition/pull/215) | 🧪 test: add edge case tests for normalizePcmInput | 98855f5e3ffe | 609e53e22b7f | 1 | CI:success |
| [214](https://github.com/asrjs/speech-recognition/pull/214) | 🧪 test: add edge case test coverage for confidenceFromLogits | 9c045d760b77 | 609e53e22b7f | 1 | CI:success |
| [213](https://github.com/asrjs/speech-recognition/pull/213) | 🧪 Add unit tests for estimateSecondsPerOutputFrame | 7634d9344db3 | 609e53e22b7f | 1 | CI:success |
| [212](https://github.com/asrjs/speech-recognition/pull/212) | Bolt: Add fast paths to avoid typed-array copies for single chunks | 1ba2ab6cf515 | 609e53e22b7f | 2 | CI:success |
| [211](https://github.com/asrjs/speech-recognition/pull/211) | Bolt: Add fast-path to array concatenation loops for single chunks | 33fc7c076dad | 609e53e22b7f | 3 | CI:success |
| [210](https://github.com/asrjs/speech-recognition/pull/210) | Bolt: Improve type safety in production pipeline types | 20acd07eea32 | 609e53e22b7f | 2 | CI:success |
| [209](https://github.com/asrjs/speech-recognition/pull/209) | Bolt: Optimize single-chunk stream buffering in UrlAssetHandle | c82a1c2ebcab | 609e53e22b7f | 1 | CI:success |
| [208](https://github.com/asrjs/speech-recognition/pull/208) | Bolt: Fix memory leak in TEN-VAD worker initialization failures | 561fee1dc1f2 | 609e53e22b7f | 1 | CI:success |
| [207](https://github.com/asrjs/speech-recognition/pull/207) | Bolt: optimize UrlAssetHandle stream cache copying | 4052397b9a37 | 609e53e22b7f | 1 | CI:success |
| [206](https://github.com/asrjs/speech-recognition/pull/206) | Bolt: Improve type safety for pipeline transcript words and segments | 93a064875aa2 | 609e53e22b7f | 2 | CI:success |
| [205](https://github.com/asrjs/speech-recognition/pull/205) | Bolt: fix TEN-VAD worker handleDispose memory leak | 67892b545516 | 609e53e22b7f | 1 | CI:success |
| [204](https://github.com/asrjs/speech-recognition/pull/204) | Bolt: optimize array copying in WhisperMelProcessor | 37686244ac6e | 609e53e22b7f | 1 | CI:success |
| [203](https://github.com/asrjs/speech-recognition/pull/203) | Bolt: Inline getReflectPaddedSample in WhisperMelProcessor | b4ed1ff38233 | 609e53e22b7f | 1 | CI:success |
| [202](https://github.com/asrjs/speech-recognition/pull/202) | Bolt: improve type safety in production whisper pipeline output | 1eb9ad0a3cb8 | 609e53e22b7f | 2 | CI:success |
| [201](https://github.com/asrjs/speech-recognition/pull/201) | Bolt: improve type safety in pipeline output types | 47f6dbd3f348 | 609e53e22b7f | 2 | CI:success |
| [200](https://github.com/asrjs/speech-recognition/pull/200) | Bolt: Optimize array copying in Whisper mel padding | deaa2f9a2865 | 609e53e22b7f | 1 | CI:success |
| [199](https://github.com/asrjs/speech-recognition/pull/199) | Bolt: Optimize cache chunk allocation for UrlAssetHandle | 8828d1f28c73 | 609e53e22b7f | 1 | CI:success |
| [198](https://github.com/asrjs/speech-recognition/pull/198) | Bolt: enforce strict typing for ProductionTranscript | f1dca6351ae9 | 609e53e22b7f | 2 | CI:success |
| [197](https://github.com/asrjs/speech-recognition/pull/197) | Bolt: Improve Whisper mel padding performance with TypedArray.set | 36d729e79e02 | 609e53e22b7f | 1 | CI:success |
| [196](https://github.com/asrjs/speech-recognition/pull/196) | Bolt: optimize TypedArray copying in padToFrames | 9071c4111305 | 609e53e22b7f | 1 | CI:success |
| [195](https://github.com/asrjs/speech-recognition/pull/195) | Bolt: Optimize padToFrames and remove redundant async wrappers | 6cff0aaada0c | 609e53e22b7f | 4 | CI:success |
| [194](https://github.com/asrjs/speech-recognition/pull/194) | Bolt: Improve type safety for production pipeline transcript words and segments | 70d843ba3aff | 609e53e22b7f | 2 | CI:success |
| [193](https://github.com/asrjs/speech-recognition/pull/193) | Bolt: optimize chunk allocations and remove redundant async closures | db3a543a0773 | 609e53e22b7f | 4 | CI:success |
| [192](https://github.com/asrjs/speech-recognition/pull/192) | Bolt: Optimize audio mel padding with TypedArray set() | ca7004a2153a | 609e53e22b7f | 1 | CI:success |
| [191](https://github.com/asrjs/speech-recognition/pull/191) | Bolt: Mock expected console.warn in cache fallback test | c3c37f8e723d | 609e53e22b7f | 1 | CI:success |
| [190](https://github.com/asrjs/speech-recognition/pull/190) | Bolt: Optimize TypedArray copy operations in audio preprocessing | 3d1b2f73dbad | 16b3772cc8fd | 2 | CI:success |
| [189](https://github.com/asrjs/speech-recognition/pull/189) | Bolt: reduce typed-array copies in padToFrames | c9785a52416f | 286799b0eba3 | 1 | CI:success |
| [188](https://github.com/asrjs/speech-recognition/pull/188) | Bolt: optimize memory by returning single stream chunks directly | 6a5f9e515bca | e840269f0d62 | 2 | CI:success |
| [187](https://github.com/asrjs/speech-recognition/pull/187) | security: sanitize spawnSync input parameters in node-nemo-shared-audio-parity | 5ed1c11fc3ea | e840269f0d62 | 4 | CI:success |
| [186](https://github.com/asrjs/speech-recognition/pull/186) | Code health: remove unused import type OrtModuleLike in nemo-tdt executor | 16f794fe24cf | e840269f0d62 | 0 | CI:failure |
| [185](https://github.com/asrjs/speech-recognition/pull/185) | Optimize smoothProb in VadPostprocessor | ca9b7532f1b2 | e840269f0d62 | 3 | CI:failure |
| [184](https://github.com/asrjs/speech-recognition/pull/184) | Verify ModelFileProgress usage in Parakeet compat | 6a4eaf8d09a3 | e840269f0d62 | 1 | CI:success |
| [183](https://github.com/asrjs/speech-recognition/pull/183) | Optimize Uint8Array chunk assembly in streamToBytes | 2a89b3f802dc | e840269f0d62 | 3 | CI:success |
| [182](https://github.com/asrjs/speech-recognition/pull/182) | remove unused OrtModuleLike import from nemo-rnnt executor | beda359ccc88 | e840269f0d62 | 1 | CI:failure |
| [181](https://github.com/asrjs/speech-recognition/pull/181) | Security Fix: Eliminate XSS risk in Browser Compact Stats Renderer | 6c3bd82c55c6 | e840269f0d62 | 3 | CI:success |
| [180](https://github.com/asrjs/speech-recognition/pull/180) | Document sequential stage rationale in runPipelineStages | 9571bf3d81a0 | e840269f0d62 | 3 | CI:success |
| [167](https://github.com/asrjs/speech-recognition/pull/167) | Bolt: Add missing vitest path aliases for pipeline and other modules | 64d323d396f7 | 850a51f159da | 1 | no workflow run returned |
| [166](https://github.com/asrjs/speech-recognition/pull/166) | Bolt: fix missing path aliases for submodules in vitest.config.ts | 74f7423ae1e2 | 850a51f159da | 1 | no workflow run returned |
| [165](https://github.com/asrjs/speech-recognition/pull/165) | Bolt: add missing module resolution aliases to vitest.config.ts | 188c28a1e3e8 | 850a51f159da | 1 | no workflow run returned |
| [164](https://github.com/asrjs/speech-recognition/pull/164) | Bolt: add missing path aliases to vitest.config.ts | dca8030da11a | 850a51f159da | 1 | no workflow run returned |
| [163](https://github.com/asrjs/speech-recognition/pull/163) | Bolt: optimize padToFrames allocation via native subarray memory set | 9647932adef2 | 850a51f159da | 2 | no workflow run returned |
| [162](https://github.com/asrjs/speech-recognition/pull/162) | Bolt: Add missing vitest path aliases for pipeline and chunking | 062d72fa48f2 | 850a51f159da | 1 | no workflow run returned |
| [161](https://github.com/asrjs/speech-recognition/pull/161) | Bolt: Add missing vitest alias for pipeline module | e87efdfb6162 | 850a51f159da | 1 | no workflow run returned |
| [160](https://github.com/asrjs/speech-recognition/pull/160) | Bolt: optimize array copying in padToFrames using native set | e78b6fc15f46 | 850a51f159da | 1 | no workflow run returned |
| [159](https://github.com/asrjs/speech-recognition/pull/159) | Bolt: optimize typed array copy in WhisperMelProcessor | 06998808c755 | 850a51f159da | 1 | no workflow run returned |
| [158](https://github.com/asrjs/speech-recognition/pull/158) | Bolt: Optimize Whisper mel padding with native TypedArray.set() | 0adf4ec82124 | 850a51f159da | 1 | no workflow run returned |
| [157](https://github.com/asrjs/speech-recognition/pull/157) | Bolt: optimize token index mapping loops in post-processing | 298b4c4e59b9 | 850a51f159da | 3 | no workflow run returned |
| [156](https://github.com/asrjs/speech-recognition/pull/156) | Bolt: Fix pipeline vitest aliases | d3bf0356d936 | 850a51f159da | 1 | no workflow run returned |
| [155](https://github.com/asrjs/speech-recognition/pull/155) | Bolt: Add missing pipeline alias to vitest config | 93870b7150e5 | 850a51f159da | 1 | no workflow run returned |
| [154](https://github.com/asrjs/speech-recognition/pull/154) | Bolt: Optimize array copying in WhisperMelProcessor | 44e61f738d07 | 850a51f159da | 2 | no workflow run returned |
| [153](https://github.com/asrjs/speech-recognition/pull/153) | Bolt: add missing vitest aliases for pipeline, quality, chunking, and post-processing modules | 1311aadc4fc2 | 850a51f159da | 1 | no workflow run returned |
| [152](https://github.com/asrjs/speech-recognition/pull/152) | Bolt: add missing test module aliases to vitest.config.ts | 02687d1d40b5 | 850a51f159da | 1 | no workflow run returned |
| [151](https://github.com/asrjs/speech-recognition/pull/151) | Bolt: add missing pipeline export aliases to vitest config | be9a32589ad5 | 850a51f159da | 1 | no workflow run returned |
| [150](https://github.com/asrjs/speech-recognition/pull/150) | Bolt: Optimize padToFrames array copying with TypedArray.set | e8a704be0471 | d6bfbd341267 | 1 | no workflow run returned |
| [149](https://github.com/asrjs/speech-recognition/pull/149) | perf: optimize whisper mel real fft performance | 1120e010f3af | d6bfbd341267 | 1 | no workflow run returned |
| [148](https://github.com/asrjs/speech-recognition/pull/148) | Bolt: optimize whisper mel processor inner loops and fft | e19521bd7b1e | d6bfbd341267 | 1 | no workflow run returned |
| [147](https://github.com/asrjs/speech-recognition/pull/147) | Fix base names conversion for fp16 model variants | 2b944dffe1a1 | d6bfbd341267 | 2 | no workflow run returned |
| [146](https://github.com/asrjs/speech-recognition/pull/146) | [Testing Improvement] Add unit tests for joinTranscriptWords | 12108694c083 | d6bfbd341267 | 1 | no workflow run returned |
| [145](https://github.com/asrjs/speech-recognition/pull/145) | [Performance] Replace synchronous file I/O with async in local model loading | 94b387078de6 | d6bfbd341267 | 7 | no workflow run returned |
| [144](https://github.com/asrjs/speech-recognition/pull/144) | [Testing Improvement] Add tests for validatePositiveNumber | 621bc79464a9 | d6bfbd341267 | 1 | no workflow run returned |
| [143](https://github.com/asrjs/speech-recognition/pull/143) | [testing improvement] Add tests for validatePositiveInteger in whisper-chunking | 21affb0c74d8 | d6bfbd341267 | 1 | no workflow run returned |
| [142](https://github.com/asrjs/speech-recognition/pull/142) | Remove redundant async wrapper in session dispose map | daa06882664e | d6bfbd341267 | 1 | no workflow run returned |

CI:success or CI:failure reports the associated GitHub Actions workflow result only. "no workflow run returned" is unverified, not a pass. Most PRs are automated Jules/Bolt work; numerous branches are stale or repeat the same small change.

## Verified issues and review decisions

- **major, [182](https://github.com/asrjs/speech-recognition/pull/182) - src/models/nemo-rnnt/executor.ts:17,40.** Removes OrtModuleLike while LoadedExecutorState.ort still uses it; GitHub CI Build fails TS2304 on Node 22/24. Decision: Reject.
- **major, [227](https://github.com/asrjs/speech-recognition/pull/227) - tests/smoke/browser-root-import-graph.mjs:52,173.** Raises regression ceiling 78 to 79 and hardcodes false around the <45 failure guard; CI success masks unmet target. Decision: Reject; false PASS.
- **major, [234](https://github.com/asrjs/speech-recognition/pull/234) - tests/smoke/browser-root-import-graph.mjs:173.** Gates target failure on ASRJS_ROOT_IMPORT_STRICT_BUDGET, which CI does not set, so process exits 0 while output says <45 UNMET. Keep this separate from its likely-equivalent TypedArray copy hunk. Decision: Reject guard under no-false-PASS requirement.
- **major, [226](https://github.com/asrjs/speech-recognition/pull/226), [211](https://github.com/asrjs/speech-recognition/pull/211), [209](https://github.com/asrjs/speech-recognition/pull/209), [207](https://github.com/asrjs/speech-recognition/pull/207), [199](https://github.com/asrjs/speech-recognition/pull/199), [193](https://github.com/asrjs/speech-recognition/pull/193) - src/io/handles.ts:432-439.** With a one-chunk response and MemoryAssetCache (which stores the supplied value by reference), the new shortcut caches the same Uint8Array yielded to the consumer. After the iterator has fully completed, mutating a retained chunk changes the cached bytes; the old copy-based path is independent after completion. Mutating before completion can affect both implementations, so that is not the new regression. Decision: Reject shortcut or snapshot with a copy before cache.set.
- **major, [216](https://github.com/asrjs/speech-recognition/pull/216) - src/models/whisper-seq2seq/enhanced-executor.ts:261-288.** Shared contextBuilder resets/reads/adds across awaits; concurrent calls can mix tokens. No concurrency test in PR. Decision: Require per-call state or serialization and test.
- **major, [145](https://github.com/asrjs/speech-recognition/pull/145) - src/models/whisper-seq2seq/local-file.ts:86-200.** Changes exported loader sync to Promise and breaks callers; comparison benchmark fails to await new async loader. Decision: Reject wholesale.
- **major, [187](https://github.com/asrjs/speech-recognition/pull/187) - tests/nemo-shared-audio-parity.test.ts:~52.** Path test expects /models unchanged; Windows path.win32.normalize returns backslash-rooted form. CI was Linux; argv-based spawnSync does not show injection. Decision: Port guards only after portable test.
- **major, [228](https://github.com/asrjs/speech-recognition/pull/228), [220](https://github.com/asrjs/speech-recognition/pull/220), [208](https://github.com/asrjs/speech-recognition/pull/208), [205](https://github.com/asrjs/speech-recognition/pull/205) - src/runtime/ten-vad-worker.ts:82-92,229-239.** Init allocates pointers before create; failed create leaves vadHandle zero and current dispose skips all frees. #228 fixes cleanup but no targeted test; CI fails lint. Decision: Adapt fix with failure/partial-allocation tests.
- **major, [206](https://github.com/asrjs/speech-recognition/pull/206), [210](https://github.com/asrjs/speech-recognition/pull/210), [202](https://github.com/asrjs/speech-recognition/pull/202), [201](https://github.com/asrjs/speech-recognition/pull/201), [198](https://github.com/asrjs/speech-recognition/pull/198), [194](https://github.com/asrjs/speech-recognition/pull/194) - src/pipeline/types.ts:29-54; src/pipeline/whisper-production-pipeline.ts:115-185.** Type variants may drop source fields; #202 fabricates probability .9, #201 options unknown, #194 asserts native shape. Decision: Source-derived types, preserve fields.
- **major, [147](https://github.com/asrjs/speech-recognition/pull/147) - tools/whisper-onnx-export/export_whisper.py:1048.** FP16 converter likely receives suffixed names although helper expects canonical graph names; PR also changes fixture to absolute /app path. Decision: Port converter hunk only with test and portable fixture.
- **minor, [231](https://github.com/asrjs/speech-recognition/pull/231) - src/models/lasr-ctc/ctc.ts:11.** Removes @deprecated editor annotation while adjacent comment recommends canonical import. Decision: Do not take absent policy change.

## Best candidates to adapt

- Test-only edge coverage: #213, #214, #215, #217. Rebase and rerun on current main.
- TEN-VAD cleanup: #228/#208 contain a useful leak fix; add a focused failed-init and partial-allocation regression test before porting.
- FP16 export correction: #147 appears useful as a narrow change; keep its test fixture portable.
- Internal micro-optimizations #212/#218 and the padToFrames copy in #234 are plausible, but speed claims are unverified. Split away unsafe cache edits and smoke-gate changes.

## Duplicate clusters

- **Safe test templates** (#213, #214, #215, #217): Best candidates to cherry-pick/rebase and rerun on current main: isolated edge-case coverage.
- **Whisper padToFrames duplicates** (#234, #204, #200, #197, #196, #192, #190, #189, #163, #160, #159, #158, #154, #150): At most one consolidated TypedArray.set/subarray code hunk. Likely correct, speedup unverified.
- **Unsafe cache alias** (#226, #211, #209, #207, #199, #193): After iterator completion, the new one-chunk shortcut keeps cache.bytes aliased to the consumer-held chunk. MemoryAssetCache stores the value by reference, so later mutation changes cached model bytes; baseline copy avoided this post-completion alias. Mutation before completion may affect both and is not the new regression. Keep snapshot copy or slice before cache write.
- **TEN-VAD cleanup** (#228, #220, #208, #205): Real leak; adapt with test. #208 guards pointers; #228 latest base but no regression test and CI lint fails.
- **Transcript types** (#210, #206, #202, #201, #198, #194): DX goal valid; adapt source-derived contract, preserve unknown fields.
- **Whisper prompt context** (#216): Shared mutable context may race. Require per-call state/serialization.
- **Aliases** (#227, #167, #166, #165, #164, #162, #161, #156, #155, #153, #152, #151): Mostly redundant; current main already has aliases.
- **FFT/mel parity** (#203, #149, #148): Do not accept until numeric parity on boundaries and representative audio.
- **Already in main** (#188, #184, #183, #180, #186): Several path/stream optimizations already exist; #186 has no diff.
- **Whisper loader API** (#145): Reject sync-to-async breaking change and invalid benchmark.

## Unverified remainder

Not every diff was line-by-line reviewed. CI status recorded for all. No workflow run returned means unverified, never PASS.
- #233-#229 and #226 latest workflow failed; exact failure causes not all inspected.
- #167-#142 had no workflow run returned; mostly stale duplicates.
- #185/#186 fail on existing X-ASR contract tests; #186 is zero-diff.
- Performance claims for copy/FFT were not independently reproduced.
- Demos has no open PRs.

No PR comments/reviews, merges, pushes, source edits, or other remote actions were made.
## Applied decisions (2026-10-22)

User authorized accept/adapt/reject. Closed #227 (353066b898dc), #234 (8a393f9f1efb) and #226 (3067dc43b2a0) after confirming unchanged head SHAs. No comments posted. #234 copy optimization remains a separate possible adaptation; closure rejects its combined unsafe patch. Test and TEN-VAD adaptations are pending independent validation; no merge claim.

### Execution update

Closed #182 after confirming its diff removes the still-used OrtModuleLike import. Ported #213 as local commit 4a9da7e5 (29 tests independently passed); #214/#215/#217 as d45993f7 (56 tests independently passed). Adapted TEN-VAD disposal idea with pointer guards and pointer reset. New RESET recreation-failure test failed before the fix and passed afterward; TEN-VAD ABI/browser + CTC tests: 58 passed. Typecheck passed; lint 0 errors/13 warnings. These are local commits, not GitHub merges. Test-source PRs remain open until publication. Broader failed-initialization/partial-allocation paths and remaining PR candidates are still pending; no full-suite or browser claim for this new VAD change.

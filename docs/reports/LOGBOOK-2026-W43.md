# Logbook — 2026 W43

## Scope

This entry records the verified October 25 continuation only. It does not infer
unrecorded work earlier in the week. Earlier October 23–24 receipts remain in the
versioned evidence archive and are not promoted to current certification.

## October 25 — modern model integrations and real-weight acceptance

Completed shared local-artifact registration for SenseVoice, GigaAM CTC/RNN-T,
X-ASR and the separate Moonshine Streaming identity. Tested the six-model matrix
including Qwen across Showcase, Realtime, Playground and Benchmark on WASM/WebGPU.
Result: 81 accepted / 108 requested flows; 8 unsupported native-streaming requests;
19 transcript-quality failures. All six models passed the file fixture across all
four apps and both backends. Native streaming passed for Moonshine and X-ASR.
Physical microphone capture was not tested; controlled Chrome audio was used.

Corrected four Moonshine tail/padding/hidden-slice contract bugs with regression
tests. Parent reran exact-PCM native controls rather than trusting agent PASS labels.
Remaining Moonshine/GigaAM failures were reproduced by official CPU controls;
X-ASR CPU also truncated VAD output but precise browser/native parity remains open.
No ONNX re-export or source-weight modifications were needed.

Repaired diagnostic callback deadlock and terminal/empty-result handling in the
browser harness. Disabled profiling by default while respecting saved boolean
choices. X-ASR reruns no longer hit the ORT profiling event-cap console error.
Failed receipts and immutable full-transcript oracle checks were retained.

Final validation: library 1961 passed / 36 skipped; demos 69 passed; harness 57 passed;
four demo builds and library build passed; lint 0 errors / 14 warnings; root import
44 modules, requested limit unchanged. Final v18 archive/install/build hashes and
served app scripts were independently checked. Durable archive stores 1078 JSON
receipt versions. Raw PCM remains local for reproducibility.

Rejected and closed bot PR 238 unmerged after reproducing its synchronous-cleanup
regression. No open PRs remained in either repository at the final connector check.
No commit, push, merge or new CI run occurred in this continuation. Existing dirty
work was preserved. Non-Turbo Large v3 and EOU120 stayed out of scope.

See [handoff](HANDOFF-MODERN-MODELS-2026-10-25.md) for exact heads, artifact hashes,
commands and next gates; [full report](../../../demos/docs/reports/newer-models-2026-10-25.md)
for per-model outcomes and evidence links.


## 2026-10-25 — Context-preserving VAD and streaming, v24

Implemented bounded shared acoustic context and a final captured-tail revision; Realtime replaces context hypotheses rather than duplicating text. Playground now retains raw PCM for all recorded-audio actions, shares exact-zero edge preparation, rebases timestamps, and invalidates stale recordings before capture. Fixed X-ASR dropping PCM when partial output is disabled and corrected final/revision state. Parent reviewed sub-agent changes and added regressions for delayed onset, actual PCM continuity, cleanup failures, session ownership, stale recording input and stop completion.

Final real-weight acceptance: Realtime six models x two backends x three flows = 36/36; Playground Moonshine/X-ASR x two backends x four flows = 16/16. Full normalized fixture oracles and artifact/backend evidence checked independently, not inferred from load success. Native CPU controls located a truncated captured tail; this was repaired in common capture and verified again in the browser. Preserved v20 71/76 and v23 VAD 34/36 failures as historical evidence.

Validation: library 1,982 passed / 36 skipped (maxWorkers=4), demos 93, harness 68; library/four app builds passed; lint 0 errors / 16 warnings, including two new max-lines warnings; root graph 44 (<45). Earlier default-parallel test timeouts remain recorded. Package v24 SHA-256: 2cbbd04f7c102c372e7f281880d8d3b6770ac5afd09ad2ce911a2700d6b38bb2. Parent verified 426 source and 1,695 distribution hashes plus 34 model files. Durable snapshot stores 780 JSON receipts with hashes and logs.

Physical microphone, broad multilingual/long-recording quality, recorded benchmark UI real-weight acceptance and CI remain unverified. No commit/push/merge in this continuation; inherited dirty work preserved. See [latest handoff](HANDOFF-MODERN-MODELS-2026-10-25.md) and [report with exact reproduction commands](../../../demos/docs/reports/context-preserving-capture-2026-10-25.md).


## 2026-10-25 — Framework bindings and shared browser cache

Added the framework-neutral controller and thin React/Svelte/Vue bindings in
the sibling frameworks workspace, with file examples, ownership tests and
documentation. All seven apps are now served under the existing 4190 origin.
Recognized HF main artifact keys use a normalized repository/revision/path
identity, including nested-path aliases. Legacy reads avoid copying model bytes;
explicit caller fallback migration remains unchanged. FireRed defaults to the
shared cache and reads an existing legacy database in place.

Parent verification: 2,011 core tests passed / 36 skipped, 10 wrapper/suite tests,
93 demo tests; all seven app builds and package import/export checks passed.
Root graph 44; lint 0 errors / 16 warnings; API docs 0 errors / 42 warnings.
Chrome proof uses a synthetic 64 KiB payload: one network fetch, six cache reads,
one persisted fixture row across seven pages, old ASR/FireRed rows read without
copying and a negative different-port isolation control. This is not model
accuracy acceptance. Preserved failed initial mock checks and an import timeout;
corrected the mock to return bytes only for the seeded legacy key, and the final
isolated suite passed with the original timeout.

New SDK archive SHA-256: c0cec2731005f6d25a3a2d852af7060f51385e9fcc3aea4be0f60e87e269b722.
All 1,711 installed dist file hashes matched. The user chose browser-only use:
no native helper/extension and no global cross-site cache claim. Old cache data
was not deleted. No commit/push/merge; inherited edits preserved. See
[cache receipts](evidence/shared-browser-cache-2026-10-25/manifest.json).

## 2026-10-25 — File/microphone composition and week close

Shared file convenience APIs reuse existing decode/downmix/resample/windowing
services. Microphone APIs compose the existing capture/controller pipeline,
retain acoustic context, drain pending transcription and release capture before
model cleanup. React/Svelte/Vue examples use the same controller contract.
Parent-verified core 2,050 passed / 36 skipped; demos 93; frameworks 10;
root graph 44; lint zero errors / 16 warnings. Real-weight scope and unresolved
model/physical-microphone gates are recorded in
[week-close handoff](HANDOFF-WEEK-CLOSE-2026-10-25.md).
User authorized commit/push and framework publication to demos/framework-examples.

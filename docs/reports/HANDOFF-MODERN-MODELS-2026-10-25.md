# Modern ASR handoff — 2026-10-25

## Latest checkpoint: browser cache and framework integration

All seven apps now run at `http://127.0.0.1:4190/` from the sibling
`frameworks/scripts/serve-suite.mjs`, with a shared browser origin. Core browser
cache remains `asrjs-cache-db` / `asset-cache`, schema 2. Recognized HF main-file
keys converge across family/helper/URL paths; old entries are read in place.
FireRed new entries use the common cache, with existing legacy DB read fallback.
No browser cache was deleted or automatically copied.

Latest four-demo package: `asrjs-speech-recognition-0.2.0-shared-cache-20261025.tgz`,
SHA-256 `c0cec2731005f6d25a3a2d852af7060f51385e9fcc3aea4be0f60e87e269b722`.
The parent checked 1,711 installed distribution file hashes against its archive.

Verification: library 2,011 passed / 36 skipped; framework/suite 10 passed;
demos 93 passed. Library, four-demo and three-framework builds passed; root
import stays 44. Lint has 0 errors / 16 warnings, API docs 0 errors / 42 warnings.
Actual Chrome IndexedDB proof: one deterministic 64 KiB fixture fetch, six cache
reads across seven app paths, no duplicate row. Legacy ASR/FireRed reads did not
copy bytes; different-port isolation passed. This is cache integration evidence,
not a fresh real-model transcription matrix. Earlier v24 model receipts remain
historical. Initial legacy-cache mock failures and an import timeout were
preserved; the final isolated full run passed without changing the timeout.

The user chose installation-free browser use. No local Model Store, extension
or universal cross-site cache is shipped. Different sites/profiles remain
isolated; concurrent first-time tab download single-flight is not implemented.
See [browser cache guide](../BROWSER_CACHE.md) and
[receipts](evidence/shared-browser-cache-2026-10-25/manifest.json).

Next startup: read framework README, use `npm run build:suite` and
`npm run preview:suite` from `../frameworks` after the demo build. Keep hostname
and port fixed. No commit/push/merge in this continuation; preserve inherited
dirty work. The SDK pin is explicit; do not silently regenerate it on every build.

## Historical checkpoint: context-preserving capture, v24

Supersedes the v18 installed-package and independent-utterance status below; older results remain historical.
[Detailed report](../../../demos/docs/reports/context-preserving-capture-2026-10-25.md).

- Real weights: 52/52 controlled flows independently verified: six models on Realtime WASM/WebGPU file, TEN-VAD and FireRed (36); Moonshine/X-ASR Playground file, file streaming, recorded transcription and recorded streaming (16).
- Opt-in shared acoustic context preserves original PCM, 600 ms preroll, 30 s cap, 3 s gap; successful flush emits one bounded captured-tail revision. Realtime replaces the hypothesis by context id/revision. Native streaming cache continuity across VAD is not claimed.
- Playground retains raw PCM, shares capture-window preparation and restores timestamp offsets. X-ASR now feeds PCM even with partial output disabled; finalization/revision semantics are tested.
- Library 1,982 passed / 36 skipped; demos 93; harness 68. Library and four app builds passed. Lint 0 errors / 16 warnings (two new max-lines warnings). Root import 44, below the unchanged 45 limit.
- All four apps pin `asrjs-speech-recognition-0.2.0-model-audit-20261024-v24.tgz`, SHA-256 `2cbbd04f7c102c372e7f281880d8d3b6770ac5afd09ad2ce911a2700d6b38bb2`. Parent checked source/distribution provenance and 34 model artifacts.
- Durable evidence: `demos/docs/reports/evidence/context-preserving-capture-2026-10-25/manifest.json` (780 JSON receipts plus test/provenance logs). Exact rerun commands and source paths are in the report.
- Physical microphone remains UNVERIFIED; controlled English/Russian fixtures only. Recorded benchmark UI real-weight acceptance was not run this turn. No new CI, commit, push or merge; inherited dirty work remains.
- Next gates: physical microphone/device coverage, longer/multilingual capture and real-weight recorded benchmark UI acceptance. Preserve earlier v20/v23 failures and raw PCM artifacts.

## Historical v18 checkpoint

The current continuation is complete as an audit, not a blanket model certification.
See [the detailed demo report](../../../demos/docs/reports/newer-models-2026-10-25.md).
Six real-weight models, four demos, both WASM/WebGPU: 108 requested flows,
81 accepted, 8 unsupported native-streaming requests, 19 transcript-quality failures.
Physical microphone input remains UNVERIFIED: capture used Chrome fake audio.
Realtime VAD runs independent utterances; Playground separately tests native streaming.

| Model | Package tested | Accepted / 18 |
| --- | --- | --- |
| Qwen3-ASR 0.6B | v16 | 16 |
| SenseVoice Small | v17 | 16 |
| GigaAM multilingual CTC | v17 | 10 |
| GigaAM v3 E2E RNN-T | v17 | 12 |
| Moonshine Streaming Tiny | v18 | 13 |
| X-ASR 160 ms | v18 | 14 |

Latest installed package in all four demos is
`vendor/asrjs-speech-recognition-0.2.0-model-audit-20261024-v18.tgz`.
SHA-256: `c85b861e576b08c237af5f5c2d982909547e9aa4777e2d5d1ee95862c7bf00b6`.
Earlier model receipts retain their real package versions. v17 to v18 SDK changes
are limited to Moonshine streaming-session JS and maps; other executors are identical.

Library HEAD: `e534fb517dcd15a52ac8b52bff85551df89d3673`, main.
Demos HEAD: `ce0cf1c0ecf0e60f8fce1e7e7f734bbf73a733a8`, main.
Both contain inherited and current dirty work. This continuation did not commit,
push, merge or run new CI. Do not stage the entire dirty trees.
Preview was serving the verified v18 build at http://127.0.0.1:4190,
exec session 2286 when this handoff was written; recheck process liveness later.

## Completed changes and checks

Shared local-artifact contracts integrate SenseVoice, both GigaAM families and
X-ASR without four app-specific loading branches. Moonshine Streaming has its own
identity and four-graph contract. Existing offline Moonshine remains available.
Four Moonshine host-contract fixes cover 80-sample padding/valid mask, frame versus
sample tail retention, exclusion of replayed lookahead, and valid hidden-row slices.
No new ONNX export or source-weight mutation was necessary.

Harness fixes preserve runtime failures and empty-output failures, and avoid a
reentrant browser-query deadlock in PCM capture. Profiling defaults off in the
three apps exposing that control, preserving explicit saved booleans. Benchmark
has no profiling control. Explicit long profiling still has the upstream event cap.

Parent independently reran subagent CPU controls and inspected receipts/hashes.
Library: 1961 passed, 36 skipped; build passed; lint 0 errors / 14 warnings.
Demos: 69 tests and four builds passed. Harness: 57 tests passed.
Root static import graph: 44, below the unchanged requested limit of 45.
The smoke tool's legacy printed ceiling of 78 is not the acceptance criterion.
49 source hashes and 1691 dist hashes match the final archive/install/build.
Both repository diff checks passed; served app HTML and entry JS match disk.

## Remaining evidence limits

Moonshine's silence-padded manual/recording input also produces empty output in
the official CPU model. Its remaining WebGPU FireRed word error is reproduced by
the official model on the captured segment. GigaAM native CPU reproduces the VAD
errors on exact captured PCM. X-ASR sherpa CPU also loses words on VAD segments,
but exact browser/native text or numerical parity is not established.
Native agreement is not a transcript-quality PASS. The fixture scope is one
English JFK clip, plus a Russian Pushkin clip for RNN-T; not a broad accuracy study.

Next work should evaluate a shared contextual VAD strategy and broader fixtures,
then physical microphone capture. Do not tune to the oracle, hardcode transcript
corrections, weaken acceptance, or assume a model graph bug without a reproduction.
Non-Turbo Whisper Large v3 and EOU120 remain excluded; preserve existing Turbo work.
Keep source weights read-only and do not publish to Hugging Face without authorization.

## Evidence and reproduction

Durable compact matrix, package and PCM checks:
`N:/github/asrjs/demos/docs/reports/evidence/modern-model-audit-2026-10-25/`.
Durable receipt archive contains 1078 current/historical JSON versions, including
failures, at `demos/docs/reports/evidence/remaining-model-audit-2026-10-24/receipts-v18.json.gz`.
SHA-256: `f8a54c7cae98e40565a17b3527af334c232c959231831af2c32529314ee17d1d`.
Historical receipts do not certify the current package. Raw PCM is intentionally
outside the archive. Preserve `demos/tmp/real-e2e/remaining-models-20261025/` and
its `mic-pcm-capture` folders, plus staged Moonshine graphs under
`demos/tmp/real-e2e/remaining-models-20261024/native-moonshine-streaming-tiny`.

CPU controls and probe source are under this repository's
`docs/reports/evidence/{moonshine-streaming,gigaam-ctc,gigaam-rnnt,xasr}-2026-10-25/`.
The Moonshine probe adds explicit BOS/EOS normalization only for token comparison;
raw outputs remain recorded. It does not establish numerical allclose.

From the library directory, separate commands:
```powershell
npm --cache tmp/npm-cache test
npm --cache tmp/npm-cache run build
npm --cache tmp/npm-cache run lint
node tests/smoke/browser-root-import-graph.mjs
```
From the demos directory:
```powershell
npm --cache tmp/npm-cache test
npm --cache tmp/npm-cache run build
C:/Python311/python.exe -m unittest discover -s scripts/e2e -p 'test_*.py'
```
Real-weight runners are retained at:
`demos/tmp/real-e2e/run-moonshine-streaming-v18-matrix.py` and
`demos/tmp/real-e2e/run-modern-four-v18-matrix.py` (X-ASR selected with `--model x-asr`).
The older v17 runner retains the other model configurations. Inspect their arguments
before use and choose fresh output/profile directories: existing output roots are
rejected intentionally. Run only one GPU model job at a time. Main reusable harness:
`demos/scripts/e2e/remaining-model-flows.py`; parent verifier:
`demos/tmp/real-e2e/verify-modern-matrix-parent.py`.

## Bot PR disposition

[PR 238](https://github.com/asrjs/speech-recognition/pull/238) is closed unmerged.
The proposed synchronous cleanup map stops after a synchronous dispose throw,
leaving later resources uncleaned. Parent real-source reproduction confirmed this.
No cosmetic subset was adopted. See [decision](bot-pr-238-decision-2026-10-25.md)
and its evidence directory. Latest connector checks found zero open PRs in both
repositories; this is the checkpoint state, not an ongoing monitoring claim.

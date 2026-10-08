# C93 — Automatic speech-detect utterances against original Orukeet

Status: seven exact-input token/text comparisons PASS. This closes C92's
independent original-engine output gate, not full model/product acceptance.

## Implementation and evidence

The existing `hf-parakeet-port/python-nemo-inspect.py` now accepts
`--input-pcm-start-frame`, `--input-pcm-end-frame` and
`--input-pcm-source-sha256`. Bounds are mono sample frames, end-exclusive.
The whole f32le source digest and selected window digest are checked separately.
The summary's `frozen_pcm` describes the effective selected waveform;
`source_slice` records whole-source hash/count and exact bounds. No intermediate
PCM files, PCM16 conversion, resampling, alternate decoder or padding was added.
The old whole-file input path remains unchanged. Invalid/incomplete bounds,
wrong hashes, nonfinite selected input and windows shorter than one second fail.

Frozen C92 receipt:
`tools/data/results/orukeet/auto-browser-c92-speech-capture-2026-10-08-02.json`,
SHA-256 `0fa5f6f1c45927480e6c4d8efee163054514aeacf8b77d81b59869b564589547`.
It contains two actual fake-device capture sessions, rough-gate utterance ASR,
real TEN diagnostics and cleanup; it is not physical microphone or learned-VAD
quality acceptance. Existing capture/code checks were re-evaluated before the
original-engine comparisons. Every saved window hash was recomputed from the
unchanged full-session bytes.

Both source sidecars contain 160,000 mono float32 samples at 16 kHz:

- Session 0 SHA-256: `d6128523c1405d62b2c2c90f75f8bd25afaea29830a71724017302fc8ec33639`.
- Session 1 SHA-256: `afda7f61aaab28722bb90533156d0c56031e47c96bb2a5c3dfd42124a849faaa`.

Seven sequential original-weight CPU jobs completed in 400.2 seconds. The
existing inspector restores the cached model per CLI call; this timing includes
restoration and three official transcriptions per window, not an ASR benchmark.
Qwen's coordination file reported its CPU/RAM window released. No new weights,
dependencies, reference environments, sibling edits or GPU jobs were needed.
The reused overlay is Python 3.10.12 / Torch 2.6.0+cu118 / NeMo 3.0.0, CPU,
batch 1, four CPU threads; it inherits system packages, not an isolated stack.

Original `.nemo` SHA-256 remains
`031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`,
revision `b59c13a733fce5cf193230d0fa317ee7f145a108`.

| Session/window | Frame range | Tokens | Captured stages | Output |
| --- | --- | ---: | ---: | --- |
| 0/0 | [0, 40960) | 11 | 151 | PASS |
| 0/1 | [39680, 78080) | 4 | 89 | PASS |
| 0/2 | [76800, 128000) | 12 | 165 | PASS |
| 0/3 | [124672, 160000) | 13 | 176 | PASS |
| 1/0 | [0, 42240) | 10 | 143 | PASS |
| 1/1 | [40960, 79360) | 4 | 89 | PASS |
| 1/2 | [78080, 160000) | 20 | 262 | PASS |

All full token sequences and exact text match. Each actual NeMo preprocessor
input matches the selected f32le hash, [1,N] shape, rate, valid length and zero
padding. All six baseline/captured/after controls pass. The existing Orukeet
`compare_capture_oracle` checks model/weights, input, instrumentation and outputs;
the caller explicitly maps window identity and separately verifies source/bounds.
No whole-session output claim is inferred from this mapping. Compressed stage
payloads were inspected for agreement with their summaries and token/text rows.

Saved aggregate:
`tools/data/results/orukeet/capture-window-original-output-c93-2026-10-08-01.json`,
SHA-256 `4852bb047d6d395761b5f72d3011cdcf971ffb5c2d7a66c265372e77240ac6ef`.
It records browser/runner/adapter/inspector/stage-tool identities, both PCM
sources, original weights/revision/environment and every summary/stage digest.
Individual outputs use `official-c93-capture-I-window-J-2026-10-08-01.json`
and `.json.gz` in the same results directory. Original summaries and full traces
remain byte-preserved local artifacts: Git would normalize the newly generated
summary line endings and invalidate their recorded byte hashes. The aggregate
is tracked, but verification still requires these local original artifacts;
it is not self-contained fresh-clone model acceptance.

## Reproduce through existing tools

For a saved window, use its own source/window hashes and exact frame bounds.
Choose fresh summary/stage names; neither originals nor receipts are overwritten.
Example for C92 session 0/window 0, from repository root:

```powershell
& tools/model-debugging/reference/nemotron-3.5-asr-streaming/.venv/Scripts/python.exe tools/model-debugging/reference/hf-parakeet-port/python-nemo-inspect.py --restore-path models/orukeet/source/orukeet-v0.1.0.nemo --model oruk/orukeet --model-revision b59c13a733fce5cf193230d0fa317ee7f145a108 --audio tools/data/fixtures/audio/jfk-short.wav --input-pcm-f32le tools/data/results/orukeet/auto-browser-c92-speech-capture-2026-10-08-02.json.capture-0.f32le --input-pcm-sample-rate 16000 --input-pcm-source-sha256 d6128523c1405d62b2c2c90f75f8bd25afaea29830a71724017302fc8ec33639 --input-pcm-start-frame 0 --input-pcm-end-frame 40960 --input-pcm-sha256 0380c57d28f52c845734779e1848a0f82333769301ae095548643606526db6f3 --device cpu --cpu-threads 4 --sample-id c93-0-0 --stage-output tools/data/results/orukeet/original-window-FRESH.json.gz --output tools/data/results/orukeet/original-window-FRESH.json
```

`--audio` is provenance only in frozen mode. The venv path is reused solely for
the NeMo environment; it does not resume Nemotron work. Repeat sequentially with
each window's actual identity, not this window's digest. Before output comparison,
verify receipt/code/cleanup and source bytes; require the summary's `source_slice`
to match source hash/count/bounds, then call the existing adapter with
`capturedSamples=window.samples`, `capturedPcmSha256=window.capturedSliceSha256`,
`tokenIds=window.tokenIds`, `text=window.text`. Require all seven returned checks.
Verify summary/stage/inspector identities separately; matching text is not enough.

## Validation and remaining scope

Python compilation, scoped diff checks and 70 lightweight methods PASS:
8 frozen-inspector, 31 automatic-browser, 7 adapter and 24 manifest tests.
Three new inspector tests cover source/window identity, signed zero, owned data,
incomplete/bool/noninteger/out-of-range bounds, wrong hashes and short/nonfinite
input. Independent read-only review approves. No core source, graph, tolerance,
original oracle or manifest code pin changed. The separate fresh Chrome manifest
regression also passes all 19 checks; see the Chrome recovery report.

This is original-engine output evidence, not numerical feature/encoder/state
parity, timestamp accuracy, fp16, held-out human-label quality, native streaming
or memory/soak acceptance. Orukeet's existing strict production feature failure
remains open. Original model output is an implementation oracle, not human gold.

Next useful product gate: combine built/packed browser imports and the default
packaged TEN worker with actual capture-to-ASR. C92 uses a source-worker override;
the separate built default-worker probe does not run ASR. A new capture must save
its own effective PCM and receive new original-engine oracles; C93's old window
rows cannot certify different captured bytes/bounds. Frozen C93 input replay is
a separate control, not proof of a new microphone capture. Keep package/worker,
provider and awaited resource-release evidence explicit.

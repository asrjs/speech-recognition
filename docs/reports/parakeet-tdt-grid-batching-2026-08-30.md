# Parakeet TDT speculative grid batching with utilization gate (2026-08-30)

Status: implemented, unit- and browser-validated, pending commit. Scope:
'src/models/nemo-tdt/executor.ts' (fused TDT decoder-joint decode loop),
'tests/nemo-tdt-grid-batching.test.ts' (new), browser harness wiring in
'webgpu-agent-test'.

## Design

One '[1, features, width]' decoder-joint run scores 'width' frames of the
encoder output against the current '(target token, GRU state)' pair. The row
scan then walks the scored grid:

- blank rows with duration 'step > 0' jump the cursor (duration-skip) and
  reuse rows that were already scored by the same run;
- a blank row with 'step == 0' advances one column;
- an emission row commits its token, transfers the grid 'output_states_*'
  tensors into the decoder state, and re-batches starting at the emitting
  frame (multi-token-per-frame parity with the sequential loop);
- a window with no emission advances the cursor past the whole window and
  doubles the next width (2 -> 32, capped by remaining frames).

## Layout bug (the long bring-up debug)

The batch tensor is declared '[1, features, width]' and the exported graph
transposes 'encoder_outputs' with perm '(0, 2, 1)', so the flat buffer must be
FEATURE-MAJOR: element '(f, w)' lives at 'f * width + w'. The original
fill was row-major ('row * featureSize + fIdx'), which silently transposes
every 'width > 1' grid. Width 1 is layout-invariant, so the layout-blind unit
mocks passed while the real browser produced an EMPTY transcript (all rows
argmax'd blank). Fixed both gather branches (BDT source: per-feature
'batchData.set(subarray, fIdx * width)'; frame-major source: scatter loop) and
hardened the unit mock to assert the per-column layout (throws
'Grid layout violation' when 'encoder.data[column] != startFrame + column').

Reusable lesson: any future batched-decode port must include the per-column
layout assertion in its mock. A width-1-passing mock proves nothing about
batch layout.

## Utilization gate

Browser A/B on the emission-dense LibriVox fixture (91 tokens / 105 decoder
visits) showed the pre-gate batcher scored 98 grid runs with decode time
within noise of the sequential baseline: row 0 of nearly every window emits,
so speculative columns beyond the first row are wasted (utilization about
'1 / width').

Gate added ('src/models/nemo-tdt/executor.ts'):

- counters 'tdtBatchColumnsScored' / 'tdtBatchColumnsUseful' reset per
  transcribe; each run adds 'width' scored and 'examined' useful;
- after a 24-column sampling window ('tdtBatchMinSampleColumns'), batching is
  disabled for the rest of the utterance when useful columns fall below 70%
  ('tdtBatchMinColumnUtilization') of scored columns;
- 'options.gridBatching: false' pins the sequential path (escape hatch for
  embedding apps).

The gate bounds wasted speculation to the sampling window while preserving
the dispatch win for blank-dominant audio (unit-proven: a 64-frame fixture
with one emission needs very few dispatches and never latches).

## Browser A/B (Chrome headless, real WebGPU host)

Setup: 'webgpu-agent-test', vite :8765, Parakeet v3, encoder fp16 WebGPU +
decoder int8 WASM, LibriVox 18.714 s fixture, '--oracle=none --repeat=3',
warmup 1, 'returnConfidence' on. Gate-on vs gate-off ('--grid-batching=off')
in the same session; NVIDIA Blackwell adapter.

| Variant | grid runs | decodeMs (runs) | wallRtfx (runs) | transcript |
| --- | --- | --- | --- | --- |
| gate-off baseline | 0 | 643.3 / 704.0 | 17.35x / 19.86x | exact 91-token expected |
| gate-on | 7 (then sequential) | 643.3 / 705.8 / 495.8 | 17.35x / 19.15x / 23.72x | identical |

Pre-gate implementation on the same fixture: 98 grid runs, decodeMs 563-900
(no win, wasted dispatches). Post-gate: 7 grid runs, parity on decode time,
identical transcript. The throughput win remains reserved for blank-dominant
audio where row scan utilization is near 100% (unit-proven geometry; no
blank-heavy long clip in the local fixture set yet).

Evidence JSONs:
'tools/data/results/nemo-tdt/parakeet-tdt-v3-grid-ab-baseline-2026-08-30.json'
and
'tools/data/results/nemo-tdt/parakeet-tdt-v3-grid-ab-gateon-2026-08-30.json'.

## Validation

- 'npx vitest run tests/nemo-tdt-grid-batching.test.ts': 10/10 (parity,
  duration-skip reuse, multi-token frames, maxSymbolsPerStep, latch on
  rejection/corruption, abort re-throw without latch, tensor disposal, gate
  latch on dense audio, gate stays open for blank-dominant audio,
  'gridBatching=false' pin).
- Full suite: 1053 passed / 18 artifact-gated skips. 'tsc --noEmit' clean.
  'npm run build' clean (dist rebuilt before browser runs).

## Next steps

1. Port the same template to 'src/models/nemo-rnnt/executor.ts' (eou-120m
   fused graph): feature-major fill pattern 'f * width + w', layout-asserting
   mock, and a utilization gate without a duration head (useful rows are
   emission-rows only; blank breaks the inner loop).
2. [Completed same day] Blank-dominant browser win measured (details in
   the addendum below).
3. If more blank-heavy clips appear, consider exposing the gate
   thresholds as preset tuning rather than constants.

## Addendum: blank-dominant browser win and the gate re-probe fix (2026-08-30 later)

Fixture: 'tools/data/fixtures/audio/librivox-blankgaps-synthetic.wav' -
SYNTHETIC measurement clip (generator:
'tools/scripts/make_librivox_blankgaps_fixture.py'): first 15 s of the
LibriVox speech with nine 3.2 s silence gaps, 40.6 s total, ~63% blank
duty. Not a benchmark-quality fixture; used only for dispatch mechanics.

First measurement with the shipped gate exposed a real design flaw: the
24-column sampling window accumulated during the OPENING SPEECH segment
(dense, ~50% utilization) and latched batching off after just 7 grid runs,
before any silence was reached - the gate never re-evaluated, so the
blank-dominant dispatch win could not materialize.

Fix shipped: the sequential fallback now counts consecutive blank visits;
after six blanks (about 0.5 s) it resets the sampling window (scored/useful
counters and width), letting the grid re-probe. A fluke re-probe costs at
most the bounded sampling window before the gate closes again; the dense
phase still latches as designed. Verified by a new unit test
('re-opens the gate after a latched dense phase once the audio goes blank').

Browser A/B after the fix (Chrome headless, real WebGPU host, TDT v3,
encoder fp16 WebGPU + decoder int8 WASM, repeat 3, warmup 1):

| Variant | grid runs | decodeMs (runs) | median | transcripts |
| --- | --- | --- | --- | --- |
| gate-off baseline | 0 | 1219.8 / 1194.3 / 990.5 | 1194.3 | identical |
| gate-on (pre-fix) | 7 | 1365.7 / 901.9 / 994.6 | ~950 | identical |
| gate-on (fixed) | 53 | 862.4 / 853.9 / 625.7 | 853.9 | identical |

Result: ~29% faster decode at the median (854 vs 1194 ms; best runs 626 vs
990 ms = 37%), RTFx median 33.2x vs 25.7x, decodeIterations unchanged
(183 - same rows examined), transcripts identical in every run. The gate
now re-opens per silence gap and crosses the gaps with widening grids.

Evidence JSONs:
'tools/data/results/nemo-tdt/parakeet-tdt-v3-blankgaps-grid-{off,on,on-pregatefix}-2026-08-30.json'.

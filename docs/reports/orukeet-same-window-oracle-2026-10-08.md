# C66 — Same-window original-engine control

Status: PARTIAL PASS, output-only on one nonzero rolling window. The long
controller/full-file text gate remains FAIL; feature numerical acceptance is
unchanged. No production code, oracle normalizer or acceptance threshold changed.

The recovered automatic Chrome receipt
`tools/data/results/orukeet/auto-browser-controller-timing-gated-long-2026-10-08-01.json`
contains request 3: frames 51,200–128,000, mono 16 kHz, 76,800 samples. Its raw
text contains the first aggregate/full-file difference, `All LibriVox` versus
`All Librivox`. The same full-file browser runs match all 722 official IDs.

## Discriminating experiment

FFmpeg losslessly sliced the existing PCM16 source with
`atrim=start_sample=51200:end_sample=128000`, without resampling. Independent
wave/struct checks established exact source-slice PCM16 bytes, then canonical
float32 little-endian identity:
`31497b359206a88e148279be1e8f7cd9b0373eab66b29014b67347b4d7b28868`.

Reused cached original weights (SHA-256
`031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`), pinned
revision `b59c13a733fce5cf193230d0fa317ee7f145a108`, and the existing NeMo 3.0 /
Torch 2.6.0+cu118 / Python 3.10.12 environment. No download or environment change.
Qwen's recorded CPU/RAM release and live process inventory were checked first;
one four-thread CPU job completed with exit 0, no other process stopped.

Unlike C65's proposed plain capture, this bounded 4.8-second control used the
existing three-pass stage capture: necessary to independently bind the actual
NeMo preprocessor input rather than just the WAV container. All six baseline /
instrumented / after controls pass. 349 stages, 31 predictor and joint calls.
Actual input has exactly the digest above, 76,800 valid samples and no padding.

```powershell
& tools/model-debugging/reference/nemotron-3.5-asr-streaming/.venv/Scripts/python.exe tools/model-debugging/reference/hf-parakeet-port/python-nemo-inspect.py --restore-path models/orukeet/source/orukeet-v0.1.0.nemo --model oruk/orukeet --model-revision b59c13a733fce5cf193230d0fa317ee7f145a108 --audio tools/data/results/orukeet/repeat8-window-51200-128000-2026-10-08.wav --device cpu --cpu-threads 4 --sample-id repeat8-window-51200-128000 --timestamps --stage-output tools/data/results/orukeet/official-repeat8-window-51200-128000-stages-2026-10-08.json.gz --output tools/data/results/orukeet/official-repeat8-window-51200-128000-summary-2026-10-08.json
```

These are historical output names; use fresh paths to reproduce. Reusing the
interpreter under the Nemotron reference folder does not resume that model.

## Verified comparisons and limits

Original NeMo and preserved Chrome raw request match exact text, all 29 token
IDs, all 29 frame indices and all 29 TDT durations:

> This is a LibriVox recording. All LibriVox recordings are in the public domain.

Thus the first casing divergence is reproduced by the original engine under
the same window context; it is not evidence of a port or merger fault at this
window. Other later aggregate differences remain unresolved. Do not normalize
them away or declare the controller accepted from this one control.

Numerical feature/encoder/head comparisons are UNTESTED for this capture.
Canonical timestamp parity is also UNTESTED: punctuation endpoints have distinct
semantics (e.g. NeMo's period ends at 1.92 s, canonical token ends at 2.08 s).
Equal token frames/durations do not prove word/segment timestamp equivalence.
The existing local-to-global timing gate remains a separate coordinate check.

18 existing automatic-runner tests were rerun and PASS. Independent read-only
receipt review confirmed request association and the first aggregate difference.

SHA-256 bindings:

- Chrome receipt: `c058931a0fc94cba0c675b030c5813732778c2844e4ea7be31fd57eb11bfd916`.
- Official summary: `ec172a64406829bd5d7936d4ad29c262c0099f6b633f2f33aa1ce410afbd23e8`.
- Official compressed stages: `95ef29964a9e59ff8cd25cb08cfd5f4261f8b18d1a2b4208c015a210d1a905b8`.

Next: replay preserved raw windows through the existing merger independently
of model inference; locate the next non-casing divergence and compare its exact
window against the original engine. Keep raw-window parity, aggregation
correctness, full-file quality, numerical conformance and timestamp semantics
as separate evidence scopes; do not run an unchanged 75-window Chrome loop.

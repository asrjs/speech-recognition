# C68 — Original engine reproduces the domain-format window difference

Status: PARTIAL PASS, one-window output control, G4/G6. Long aggregate/full-file
text gate remains FAIL; feature numerical conformance remains unchanged. No
production fix, oracle normalization or acceptance relaxation.

C67 traced `Libravox dot org` to request 6 in the saved long Chrome controller
receipt, before merger processing. A lossless FFmpeg sample slice of the existing
mono 16 kHz PCM16 fixture covers frames 130,560–224,000 (8.16–14.0 s).
Independent wave/struct verification confirms all 93,440 sample values and
canonical f32le SHA-256
`665df291b767d8ae37afd6e1ed6f6ea7b9a0e4ef2780ab09a908620ce78007fb`.

One four-thread CPU original-weight run used the existing NeMo interpreter and
inspector after checking Qwen coordination and live Python processes. No
download, install or environment change; no other process stopped. Three
baseline/instrumented/after passes completed with exit 0, 339 stages, 29 predictor
and 33 joint calls. All six instrumentation controls pass. Actual preprocessor
input matches that exact digest, rate, channel count and valid length, with no
padding. Original weights retain SHA-256
`031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`.

Original NeMo and preserved Chrome request 6 agree exactly:

> For more information, or to volunteer, please visit Libravox dot org. Read by

All 27 token IDs, token frame indices and TDT durations match. The original
engine on this window itself differs from its full-file `LibriVox.org` output.
Thus neither this phrase nor the earlier casing difference (C66) justifies a
decoder/merger workaround. This does not certify all remaining windows or final
aggregation quality. Canonical timestamp semantics and numerical stages are
UNTESTED in this comparison; instrumentation alone is not numerical comparison.

## Reproducible recipe and artifacts

Run from the repository root, choosing fresh output names when repeating:

```powershell
ffmpeg -nostdin -hide_banner -loglevel error -n -i tools/data/results/orukeet/librivox-repeat8-16k-pcm16-2026-10-02.wav -af atrim=start_sample=130560:end_sample=224000 -c:a pcm_s16le tools/data/results/orukeet/repeat8-window-130560-224000-2026-10-08.wav
& tools/model-debugging/reference/nemotron-3.5-asr-streaming/.venv/Scripts/python.exe tools/model-debugging/reference/hf-parakeet-port/python-nemo-inspect.py --restore-path models/orukeet/source/orukeet-v0.1.0.nemo --model oruk/orukeet --model-revision b59c13a733fce5cf193230d0fa317ee7f145a108 --audio tools/data/results/orukeet/repeat8-window-130560-224000-2026-10-08.wav --device cpu --cpu-threads 4 --sample-id repeat8-window-130560-224000 --timestamps --stage-output tools/data/results/orukeet/official-repeat8-window-130560-224000-stages-2026-10-08.json.gz --output tools/data/results/orukeet/official-repeat8-window-130560-224000-summary-2026-10-08.json
```

Summary: `tools/data/results/orukeet/official-repeat8-window-130560-224000-summary-2026-10-08.json`.
Original working-file SHA-256 (before Git newline normalization):
`27ff22fbdf74cf2f552621be37229e969e95d3b72fa7db51441bbf3fc36012ac`.
Large compressed stages and generated WAV remain ignored local evidence.
Source browser receipt SHA-256:
`c058931a0fc94cba0c675b030c5813732778c2844e4ea7be31fd57eb11bfd916`.

Next: end this bounded two-window attribution experiment without declaring the
long gate passed. Use C67 replay for a concrete overlap-loss/duplication hypothesis
if one is found; otherwise advance independent agreed consumer/runtime work.
No unchanged full Chrome run or all-window oracle sweep is needed just to
reproduce already-explained context differences. Keep model numerical, quality,
timestamp and long aggregate acceptance unresolved and visible.

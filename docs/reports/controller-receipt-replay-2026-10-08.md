# C67 — Weight-free replay of recorded browser controller behavior

Status: receipt regression PASS, G4/G6; long controller/full-file text remains
FAIL. No production implementation or acceptance criteria changed.

`tests/windowed-controller-replay.test.ts` reuses `runWindowedController`, which
calls the actual public controller. Explicit receipt path, recorded SHA-256 and
source WAV are required. The shared WAV decoder provides PCM; canonical f32le
hashes check every callback against the saved input. Captured local details are
cloned into callbacks. All frame ranges, reasons, timing payloads, partial /
committed / preview text, inference links, final text and reset are compared.
Only nondeterministic latency/metrics are excluded. Original files are read-only.

The recorded failing long run (`auto-browser-controller-timing-gated-long-2026-10-08-01.json`,
SHA-256 `c058931a0fc94cba0c675b030c5813732778c2844e4ea7be31fd57eb11bfd916`)
replays all 75 inferences and 76 updates exactly, about 0.34 s in this run.
This reproduces recorded behavior, including its 1,924-character aggregate,
not desired correctness or model inference. The official full-file oracle is
1,899 characters; its 722-token browser controls already match independently.

Validation: 7 focused controller/replay/offset tests PASS with opt-in replay;
without variables 5 demo tests PASS and replay SKIP. Typecheck and file ESLint
PASS. Wrong digest and missing WAV each FAIL (exit 1), rather than SKIP/PASS.
Any supplied replay variable requests the check; absent companions fail closed.
The runnable recipe is in `examples/demo/README.md`.

## Next material difference localized

Independent read-only trace, checked against raw receipt fields: after the
same-window casing control in C66, the next non-casing difference includes
`please visit Libravox dot org` versus full-file `please visit LibriVox.org`.
It is already present in request 6, not introduced by merging:

> For more information, or to volunteer, please visit Libravox dot org. Read by

Request 6 spans frames 130,560–224,000 (8.16–14.0 s, 93,440 samples). PCM SHA:
`665df291b767d8ae37afd6e1ed6f6ea7b9a0e4ef2780ab09a908620ce78007fb`.
Update 6 commits the same `dot org.` phrase and keeps `Read by` as preview.
This excludes a merger-introduced change for that phrase, but its official
same-window output is still UNTESTED. Case-insensitive/difflib analysis was
diagnostic localization only; the exact gate remains FAIL without normalization.

Next bounded step: losslessly slice that window and capture original-engine
PCM/token/text like C66. Independently audit later overlap passages before any
merger patch. Do not infer all 75 windows match the original engine from replay;
no numerical, timestamp-accuracy, new GPU execution or quality PASS is claimed.

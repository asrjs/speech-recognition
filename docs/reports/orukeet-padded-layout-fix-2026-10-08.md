# C62 — Explicit NeMo TDT encoder layout fixes short padded windows

Status: bounded product workflow PASS; whole Orukeet numerical/model acceptance
remains incomplete. Track G4/G6, dependency C61. No new model lane or default
integer precision. User-visible outcome: the automatic Chrome controller example
now transcribes the short chunk-fed JFK workflow rather than rejecting its first
valid padded encoder output.

## Contract and implementation

C61 retained the real failure: physical `[1,1024,26]`, valid `encoded_lengths=25`.
Valid length cannot identify layout when both physical axes exceed it. The exact
packed Orukeet encoder and decoder graph metadata have feature axis 1 (1024),
not axis 2. Existing graph receipt:
`tools/data/results/orukeet/browser-real-encoder-jfk-final-2026-10-03.json`;
encoder SHA-256 `9924d5c30da5d54454dd3de6dbfc177aa771205eaac5da55d8f05ec813036f17`.

`NemoTdtModelConfig.encoderOutputLayout` is an optional family contract, BDT or
BTD. Only the audited Orukeet preset sets BDT. Existing preset factory merging
routes it into local-folder and ordinary public loads; no shared runtime, local
adapter, or demo workaround was introduced. Other family/preset defaults are
unchanged. Custom exports must declare their own layout through config.

Explicit layout bounds the positive integer valid length against the physical
time axis. Missing length output uses the physical extent; present malformed
length output fails closed. Decoder processing uses only valid frames and
existing tensor addressing retains physical padded stride. Tensor outputs are
disposed before layout validation failures. Unconfigured ambiguous axes still
fail closed. No feature math, weights, tokenization, tolerances or transcript
assembly changed.

## Validation

- TypeScript `node node_modules/typescript/bin/tsc --noEmit`: PASS.
- Executor and preset regressions first reproduced four decode/layout failures
  and one missing manifest-contract failure before implementation; then passed.
- Final focused tests: 43 PASS across executor (25), grid batching (13), manifest
  (5). Both BDT/BTD short padding and overlong lengths, zero/negative lengths,
  unconfigured ambiguous axes, physical row stride, cropping, actual grid use
  without silent fallback, and factory routing are covered. Grid mode uses a
  sentinel padded frame; passing simulation is not engine numerical parity.
- Full Vitest run before the last two added grid tests: 194 files PASS, 6 files
  SKIP; 1489 tests PASS, 33 SKIP. Skipped opt-in real engines are not accepted.
  The subsequent focused run covers the two added grid regressions.
- Actual installed Chrome 154.0.8037.98:
  `tools/data/results/orukeet/auto-browser-controller-jfk-2026-10-08-02.json`,
  `PASS_AUTO_BROWSER_SAME_PCM_TOKEN_TEXT_SMOKE`.
  Same official f32le PCM digest, 176000 samples, six contiguous ingestion chunks,
  seven actual window requests, seven partial/final updates. First window now
  processes 25 valid frames. Final merged text matches the full-clip official
  oracle; both ordinary full-clip repeats match all 38 IDs and exact text.
  Actual hybrid backend labels, delivered-code hashes, controller reset, model
  disposal and browser closure pass. Prior failed receipt remains unchanged.
- Fresh actual Chrome long control:
  `auto-browser-layout-long-2026-10-08-01.json` PASS, two exact 722-token/text
  repeats on the official 2395400-sample PCM (149.7125 seconds). Same code binding
  and awaited cleanup pass; controller is explicitly not requested. This guards
  the existing long full-clip layout path, not long rolling-window acceptance.

Recipe: use the existing command in `examples/demo/README.md`, append
`--verify-controller` and the official JFK `--expected-pcm-sha256`, choosing a
fresh output filename. This reuses the installed Chrome runner, no downloads.

## Limits and next step

This is chunk-fed windowed offline ASR, not native streaming or microphone
capture. All short windows begin at zero in this fixture; nonzero mature-cursor
rollover, timestamps across long windows, per-window original-engine token and
numerical parity remain unverified. Full-clip token/text equality does not close
the existing feature numerical FAIL or held-out quality/fp16/soak gates.
Next validate a bounded rolling-window input with actual nonzero window starts
and independent input/offset checks, while retaining these explicit limitations.

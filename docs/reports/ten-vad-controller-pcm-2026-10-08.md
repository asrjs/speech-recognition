# C87 — Public speech-detection controller with real TEN diagnostics

Status: bounded G4/G5/G6 browser integration PASS; whole-product/model gates open.

## User workflow and scope

The existing offline Chrome runner now accepts `--scenario ten-vad-controller`.
It calls the public `createBrowserRealtimeMicrophoneController` in speech-detect
mode with explicit TEN backend and rough-only gating. The supported startCapture
input boundary supplies identified local file PCM, not a simulated inference
engine. The default starter, detector, packaged WASM worker, probability buffer,
monitor and utterance publication execute unchanged. No production code, detector
thresholds, model math, original fixture or acceptance tolerance was changed.

TEN is intentionally diagnostics-only in this detector: segment boundaries use
the rough energy gate. The probe records this provenance and does not claim
model-driven segmentation. No transcribe callback is installed; this is PCM
utterance delivery, not ASR. Microphone capture is separately tested, not injected
file acceptance. No downloads, installations, sibling writes or GPU jobs.

## Verified integration

Chrome 154.0.8037.98 decodes JFK to 176000 samples at 16 kHz, canonical f32le
PCM SHA `ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0`.
Two controller starts process the same file through one reused real worker.
Every one of the 687 model hops has exact contiguous offsets and valid
probability/flag counts. The public probability buffer contains those exact
float32 values at the corresponding frames, not just a nonempty plot.
Direct-output and buffer hashes agree across both starts and with C86:
`dcc82cc557bb0c19f5be11d8101384bce25751086d4867563904be7cdc3360c6`.
Both runs have 463 model speech flags.

Each start publishes the same four rough-gate segments/utterances. The probe
checks each extracted PCM sample against the canonical input slice, bounds,
sample rate, reason and duration; it stores each slice hash. These are observed
deterministic windows, not human-annotated speech boundaries:

| Start frame | End frame (exclusive) | Reason |
| --- | --- | --- |
| 0 | 43520 | silence |
| 42240 | 80640 | silence |
| 79360 | 161280 | max-duration |
| 157952 | 176000 | stop |

Overlap is the existing detector extraction contract, not lost frame accounting.
A third start processes 16000 synthetic zero samples: 62 complete model hops,
zero speech flags and no utterances. Probability SHA matches C86 fresh silence:
`46b701af636ebce57f137b91ba52f21759324ef868988f1817ca053fd932b5ae`.
The incomplete 128-sample tail in each run is intentionally discarded by reset
or disposal, not called a finalized model frame. Each injected input handle stops
once; a stale callback after stop cannot advance the probability buffer or
publish an utterance. Disposal has a real successful worker acknowledgement,
one worker termination and rejected restart. The existing starter performs two
resets per start; this is recorded, not an efficiency improvement claim.

## Evidence and reproduction

Final source/probe-bound PASS receipt:
`tools/data/results/harness/ten-vad-controller-pcm-2026-10-08-03.json`,
SHA `309e48cd357c48040e4b5aeb362d17e29229994504a6601e47f6bcd35eb6ae0b`.
The receipt binds the runner, probe, relevant TS sources, local audio, packaged
assets and all delivered modules; every hash was independently checked against
current files. Real worker control responses and input stop counts are saved.
The C86 report describes the asset identity/version-pointer limitation.

Negative control: passing pure silence as the required speech fixture exits 1,
writes FAIL with `Unexpected speech/silence segmentation`, and closes Chrome:
`tools/data/results/harness/ten-vad-controller-negative-speech-fixture-2026-10-08-01.json`,
SHA `85ae86949a5990154b24510c1daefcb309cd225b305398ee5166af5d6f074d9c`.
A loaded model alone cannot pass this workflow. Earlier -01/-02 receipts remain,
but only -03 binds the final strengthened/formatted probe.

68 focused controller/starter/monitor/detector/TEN tests PASS; TypeScript build,
Python/JS syntax, formatting and scoped diff checks PASS. Existing actual
fake-device manual-controller Chrome regression PASS with three stopped handles,
closed contexts, ended tracks and controlled pending handoff:
`tools/data/results/harness/manual-controller-post-ten-integration-2026-10-08-01.json`.
The earlier direct TEN model scenario also re-passes with the updated runner:
`tools/data/results/harness/ten-vad-post-controller-integration-2026-10-08-01.json`.
Independent read-only review approves. Full-suite C86 result is historical, not
a newly rerun full suite for C87 (production source is unchanged in C87).

```powershell
node node_modules/typescript/bin/tsc -p tsconfig.json
node scripts/copy-runtime-assets.mjs
python tools/browser-validation/capture-lifecycle-smoke.py --scenario ten-vad-controller --browser "C:/Program Files/Google/Chrome/Application/chrome.exe" --audio tests/fixtures/jfk2.en.wav --output tools/data/results/harness/ten-controller-FRESH.json
```

Choose a new output path each time. No Vite server required. Tests use the public
built browser API; refresh dist first. This does not prove physical capture,
live capture plus model integration, model-based segmentation, original-engine
parity, VAD accuracy, transcription, native streaming or memory reclamation/soak.
Next integrate actual capture with this model-backed controller, retaining
captured-PCM identity separately from canonical file identity; audit misleading
hardcoded FireRed status messages without changing the intentional rough-gate
contract. Orukeet numerical gates and the confirmed next-model order remain open.

# asr.js Whisper 4-Graph Browser Demo

Runs the Whisper large-v3-turbo 4-graph splitgraph path in a browser through the
library pipeline. The demo uses the core package for model loading, mel
processing, decode policy, KV cache handling, and transcript mapping.

```bash
cd examples/demo
npm run dev
```

Open the printed local URL in a real Chrome/Edge window with WebGPU enabled for
WebGPU profiles. WASM profiles require cross-origin isolation for threaded WASM;
the Vite config sets the required headers.

Default model repo:

```text
ysdede/whisper-large-v3-turbo-onnx-4graph
```

The q8 WASM preset is the safest browser fallback. The fp16io encoder + fp32
decoder preset is the current WebGPU target.

Canary 180M Flash smoke page:

```text
https://localhost:8765/canary-smoke.html
```

It uses the approved local smoke bundle under `N:/models` and the fixed
`jfk-short.wav` fixture. The Vite filesystem allow-list is intentionally
limited to this repository and `N:/models`.

Orukeet local-folder smoke page:

```text
https://localhost:8765/orukeet-smoke.html
```

Pick an Orukeet ONNX folder, a 16 kHz mono WAV, a runtime backend and an
encoder precision. It loads through the public browser API, transcribes the
fixture repeatedly, awaits disposal, and prints the receipt as JSON. Real local
inference only; it is not numerical parity against the original NeMo engine.

## Automatic Chrome model test and JSON capture

Latest recovery and reviewed current-code pin:
[Chrome automation recovery](../../docs/reports/chrome-auto-recovery-2026-10-08.md).
The manifest uses the separately reviewed C83 snapshot; earlier snapshots are
preserved. Source/server changes must still fail code identity until reviewed.

The reviewed manifest also provides `chrome-oracle-jfk-webgpu-hybrid` and
`chrome-oracle-pure-silence-webgpu-hybrid`. Each case checks two complete outputs,
original effective PCM, pinned delivered code and awaited cleanup. With the
server below running, choose one case and a fresh output filename:

```powershell
python tools/model-debugging/scripts/port_manifest.py --manifest tools/model-debugging/ports/oruk-orukeet.port.json --run chrome-oracle-pure-silence-webgpu-hybrid --output tools/data/results/orukeet/manifest-silence-new.json
```

Valid silence means explicit empty token IDs and text, not missing inference.
Code-snapshot mismatches fail closed; refresh requires a separate source review,
not automatic regeneration. These cases are output smokes, not numerical parity.

The sibling `webgpu-agent-test` contains the historical `?auto=1` Chrome
runner and `/__test_result__` JSON capture system. Stock Parakeet v3 was
rechecked successfully on 2026-10-08. The tracked Orukeet demo now supports
the same automatic URL workflow, using its existing public local-folder API.
The runner selects local files, waits for two transcriptions and awaited
disposal, compares token IDs and text with the official oracle, closes its
Chrome instance, and saves a fresh PASS/FAIL JSON receipt. No model download.

From the repository root, start the demo (8767 avoids an existing sibling
server on 8765):

```powershell
node node_modules/vite/bin/vite.js examples/demo --host 127.0.0.1 --port 8767 --strictPort
```

In another terminal at the repository root:

```powershell
python tests/smoke/orukeet-auto-browser.py --url https://127.0.0.1:8767/orukeet-smoke.html --model-root models/orukeet/onnx/fp32-packed-2026-10-01 --audio tools/data/fixtures/audio/jfk-short.wav --oracle tools/data/results/orukeet/official-jfk-stage-summary-2026-09-30.json --output tools/data/results/orukeet/auto-browser-new-run.json
```

Use the server's printed scheme (`http` if local certificates are absent).
Default runner URL uses 8765, matching `npm run dev`. Python Playwright and
installed Chrome are prerequisites; the runner never installs them. Use a
new output filename for each run; existing receipts are protected.
The Orukeet ORT sidecar prefix and repository allow-list now follow the Vite
config's location, not a hardcoded checkout path. This is a source-checkout
development workflow (`/@fs/`), not a deployable production asset URL. The
separate Canary fixture still requires its approved `N:/models` directory;
the allow-list has not been widened to an entire drive or workspace parent.
`--backend wasm`, `--precision fp16`, and `--repeat 1` are explicit options,
not automatic fallback policies.

Verified receipt: `tools/data/results/orukeet/auto-browser-jfk-2026-10-08-03.json`:
fp32 WebGPU hybrid, two runs, 38 oracle token IDs and exact text, awaited
model disposal and browser closure. This is a short-fixture output smoke,
not feature/encoder numerical parity, long-audio, streaming, or soak acceptance.
The first receipt (`...-01.json`) remains a FAIL: text matched but the default
canonical segment detail omitted tokens; the demo now uses the supported
`canonical+native` envelope to capture native debug IDs and canonical metrics.

### Optional actual microphone capture to ASR

Add `--verify-capture` to the JFK command and choose a fresh output filename.
Use `--expected-pcm-sha256 ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0`
to retain the separate official file-PCM gate. After normal oracle runs and
disposal, the runner loads another local model and tests two real fake-device
microphone sessions through the public manual controller. Exact captured PCM
must reach each stop-flushed utterance; ASR must be nonempty and match another
public API transcription of that same captured PCM. Contexts/tracks, controller,
model and browser must close. Missing inference cannot pass as a HUD-only test.

Verified capture receipt: `auto-browser-c90-capture-2026-10-08-01.json` under
`tools/data/results/orukeet`. Both sessions delivered 32,000 samples and produced
11 tokens, “And so, my fellow Americans.” The capture hashes differ because
browser capture is not the deterministic file decoder. This proves combined
manual capture/ASR and replay consistency, not official-engine capture parity,
speech-detect ASR, native streaming, physical microphone or soak.
See [C90 report](../../docs/reports/microphone-capture-asr-2026-10-08.md).

### Long-input, same-PCM browser control

The demo now defaults to the existing deterministic WAV decoder exposed by
`browser/media` (using the public entrypoint's source in this source-checkout
demo). It requires mono/16 kHz input and never silently resamples. The previous
AudioContext path remains an explicit `--audio-strategy audiocontext` control.
Supply `--expected-pcm-sha256` to require the official effective float32 PCM;
without it the status explicitly says `PCM_UNBOUND`, even if tokens/text match.
Served JS/TS/runtime response bytes are hashed at the fulfillment boundary,
not taken from Chrome's evictable inspector cache. Runner source is also bound.

```powershell
python tests/smoke/orukeet-auto-browser.py --url https://127.0.0.1:8767/orukeet-smoke.html --model-root models/orukeet/onnx/fp32-packed-2026-10-01 --audio tools/data/results/orukeet/librivox-repeat8-16k-pcm16-2026-10-02.wav --oracle tools/data/results/orukeet/official-long-repeat8-summary-2026-10-02.json --expected-pcm-sha256 caf86c6b47b9e82ea81202df6912366802bed190272485db2329c30e27ea4b46 --output tools/data/results/orukeet/auto-browser-long-new-run.json
```

Verified two-repeat result: `auto-browser-long-repeat8-2026-10-08-06.json`.
All 722 tokens and text match, PCM identity matches, component backend labels
match, code capture and awaited cleanup pass. See
[the scoped report](../../docs/reports/orukeet-auto-browser-long-2026-10-08.md).
This does not pass feature numerical, streaming, fp16, held-out quality or soak gates.

### Cooperative decode cancellation and reuse

Add `--verify-cancellation` to the long-input command and choose a fresh output
filename. Keep `--repeat 2` or greater. The page uses `cancel=1` to request
AbortController cancellation only after a nonzero, incomplete decode-progress
event. It requires `PipelineAbortedError` at the decode stage without a complete
event, then transcribes again on the same model and PCM before awaited disposal.
The two normal results (one before and one after cancellation) must still match
every original-engine token ID and full text; a successful abort alone cannot pass.
Without the flag this optional lifecycle control is not exercised.

Cancellation is cooperative at decoder boundaries, not hard interruption of an
in-flight ORT/GPU call. This source-checkout example is separate from the classic
sibling browser-demo's user cancellation/unload UI and from GPU reclamation/soak.

### Optional chunk-fed offline controller regression

Add `--verify-controller` to the JFK command, with a fresh output filename and
`--expected-pcm-sha256 ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0`.
This feeds 2-second PCM chunks to the public `RealtimeTranscriptionController`
and invokes the same loaded model through `transcribeMonoPcm`. Receipts preserve
chunk coverage, window PCM hashes, native tokens, backend metrics, partial/final
text and reset state. The final text must match the supplied official oracle.
This is windowed offline inference, not microphone capture or native streaming;
individual window token/feature parity is not certified by the full-clip oracle.

The first real run is a preserved FAIL:
`auto-browser-controller-jfk-2026-10-08-01.json`. Normal full-clip repeats match
all 38 oracle tokens and text, but the first 2-second window is rejected because
physical encoder shape `[1, 1024, 26]` does not match valid `encoded_lengths=25`
under the current layout resolver. Model disposal, controller reset and browser
closure still complete. Do not use this optional mode as a passed streaming gate.
See [the scoped report](../../docs/reports/orukeet-windowed-controller-2026-10-08.md).

The subsequent `auto-browser-controller-jfk-2026-10-08-02.json` passes after
the family-specific layout fix. Orukeet's audited export now declares BDT in
its preset; decoder iteration uses valid lengths while tensor addressing uses
physical padded stride. Six chunks / seven requests produce exact final oracle
text and clean reset/disposal. Other presets retain their previous defaults;
custom TDT exports may supply `config.encoderOutputLayout: 'bdt' | 'btd'`
according to their actual graph contract. See
[the fix report](../../docs/reports/orukeet-padded-layout-fix-2026-10-08.md).

Controller inference now explicitly requests `detail: 'detailed'`. Receipts
include cloned window-local timing in `controller.inferenceResults[].localTiming`
and global canonical/partial timing in `controller.updates[]`. `inferenceIndex`
links each inference-backed update to its input/result; `null` identifies a
finalization-only update with no new inference. Missing timing fields remain
`null`, not invented timestamps.

`--verify-controller` also gates `controller_timing_valid`: local-to-global
coordinate and unchanged-payload transfer for words, segments, sentences and
tokens, matching partial detail and strict inference/window associations.
It rejects missing records and nonfinite or fabricated timestamps. JSON `4`
and `4.0` represent the same number; booleans are not numeric substitutes.
For a long fixture add `--require-controller-nonzero-origin` (requires
`--verify-controller`): all four fields must include observed timed detail at
nonzero origins, not just empty shifted windows. Without controller verification
the timing scope is `SKIP_NOT_REQUESTED`. These gates supplement, rather than
replace, full token/text/oracle and cleanup checks; they are not model timestamp
accuracy or original-engine numerical parity.

The 149.7-second repeat fixture exercises nonzero window starts, but its first
timestamp-capture run remains **FAIL**: two full-clip transcriptions match all
722 original-engine token IDs/text, while the controller's accumulated windowed
text differs from the full-clip oracle. Smaller windows change model context;
do not normalize away the mismatch, call this a streaming PASS, or infer a
decoder/merger defect from the full-clip comparison alone. Preserve this failed
control and compare identical window PCM against the original engine to localize
recognition versus merging. Timing capture itself is not a per-window
original-engine timestamp or numerical parity check.

### Replay recorded controller behavior without model inference

From the repository root, the opt-in Vitest test reuses the actual controller
example and shared WAV decoder, checks receipt/audio/full PCM/window PCM
identities, then supplies captured local transcript payloads. It compares every
window, update, final text and reset; no GPU, weights or network are used.

```powershell
$env:ASRJS_CONTROLLER_REPLAY_RECEIPT = 'tools/data/results/orukeet/auto-browser-controller-timing-gated-long-2026-10-08-01.json'
$env:ASRJS_CONTROLLER_REPLAY_SHA256 = 'c058931a0fc94cba0c675b030c5813732778c2844e4ea7be31fd57eb11bfd916'
$env:ASRJS_CONTROLLER_REPLAY_AUDIO = 'tools/data/results/orukeet/librivox-repeat8-16k-pcm16-2026-10-02.wav'
node node_modules/vitest/vitest.mjs run tests/windowed-controller-replay.test.ts
```

Unset these task-specific variables afterward, or use a dedicated shell. With
none supplied the test SKIPs; supplying any requests replay and incomplete,
missing or corrupt inputs FAIL. Use a verified receipt digest, not an implicit
latest-file lookup. PASS means recorded behavior is reproducible, including its
known wrong/full-file-different text. It does not accept model quality, numerical
parity or the original long controller smoke; those gates remain separate.

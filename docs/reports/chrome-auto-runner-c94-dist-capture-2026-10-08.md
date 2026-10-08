# C94 — Built dist speech-detect capture resolved

Status: PASS with documented limits. The paused handoff
[HANDOFF-chrome-auto-runner-2026-10-08.md](HANDOFF-chrome-auto-runner-2026-10-08.md)
is resumed and closed for its acceptance goal; whole-product/model parity stays
open. The preserved C94-01 failure receipt
`auto-browser-c94-dist-capture-2026-10-08-01.json` remains unchanged.

## Root cause of the C94-01 `Speech ASR missing output`

An instrumented reproduction (same flags, same routes, same model/audio, extra
per-utterance recording; no gate changed) showed the built dist pipeline itself
works: through `dist/browser.js` the first three rough-gate utterances decoded
exactly ("And so, my fellow Americans." / "Ask not." / "What your country can
do for you, ask what you can do for your country.") on the WebGPU
encoder + WASM decoder/joint. Iteration 0 then ended with a stop-flush tail of
4608 samples (0.288 s, max amplitude 0.616, all samples nonzero). The official
model decodes that sub-word remnant in 4 iterations and emits **0 tokens**
(`canonical.text: ""`, `native.debug.tokenIds: []`); the helper's requirement
that every utterance window carry non-empty ASR output then threw, so iteration
1 never ran. C92 source-mode receipts had only passed because their final
windows happened to be 35k+ samples; segmentation boundaries vary with real TEN
probabilities, so the old gate was intermittently fatal, not evidence of a
capture or ASR integration defect.

## Change (evidence-based gate correction, safety rule 18)

`ASR_MIN_WINDOW_SAMPLES = 16000` (1 s at 16 kHz) is now defined in
`examples/demo/src/capture-controller.js` and mirrored in
`tests/smoke/orukeet-auto-browser.py`. For every window, `text` must be a
string, `tokenIds` an array, and same-PCM public replay equality still required.
Non-empty text/tokens is required only from windows >= 16000 samples; shorter
stop-flush tails may legitimately decode to zero tokens. No lifecycle, PCM,
provenance, worker, code-binding or replay gate was weakened; the acceptance
status string is unchanged.

## Validation ladder

- Python targeted suite: 35 PASS (34 prior + new
  `test_speech_capture_allows_empty_asr_only_for_short_windows` covering
  empty-short acceptance, replay/type rejection and long-empty rejection).
- Vitest: 22 PASS (`capture-demo-built-runtime` 4, `orukeet-smoke-demo` 13,
  `windowed-controller-demo` 5).
- Fresh real Chrome (Chrome 154.0.8037.98, WebGPU hybrid, fp32), fresh receipts:
  - Dist capture `auto-browser-c94-dist-capture-2026-10-08-02.json`:
    `PASS_AUTO_BROWSER_SAME_PCM_TOKEN_TEXT_SMOKE`, `capture_valid`,
    `capture_build_valid`, one terminated default worker, observer restored,
    contexts closed, PCM sidecars saved.
  - Source regression `auto-browser-c94-source-capture-2026-10-08-01.json`:
    same full PASS, and it actually contains the previously fatal case — run 0
    window `[154112, 160000)` is 5888 samples and decodes empty — so the
    corrected gate is exercised end-to-end in real Chrome.
- Manifest: runner pins reconciled to the reviewed runner
  `04cf682b7b0dcfd1d8d1a2e3d4aebf82578d8ed00d31d6f78de90f3a9dcc86a6`; both
  `chrome-oracle-jfk-webgpu-hybrid` and `chrome-oracle-pure-silence-webgpu-hybrid`
  PASS with all 19 checks true
  (`manifest-chrome-c94-jfk-2026-10-08-02.json`,
  `manifest-chrome-c94-silence-2026-10-08-02.json`); 31 manifest/port unit
  tests PASS.

## Original-weights oracle on the NEW captured windows

Following the C93 recipe, each new window (>= 1 s) from both C94 receipts was
transcribed by the original `.nemo` weights on the exact captured slice
(inspector `7f7b29db55232b34…`, adapter `267b097a252c56c7…`, CPU, 4 threads,
927.7 s). All 14 comparisons PASS with exact token/text equality; the 5888-sample
sub-second tail is excluded by the inspector's one-second minimum, consistent
with the receipt-gate rule. Tracked aggregate with all identities:
`capture-window-original-output-c94-2026-10-08-01.json`.

| Receipt/session | Frame range | Tokens | Original-engine output |
| --- | --- | ---: | --- |
| dist 0 | [0, 39680) | 11 | PASS |
| dist 0 | [38400, 76800) | 4 | PASS |
| dist 0 | [75520, 126720) | 12 | PASS |
| dist 0 | [123392, 160256) | 13 | PASS |
| dist 1 | [0, 19200) | 3 | PASS |
| dist 1 | [17920, 56320) | 4 | PASS |
| dist 1 | [55040, 106240) | 12 | PASS |
| dist 1 | [102912, 179968) | 24 | PASS |
| src 0 | [0, 39680) | 10 | PASS |
| src 0 | [38400, 76800) | 4 | PASS |
| src 0 | [75520, 157440) | 20 | PASS |
| src 1 | [0, 42240) | 10 | PASS |
| src 1 | [40960, 79360) | 4 | PASS |
| src 1 | [78080, 160000) | 20 | PASS |

## Browser code contract refresh (C78 precedent)

The long-lived Vite dev server referenced by the handoff no longer exists; a
fresh server was started with the documented command. Vite therefore serves
clean transforms without the HMR `?t=` import stamps that were embedded in the
c83 contract rows (the C94-01 receipt matched c83 exactly, proving both came
from the same server lifetime). The first manifest rerun failed only
`expected_code` (18/19 checks true), with a 42-row delta consisting solely of
stamp-carrying importer rows plus the preserved uncommitted `nemotron-rnnt`
working-tree rows; the path set stayed identical (281 rows). Following the
reviewed manual-refresh procedure from
[browser-code-snapshot-c78-2026-10-08.md](browser-code-snapshot-c78-2026-10-08.md),
`browser-code-contract-c94.json` (SHA-256
`088e0392d05334061cf61a7a4a59f9817551de3628d070e15ff46e8037727fd5`) was frozen
from the fresh seed receipt `manifest-chrome-c94-jfk-2026-10-08-01.json.browser.json`
and bound in the port manifest; c83 is preserved as predecessor. No verifier
code or threshold changed.

## Limits

- Dist capture is Vite-served `dist/` evidence, not installed/packed npm
  acceptance (unchanged from the handoff).
- Original-engine window comparisons are exact-input output evidence, not
  numerical feature/encoder/state parity; Orukeet's strict production feature
  gate remains open.
- The c94 contract freezes the served bytes of preserved uncommitted
  `nemotron-rnnt` WIP rows; any later change to those files (or another server
  lifetime with new HRM stamps) will correctly fail `expected_code` again.
- f32le sidecars, `original-c94-*` summaries/stage payloads and the failed
  contract-transition receipts (`manifest-chrome-c94-*-2026-10-08-01.json`)
  remain byte-preserved local artifacts, like C93's.

## Reproduce

Server: `node node_modules/vite/bin/vite.js examples/demo --host 127.0.0.1
--port 8767 --strictPort`. Then the handoff's C94 command with a fresh output
suffix, the same command without `--capture-library dist` for the source
regression, and `python tools/model-debugging/scripts/port_manifest.py
--manifest tools/model-debugging/ports/oruk-orukeet.port.json --run
chrome-oracle-jfk-webgpu-hybrid --output <fresh>` for the manifest gate.

# C91 — Actual browser capture PCM against original Orukeet engine

Status: bounded exact-input token/text PASS; whole-product/model acceptance
remains incomplete. C90's same-library replay now has an independent original
NeMo engine comparison. No runtime/model math, tolerance or weights changed.

## Changes and evidence

The existing Chrome runner adds `--save-capture-pcm`, which requires
`--verify-capture`. A caller-owned optional callback receives a clone of actual
captured PCM from the model-neutral capture example. The bridge serializes
little-endian float32 bytes explicitly, preserving signed zero. Python checks
both sessions' lengths, finite values and recorded SHA-256 before exclusively
creating sidecars. All outputs are preflighted before launching Chrome. A write
failure remains FAIL and may retain a partial new artifact; old files are never
deleted or overwritten. No resampling, PCM16 conversion or fixture substitution.

The existing `python-nemo-inspect.py` adds a frozen-f32le input option. Official
NeMo 3.0 `transcribe` accepts arrays and uses its tensor dataset, preprocessor
and decoder. Each baseline/captured/after call receives a fresh owned NumPy
float32 array. CPU, local original weights, batch size 1 and stage capture are
required. Processor and tensor-dataset sample rates must match the declared
input. Inputs shorter than one second are explicitly rejected because this
NeMo dataset pads them; padded input cannot silently count as exact input.
The existing waveform hook checks actual preprocessor hash, shape, valid/total
length, dtype, rate and zero padding before accepting the oracle.

The Orukeet adapter gains `compare_capture_oracle`, a small output comparison,
not another inference runner or decoder. The caller must separately verify
receipt/file digests and parent capture lifecycle. Seven checks cover original
model/weights, frozen identity, effective PCM, six instrumentation controls,
full token IDs and exact text. Numerical stages are deliberately not certified.

Local receipts under `tools/data/results/orukeet/`:

- `auto-browser-c91-capture-pcm-2026-10-08-01.json`: Chrome PASS, two base file
  oracle repeats and two real fake-device manual capture sessions; cleanup PASS.
- Its `.capture-0.f32le` and `.capture-1.f32le`: 128,000 bytes each, 32,000
  samples, mono 16 kHz; exact capture/utterance identity.
- `official-c91-capture-0-summary-2026-10-08-01.json` and `...-1-summary-...`:
  real CPU original-weight NeMo 3.0 / Torch 2.6.0+cu118 / Python 3.10.12 runs,
  all six instrumentation controls PASS; 151 and 154 captured stages respectively.
  Full stages are in the correspondingly named `.json.gz` files.
- `capture-original-output-c91-2026-10-08-01.json`: both seven-check rows PASS;
  browser/summary/stage/adapter identities recorded. Browser receipt SHA-256:
  `ed608afa10f3c649100243ce74de783d4e25a9dbce35dcdf8822bc46bad20ab3`.
- `manifest-chrome-c91-regression-2026-10-08-01.json` and its `.browser.json`:
  independent base Chrome regression PASS, all 19 checks, unchanged C83 pin,
  two exact 38-token official file outputs and actual cleanup.

Both sessions match all 11 original tokens and exact text:
“And so, my fellow Americans.” The official preprocessor receives exactly the
browser bytes, including sample count and no padding. Actual PCM SHA-256:

- Session 0: `e86ce005a089dfb2eb18fb78259e8b0dd8e7d47c7d5685d0e2e40acd64fe1db2`
- Session 1: `7fb1d73a1d4fb3432185a0060c50d72f182cbfa1e143a1f8c6fefd0b1f4fe220`

These observed capture hashes are not future goldens. Original `.nemo` SHA-256
remains `031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56`;
revision `b59c13a733fce5cf193230d0fa317ee7f145a108`. Cached original weights and
the existing NeMo overlay were reused. Qwen's report released its CPU/RAM window;
the two original jobs ran sequentially. No downloads, installs or sibling writes.

## Repeat through existing tools

With the existing demo on 8767, choose a fresh output name:

```powershell
python tests/smoke/orukeet-auto-browser.py --url https://127.0.0.1:8767/orukeet-smoke.html --model-root models/orukeet/onnx/fp32-packed-2026-10-01 --audio tools/data/fixtures/audio/jfk-short.wav --oracle tools/data/results/orukeet/official-jfk-stage-summary-2026-09-30.json --expected-pcm-sha256 ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0 --verify-capture --save-capture-pcm --output tools/data/results/orukeet/capture-FRESH.json
```

For each saved session, use its recorded SHA-256 and actual path, not the old
hash above. `--audio` is source provenance only in frozen mode; it is not fed
through the file decoder. Choose fresh summary/stage paths:

```powershell
& tools/model-debugging/reference/nemotron-3.5-asr-streaming/.venv/Scripts/python.exe tools/model-debugging/reference/hf-parakeet-port/python-nemo-inspect.py --restore-path models/orukeet/source/orukeet-v0.1.0.nemo --model oruk/orukeet --model-revision b59c13a733fce5cf193230d0fa317ee7f145a108 --audio tools/data/fixtures/audio/jfk-short.wav --input-pcm-f32le tools/data/results/orukeet/capture-FRESH.json.capture-0.f32le --input-pcm-sample-rate 16000 --input-pcm-sha256 RECORDED_SESSION_0_SHA256 --device cpu --cpu-threads 4 --sample-id capture-0 --stage-output tools/data/results/orukeet/original-capture-0-FRESH.json.gz --output tools/data/results/orukeet/original-capture-0-FRESH.json
```

Repeat sequentially for session 1 with its own PCM/hash and fresh output paths.
Compare each saved browser run against its original summary using
`browser_manifest.py::compare_capture_oracle(run, oracle, manifest['originalWeights']['sha256'])`.
Before using that pure comparison, verify saved PCM size/SHA against its receipt,
parent capture/code/cleanup gates, current recorded runner/inspector identity and
the summary/stage hashes. This does not rerun capture; a new capture would be a
different input. Keep the saved pair immutable.

For the saved C91 pair, this executable offline check uses the existing adapter
and prints both rows (no inference or output overwrite). Replace paths for a
newly captured pair; preserve its independently recorded receipt digest:

```powershell
python -c @'
import importlib.util, json
from pathlib import Path
p = Path('tools/model-debugging/reference/orukeet/browser_manifest.py')
spec = importlib.util.spec_from_file_location('capture_compare', p)
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
bpath = Path('tools/data/results/orukeet/auto-browser-c91-capture-pcm-2026-10-08-01.json')
assert a.smoke.helpers.digest(bpath) == 'ed608afa10f3c649100243ce74de783d4e25a9dbce35dcdf8822bc46bad20ab3'
b = json.loads(bpath.read_text())
assert b['status'] == 'PASS_AUTO_BROWSER_SAME_PCM_TOKEN_TEXT_SMOKE' and b['browser_closed'] is True
assert a.smoke.check_capture(b['result'], True, 'webgpu-hybrid') and a.smoke.check_code_bindings(b)
assert b['runner_sha256'] == a.smoke.helpers.digest(a.RUNNER)
m = json.loads(Path('tools/model-debugging/ports/oruk-orukeet.port.json').read_text())
assert a.smoke.helpers.digest(Path(m['originalWeights']['path'])) == m['originalWeights']['sha256']
rows = []
for i, run in enumerate(b['result']['capture']['runs']):
    pcm = b['capture_pcm_artifacts'][i]; path = Path(pcm['path'])
    assert path.stat().st_size == run['capturedSamples'] * 4
    assert a.smoke.helpers.digest(path) == pcm['sha256'] == run['capturedPcmSha256']
    opath = Path(f'tools/data/results/orukeet/official-c91-capture-{i}-summary-2026-10-08-01.json')
    o = json.loads(opath.read_text())
    assert o['meta']['inspect_tool_sha256'] == a.smoke.helpers.digest(Path('tools/model-debugging/reference/hf-parakeet-port/python-nemo-inspect.py'))
    rows.append(a.compare_capture_oracle(run, o, m['originalWeights']['sha256']))
print(json.dumps(rows, indent=2))
raise SystemExit(0 if all(r['status'] == 'PASS_CAPTURE_ORACLE_OUTPUT' for r in rows) else 1)
'@
```

The preserved aggregate receipt additionally records summary/stage digests.
This invocation recipe is external orchestration of the tested comparator, not
an automatic original-engine step inside the Chrome runner.

## Validation and remaining scope

25 auto-browser, 5 frozen-inspector, 7 browser-adapter and 24 manifest unittest
methods PASS (61 total, excluding subcases). Byte/signed-zero, corrupt/nonfinite
payload, existing sidecar, old file route, owned-array input, missing boundary
fields, wrong weights/output and failed instrumentation are covered. Node
syntax, Python compilation, example TypeScript and scoped diff checks PASS.
Independent read-only review approves the frozen-input changes.

Runner SHA-256 is now
`e7e6c68506a9427e75d33a99e6962949001040963c028318a96ae3aff431dfbb`;
only its two manifest bindings were updated. C83 base delivered-code contract,
artifacts, official file oracles and tolerances remain unchanged.

This is original-engine **token/text** evidence on the captured input, not
feature/encoder/state numerical parity, human-label quality, physical microphone,
speech-detect ASR, native streaming, performance gain or memory/soak acceptance.
Orukeet's existing strict feature failure and fp16 gates remain open. The cached
NeMo environment still inherits system packages; it is not an isolated newly
provisioned reference stack. Chrome console retains development-server warnings
and the first capture process inherited a TLS-disable environment warning;
the runner rejected non-loopback requests and did not fetch weights externally.
No global setting was changed. Next: speech-detect utterances to actual ASR,
using saved exact PCM/original oracles rather than inferring it from manual mode.

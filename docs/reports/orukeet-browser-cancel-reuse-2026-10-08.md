# C57 — Orukeet real browser partial-decode cancellation and reuse

Status: PASS for this bounded fp32 browser lifecycle/output control. Orukeet
production feature numerical acceptance and general model promotion remain open.

The existing local-folder demo/automatic Chrome runner now accepts optional
`--verify-cancellation` (`cancel=1` on the page URL). All inference, cooperative
cancellation and decoding remain in the public library API. No production math,
original weights, oracle, tolerance or sibling app changed.

## Real evidence

Receipt: `tools/data/results/orukeet/auto-browser-long-cancel-reuse-2026-10-08-01.json`.
Raw SHA256: `206534a9de69304d313a457f5a0d173806800839697e430e64c2e0876037f76f`.
Runner SHA256: `59e6e22428501b10ef501868047500f7645f41d4e054aeaaec0f4e34ea1b0668`.
The raw receipt's bytes are preserved by the narrow dated Git binary attribute.

Chrome 154.0.8037.98 / NVIDIA Blackwell, cached own Orukeet fp32 graphs from
`models/orukeet/onnx/fp32-packed-2026-10-01`; canonical metrics report WebGPU
encoder, WASM decoder/joint and JS processor. Served code/runtime bytes and
selected graph/tokenizer, audio and original-engine oracle hashes are recorded.

The 149.7125-second waveform has 2,395,400 valid mono samples at 16 kHz, no
padding/resampling, effective f32le SHA256
`caf86c6b47b9e82ea81202df6912366802bed190272485db2329c30e27ea4b46`.
One complete transcription precedes the cancellation attempt. Abort is requested
after 4 of 1872 encoder frames are reported decoded; the public call rejects as
`PipelineAbortedError` at stage `decode`, with no complete event. A subsequent
complete transcription on the same loaded model and PCM succeeds. Both complete
results match all 722 official NeMo token IDs and full text. Model disposal is
awaited and Chrome closure recorded.

Observed load: 17,803 ms. Complete calls: 7,389 ms / RTFx 20.26 before abort and
5,245 ms / RTFx 28.54 after. These are observations, not an optimization claim or
matched A/B speedup. Cancellation is cooperative between decoder iterations;
the currently running ORT call is not forcibly interrupted.

## Reproduce

With the existing `examples/demo` server on 8767 (see its README):

```powershell
python tests/smoke/orukeet-auto-browser.py --url https://127.0.0.1:8767/orukeet-smoke.html --model-root models/orukeet/onnx/fp32-packed-2026-10-01 --audio tools/data/results/orukeet/librivox-repeat8-16k-pcm16-2026-10-02.wav --oracle tools/data/results/orukeet/official-long-repeat8-summary-2026-10-02.json --expected-pcm-sha256 caf86c6b47b9e82ea81202df6912366802bed190272485db2329c30e27ea4b46 --verify-cancellation --output tools/data/results/orukeet/auto-browser-long-cancel-reuse-FRESH.json
```

Always use a fresh output. The flag requires at least two successful repeats;
absence of the flag does not exercise cancellation. A successful cancellation
alone cannot grant PASS: repeat count, actual component labels, oracle tokens/
text, PCM, served-code identity and awaited disposal are still required.

Eleven automatic Orukeet runner tests, twelve browser helpers and fourteen
local-Parakeet tests pass (37 total). New negative cases reject missing abort
evidence, completed/before-decode attempts, wrong error/stage, invalid counters
and repeat=1. Python compilation, JS syntax and targeted diff checks pass.
Unit metadata checks do not establish runtime behavior; the real receipt does.

This closes one public API cancel/reuse/dispose browser row, not the classic
sibling demo's absent user cancel/unload controls, load interruption, every
partial acquisition failure, GPU memory reclamation, streaming, fp16, quality or
soak. Production feature mismatch gates remain unchanged. Next model delivery
work must address those numerical/quality gaps, not count more output smokes as
general model acceptance. Sibling UI implementation still awaits write authority.

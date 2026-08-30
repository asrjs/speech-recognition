# X-ASR backend placement after speculative joiner batching (2026-08-30)

## Scope

The X-ASR 160 ms Zipformer2 streaming path is now measured on the same
Chrome-headless harness with an explicit `backend=wasm` control. This closes
the placement question for the current local artifact and ORT Web 1.29 stack;
it does not change the library default or claim cross-device behavior.

Audio: `/gigaam-audio/jfk-short.wav`, 11.29 s, 55 x 200 ms streaming chunks.
Browser: Chrome headless new, NVIDIA Blackwell, ORT Web 1.29.0, one same-session
warm-up and three timed runs, exact fixed transcript oracle. The WebGPU result
includes the shipped speculative batched joiner decode.

## Results

| composition | median transcribeMs | median RTFx | oracle |
| --- | ---: | ---: | --- |
| WebGPU encoder/state + WASM decoder/joiner | 8336 | 1.3196x | exact |
| WASM encoder/state/decoder/joiner | 6822 | 1.6124x | exact |

WebGPU measured runs: 8443, 8252, 8336 ms (1.303, 1.333, 1.320x).
WASM measured runs: 7124, 6469, 6822 ms (1.544, 1.700, 1.612x).
Both paths return the byte-identical 55-token JFK transcript.

## Phase attribution

A one-warm-up profiled run on each backend (two passes, so phase totals are
approximately two transcriptions):

- WebGPU: encoder 136 runs / 14,983 ms (110.17 ms/run), decoder 188 /
  757 ms, joiner 175 / 626 ms; 622 joiner rows.
- WASM: encoder 136 runs / 13,266 ms (97.54 ms/run), decoder 188 / 97 ms,
  joiner 175 / 265 ms; 622 joiner rows.

The encoder dominates both backends. The WASM encoder is ~11% faster per
stateful run on this host, and the WASM decoder/joiner is also cheaper; the
GPU path has no compensating advantage for this small, dispatch-heavy
streaming workload. This agrees with the goal's workload-specific placement
rule and the previous 116-state GPU-residency finding: keeping state on GPU
avoids a much worse CPU-state WebGPU path, but does not make WebGPU the fastest
provider for this artifact.

## Decision

- Keep the existing explicit backend controls and exact WebGPU path.
- For this X-ASR 160 ms model on the measured Blackwell/ORT 1.29 setup, prefer
  **WASM** when throughput is the priority; retain WebGPU for applications
  that need GPU residency or when a different adapter/workload reverses the
  result. Do not silently change the public default based on one adapter.
- The next optimization target remains the stateful Zipformer2 encoder itself:
  investigate why a 116-state encoder run costs ~100 ms (kernel/dispatch versus
  state tensor traffic) before changing chunk semantics or fbank code.

## Harness change

`webgpu-agent-test/src/x-asr-webgpu.js` accepts `?backend=wasm` and reports
the selected backend; `scripts/run-xasr-webgpu.mjs --backend=wasm` supplies
the control. `--backend=webgpu` remains the default. Profile mode is
`--profile` and records per-session encoder/decoder/joiner run counts and
timings in the result log.

Evidence:

- `tools/data/results/x-asr/x-asr-zh-en-160ms-jfk-short-webgpu-stream-chrome.json`
- `tools/data/results/x-asr/x-asr-zh-en-160ms-jfk-short-wasm-stream-chrome.json`
- `tools/data/results/x-asr/x-asr-zh-en-160ms-jfk-short-wasm-stream-profiled-chrome.json`
- `docs/reports/x-asr-joiner-batching-2026-08-30.md`


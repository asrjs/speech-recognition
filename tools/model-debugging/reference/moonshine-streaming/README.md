# Streaming Moonshine official oracle (project Transformers 5 stack)

Local snapshot `.cache/moonshine-streaming-tiny` is byte-identical to
`moonshine-ai/moonshine-streaming-tiny` at revision `f8e9dfd8c562…`
(size-verified per file; official org, not a community conversion). The old
`.cache/moonshine-streaming-remapped` layout and
`model.remapped.safetensors` are the pre-5.18 4.57-era layout (19/19 key names
differ) — do not use them with transformers 5; the original `model.safetensors`
matches `MoonshineStreamingForConditionalGeneration` exactly
(161 tensors, 0 missing / 0 unexpected / 0 shape mismatch).

## Oracle result (2026-10-08)

`official_oracle.py` (run with `tools/model-debugging/reference/transformers5`):

```powershell
tools/model-debugging/reference/transformers5/Scripts/python.exe tools/model-debugging/reference/moonshine-streaming/official_oracle.py --output tools/data/results/model-candidates/moonshine-streaming-official-oracle-jfk-FRESH.json
```

On the pinned JFK effective PCM (`ebd52851100536db02d12c49fddd010372dcdc70243562e057553d476b706ae0`):
`PASS_OFFICIAL_STREAMING_ORACLE` — 28 tokens, canonical transcript, decoder
cache stepwise==one-shot, fresh-instance reset equality, encoder hidden states
`[1, 550, 320]` captured with SHA-256 for later export parity. Receipt:
`tools/data/results/model-candidates/moonshine-streaming-official-oracle-jfk-2026-10-08.json`.

## Contract finding: the attention mask is required

Generating without the processor's `attention_mask` degenerates into repeated
tokens on this whole-utterance path, while the offline moonshine model on the
same fixture/stack is mask-insensitive. The streaming encoder must receive the
padding mask; any faithful ONNX export or library path must carry mask
semantics (or bake all-ones behaviour with explicit length handling).

## Re-export progress (2026-10-08)

The old export blocker ("dynamo needs onnxscript, not installed") no longer
holds: transformers5 venv has torch 2.14.1+cpu with onnxscript 0.7.2.

**Encoder done — `PASS_STREAMING_ENCODER_FP32`**
(`export_streaming_encoder.py`, receipt
`tools/data/results/model-candidates/moonshine-streaming-encoder-fp32-2026-10-08.json`):
our own fp32 ONNX graph at `.cache/moonshine-streaming-onnx/encoder_model.onnx`
(opset 18, 377 nodes, dynamic `batch`/`time`, inputs `input_values` FLOAT +
`attention_mask` INT32 — the required mask from the oracle finding). The legacy
TorchScript tracer still fails (`invalid unordered_map<K,T> key`); the torch
dynamo exporter works. Parity: onnxruntime fp32 hidden vs torch fp32 hidden
max abs 4.26e-5 / mean 1.0e-6 / cosine 0.99999996, allclose atol/rtol 1e-4, and
end-to-end greedy tokens through the ONNX encoder equal the official oracle's
28 token ids and canonical text. Gate rationale recorded in the receipt: the
torch fp32 encoder is bit-stable in-process but not across processes (oneDNN
thread scheduling), so digests are identity evidence while the gate is explicit
fp32 tolerances plus exact token/text equality.

**Decoder done — `PASS_STREAMING_DECODER_FP32`**
(`export_streaming_decoder.py`, receipt
`tools/data/results/model-candidates/moonshine-streaming-decoder-fp32-2026-10-08.json`):
two fp32 graphs at `.cache/moonshine-streaming-onnx/` with external data —
`decoder_prefill_model.onnx` (seed: one BOS token, empty past) and
`decoder_step_model.onnx` (one token + per-layer past K/V `[batch, 8, past>=1,
40]`, positions from the cache length). Cross-attention K/V are computed inside
each graph from `encoder_hidden_states`, so the pair is stateless across
utterances. Both graphs take exactly one token: that keeps `q_len > 1` a Python
bool so SDPA's `is_causal` never becomes symbolic, which is precisely greedy
causal semantics — no library code is patched. The emitter's legacy
`Split num_outputs` attributes (rejected by ORT under opset >= 13) are dropped
by a recorded sanitizer, and the parity gates re-verify the pipeline after it.
Gates: per-step final-position logits vs a torch reference greedy loop max abs
2.67e-5 over 27 steps (atol 1e-4), and the fully-ONNX pipeline (exported
encoder -> seed/step graphs) reproduces the official oracle's 28 token ids and
canonical text exactly.

**fp16 set done — `PASS_STREAMING_FP16`**
(`export_fp16.py`, receipt
`tools/data/results/model-candidates/moonshine-streaming-fp16-2026-10-09.json`):
all three graphs re-exported directly from torch with `dtype=float16`
(the measured Qwen lesson: ORT's float16 converter breaks graphs; never
post-convert) into `.cache/moonshine-streaming-onnx-fp16/` (284 MB vs 566 MB
fp32). The fully-fp16 pipeline reproduces the official oracle's 28 token ids
and the fp32 pipeline's tokens exactly; drift evidence: encoder hidden cosine
0.9999876 (max abs 5.8e-2), per-step final logits cosine 0.9999996 (max abs
8.8e-2 on large-magnitude logits). Everything is fp16 in this pass including
norms/softmax; the controlled mixed-precision fallback was not needed on this
fixture and stays documented, not silently applied. No WebGPU run yet: browser
speed/memory and the fp16 WebGPU provider path are unmeasured.

**Chunked-state equivalence proven — `PASS_CHUNKED_EQUIVALENCE`**
(`chunked_encoder_probe.py`, receipt
`tools/data/results/model-candidates/moonshine-streaming-chunked-equivalence-2026-10-09.json`):
feeding audio in chunks with explicit state reproduces the whole-utterance
encoder exactly — embedder stage max abs <= 1.4e-6, full encoder cosine 1.0
with max abs <= 3.2e-5 (allclose atol/rtol 1e-4, the fp32 tolerance calibrated
for this model's cross-run attention reordering) over chunk sizes 16/64/137/512
on the JFK fixture and a synthetic odd-length waveform. Proven state contract:

- per-frame CMVN/asinh/linear front-end is stateless per 5 ms frame;
- both causal stride-2 convs carry only their last 4 input frames and an
  output-phase counter (output j uses inputs [2j-4 .. 2j]; zeros only before
  the stream start);
- the six attention layers have no absolute positions (relative windows
  (16,4)/(16,0)): each output frame needs 15 left and up to 3 right context
  frames per layer — a runtime carries 15 hidden frames per layer, with
  emission delayed by the lookahead; the proof re-runs slices with the
  compounded 90-left/24-right context instead.

This is the correctness foundation for a chunked ONNX encoder graph
(state-in/state-out); that export plus the library integration and the real
WebGPU browser validation of both precision sets remain.

**Chunked encoder graph — `PASS_CHUNKED_ENCODER_GRAPH`**
(`export_chunked_encoder.py`, receipt
`tools/data/results/model-candidates/moonshine-streaming-chunked-encoder-graph-2026-10-09.json`):

`.cache/moonshine-streaming-onnx/encoder_chunked_model.onnx` is a PURE
FUNCTION: (audio window, explicit state) -> (hidden frames, new state). It
keeps no memory; juniors can reason about each call in isolation.

CONTRACT (read this before touching the runtime):
- Inputs: `input_values [1, n, 80]` (n = emit+60; emit a multiple of 4,
  emit >= 16), `embedded_context [1, 90, 320]`, `context_len` i64,
  `conv1_tail [1, 320, 4]` (conv1-INPUT domain), `conv2_tail [1, 640, 4]`
  (silu(conv1) domain), `frames1/frames2` i64 bookkeeping.
- Outputs: `hidden [1, n/4, 320]` — VALID for the first `emit/4` embedded
  frames; plus every state tensor mirrored (`new_*`).
- All state is RIGHT-ALIGNED; zero-fill exists only before the stream start.
- Window rule: a call's trailing 60 audio frames (15 embedded x 6 layers of
  right context) are lookahead — RE-FED next call; the window advances by
  `emit`, and the conv tails roll back to the emission cut, which is exact
  because the convs are deterministic local ops.
- FINAL FLUSH: do not use this graph. Run the whole-utterance encoder graph
  on an audio tail (>= 480 frames before the cut) and keep the tail hiddens;
  positions are relative, so the tail run equals the stream there.
- Latency: emission lags input by 60 frames = 1.2 s (model-inherent, from the
  4-frame per-layer right window).

Parity: torch-module and ONNX chunked runs vs the whole-utterance official
encoder on JFK + synthetic at emit 20/64/256 (max abs 1.2e-5 / 3.9e-5,
allclose 1e-4), and the fully-ONNX chunked pipeline (chunked encoder +
whole-encoder final flush + fp32 decoder graphs) reproduces the official
oracle's 28 token ids exactly.

**Library integration — `MoonshineStreamingSession`**
(`src/models/moonshine/streaming-session.ts`, exported from the family index):
the runtime-side consumer of the chunked contract, deliberately junior-scale —
injected ORT-session runners (no I/O, no framework), explicit state, the
geometry (context 90 / lookahead 60 / emit) as options, and plain text results
(canonical PartialTranscript assembly stays a runtime-lane wrapper). Window
bookkeeping lives in ~60 readable lines: buffer samples, run a chunked call per
full `emit+60` window, keep the first `emit/4` hidden frames, advance by
`emit`, retain the last 480 frames for the whole-encoder final flush. Verified
twice:

- `tests/moonshine-streaming-session.test.ts` (9): scripted index-tagged mocks
  pin the window rules, state carry, retention bound, decoder past wiring,
  preview repeatability and reset/dispose.
- `tests/moonshine-streaming-graphs.test.ts` (3, real fp32 graphs in Node ORT,
  skipped when graphs are absent): one-second pushAudio chunks finalize to the
  exact official oracle transcript with whole-utterance framing; a mid-stream
  preview decodes; reset/reuse at emit=64 reproduces identical tokens.

Bugs the gates caught during integration: window advance/counters in samples
vs frames, whole-window hiddens stored instead of the first emit/4, the
chunked graph's rank-3 `[1, frames, 80]` input, the whole-encoder's required
attention mask, spread-push stack overflow, and a naive 44-byte WAV parse in
the test fixture loader.

**fp16 chunked graph done** (`export_fp16.py` extended; receipt
`tools/data/results/model-candidates/moonshine-streaming-fp16-chunked-2026-10-09.json`,
superseding the three-graph fp16 receipt): the fp16 set now includes
`encoder_chunked_model.onnx` with the whole-encoder time dim dynamic (the
final-flush tails need arbitrary lengths). Gates: the fully-fp16 chunked
pipeline (chunked encoder at emit=64 + fp16 final flush + fp16 decoder graphs)
reproduces the official oracle's 28 token ids exactly, alongside the existing
whole-file and fp32-equality gates.

The session's `dtype: 'float16'` option feeds fp16 graphs (raw half bits via a
round-to-nearest-even converter). Node-side fp16 execution is blocked by an
ORT-vs-Node-26 interaction, root-caused 2026-10-09 with a minimal Relu graph:

- feeding ANY float16 input fails with "not enough space: expected 8, got 0"
  on onnxruntime-node 1.25.0-dev, 1.30.0 and 1.31.0-dev alike;
- the JS wrapper converts Uint16Array to Node's Float16Array whenever the
  global exists (Node >= 24) and the native NAPI bridge cannot read
  Float16Array; hiding the global makes the same feed work (raw half bits
  decode correctly), proving the diagnosis;
- fp32-input/fp16-output works fine on the same builds, so only the input
  transfer is affected.

onnxruntime-node was therefore bumped from the 1.25.0-dev pin to stable
1.30.0 (peer + dev; onnxruntime-web stays 1.29.0) with the ORT-dependent
suites re-run for regressions. The real-graph fp16 test probes once, logs the
reason and activates automatically when a build that reads Float16Array
lands. Export-level fp16 evidence comes from Python ORT, and the browser
validation slice uses onnxruntime-web, whose tensor path is separate.

**Canonical transcriber done**
(`src/models/moonshine/streaming-transcriber.ts`):
`MoonshineStreamingTranscriber` implements the core `StreamingTranscriber`
contract over the session — pushAudio buffers (and preview-decodes when
`emitPartials` is on), `previewText` mirrors the latest decode,
`committedText` publishes at finalize, revisions advance per update,
reset/dispose follow the contract and non-16 kHz input is rejected rather
than silently resampled. Seven unit tests pin the contract mapping; one
real-graph test pushes two-second chunks and asserts the final
`committedText` equals the official oracle transcript with canonical meta
(modelFamily moonshine-streaming, isFinal, sampleRate, durationSeconds).

**Preset asset routing done** (`src/models/moonshine/streaming-loader.ts`,
adapter in `src/presets/moonshine-local-adapter.ts`): `loadMoonshineStreaming`
creates the four ORT sessions (onnxruntime-web by default, injectable for
tests and the browser demo), fetches the tokenizer and returns a
transcriber + session + dispose that releases every session. The local
adapter resolves `builtInLoadOptions.streaming` when a folder carries the
streaming graphs (one locator per file; the whole-utterance encoder doubles
as the final-flush graph) and omits it for offline-only folders. Five unit
tests cover session creation/provider selection/tokenizer-failure release
and both adapter routings; the full suite is 1705 passed / 0 failed.

**Remaining**: real browser (WebGPU) validation of all graph sets — the
first real fp16 execution, via onnxruntime-web's separate tensor path.
Invalid historical streaming graphs stay excluded.

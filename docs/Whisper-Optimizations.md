## Whisper Large V3 Turbo WebGPU optimization report for `asrjs/speech-recognition`

Your preflight agent is aiming at the right bottleneck class, but the implementation details need tightening. The biggest practical opportunity is **not “make it more like CTranslate2” in the abstract**; it is to stop the decoder loop from repeatedly moving KV cache and logits through JS-owned typed arrays.

Whisper Large V3 Turbo is already a good target for this because it keeps the large-v3 encoder but reduces the decoder from 32 layers to 4, which makes decode-loop overhead proportionally more visible. ([Hugging Face][1]) Faster-whisper/CTranslate2 gets its gains from a fast transformer runtime, lower precision/quantization, and reduced memory movement; its README claims up to 4× speedup over OpenAI Whisper and further gains from 8-bit quantization. ([GitHub][2])

### Executive conclusion

Your current code already has the right 4-graph architecture, KV cache, beam search, timing metrics, and splitgraph flow. But the hot path still behaves like a CPU-orchestrated decoder:

* `runDecoderStepSplit()` reconstructs input tensors from JS typed-array data every step and then reads `logitsTensor.data` plus `presentKv` back into typed arrays.
* `transcribeWithSplitGraph()` passes `encoderHiddenStates.data` into the splitgraph decode loop and converts `presentKv` tensors into raw `.data` objects across init/step boundaries.
* Stable beam search still runs one decoder step per active beam, but candidate
  expansion now uses bounded top-k selection and does not allocate a full
  vocabulary log-softmax array per beam. Batched beam remains opt-in.

That means the next speedup should focus on **GPU-resident tensors + scalar-only readback + batched beam execution**, before spending too much effort on speculative fused-attention exports.

---

## Reality check on the other agent’s claims

| Claim                                                                      | Verdict                                                                                                                                                                                                                                                                                 | What to do instead                                                                                                  |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| “Use `decoderSession.createIOBinding()` in ORT Web”                        | Directionally right, API shape is misleading. ORT Web documents GPU-buffer tensors, preallocated output tensors, and `preferredOutputLocation`, not the native C++-style API shown in the snippet. ([onnxruntime.ai][3])                                                                | Use `Tensor.fromGpuBuffer()`, `preferredOutputLocation: 'gpu-buffer'`, and `fetches` with preallocated GPU tensors. |
| “Set `onnxruntime.ep.webgpu.past_present_share_buffer=1`”                  | I would not trust this for ORT Web. I did not find this as a documented ORT Web option; ORT Web’s documented knobs are `externalData`, `freeDimensionOverrides`, `enableGraphCapture`, and `preferredOutputLocation`. ([onnxruntime.ai][4])                                             | Implement your own static KV-cache export or GPU-buffer ping-pong path.                                             |
| “Fuse Whisper attention to `MultiHeadAttention` and WebGPU will be faster” | Possible, but not guaranteed. ORT WebGPU’s generated operator table lists `Attention` and `MultiHeadAttention`, but comments say mask and past/present still need implementation; `Reshape` and `Shape` also show no GPU kernel, and `Transpose` needs perf optimization. ([GitHub][5]) | Profile the current primitive graph first, then try fused variants as a separate export experiment.                 |
| “Custom WebGPU top-k/beam kernel”                                          | Correct for beam search, especially because your current CPU top-k/full-sort path is expensive.                                                                                                                                                                                         | Start with greedy GPU ArgMax helper; then add WebGPU top-k for beam.                                                |

---

## Priority 0: keep decoder state and logits on GPU

ORT WebGPU explicitly says default inputs/outputs are CPU tensors, causing copies to/from GPU, and recommends IO-binding-style GPU tensors for transformer loops where the previous output becomes the next input. ([onnxruntime.ai][3]) That matches your exact decoder-step loop.

### Target change

For WebGPU backend, add a separate fast path:

```ts
interface GpuWhisperStepState {
  logitsGpu: unknown;        // ort.Tensor location gpu-buffer
  kvGpu: Record<string, unknown>;
  tokenGpuOrCpuScalar: unknown;
}
```

Use session options like:

```ts
const sessionOptions = {
  executionProviders: [{
    name: 'webgpu',
    powerPreference: 'high-performance',
    validationMode: 'disabled', // production only; keep basic/full for debugging
  }],
  graphOptimizationLevel: 'all',
  preferredOutputLocation: 'gpu-buffer',
};
```

ORT Web’s WebGPU EP options include `powerPreference`, `device`, `preferredLayout`, `forceCpuNodeNames`, and `validationMode`; using `validationMode: 'disabled'` should be a benchmarked production toggle, not a default during parity work. ([onnxruntime.ai][6])

### Why this is first

Right now `OrtTensorLike` only models CPU-style `.data` tensors.  But ORT Web supports tensors created from WebGPU buffers and lets outputs remain on GPU through `preferredOutputLocation` or preallocated fetch tensors. ([onnxruntime.ai][3])

For greedy decode, you do not need the full `[1, 1, vocab]` logits on CPU. You only need:

1. next token id,
2. maybe its logprob/confidence,
3. EOS check.

So the fastest path should read **one token id**, not 51k logits.

### First implementation step

Add a `decoder_step_argmax.onnx` or tiny WebGPU shader:

`logits_gpu -> suppress/timestamp mask -> argmax -> token_id`

For greedy mode, ORT WebGPU already supports `ArgMax`, `Softmax`, many reductions, and common math ops. ([GitHub][5]) For beam mode, do not assume ONNX `TopK` is available in browser WebGPU; the current browser operator table I found does not list `TopK`, so custom WGSL top-k is safer.

---

## Priority 1: eliminate KV cache CPU round-trips

Your current step graph returns `present.*`, then code converts it into typed arrays and feeds it back as `past_key_values.*` on the next step.  That is exactly the memory churn CTranslate2 avoids.

### Two viable designs

**Design A — lower risk: GPU ping-pong KV tensors**

Keep your existing dynamic-shape decoder graph, but set `preferredOutputLocation` for all `present.*` outputs to `gpu-buffer`. Feed those GPU tensors directly into the next step as `past_key_values.*`. This avoids CPU copies but still allocates growing present buffers.

This is the best first experiment because it does not require a new ONNX export.

**Design B — higher upside: static full-length KV cache graph**

Export a `decoder_step_static_cache.onnx` where the KV inputs are fixed-size:

```txt
self_kv: [batch, heads, 448, head_dim]
cache_position: scalar/int32
input_ids: [batch, 1]
```

The graph writes the new K/V slice into position `cache_position` and returns the same fixed-shape cache. This is the closest browser equivalent to CTranslate2’s preallocated KV cache. It also makes ORT WebGPU graph capture much more realistic because ORT’s docs say graph capture helps when shapes are static and all kernels run on WebGPU. ([onnxruntime.ai][4])

For Turbo, the self-attention KV cache is not huge: 4 decoder layers × K/V × 20 heads × 448 tokens × 64 head dim. The bigger pain is not final memory size; it is repeated allocation/copy/reconstruction every token.

---

## Priority 2: batch beam search

Your current beam path runs `session.runStep()` separately for each active beam.  With `beamSize=5`, that can approach 5 decoder-step ORT calls per generated token.

### Better design

Export or adapt decoder step to accept:

```txt
input_ids: [beam, 1]
past_key_values.*: [beam, heads, seq, head_dim]
```

Then one ORT call computes all active beams. After selecting survivors, reorder KV by beam parent.

This is likely your biggest beam-search speedup. Even if greedy is already 5× RTFx, beam search can be much slower because the code currently multiplies ORT calls by active beams.

### Immediate CPU fallback improvement

While GPU top-k is being built, replace:

```ts
Array.from(logProbs).sort(...)
```

with a fixed-size min-heap or partial-selection top-k over `Float32Array`. That will not fix GPU readback, but it removes a lot of JS allocation and full-vocab sorting.

---

## Priority 3: use ORT WebGPU graph capture and free-dimension overrides carefully

ORT Web recommends profiling first, then trying graph capture, free-dimension override, and GPU-resident tensors for WebGPU performance. ([onnxruntime.ai][7]) Your implementation already records useful timing buckets like `decoderStepFeedBuildMs`, `decoderStepRunMs`, `decoderStepOutputMs`, p50/p95/max, and decode iteration count.

### Experiment matrix

Run each variant on the same 30-second fixture:

| Variant                                                           | Expected signal                                                                  |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Current WebGPU baseline                                           | Establish `decoderStepRunMs` vs feed/output overhead                             |
| `preferredOutputLocation: 'gpu-buffer'` for logits + KV           | Should reduce output/download time                                               |
| Preallocated fetches for known-shape outputs                      | Should reduce allocator pressure where shapes are fixed                          |
| Static-cache graph + `enableGraphCapture: true`                   | Best chance to reduce command preparation overhead                               |
| `freeDimensionOverrides` for batch=1, seq=1, max cache length=448 | May help shape inference; ORT warns it is model-dependent. ([onnxruntime.ai][4]) |

Graph capture should not be forced blindly; ORT says it can fail for some models and dynamic decoder models are specifically a caution area. ([onnxruntime.ai][4])

---

## Priority 4: export-level graph optimization, but validate every fusion

Use ORT’s transformer optimizer as an offline experiment. ORT says transformer-specific optimization helps when runtime load-time optimization misses fusions, when fp16 helps GPU performance, and when dynamic axes block shape-based optimizations. ([onnxruntime.ai][8])

But for Whisper WebGPU, do not assume `MultiHeadAttention` fusion is automatically better. The current WebGPU operator table’s comments around `Attention`/`MultiHeadAttention` say mask and past/present still need implementation, while `Shape`/`Reshape` can fall back or warn. ([GitHub][5])

### Export validation checklist

For each graph variant:

1. Run `onnx.checker`.
2. Run native ORT parity.
3. Run ORT Web WASM parity.
4. Run ORT WebGPU parity.
5. Capture ORT verbose logs and confirm no important nodes fall back to CPU.
6. Compare node counts and op types before/after optimization.
7. Keep fused and primitive variants side-by-side in the manifest until benchmarked.

---

## Priority 5: fix WebGPU packaging/import details

Your `initWhisperOrt()` currently imports `onnxruntime-web`, while ORT’s WebGPU docs tell web apps to import `onnxruntime-web/webgpu` or use `ort.webgpu.min.js` for WebGPU. ([onnxruntime.ai][3]) Your code still creates `executionProviders: [{ name: 'webgpu', ... }]`, so it may work depending on bundler/package resolution, but I would make the WebGPU import explicit to avoid accidentally shipping a WASM-only bundle.

Recommended narrow change:

```ts
const imported = backendId.startsWith('webgpu')
  ? await import('onnxruntime-web/webgpu')
  : await import('onnxruntime-web');
```

---

## What not to prioritize yet

I would not start with q8/q4. CTranslate2’s quantization support is a major reason faster-whisper is efficient, but your previous parity work already showed q8 decoder sensitivity. CTranslate2 supports INT8/FP16/BF16/4-bit AWQ modes, but browser ORT WebGPU quantized transformer coverage is not identical to CTranslate2 CUDA. ([opennmt.net][9]) For your library, correctness and full Whisper controls are the differentiator; keep fp16 WebGPU as the main fast path, then revisit q4/q8 as optional model variants.

I also would not ask an agent to “fuse everything” broadly. ORT’s optimizer docs explicitly warn that many optimizations require exact subgraph matches and can be affected by export/layout differences. ([onnxruntime.ai][8])

---

## Recommended implementation order

1. **Add WebGPU profiling harness.** Use `ort.env.webgpu.profiling`, `enableProfiling`, and your existing decoder timing metrics. ([onnxruntime.ai][7])
2. **Explicit WebGPU import.** Switch to `onnxruntime-web/webgpu` only for WebGPU backend.
3. **GPU output retention.** Add `preferredOutputLocation` for encoder output, decoder logits, and `present.*` KV outputs.
4. **Greedy scalar readback.** Add ArgMax helper graph or WGSL shader so JS reads only `nextTokenId`.
5. **GPU KV bridge.** Feed GPU `present.*` tensors directly into next step as `past_key_values.*`; avoid `.data` for WebGPU.
6. **Batched beam step.** One decoder-step call per token for all beams, plus KV reorder.
7. **Static-cache export.** Fixed `[1 or beam, heads, 448, head_dim]` cache + `cache_position`; try graph capture.
8. **Offline optimizer variants.** Try fused attention/layernorm models only after profiling primitive graph bottlenecks.

---

## Tight coding-agent prompt

```text
We need one focused performance experiment in asrjs/speech-recognition.

Goal:
Optimize Whisper splitgraph WebGPU decode by keeping decoder outputs on GPU instead of reading logits/KV back to JS each token.

Scope:
- Only touch src/models/whisper-seq2seq/ort.ts and the splitgraph WebGPU path in executor.ts/core adapters as needed.
- Do not change Whisper semantics, token suppression rules, beam search behavior, timestamps, chunking, or public API.
- Do not refactor unrelated model families.
- Do not modify exporter yet.

Tasks:
1. Make WebGPU backend import `onnxruntime-web/webgpu`; keep WASM import unchanged.
2. Add WebGPU-only session options for `preferredOutputLocation`, initially for decoder_step outputs and encoder output.
3. Preserve the existing CPU/WASM path exactly.
4. Add instrumentation that reports how many decoder tensors are CPU `.data` tensors vs GPU-buffer tensors per decode.
5. Benchmark one 30s fixture greedy mode before/after and report:
   - totalMs / rtfx
   - decodeMs
   - decoderStepFeedBuildMs
   - decoderStepRunMs
   - decoderStepOutputMs
   - p50/p95 step
6. Keep parity: generated token IDs must match baseline for the same fixture.

Hard rules:
- No broad architecture changes.
- No q8/q4 work.
- No fused-attention export work.
- No beam rewrite in this pass.
```

Bottom line: your next meaningful speedup is probably hiding in the JS/WebGPU boundary, not in the ONNX math itself. First make greedy decode GPU-resident and scalar-readback only; then batch beam search. That gives you a clean path toward CTranslate2-style execution while staying compatible with ORT Web.

[1]: https://huggingface.co/openai/whisper-large-v3-turbo?utm_source=chatgpt.com "openai/whisper-large-v3-turbo"
[2]: https://github.com/SYSTRAN/faster-whisper?utm_source=chatgpt.com "Faster Whisper transcription with CTranslate2"
[3]: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html "Using WebGPU | onnxruntime"
[4]: https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html "The ‘env’ Flags and Session Options | onnxruntime"
[5]: https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md "onnxruntime/js/web/docs/webgpu-operators.md at main · microsoft/onnxruntime · GitHub"
[6]: https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.WebGpuExecutionProviderOption.html "WebGpuExecutionProviderOption | ONNX Runtime JavaScript API"
[7]: https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html "Performance Diagnosis | onnxruntime"
[8]: https://onnxruntime.ai/docs/performance/transformers-optimization.html "Transformers optimizer | onnxruntime"
[9]: https://opennmt.net/CTranslate2/quantization.html?utm_source=chatgpt.com "Quantization — CTranslate2 4.8.0 documentation"

---

## Codex validation addendum — 2026-06-14

### Current optimization state — 2026-06-14 late update

The working fast path is now:

```text
fp16io encoder + fp16 decoder + WebGPU EP + experimentalGpuKvCache
```

The important measured win is the GPU-resident KV bridge, not ONNX graph
surgery. Decoder KV tensors stay as ORT `gpu-buffer` tensors between
`decoder_init` and repeated `decoder_step` calls, while logits intentionally
remain CPU outputs so the existing Whisper logit processor keeps exact
suppression and no-timestamp semantics.

Latest known 29.9s Chrome WebGPU run (`fp16io-fp16-webgpu`, 50-token cap,
`experimentalGpuKvCache=true`, `experimentalWebGpuEncoderGraphCapture=false`):

| Metric | Value |
| --- | ---: |
| Preprocess | `335.83ms` |
| Encode | `1980.085ms` |
| Decode | `771.56ms` |
| Decoder init run | `72.08ms` |
| Decoder step run | `685.14ms` |
| Step p50 / p95 / max | `11.55ms` / `30.615ms` / `53.42ms` |
| Logit processing | `2.025ms` |
| Output handling | `1.525ms` |
| GPU tensor downloads | `0` |
| Total | `3095.505ms` |
| RTFx | `9.6606x` |

This changes the next-optimization priority:

1. Do not assume logit download is still a 75-100ms bottleneck. The current
   per-output placement already reports `0` GPU downloads and only a few
   milliseconds in visible JS logit handling.
2. GPU-side ArgMax is still worth an isolated A/B, but only as a masked
   no-timestamps greedy experiment in an alternate model artifact. It is not a
   safe direct patch to the current production model.
3. Encoder work is now competitive with decode time and should be measured with
   encoder graph capture and static-shape export experiments before changing
   decoder graph semantics.
4. Beam search remains supported, but the measured fast path is greedy-only.
   Batched beam decode is the next quality-mode optimization, not part of the
   `11x` claim.

### ArgMax experiment guardrail

A plain ONNX `ArgMax(logits)` output is semantically unsafe for Whisper. The
runtime currently mutates logits before token selection:

- `suppress_tokens` are suppressed every step.
- `begin_suppress_tokens` are suppressed for the first generated token.
- `no_timestamps` suppresses every timestamp token.
- Timestamped mode has dynamic state rules based on previously emitted tokens.

Therefore an ArgMax graph experiment must satisfy all of these conditions:

- Publish or serve it as an alternate artifact, not by overwriting
  `ysdede/whisper-large-v3-turbo-onnx-4graph`.
- Limit the first version to greedy decode with `noTimestamps=true`,
  `temperature=0`, `numBeams=1`, `bestOf=1`, and no `onTokenLogits` callback.
- Add a static mask for `suppress_tokens` plus timestamp-token suppression.
- Keep `decoder_init` token selection on the existing CPU logit path unless a
  separate masked init graph is validated.
- Request only `next_token_id` plus `present.*` outputs through ORT `fetches`
  for the speed A/B; otherwise the graph may still materialize CPU logits.
- Reject the experiment on any token mismatch, transcript mismatch, session
  creation failure, or WebGPU memory growth.

Expected upside is unknown until measured. The latest counters show only about
`3.5ms` total in visible output/logit JS work, so the only plausible larger win
would be hidden inside `decoder_step` `session.run`.

### Safety baseline

Before starting optimization experiments, the known-good fp16 WebGPU state was
preserved as:

- Git commit: `d6bfbd341267e12808bd06bdf51bcf08abdc2fad`
- Git tag: `backup/whisper-fp16-webgpu-working-2026-06-14`
- Branch for experiments: `perf/whisper-webgpu-decode`
- Current production HF model repo to keep unchanged:
  `ysdede/whisper-large-v3-turbo-onnx-4graph`

Do not overwrite the current 4-graph HF repo for performance experiments. If an
exported graph changes, publish it to a separate repo, for example:

```text
ysdede/whisper-large-v3-turbo-onnx-4graph-webgpu-opt
```

### Research cross-check

The main direction in this report is confirmed by current ORT Web docs:

- ORT WebGPU docs recommend importing `onnxruntime-web/webgpu` and explicitly
  selecting the `webgpu` EP.
- ORT WebGPU docs state that default tensors use CPU memory and are copied to
  GPU for WebGPU runs, then copied back to CPU for outputs.
- ORT WebGPU docs specifically call GPU-resident tensors useful for transformer
  loops where previous output becomes next input.
- ORT WebGPU supports `preferredOutputLocation: 'gpu-buffer'` and preallocated
  GPU output tensors.
- ORT Web performance docs expose `ort.env.webgpu.profiling` and
  `enableProfiling`.

One correction: the current ORT Web API page for `WebGpuExecutionProviderOption`
lists `device`, `forceCpuNodeNames`, `preferredLayout`, and `validationMode`.
It does not list `powerPreference`. Treat any `powerPreference` setting as
unverified for ORT Web until measured in the actual bundle.

Faster-whisper/CTranslate2 confirms the broader playbook: fp16/int8 execution,
batching, and reduced runtime overhead. But CTranslate2 gains do not map 1:1 to
ORT WebGPU; browser WebGPU operator coverage and tensor lifetime rules are the
limiting factor.

whisper.cpp confirms similar themes from a different runtime: mixed precision,
integer quantization, GPU backends, and zero runtime allocations. For our WebGPU
path, "zero allocation" translates most directly into static KV cache or
preallocated GPU tensors.

### Applicability to current code

Current hot path in `src/models/whisper-seq2seq/executor.ts`:

- `transcribeWithSplitGraph()` passes `encoderHiddenStates.data` into the
  decode loop even though `runDecoderInit()` later uses the original ORT tensor.
- `runDecoderInit()` reads logits and KV through `.data`.
- `runStep()` reconstructs ORT tensors from raw KV `.data` every token.
- `runDecoderStepSplit()` clones/wraps every KV tensor, runs ORT, reads logits,
  and returns `presentKv` tensors whose `.data` is then extracted again.

Latest measured 29.9s Chrome WebGPU fp16 run:

| Metric | Time |
| ------ | ---- |
| Encode | `1719ms` |
| Decode | `3925ms` |
| Decoder step ORT run | `3737ms` |
| Step feed build | `0.76ms` |
| Step tensor clone/wrap | `1.13ms` |
| Step output handling | `1.56ms` |
| Step p50 / p95 / max | `76.18ms` / `80.15ms` / `89.58ms` |

Important nuance: the current JS timing counters show only a few milliseconds in
explicit JS feed/output handling. GPU-resident tensors may still help, but the
first proof must come from ORT/WebGPU profiling and a measured A/B, not from
assuming that typed-array glue is the dominant cost.

### First implemented experiment

`initWhisperOrt()` now imports `onnxruntime-web/webgpu` when the resolved ORT
backend is WebGPU, and keeps `onnxruntime-web` for WASM. This follows ORT's
documented WebGPU entrypoint and avoids accidentally bundling a WASM-only entry.

The local `webgpu-agent-test` app aliases ORT to a custom WebGPU build, so its
Vite alias must map both:

```js
find: /^onnxruntime-web(\/webgpu)?$/
```

to the same local `ort.webgpu.min.mjs` file.

Verification after this import change:

- Demo build: passed.
- Chrome WebGPU smoke: `fp16io-fp16-webgpu` completed with the same 50 token IDs
  and transcript prefix.
- Timing was within normal variance, not an improvement:
  `decodeMs=4034.72ms`, `decoderStepRunMs=3833.98ms`,
  `decoderStepP50Ms=78.23ms`, `decoderStepP95Ms=82.94ms`, `rtfx=4.961`.

Conclusion: explicit WebGPU import is a packaging/correctness prerequisite, not
the speedup.

`enableProfiling` on Whisper artifact sources now also enables
`ort.env.webgpu.profiling = { mode: 'default' }` when the resolved ORT backend is
WebGPU. This is opt-in and should be used in the demo/source request for the
next profiling run.

Profiled smoke (`?profiling=1`) also completed with the same 50 token IDs and
transcript prefix. Profiling adds overhead and should not be used for speed
comparisons:

| Mode | Decode | Step ORT run | Step p50 / p95 | RTFx |
| ---- | ------ | ------------ | -------------- | ---- |
| Profiling off | `4034.72ms` | `3833.98ms` | `78.23ms` / `82.94ms` | `4.961` |
| Profiling on | `4405.08ms` | `4214.56ms` | `85.64ms` / `91.10ms` | `4.662` |

### Next experiment order

1. Enable optional ORT WebGPU profiling in the demo and inspect console output
   with the existing `_results/*.json` timings.
2. Add a WebGPU-only `preferredOutputLocation` experiment for decoder-step KV
   outputs, but keep logits on CPU until logit processing moves to GPU.
3. If GPU KV tensors can be fed back into `decoder_step` without `.data`, add
   tensor-location metrics and parity-gate it behind an opt-in flag.
4. Only after that, export an alternative static-cache graph to a new HF repo.
5. Beam batching is separate: do not mix it with greedy GPU-resident KV work.

### Stop conditions

- Any token mismatch on the fixed 29.9s fixture stops the experiment.
- Any Vite import/bundle regression stops the experiment.
- Any WebGPU memory leak or unreleased GPU tensor growth stops the experiment.
- If `decoderStepRunMs` does not improve, do not keep complexity just because it
  looks closer to CTranslate2.

### Experiment 2: opt-in GPU KV bridge

Implemented an opt-in splitgraph source flag:

```ts
experimentalGpuKvCache: true
```

When this is enabled and the resolved decoder backend is WebGPU, decoder init
and decoder step sessions use:

```ts
preferredOutputLocation: 'gpu-buffer'
```

The greedy decode path keeps `present.*` / `past_key_values.*` KV tensors as ORT
GPU-buffer tensors and feeds them directly into the next `decoder_step` call.
Logits are still processed on CPU for now. GPU KV tensors are disposed as they
are replaced. This path is intentionally limited to greedy argmax decoding. Beam
search, `best_of`, and temperature sampling still use the stable CPU-KV bridge.

New transcript metrics expose whether the path actually used GPU tensors:

- `decoderGpuTensorInputs`
- `decoderCpuTensorInputs`
- `decoderGpuTensorOutputs`
- `decoderCpuTensorOutputs`
- `decoderGpuTensorDownloads`
- `decoderKvCacheLocation`

Chrome WebGPU smoke on the 29.9s fixture, `fp16io-fp16-webgpu`,
`maxNewTokens=50`:

| Mode | Decode | Step run | Step p50 / p95 | RTFx | Tensor path |
| ---- | ------ | -------- | -------------- | ---- | ----------- |
| CPU KV bridge | `4365.735ms` | `4169.225ms` | `84.395ms` / `93.865ms` | `4.6793` | `cpu` |
| GPU KV bridge | `608.53ms` | `389.155ms` | `10.555ms` / `15.285ms` | `10.7035` | `gpu-buffer` |

Parity result:

- Same 50 token IDs.
- Same transcript prefix.
- CPU-KV run: `0` GPU inputs, `835` CPU inputs, `458` CPU outputs.
- GPU-KV run: `784` GPU inputs, `51` CPU inputs, `458` GPU outputs, `50`
  GPU downloads.

Conclusion: this is a real speedup and should be kept behind the experimental
flag while we gather more fixtures. The next clear optimization is not another
KV bridge change; it is to avoid downloading full logits once per token by
moving logit processing + argmax/top-k to GPU.

### Experiment 3: per-output placement for CPU logits + GPU KV

The first GPU-KV implementation used `preferredOutputLocation: 'gpu-buffer'`
for all decoder outputs, which kept KV on GPU but also made logits GPU tensors.
Because JS still needs CPU logits for timestamp suppression and argmax, that
created one explicit `getData(true)` download per token.

ORT Web supports per-output location maps, so the decoder sessions now request:

```ts
{
  logits: 'cpu',
  'present.0.decoder.key': 'gpu-buffer',
  'present.0.decoder.value': 'gpu-buffer',
  // ...
}
```

`decoder_init` maps both decoder and encoder KV outputs to `gpu-buffer`;
`decoder_step` maps decoder KV outputs to `gpu-buffer`. Logits remain CPU
outputs until logit processing moves to GPU.

Chrome WebGPU smoke on the same 29.9s fixture, `fp16io-fp16-webgpu`,
`maxNewTokens=50`:

| Mode | Decode | Step run | Step output | Step p50 / p95 | RTFx | Downloads |
| ---- | ------ | -------- | ----------- | -------------- | ---- | --------- |
| CPU KV bridge | `3979.045ms` | `3783.915ms` | `2.19ms` | `77.02ms` / `83.315ms` | `4.8285` | `0` |
| GPU KV, CPU logits | `575.785ms` | `499.365ms` | `1.75ms` | `9.66ms` / `12.51ms` | `11.3872` | `0` |

Parity result:

- Same 50 token IDs.
- GPU-KV run: `784` GPU inputs, `51` CPU inputs, `408` GPU outputs, `50`
  CPU outputs, `0` GPU downloads.

Conclusion: per-output placement is safer than all-output GPU placement for the
current CPU logit-processing design. It removes the explicit logits download
without changing suppression, timestamps, or argmax semantics.

### Follow-up audit: beam state, encoder work, and browser testing

The measured `11x` WebGPU path is currently the greedy-only
`experimentalGpuKvCache` path. Beam search is still implemented and exposed by
the stable splitgraph decoder path, but it is not yet accelerated by the
GPU-resident KV bridge. Keep this distinction explicit in demos and reports:

| Decode mode | Current path | Status |
| --- | --- | --- |
| Greedy, `temperature=0`, `numBeams=1`, `bestOf=1` | WebGPU GPU-KV bridge | Fast path, measured around `11x` RTFx on the 29.9s fixture |
| Beam search | CPU/WASM-style KV bridge | Supported, not part of the measured `11x` path |
| `best_of` / temperature sampling | CPU/WASM-style KV bridge | Supported, not part of the measured `11x` path |

**Update 2026-08-30: GPU-KV beam search is now available behind
`experimentalGpuKvBeam`.** Setting both `experimentalGpuKvCache` and
`experimentalGpuKvBeam` (source options; harness `?gpuKv=1&gpuKvBeam=1`)
keeps the beam caches GPU-resident across beam steps and removes the
per-step full-KV CPU snapshot round-trip. Measured on jfk-30s (29.9 s,
fp16io encoder + fp16 decoder, ORT Web 1.29, Blackwell): beam-2 decode
drops from 5418 ms to a 1585-1938 ms band (~17.7-18.9x RTFx vs 5.52x) with
a byte-identical transcript vs the CPU-KV stable beam control measured
back-to-back. Wider beams are validated too after removing a stale
`numBeams <= 2` guard (earlier "beam-5 validated" runs had actually been
rejected by that guard; the evidence JSONs recorded numBeams=2): with the
3x-numBeams prune lag, genuine beam-3 measures 2234 ms (~13.4x RTFx) and
genuine beam-5 3090 ms (~9.7x) on jfk-30s, both 50 tokens byte-identical
to the beam-2 GPU-KV transcript. Beam-2 + word timestamps also passes.
The prune lag must exceed numBeams x 3 generations because beam position
shifts between strategy steps delay a live beam's next feed. `temperature > 0` and
`bestOf > 1` remain rejected; `experimentalBatchedBeam` is ignored while
the flag is active. Details: docs/reports/whisper-beam2-gpu-kv-2026-08-30.md.

This mirrors the broader transformer-inference pattern: fast paths usually
require static or carefully managed cache state. Hugging Face Transformers
documents separate dynamic, static, and quantized KV cache strategies, where
static cache is the compile-friendly/high-memory option. CTranslate2 exposes
Whisper generation with `beam_size`, fp16/int8 compute types, worker queues, and
optional flash attention; faster-whisper's largest benchmark wins also come from
batching. In browser ORT WebGPU, the closest portable equivalents are:

- keep KV tensors on GPU between decoder calls,
- batch independent decode work when the graph supports it,
- use static shapes/graph capture only after a measured A/B,
- avoid full-vocabulary JS allocation in CPU fallback paths,
- export alternate model repos for static-cache or fused-attention experiments.

Encoder-side optimization is now the next likely source of gains for the greedy
path because decode has dropped below one second on the 29.9s fixture. The safe
candidate experiments are:

1. Try `enableGraphCapture` on the encoder only, gated by an explicit option,
   because ORT WebGPU says graph capture is model-dependent and session
   creation can fail.
2. Export an alternate encoder graph with fixed input shapes and fewer dynamic
   shape ops, then publish it to a separate HF repo rather than overwriting
   `ysdede/whisper-large-v3-turbo-onnx-4graph`.
3. Benchmark encoder output placement before changing it. The current fp16
   decoder bridge still needs dtype handling, so keeping encoder output on GPU
   is only useful if the encoder output dtype and decoder input contract line up
   without a CPU cast.
4. Keep mel optimization separate. The Whisper mel processor should be compared
   against the optimized mel processors in `speech-recognition` and
   `ysdede/parakeet.js`, but the current 30s WebGPU profile is dominated by
   encoder/decode rather than mel.

Immediate code cleanup landed for the beam fallback: beam candidate ranking no
longer materializes every vocabulary token as an object and full-sorts the whole
candidate list. It keeps only the best `beamWidth` candidates while scanning
logits, preserving deterministic tie order. This does not change the GPU-KV
fast path; it reduces JS allocation for the supported beam path while a proper
batched-beam graph remains future work.

Local Node micro-benchmark, using 5 active beams and a 51,865-token vocabulary:
old full-materialize/full-sort ranking averaged `182.9ms`, while the partial
selection implementation averaged `9.0ms` over 5 measured runs
(`~20.3x` faster for the ranking helper itself). This is a helper-level result,
not an end-to-end beam transcription result.

For Chrome testing, avoid shell commands that launch a fresh Chrome tab for
each smoke run. Reuse an existing localhost tab through the Chrome extension
automation session, or navigate the current controlled test tab. If a scripted
smoke runner is added later, it should claim an existing
`http://localhost:8765/` tab first and only create a new tab when no controlled
test tab exists.

### Experiment 4: opt-in encoder graph capture

ORT WebGPU documents graph capture as a potential win when shapes are static
and all kernels run on WebGPU, but also says some models fail session creation
and decoder-style dynamic shapes are a caution case. The next experiment is
therefore encoder-only:

```ts
experimentalWebGpuEncoderGraphCapture: true
```

When this source flag is enabled, the Whisper executor passes
`enableGraphCapture: true` only to the encoder WebGPU session. Decoder init,
decoder step, and alignment sessions do not receive graph capture in this pass.
The browser demo exposes the experiment with:

```text
?encoderGraphCapture=1
```

Recommended A/B URLs:

```text
http://localhost:8765/?auto=fp16io-fp16-webgpu&maxNewTokens=50&gpuKv=1
http://localhost:8765/?auto=fp16io-fp16-webgpu&maxNewTokens=50&gpuKv=1&encoderGraphCapture=1
```

Keep or reject criteria:

- Reject if session creation fails.
- Reject if token IDs differ from baseline.
- Reject if `encodeMs`, total time, or p95 step latency regresses materially on
  the same fixture.
- Keep only if the encoder timing improves enough to justify the extra option.

Status: source plumbing and unit coverage are implemented. The selected Chrome
profile now has the Codex Chrome Extension installed and enabled, and the native
host manifest checks pass, but the extension still is not accepting automation.
Per the Chrome workflow, the next step is to open a Chrome window for the
selected profile and retry only after user approval.

### Experiment 5: skip greedy score pass unless best-of needs it

The generic splitgraph greedy path used to compute cumulative log probability
for every token even when `bestOf=1`. That score is only needed when multiple
candidate decodes must be ranked. For ordinary greedy decoding, token selection
depends only on the processed logits and argmax/sampling, so the full-vocabulary
`logProbOfToken()` pass was wasted CPU work.

Implemented change:

```ts
whisperGreedyDecode(..., { trackScore: true })
```

is now used only by `bestOf`; default greedy decoding returns no `score`.

Local helper benchmark, 50 generated tokens and a 51,865-token vocabulary:

| Mode | Avg | p50 | Notes |
| --- | ---: | ---: | --- |
| Before | `49.18ms` | `53.90ms` | Full-vocab log-prob pass per token |
| After | `2.54ms` | `2.49ms` | Score pass skipped for `bestOf=1` |

This is a stable-path/helper optimization. It does not change the WebGPU
GPU-KV fast path, which already bypasses the generic core greedy loop. It does
reduce CPU work for WASM/CPU-KV greedy decode and keeps `bestOf` scoring intact.

### Experiment 6: GPU ArgMax on decoder_step (2026-06-14)

Appended `ArgMax(axis=-1, keepdims=0)` + `Cast(INT32)` nodes after the logits
output of `decoder_step.onnx`. The JS executor reads `outputs.next_token_id`
(INT32 scalar, 4 bytes) instead of computing `argmax(logits)` over 51k floats.

**Tooling:** `tools/whisper-onnx-export/append_argmax_to_decoder.py` (reusable
surgery script). Python ORT parity verified via `verify_argmax_parity.py` —
`next_token_id` matches `np.argmax(logits)` bit-exact.

**Result: counterproductive standalone.** The ArgMax model was ~20% slower than
the original (819ms vs 656ms decode). Root cause: the timestamp logit processor
still requires downloading the full 207KB logits every step. GPU ArgMax dispatch
overhead exceeds JS `argmax()` savings. The GPU ArgMax only becomes a net win
when **logit processing (timestamp suppression, EOS suppression) also moves to
GPU**, eliminating the 207KB/token download entirely.

**Critical pitfall discovered:** New ONNX outputs added via graph surgery are NOT
automatically mapped to CPU in ORT's `preferredOutputLocation` per-output map.
They default to GPU-buffer, making `.data` unreadable in JS. Without explicitly
adding `next_token_id: 'cpu'` to `createWhisperGpuKvOutputLocation()`, the
ArgMax kernel still dispatches on GPU while JS silently falls back to
`argmax(logits)` — a net loss of ~35%.

**Verdict:** Infrastructure committed (executor.ts `nextTokenId` plumbing,
surgery scripts). Skip standalone deployment. Bundle with GPU logit processing
as a single combined change.

### Experiment 7: GPU encoder→decoder Cast bridge (2026-06-14) ✅ DEPLOYED

Eliminated the CPU f32→f16 cast round-trip for encoder hidden states by
injecting a native ONNX `Cast` node into `decoder_init.onnx`.

**ONNX surgery** (`tools/whisper-onnx-export/inject_decoder_init_cast.py`):
- Changed `encoder_hidden_states` input type from `FLOAT16` → `FLOAT` (fp32)
- Inserted `Cast(f32→f16)` node at graph entry position 0
- Redirected 8 downstream cross-attention MatMul references
- Preserved original external data (no weight re-packing — see pitfall below)
- Parity verified via `tools/whisper-onnx-export/verify_cast_parity.py`

**JS changes** (`executor.ts`):
- Encoder session gets `preferredOutputLocation: 'gpu-buffer'` when
  `experimentalGpuKvCache` + WebGPU (encoder output stays on GPU)
- `maybeCastEncoderHiddenStates()` now `async`: returns immediately (no-op)
  when decoder_init accepts fp32 (Cast model). For original fp16 model,
  downloads GPU tensor via `getData()` and does CPU cast.
- Two call sites updated with `await`.

**Benchmark** (RTX 5060 Ti, 29.9s JFK, fp16io-fp16, greedy, maxNewTokens=50,
3 runs, warm-up discarded):

| Metric | CPU Cast (before) | GPU Bridge (after) | Delta |
|---|---|---|---|
| Encode | ~1900ms | **336ms avg** | **5.7× faster** |
| Decode | 660-820ms | 773ms avg | within range |
| Step P50 | 11-14ms | 10.7ms | same |
| RTFx | ~10× | **21.5×** | 2.15× overall |
| Token parity | ✅ | ✅ | identical transcript |

The encode speedup is from **eliminating the GPU pipeline stall**:
`encoderSession.run()` with CPU output blocks on the 7.68MB GPU→CPU download
(flushes the WebGPU command queue). With `preferredOutputLocation: 'gpu-buffer'`,
`run()` returns as soon as commands are dispatched — JS no longer waits for the
readback. Actual GPU compute is unchanged; measured wall time drops because the
stall is removed.

**Feature parity preserved:** Timestamps, logits, beam search, temperature
sampling, and token suppression all unchanged.

**ONNX external data pitfall (CRITICAL):** When saving a modified ONNX graph that
uses external data, do NOT use `onnx.save(save_as_external_data=True)` — it
re-packs all weights and can produce different file sizes (corruption risk).
Instead: (1) load with `load_external_data=False`, (2) modify graph, (3) save
with plain `onnx.save(model, path)`, (4) copy original `.data` file alongside.
The internal `external_data.location` references stay intact. The deployed
`.onnx` filename must match the internal location reference.

### Final optimization summary (perf/whisper-webgpu-decode branch)

| # | Experiment | Impact | Status |
|---|---|---|---|
| 1 | GPU KV cache bridge | Decode: 4.8× → 11× RTFx | ✅ deployed |
| 2 | Beam candidate ranking | Helper: 20× faster (182→9ms) | ✅ deployed |
| 3 | Skip greedy score pass | CPU work: 49→2.5ms per run | ✅ deployed |
| 4 | Encoder graph capture | Session creation fails | ❌ blocked (Reshape/Shape ops) |
| 5 | GPU ArgMax | Counterproductive standalone | ⚠️ infra committed, skip solo |
| **6** | **GPU encoder→decoder Cast** | **Encode: 5.7×, RTFx: 10→21.5×** | **✅ deployed** |

**Cumulative improvement from baseline (fp16io-fp16-webgpu, greedy, 29.9s JFK, RTX 5060 Ti):**
- Preprocess: ~240ms → ~83ms (2.9× faster, fast mel N_FFT=512)
- Encode: ~1900ms → ~336ms (5.7× faster, GPU Cast bridge)
- Decode: ~4000ms → ~773ms (5.2× faster, GPU KV bridge)
- **Total: ~6140ms → ~1192ms (5.1× faster, combined)**
- **RTFx: ~4.8× → 25.3× (5.3× throughput improvement)**

### Experiment 8: fast mel — power-of-2 FFT replacing Bluestein (2026-06-14) ⚠️ OPT-IN ONLY

Replaced the expensive Bluestein (chirp Z-transform) algorithm required by the
non-power-of-2 N_FFT=400 with zero-padded 512-point standard radix-2 FFT.
The 400-point Hann window is centered in a 512-point buffer (56 zeros each side).
Mel filterbank adapts automatically (defined in Hz, not bin indices).
Frame alignment preserved (reflect pad=200, same as original).

**Chrome benchmark:** preprocessMs: 237ms → **83ms avg** (2.85× faster).
Transcript: identical (token parity confirmed).
Gate: `fastFft: true` is an explicit experiment. The shipped default is the
exact 400-point Bluestein path (`fastFft: false`) because 512-point zero
padding changes Whisper's frequency-bin grid and model-input contract.

### Experiment 9: shared WebGPU device (2026-06-14)

Set `ort.env.webgpu.device` to a single GPUDevice created at ORT init time
via `navigator.gpu.requestAdapter()/requestDevice()`. All encoder/decoder
sessions share this device, avoiding per-session device creation.

**Result:** initRun regression (197ms) persists — the overhead is from the Cast
node or first-run GPU tensor handoff, not cross-device copies. Further
investigation needed.

### Final optimization summary (perf/whisper-webgpu-decode branch)

| # | Experiment | Impact | Status |
|---|---|---|---|
| 1 | GPU KV cache bridge | Decode: 4.8× → 11× RTFx | ✅ deployed |
| 2 | Beam candidate ranking | Helper: 20× faster | ✅ deployed |
| 3 | Skip greedy score pass | CPU work: 49→2.5ms | ✅ deployed |
| 4 | Encoder graph capture | Session creation fails | ❌ blocked |
| 5 | GPU ArgMax | Counterproductive standalone | ⚠️ infra committed |
| 6 | GPU encoder→decoder Cast | Encode: 5.7×, RTFx: 10→21.5× | ✅ deployed |
| 7 | Fast mel N_FFT=512 | Preprocess: 2.85× in the historical A/B | ⚠️ opt-in only |
| 8 | Shared WebGPU device | Init regression persists, step regression | ❌ rejected, code removed |
| 9 | Stripped fp16 encoder | Encode: 6.4× (1900→296ms), no Cast nodes | ✅ deployed |
| 10 | Fused encoder_decoder_init | Slower than separate, +VRAM | ❌ rejected (perf/fused-encoder-decoder-init) |
| 11 | GPU ArgMax Phase 2 (suppression mask + ArgMax) | Token parity ✅, step regression on long audio | ❌ rejected as default; infra kept (perf/gpu-argmax) |

**Note:** Experiment 9 supersedes Experiment 6 (GPU encoder→decoder Cast). The stripped
fp16-output encoder + original fp16-input decoder_init is the cleaner architecture —
zero dtype casts anywhere in the pipeline, no ONNX modifications to decoder_init.

**Cumulative improvement from baseline (fp16io-fp16-webgpu, greedy, 29.9s JFK, RTX 5060 Ti, warm):**
- Preprocess: ~240ms → ~81ms (3.0× faster, fast mel N_FFT=512)
- Encode: ~1900ms → ~277ms (6.9× faster, stripped fp16 encoder)
- Decode: ~4000ms → ~698ms (5.7× faster, GPU KV bridge)
- **Total: ~6140ms → ~1056ms (5.8× faster, combined)**
- **RTFx: ~4.8× → 27.6× (5.8× throughput improvement)**
- **Step P50: 80ms → 9.5ms (8.4× faster per token)**

**Remaining high-impact opportunities:**
1. ~GPU logit processing + ArgMax combined~~ — **TESTED, REJECTED** (perf/gpu-argmax). Token parity perfect but step regression on long audio (+11% Turkish). GPU-side Add(mask)+ArgMax overhead exceeds CPU-side JS argmax on already-downloaded fp16 logits. Future: needs custom WGSL shader.
2. Batched beam graph (beam_size=5 → 5× fewer ORT calls)
3. Static KV cache + graph capture (requires new ONNX export)
4. Resolve decoder init regression (195ms → 69ms, cross-session GPU handoff)
5. Reduce VRAM: gpu-argmax adds ~620MB (2 extra decoder_step sessions). Even without it, baseline is ~1.85GB + runtime = ~2.4GB peak — investigate whether encoder/decoder sessions can share weight buffers.

---

## Lessons Learned — Pitfalls & Discoveries

### ONNX graph surgery

1. **`onnx.save(save_as_external_data=True)` corrupts weights.** When modifying an ONNX
   graph that uses external data, load with `load_external_data=False`, modify the
   graph structure, save with plain `onnx.save(model, path)`, and copy the original
   `.data` file alongside. Re-packing via `save_as_external_data=True` produces
   different file sizes (54MB discrepancy observed) and breaks ORT deserialization.

2. **Deployed `.onnx` filename must match internal `external_data.location`.**
   The ONNX graph stores a relative path to its weight file. If you rename the
   `.onnx` file, ORT looks for the old `.data` filename and fails with "Failed to
   load external data." Either save with the final filename or patch the internal
   reference.

3. **New ONNX outputs default to GPU-buffer in per-output maps.** When using
   `preferredOutputLocation` as a per-output record, any output NOT explicitly
   listed defaults to `'gpu-buffer'`, NOT `'cpu'`. If graph surgery adds a new
   output that JS needs to read (e.g., `next_token_id` from ArgMax), explicitly
   add it as `'cpu'` in the location map. Otherwise `.data` is unreadable and
   JS silently falls back while still paying GPU dispatch cost — a ~35% overhead.

4. **Symbolic dimensions must be preserved with `dim_param`, not `dim_value=0`.**
   When creating new graph inputs/outputs via `onnx.helper.make_tensor_value_info()`,
   use `dim_param="batch"` for symbolic dimensions. Setting `dim_value=0` makes it a
   literal zero-dimension that ORT rejects at runtime.

### WebGPU execution

5. **`preferredOutputLocation: 'gpu-buffer'` eliminates GPU pipeline stalls.**
   Without it, `session.run()` blocks on the output download (GPU→CPU DMA). With it,
   `run()` returns as soon as commands are dispatched. The measured encode wall time
   dropped from ~1900ms to ~277ms — not because the GPU is faster, but because JS
   no longer waits for the 7.68MB readback. Actual GPU compute time is unchanged.

6. **Cross-session GPU tensor handoff adds ~125ms overhead.** When encoder output
   is on GPU (`preferredOutputLocation: 'gpu-buffer'`) and passed to decoder_init,
   init time increases from ~69ms to ~195ms. This is a one-time per-transcription
   cost from ORT setting up buffer sharing between separate sessions. Not from
   dtype casting or shader compilation.

7. **Shared WebGPU device needs `shader-f16` feature.** Creating a GPUDevice via
   `navigator.gpu.requestAdapter().requestDevice()` without `requiredFeatures:
   ['shader-f16']` causes ORT's fp16 Cast/Mul/MatMul kernels to fail at runtime
   with "requires f16 but the device does not support it." ORT's default per-session
   device creation includes fp16 automatically. If pre-creating a shared device,
   always request `shader-f16`.

8. **Encoder graph capture fails on current export.** The encoder ONNX graph
   contains `Reshape`/`Shape` ops that have **no GPU kernel** per the WebGPU
   operator table. ORT rejects session creation with `ERROR_CODE: 1`. Graph
   capture requires a static-shape export eliminating those ops.

### Optimization methodology

9. **GPU ArgMax alone is counterproductive (+20% overhead).** The timestamp logit
   processor still requires downloading the full 207KB logits every step. GPU
   ArgMax dispatch overhead exceeds JS `argmax()` savings. Only becomes net-positive
   when combined with GPU-side logit processing (suppression masks + argmax on GPU).

10. **Stripped fp16 encoder > Cast-injected decoder_init.** The fp16_iofp32 encoder
    already computes internally in fp16. It has a final `Cast(f16→f32)` from
    `keep_io_types=True`. Removing this Cast and exposing fp16 output directly,
    paired with the original fp16-input decoder_init, produces a cleaner pipeline
    with zero dtype casts anywhere. Simpler graph surgery (remove 1 node vs add 1
    node + redirect 8 refs) and no cross-session Cast synchronization.

11. **Always warm up before benchmarking.** First run after model load includes
    ORT session creation, shader compilation, and GPU buffer allocation. Second run
    in the same tab shows true inference performance. Cold runs include 2-3×
    overhead from model loading.

12. **Reuse browser tabs; don't launch new Chrome windows.** Each new Chrome tab
    creates fresh ORT sessions and allocates new VRAM. Use `browser_navigate` on an
    existing `localhost:8765` tab. Kill previous tabs to free VRAM before
    benchmarking.

### Vite dev server

13. **Vite `.vite` cache corruption causes 504 errors.** After multiple kill/restart
    cycles, Vite's pre-bundled dependency cache gets out of sync. Symptom:
    `GET /node_modules/.vite/deps/pako.js 504 (Outdated Optimize Dep)`. Fix: kill
    all node processes, `rm -rf node_modules/.vite`, restart with `--force`.

14. **Vite on Windows needs PTY.** Background Vite processes exit with "stdin is
    not a tty" unless launched with `pty=true`. Use `npx vite --host 0.0.0.0 --port
    8765 --strictPort --force` with `background=true, pty=true`.

## Compatibility resume - 2026-08-23

Optimization was intentionally paused to repair reference semantics around the
working ~25x greedy WebGPU path. The custom splitgraph target is still
`ysdede/whisper-large-v3-turbo-onnx-4graph`; merged `onnx-community` decoders are
secondary compatibility targets.

### Root causes corrected

1. The beam loop allowed completed EOS hypotheses to consume active slots.
2. `patience` counted consecutive completed steps instead of setting Whisper's
   `round(beamSize * patience)` finished-candidate budget.
3. Survivor KV caches were matched by token prefix instead of explicit parent
   indexes.
4. Final beam ranking used the old `length^alpha` helper and raw-score default
   instead of Whisper's default length normalization / Google NMT formula.
5. Timestamp processing omitted `<|notimestamps|>` suppression and the
   aggregate timestamp-probability rule.
6. The reference harness treated manifest `max_source_positions=1500` as 3000
   mel input frames. The actual graph contract is input `[B, 128, 3000]` and
   output `[B, 1500, 1280]`.
7. The reference generator omitted `<|notimestamps|>` from its manual prompt,
   did not persist the exported mel path, and assumed encoder/decoder graphs
   shared one directory.

### Verification

- Full Vitest: 112 files passed, 1 skipped; 662 tests passed, 4 skipped.
- TypeScript typecheck, build, and Python `py_compile` pass; lint reports zero
  errors and six existing warnings.
- A cached HF `openai/whisper-large-v3-turbo` oracle on `jfk2.en.wav` matches
  local fp32 splitgraph decode exactly:
  - Python mel input: 31/31 normalized tokens, exact text.
  - TypeScript WAV/mel input: 31/31 normalized tokens, exact text.
- Python ORT 1.22 rejects these IR-v13 graphs, so the generator now supports
  `--skip-onnx` and the Vitest gate executes them with Node ORT 1.26.
- Node ORT cannot materialize this artifact's fp16 encoder output on the host
  (`expected 3840000, got 0`); Chrome WebGPU remains the fp16 execution gate.

### Architecture decision

Greedy GPU-KV remains untouched and is still the speed path. Stable CPU-KV beam
is the correctness oracle. Experimental batched beam shares the same corrected
candidate lifecycle and may be promoted only after stable/batched token parity
across beam sizes, EOS/timestamp cases, English and Turkish fixtures, followed
by wall-time measurement.

The next compatibility issue is metric provenance: OpenAI measures no-speech
from raw decoder-init logits at the SOT position before suppression. The current
generic quality gate uses a hard-coded token and processed next-token logits;
selected-beam logprob/entropy collection also needs a memory-conscious design.

### Browser revalidation

An independent headless Chromium run on the NVIDIA WebGPU adapter confirmed
exact 50-token parity across greedy GPU-KV, stable beam, and batched beam. The
paired beam measurements were:

| Mode | Total | Decode | RTFx | Step ORT calls |
| ---- | ----: | -----: | ----: | -------------: |
| Stable CPU-KV beam | `14126.025ms` | `12577.285ms` | `2.1192` | `98` |
| Batched CPU-KV beam | `11841.205ms` | `10609.685ms` | `2.5276` | `49` |

Batched beam therefore halved ORT calls and improved paired decode time by
15.64% with no token change. Greedy GPU-KV completed at `3291.080ms` total,
reported `9.1304` RTFx, and kept GPU tensor downloads at zero. These absolute
numbers are a new-run observation, not a replacement for the faster historical
warm baseline; use paired measurements for optimization claims.

## Healthy GPU rerun and quality provenance - 2026-08-23

The workstation restart restored the healthy NVIDIA WebGPU adapter. The active
browser target is the custom `ysdede/whisper-large-v3-turbo-onnx-4graph` repo,
not the merged `onnx-community/*` decoder family. Its remote preset uses the
`fp16_iofp32/encoder_model.onnx` artifact; the local harness maps that to the
optimized fp16-output copy `fp16_iofp32_fp16out`, paired with the `fp16` decoder.
A warmed headless Chrome run on
the 10-second JFK fixture measured `863.540ms` total and `11.7391x` RTFx. The
30-second fixture measured `1328.070ms` total and `22.7617x` RTFx; a profiling
run with explicit encoder queue drain measured `1357.655ms` and `22.2738x`.
Both 30-second runs had 49 decoder steps, zero GPU tensor downloads, and the
GPU-KV cache remained on `gpu-buffer`. The drain is a metric-attribution flag,
not a production setting. An independent manual repeat on the optimized local
variant reached `1175.81ms` total and `25.6993x` RTFx on the 29.9043-second JFK
clip, with `183.49ms` encoder time, `49` GPU-KV steps, p50/p95 step time of
`13.395/15.430ms`, and zero downloads. Its 10-second repeat reached
`13.856x`.

ONNX inspection confirmed genuine FP16 weights in the active files: 487
`FLOAT16` encoder initializers, 101 decoder-init initializers, and 88
decoder-step initializers. Decoder logits/KV interfaces are FP16 while the mel
input remains FP32.

The earlier `~8x` measurement was correctly classified as degraded GPU state.
The historical `25-28x` results are now corroborated by the independent
`25.6993x` repeat, while `22-23x` remains valid for the alternate warmed run.
Longer clips are the right throughput test because fixed preprocessing, encoder,
and decoder-init costs are amortized.

The next compatibility implementation also landed here: Whisper no-speech
probability now comes from a copied raw decoder-init logit vector before
suppression, with the no-speech token resolved from generation config or the
tokenizer. Generic quality-gate callers retain `50362` as a compatibility
fallback. Selected-beam logprob/entropy collection now uses scalar traces from
the winning sequence, and beam expansion avoids a full-vocabulary temporary
log-softmax array.

### Beam candidate selector and compatibility matrix (2026-08-24)

The selector computes log-sum-exp, entropy, and the bounded `beamSize + 1`
candidate list in one pass after normalization. It keeps the previous
Float32-ranked log-probabilities so candidate ordering remains compatible.

The deterministic regression command is:

```powershell
npm run benchmark:whisper-beam
```

On the custom FP16 splitgraph model in headless Chrome, stable versus batched
beam produced exact parity for English beam 5 (245 vs 49 step calls), English
timestamped beam 2 (40 vs 20), and Turkish auto beam 2 (158 vs 79). All beam
runs kept KV on CPU and reported zero GPU tensor downloads. These results are
compatibility evidence; the experimental batched path remains opt-in.

The runner and enhanced executor now use the same quality provenance: raw
decoder-init logits for no-speech plus selected-sequence scalar traces for
logprob/entropy fallback gates. VAD chunks preserve the `AudioBufferLike`
contract, Whisper-native timings/warnings survive merging, and overlapping
native words keep the higher-confidence copy. Real EN/TR runner fixture
validation remains a separate report-only task.

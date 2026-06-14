## Whisper Large V3 Turbo WebGPU optimization report for `asrjs/speech-recognition`

Your preflight agent is aiming at the right bottleneck class, but the implementation details need tightening. The biggest practical opportunity is **not “make it more like CTranslate2” in the abstract**; it is to stop the decoder loop from repeatedly moving KV cache and logits through JS-owned typed arrays.

Whisper Large V3 Turbo is already a good target for this because it keeps the large-v3 encoder but reduces the decoder from 32 layers to 4, which makes decode-loop overhead proportionally more visible. ([Hugging Face][1]) Faster-whisper/CTranslate2 gets its gains from a fast transformer runtime, lower precision/quantization, and reduced memory movement; its README claims up to 4× speedup over OpenAI Whisper and further gains from 8-bit quantization. ([GitHub][2])

### Executive conclusion

Your current code already has the right 4-graph architecture, KV cache, beam search, timing metrics, and splitgraph flow. But the hot path still behaves like a CPU-orchestrated decoder:

* `runDecoderStepSplit()` reconstructs input tensors from JS typed-array data every step and then reads `logitsTensor.data` plus `presentKv` back into typed arrays.
* `transcribeWithSplitGraph()` passes `encoderHiddenStates.data` into the splitgraph decode loop and converts `presentKv` tensors into raw `.data` objects across init/step boundaries.
* Beam search in `core.ts` runs one decoder step per active beam, clones every KV cache into `new Float32Array(...)`, builds `Array.from(logProbs)`, and full-sorts the whole vocabulary for top-k.

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

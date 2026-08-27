# Qwen3-ASR-0.6B official export

Use this for the official `qwen-asr` 0.0.6 chain. Do not start from
`goryodog/…` or `andrewleech/qwen3-asr-onnx`.

## Ladder

1. Official snapshot `Qwen/Qwen3-ASR-0.6B` @ `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`
2. Official `qwen-asr==0.0.6` CPU capture (jfk-short)
3. Static encoder ONNX (T % 100 == 0) from official `audio_tower` weights
4. Explicit stacked-KV prefill + step from official thinker weights
5. Native ORT greedy vs oracle text
6. WASM sequential sessions (encoder, then prefill, then step; never all three)
7. Native fp16 decoder (`--dtype float16` PyTorch `.half()`, not `convert_float_to_float16`). onnxruntime-web loads these graphs. Shared `decoder-fp16.onnx.data` ~1.503 GB.
8. Dynamic encoder ONNX (`audio-encoder-dynamic.onnx`) with T % 100 == 0 after JS pad-to-chunk. T=1050 pad/crop greedy matches JFK; embeddings are not bit-exact vs the official ragged last chunk.
9. Chrome WebGPU and Chrome sequential WASM via `webgpu-agent-test` `:8765`
10. Preset only after a product decision; family stays experimental. Dynamic encoder is exported but JFK harness still loads the static T=1100 graph.

## Failure classes seen

- `EXPORT_BLOCKED` — unmodified encoder `aten::pad_sequence`; unmodified decoder `DynamicCache` / `create_causal_mask`
- `WASM_MEMORY_LIMIT` — `std::bad_alloc` when encoder + 3 GB decoder are loaded together (fixed by sequential sessions)
- `ORT_WEB_UNSUPPORTED_OP` — `convert_float_to_float16` LayerNorm Cast fusion; native `.half()` export does load
- `WEBGPU_NO_ADAPTER` — Node/vitest has no WebGPU device
- Chrome NVIDIA Blackwell WebGPU: exact JFK
- Chrome sequential WASM fp16 and fp32: exact JFK

See `tools/model-debugging/reference/qwen3-asr-0.6b/README.md`.

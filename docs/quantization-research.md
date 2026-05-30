# Whisper ONNX — Quantization Research & Roadmap

**Author**: Flexo (2026-06-01)
**Branch**: `main` (`b7c2016`)
**Repo**: `asrjs/speech-recognition`

## Current State

large-v3-turbo (809M params) currently has 3 model variants on disk, plus a mixed variant:

| Variant | Encoder | Decoder Init | Decoder Step | Total | End-to-End Time | Accuracy |
|---------|---------|-------------|--------------|-------|----------------|----------|
| fp32    | 2.5 GB  | 0.5 GB      | 0.25 GB      | 3.4 GB | 9708ms (1.0x) | Reference |
| fp16    | 1.2 GB  | 0.25 GB     | 0.13 GB      | 1.7 GB | ? | Identical tokens |
| q8      | 616 MB  | 228 MB      | 415 MB       | 1.4 GB | 6911ms (1.40x) | Identical tokens |
| **Mixed** (q8 enc + fp32 dec) | 616 MB | 0.5 GB | 0.25 GB | **~1.4 GB** | **6644ms (1.46x fastest)** 🏆 | Identical tokens |

**Mixed precision is the fastest variant**: q8 encoder provides 24% faster encode, fp32 decoder provides 2x faster per-step decoding. Combined = 1.46x total speedup over pure fp32.

**Current toolchain**: `quantize_dynamic(src, dst, weight_type=QuantType.QInt8)` — ONNX Runtime dynamic quantization, weights only.

## Benchmark: Where does the time go?

Measured on P520 (WSL2, RTX 5060 Ti, onnxruntime-node CPU, 5 iterations):

| Phase | fp32 | q8 | Δ | Analysis |
|-------|------|----|---|----------|
| **Encoder** (full 30s window) | 6797ms | **5169ms** | **-24%** 🟢 | Q8 smaller → less memory bandwidth → faster |
| **Decoder Init** (1 prompt pass) | 151ms | **98ms** | **-35%** 🟢 | Same — smaller weights = faster |
| **Decoder Step** (per token, avg) | **2.6ms** | 5.4ms | **+108%** 🔴 | Q8 dequant overhead dominates small matmuls |

**Key insight**: q8 encoder is 24% faster, but q8 decoder step is 2x slower. For long transcriptions with many decoder steps, q8 total decode time may be WORSE than fp32.

### Root cause: On-the-fly dequantization cost

```
FP32 decoder step:  load(fp32 weights) → matmul → store(fp32 KV)
Q8 decoder step:    load(int8 weights) → dequant → matmul → store(fp32 KV)
                    ^^^^^^^^^^^^^^^^^
                    Extra step: int8→fp32 dequant for EVERY weight
```

For encoder (large matmuls), the dequant cost is amortized over millions of multiply-adds. For decoder step (tiny matmuls, 1 token × 1280 hidden), the dequant cost dominates.

## Quantization Techniques — Full Catalog

### 1. Dynamic Quantization (current q8 approach)

| Property | Value |
|----------|-------|
| **What** | Weights quantized to int8 post-training. Activations stay fp32. |
| **Tool** | `onnxruntime.quantization.quantize_dynamic` |
| **Types** | QInt8 (signed), QUInt8 (unsigned), QInt16, QUInt16 |
| **KV cache** | ❌ Not quantized (stays fp32) |
| **Pros** | Simple, no calibration data, works on any model |
| **Cons** | Decoder step is slower due to per-weight dequant overhead |
| **Best for** | Encoder (large matmuls) and memory-constrained deployments |

### 2. Static Quantization (next step)

| Property | Value |
|----------|-------|
| **What** | Weights AND activations quantized to int8. Requires calibration data. |
| **Tool** | `onnxruntime.quantization.quantize_static` |
| **KV cache** | ❌ Activations in KV cache still float32 if not explicitly targeted |
| **Pros** | Faster inference (no per-layer dequant), smaller memory |
| **Cons** | Needs calibration dataset (~100 samples), may have accuracy loss |
| **Best for** | Deployments where accuracy loss is acceptable and speed is critical |

### 3. Quantized KV Cache

| Property | Value |
|----------|-------|
| **What** | KV cache stored in int8 (or fp8) instead of fp32 |
| **Technique** | Per-token or per-head quantization of K and V tensors |
| **Tools** | Custom ORT graph modification or wrapper layer |
| **Pros** | 2-4x smaller KV cache, faster memory-bound decoder steps |
| **Cons** | ORT doesn't natively support quantized KV cache — needs custom work |
| **State of art** | KIVI, KVT (KV cache quantization for LLMs), not yet standard in ORT |

**Why it helps**: The decoder step reads ALL previous KV cache tokens for attention. With 1500-token context, that's 1500 × 1280 × 4 bytes = 7.7 MB per layer, read every single step. Quantizing to int8 cuts this to 3.8 MB, reducing memory bandwidth by 50%.

### 4. Weight-Only Quantization (GPTQ / AWQ style)

| Property | Value |
|----------|-------|
| **What** | Weights quantized to 4-bit with group-wise scaling |
| **Technique** | GPTQ, AWQ — requires calibration data |
| **ORT support** | ❌ Not natively supported in onnxruntime-quantization |
| **Pros** | 4x smaller weights (compared to fp32) |
| **Cons** | Needs custom CUDA kernels for fast dequant, ORT integration is complex |

### 5. fp8 (Float8 — E4M3 / E5M2)

| Property | Value |
|----------|-------|
| **What** | 8-bit floating point format (not int8) |
| **Hardware** | H100/H200 native, Blackwell supports |
| **ORT support** | Experimental in onnxruntime (fp8 matmul) |
| **Pros** | Better accuracy than int8 for same bit width, no dequant needed on H100 |
| **Cons** | Requires H100+ GPU. Not useful on consumer GPUs or CPU. |

### 6. Mixed Precision (hybrid — recommended)

| Component | Recommended | Reason |
|-----------|-------------|--------|
| **Encoder weights** | q8 | 24% faster encode, no accuracy loss |
| **Decoder weights** | fp16 or fp32 | q8 decoder step is 2x slower — not worth it |
| **Decoder Init weights** | fp16 | 35% faster init |
| **KV Cache** | fp32 (or quantized custom) | ORT doesn't support quantized KV natively |

**This is the optimal configuration for whisper-large-v3-turbo on current hardware.**

Implementation: Load encoder at q8, decoder at fp16 or fp32. This gives:
- Fast encoder (q8, 5.2s vs 6.8s)
- Fast decoder init (fp16, similar to q8 98ms)
- Fast decoder step (fp32, 2.6ms vs 5.4ms q8)

## KV Cache Quantization — Deep Dive

### Why KV cache matters for Whisper

Whisper's decoder uses cross-attention and self-attention. The KV cache stores:
- **Self-attention KV**: Previous generated token representations (grows with each step)
- **Cross-attention KV**: Encoder output projection (constant, 1500×1280 per layer)

For large-v3-turbo (20 decoder layers, 1280 hidden, 64 head dim):
```
Self-attention per layer = 2 × (seq_len × 1280) = grows to 2 × 29 × 1280 ≈ 74 KB
Cross-attention per layer = 2 × (1500 × 1280) = 3.84 MB (constant)
Total KV cache = 20 × (0.07 + 3.84) ≈ 78 MB
```

Quantizing to int8 = 39 MB. For streaming with 30s windows, this is modest.

### How to implement KV cache quantization

**Option A: Custom ORT wrapper**
- Wrap the decoder_step model with pre/post processing nodes
- Insert QuantizeLinear/DequantizeLinear around KV cache I/O
- Requires ONNX graph editing (Python + onnxscript or manual node insertion)

**Option B: Application-level**
- Convert KV cache tensors to int8 in JS before feeding to next step
- Apply per-token scaling factors
- Trade-off: 2x memory savings, but adds JS-side quantize/dequant cost

**Option C: Wait for ORT native support**
- ONNX Runtime SIG is working on native KV cache quantization
- Expected in onnxruntime 1.20+ (not yet released as of mid-2026)
- Would be transparent: pass `int8` KV tensors directly

## Newer Quantization Approaches

### Q4 (4-bit) quantization for Whisper

4-bit quantization would reduce model size by 4x (vs fp32) or 2x (vs q8):

| Variant | Encoder size | Est. quality | Feasibility |
|---------|-------------|--------------|-------------|
| q4 (GPTQ) | ~300 MB | Slight degradation | ❌ ORT doesn't support 4-bit natively. Needs GGUF/llama.cpp or custom kernels. |
| q4 (NF4) | ~300 MB | Better than q4 | ❌ Same — ORT limitation |
| q4_0 (GGML) | ~350 MB | Minimal degradation | ✅ Via llama.cpp GGUF format, but requires different inference engine |
| q4_K_M (GGML) | ~350 MB | Near fp16 | ✅ Via llama.cpp |

**Verdict**: q4 via ORT is not feasible without custom CUDA kernels. Switching to GGUF/llama.cpp would mean abandoning ORT entirely — a massive engineering effort.

### Int2, Int3, Int4 via GGUF/llama.cpp

If we moved to llama.cpp (ggml backend):
- Q2_K: ~2.2 bits per weight, ~220 MB encoder
- Q3_K: ~3.4 bits, ~340 MB
- Q4_K_M: ~4.5 bits, ~450 MB (whisper.cpp community reports "near-transparent" quality)
- Q5_K_M: ~5.5 bits, ~550 MB (transparent quality, 36% of fp32 size)
- Q6_K: ~6.6 bits, ~660 MB (very close to fp16)
- Q8_0: ~8.25 bits, ~825 MB (basically lossless)

**Known issue**: whisper.cpp (llama.cpp's whisper backend) doesn't yet support the kv-cache splitgraph (4-graph export). It uses a merged decoder. This would require separate development.

### Quantization-Aware Training (QAT)

- Train with simulated quantization to make the model robust to it
- Post-training quantize with minimal accuracy loss
- Requires full training pipeline (data, GPU time, expertise)
- **Not practical** for asrjs — we use pre-existing Whisper models

### fp16 with external data (existing, deployed)

Already done and working:
- Encoder: 1.2 GB (external data)
- Decoders: ~0.38 GB total
- Hardware requirement: fp16 tensor support (WebGPU ✅, ORT WASM ❌, ORT Node ❌)

## Recommended Roadmap

### Phase 1: Mixed Precision (high impact, low effort) ⬅️ DO THIS NEXT

| Change | Effort | Speedup | Risk |
|--------|--------|---------|------|
| Encoder: q8 weights | Already done | Encoder 24% faster | None (verified) |
| Decoder: fp16 weights | Medium | Decoder steps ~2x faster | Need to export fp16 decoder variants |
| KV cache: stay fp32 | None | — | Baseline |

**Result**: Encoder at q8 speed + Decoder at fp32 speed = best of both worlds.

### Phase 2: Static Quantization (medium impact, medium effort)

| Change | Effort | Speedup | Risk |
|--------|--------|---------|------|
| Encoder: static int8 | Medium | Additional 5-10% | Needs calibration data |
| Decoder: static int8 | Hard | May fix decoder step slowdown | Calibration + accuracy validation |

### Phase 3: Quantized KV Cache (lower impact for Whisper, high for streaming)

| Change | Effort | Speedup | Risk |
|--------|--------|---------|------|
| Custom KV cache quant | Hard | 2x less KV mem, 10-20% step speedup | Accuracy loss, custom code |

### Phase 4: 4-bit via GGUF/llama.cpp (experimental)

| Change | Effort | Speedup | Risk |
|--------|--------|---------|------|
| Switch to GGUF backend | Very large | Model 4x smaller | Complete ORT → ggml migration |

### Phase 5: Future techniques to watch

- **FP8 inference**: When consumer GPUs support native fp8 (RTX 5090+)
- **Sparse attention**: Reduce KV cache compute for long audio
- **Speculative decoding**: Use a small draft model for faster Whisper decoding

## Summary

| Approach | Encoder speed | Decoder speed | Model size | Total time | Recommended? |
|----------|--------------|--------------|------------|-----------|-------------|
| fp32 (baseline) | 1.0x | 1.0x | 3.4 GB | 9708ms | ✅ Works |
| q8 (current) | **1.24x** 🟢 | **0.48x** 🔴 | 1.4 GB | 6911ms | ⚠️ Decoder regression |
| fp16 | ~1.1x 🟢 | ~1.5x 🟢 | 1.7 GB | ? | ✅ Good for WebGPU |
| **Mixed (q8 enc + fp32 dec)** | **1.24x** 🟢 | **1.0x** | **~1.4 GB** | **6644ms 🏆** | **RECOMMENDED** |
| Static quantization | 1.3x? | ~0.8x? | ~1.2 GB | ? | 🔬 Needs research |
| KV cache quantization | +5% | +15-20% | ~1.2 GB | ? | 🔬 Experimental |
| GGUF Q4_K_M | 2-3x | 2-3x | ~450 MB | ? | 🚀 Future work |

**Bottom line**: The mixed-precision approach (q8 encoder + fp16 decoder) is the highest-ROI next step. It gives encoder speedup without the decoder step regression. KV cache quantization would help streaming but requires custom ORT work that's not natively supported.

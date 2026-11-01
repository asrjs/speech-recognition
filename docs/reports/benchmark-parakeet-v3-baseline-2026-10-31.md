# Milestone G3 Verification Report: Parakeet TDT 0.6B v3 Baseline Benchmark

**Date**: 2026-10-31  
**Milestone**: G3 (Performance, Accuracy & Reproducible Benchmark Program)  
**Profile**: `parakeet-tdt-0.6b-v3`  
**Execution Backend**: `webgpu-hybrid`  
**Overall Verdict**: **PASS**

---

## 1. Executive Summary

Milestone G3 establishes the reproducible performance and accuracy benchmark baseline for the flagship multilingual model **`parakeet-tdt-0.6b-v3`** on real browser hardware (NVIDIA Blackwell GPU via Chromium WebGPU and WASM SIMD).

The benchmark program executed genuine public API flows through the automated browser harness, enforcing:
1. **Cold Load & Inference vs. Warm Session Reuse**: Full separation of initial artifact resolution/compilation from warm cached model invocation.
2. **Component Execution Provider & Precision Tracking**: Independent auditing of the FastConformer encoder, TDT decoder/joint network, and audio preprocessor.
3. **Hardware GPU Workgroup Dispatch Instrumentation**: Direct observation of GPU compute shader dispatches on `GPUComputePassEncoder.prototype.dispatchWorkgroups`.
4. **Oracle Alignment & Transcription Accuracy**: Verification against the standard JFK audio benchmark oracle with word-level confidence and timestamp alignment.

---

## 2. Component Execution & Precision Matrix

The approved `parakeet-tdt-0.6b-v3` profile executes in **`webgpu-hybrid`** mode, balancing GPU throughput for large convolution/attention blocks with low-latency WASM execution for sequential autoregressive decoding:

| Component | Topology / Function | Execution Provider | Precision | Artifact | Size |
|---|---|---|---|---|---|
| **Encoder** | FastConformer (17 subsampling + conformer blocks) | **WebGPU** | **FP16** | `encoder-model.fp16.onnx` | 1,238,960,452 B (~1.15 GB) |
| **Decoder / Joint** | Token-and-Duration Transducer (TDT) | **WASM (CPU)** | **INT8** | `decoder_joint-model.int8.onnx` | 18,202,004 B (~17.3 MB) |
| **Preprocessor** | NeMo 128 Mel Spectrogram Filterbank | **JavaScript / Web Audio** | Float32 | `vocab.txt` | 102,132 B (1,024 tokens) |

---

## 3. Hardware & Environment Context

- **GPU**: NVIDIA Blackwell Architecture (`adapter.vendor: nvidia`, `architecture: blackwell`)
- **WebGPU Implementation**: Dawn via Direct3D 11 backend (`--use-angle=d3d11`)
- **Browser**: Google Chrome 156.0.8078.76 (Headless)
- **Host OS**: Microsoft Windows
- **Isolation Context**: Cross-Origin Isolated (`crossOriginIsolated: true`, `COOP: same-origin`, `COEP: require-corp`)
- **Storage Buffer Limits**: `maxBufferSize: 2,147,483,648 B` (2 GiB), `maxStorageBufferBindingSize: 2,147,483,644 B`

---

## 4. Benchmark Measurement Results

### Test Audio Fixture
- **Fixture**: `jfk-short.wav`
- **SHA-256**: `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`
- **Duration**: 11.00 seconds (176,000 samples @ 16,000 Hz, 16-bit mono PCM)

### Performance Comparison

| Metric | Cold Run (Initial Load) | Warm Run (Session Reuse) | Delta / Speedup |
|---|---|---|---|
| **Model Load Latency** | **11,925.3 ms** | **0.0 ms** (instant cache hit) | $\infty$ (Cached) |
| **Inference Latency** | **1,763.3 ms** | **641.2 ms** | **2.75x faster** |
| **Total Wall Time** | **13,688.6 ms** | **641.2 ms** | **21.35x faster** |
| **Real-Time Factor (RTFx)** | **6.2x** | **17.2x** | **+177% RTFx** |
| **Real-Time Factor (RTF)** | 0.160 | 0.058 | Lower is better |
| **GPU Workgroup Dispatches** | **1,281** | **1,276** | Measured |
| **Session Instance ID** | `browser-session-1` | `browser-session-1` | Verified identical |
| **Cache Status** | `cached: false` | `cached: true` | Verified |

---

## 5. GPU Compute Dispatch Observation

GPU activity was intercepted and counted using `GPUComputePassEncoder.prototype.dispatchWorkgroups`:
- **Cold Run GPU Dispatches**: 1,281 workgroup dispatches observed during encoder execution and feature projection.
- **Warm Run GPU Dispatches**: 1,276 workgroup dispatches observed.
- **WASM Fallback Verification**: Zero silent WASM fallback under the WebGPU encoder request (`reportedBackendId: webgpu`, `evidence: gpu-compute-observed`).

---

## 6. Transcription Accuracy & Oracle Alignment

- **Reference Hypothesis**: `"And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country."`
- **Oracle Comparison**: Normalized exact match (`comparison: normalized-text-exact`, `status: PASS`).
- **Word Count**: 22 words detected with full alignment and timestamp bounds:

| Word # | Token | Start Time | End Time | Confidence | Token Indices |
|---|---|---|---|---|---|
| 0 | `And` | 0.24s | 0.56s | 0.9997 | [0] |
| 1 | `so,` | 0.56s | 1.04s | 0.9226 | [1, 2] |
| 2 | `my` | 1.04s | 1.28s | 0.9994 | [3] |
| 3 | `fellow` | 1.28s | 1.76s | 0.9980 | [4, 5, 6] |
| 4 | `Americans,` | 1.76s | 3.28s | 0.9722 | [7, 8, 9, 10] |
| 5 | `ask` | 3.28s | 3.92s | 0.9979 | [11, 12] |
| 6 | `not` | 4.24s | 4.56s | 1.0000 | [13] |
| 7 | `what` | 5.20s | 5.52s | 0.9995 | [14] |
| 8 | `your` | 5.68s | 6.00s | 0.9999 | [15] |
| 9 | `country` | 6.00s | 6.56s | 0.9996 | [16, 17, 18, 19] |
| 10 | `can` | 6.56s | 6.80s | 1.0000 | [20] |
| 11 | `do` | 6.80s | 7.12s | 1.0000 | [21] |
| 12 | `for` | 7.12s | 7.44s | 1.0000 | [22] |
| 13 | `you,` | 7.44s | 8.16s | 0.9349 | [23, 24] |
| 14 | `ask` | 8.16s | 8.72s | 0.9994 | [25, 26] |
| 15 | `what` | 8.72s | 8.96s | 0.9999 | [27] |
| 16 | `you` | 8.96s | 9.28s | 0.9996 | [28] |
| 17 | `can` | 9.28s | 9.52s | 0.9999 | [29] |
| 18 | `do` | 9.52s | 9.68s | 0.9999 | [30] |
| 19 | `for` | 9.68s | 9.92s | 1.0000 | [31] |
| 20 | `your` | 9.92s | 10.08s | 0.9996 | [32] |
| 21 | `country.` | 10.08s | 10.88s | 0.9883 | [33, 34, 35, 36, 37] |

---

## 7. Benchmark Demo UI Export (Task 1 Delivery)

In `demos/apps/benchmark/src/App.jsx`:
- Added a clean **Export JSON** button in Tab 5 ("Run Data & Export") and Tab 4 ("Compare & Pivot").
- Strict adherence to the UI rule: **zero taglines, zero helper text, zero secondary labels**.
- Formatted `JSON.stringify(snapshots, null, 2)` export with MIME type `application/json`.
- Distinct capture and segregation of cold runs (`phase: 'cold'`) and warm repeat runs (`phase: 'warm'`), as well as dedicated summary fields (`coldInferenceMean`, `warmInferenceMean`).
- Full round-trip fidelity between `Export JSON` and `Import JSON Snapshot`.

---

## 8. Reproducibility & Independent Verification

The benchmark measurements can be independently reproduced on any machine with Chrome WebGPU and Node.js using:

```bash
# Execute cold and warm benchmark in headless Chrome WebGPU
node speech-recognition/tools/browser-harness/cli.mjs \
  --local-dir __external_models__/onnx/nemo/parakeet-tdt-0.6b-v3-onnx \
  --backend webgpu-hybrid \
  --mode offline \
  --fixture jfk-short.wav \
  --expect-jfk \
  --out docs/reports/benchmark-parakeet-v3-baseline-2026-10-31.json
```

**Associated Raw Evidence**:
- `docs/reports/benchmark-parakeet-v3-baseline-2026-10-31.json`
- `speech-recognition/docs/reports/benchmark-parakeet-v3-baseline-2026-10-31.json`

# Cross-model short-window benchmark and RTFx regression probe

Date: 2026-08-29  
Host: Windows, Chrome headless `--enable-unsafe-webgpu`, NVIDIA Blackwell,
ONNX Runtime Web 1.29.0, `N:\github\asrjs\webgpu-agent-test`  
Evidence manifest: `cross-model-audio-window-benchmark-2026-08-29.json`

## Why this probe exists

The historical Parakeet.js benchmark warns that short clips make fixed
preprocessing, session, and first decoder-step costs dominate RTFx. Its
15–30-second subset reports a median of 80.71x (74 runs, fp32 WebGPU encoder,
int8 WASM decoder, JS preprocessor). The current library had been reporting
much smaller values, so this probe separates audio-window effects from runtime
and configuration effects.

The benchmark now uses one shared clip below the model limits, executes one
same-session warm-up before measured repetitions, and records the warm-up
separately. A labeled transcript is a correctness oracle only; throughput runs
without a label use `qualityOracle: null` and are not WER claims.

## Audio inventory and limits

| Asset | Duration | Use |
| --- | ---: | --- |
| `tools/data/fixtures/audio/librivox.org.wav` | 18.714 s, 22.05 kHz mono | Shared cross-model benchmark and Parakeet v3 labeled oracle |
| `tools/data/fixtures/audio/jfk-short.wav` | 11.000 s, 16 kHz mono | Existing family-specific labeled smoke fixture |
| `tools/data/fixtures/audio/JFK.ogg` | 146.326 s, 16 kHz mono | Long source provenance; clip only, do not pass whole to static graphs |

The Parakeet v3 encoder rejects the complete JFK source (`1830` frames versus
the graph's `1024`-frame limit). A 30- and 40-second clipped exploratory run
completed, but the final cross-model contract uses the 18.714-second window so
Whisper and other models can share it; all shared windows remain below 30 s.

## Warmed browser results on the shared clip

All runs used the real Chrome/WebGPU harness and NVIDIA adapter. `loadMs` is
reported separately; the RTFx values below are the executor's native phase
metric, while `wallRtfx` is calculated from measured transcription wall time.

| Family / composition | Correctness | Warm-up | Measured median | Native RTFx | Wall RTFx | Dominant measured phase |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Parakeet TDT v3, fp16 WebGPU encoder + fp32 WASM decoder + ONNX preprocessor | exact (91 tokens) | 1456.8 ms | 1040.9 ms | 18.07x | 17.98x | decoder 750.8 ms |
| Parakeet TDT v2, fp16 WebGPU encoder + int8 WASM decoder + JS preprocessor | normalized (capitalization differs) | 1080.6 ms | 666.8 ms | 28.24x | 28.07x | decoder 380.7 ms |
| Whisper Large V3 Turbo 4-graph, fp16-I/O WebGPU encoder + fp32 CPU-KV decoder | throughput-only; transcript present | 3895.4 ms | 1526.6 ms | 12.37x | ~12.25x | decoder 976.9 ms |
| Whisper Large V3 Turbo 4-graph, same clip, GPU-KV candidate | throughput-only; transcript present | 2537.4 ms | 2237.6 ms | 8.42x | ~8.36x | decoder 1264.6 ms |

The Whisper transcript on this LibriVox clip is not the JFK labeled sentence,
so both Whisper rows intentionally have no quality oracle. The GPU-KV result
is a negative, workload-specific result on this clip (about 47% slower native
total than CPU-KV); it does not invalidate the earlier JFK GPU-KV win.

## Interpretation

1. The short-audio hypothesis is real, but it does not explain the entire
   Parakeet gap. Current v2 is about 28x on a comparable 18–20-second window,
   versus the historical 15–30-second median of 80.71x.
2. The historical configuration is not identical: it used a fp32 encoder and
   int8 decoder, while the browser fp32 external-data encoder currently fails
   under ORT Web 1.29 (`Module.MountedFiles is not available`). This is an
   explicit compatibility boundary, not a silent substitution.
3. The current browser runner forced `cpuThreads: 1`. Probes with values above
   one stalled because the harness served `/ort-dist/*` without a
   `Cross-Origin-Embedder-Policy: require-corp` header, so ORT's module workers
   were blocked with `coep-frame-resource-needs-coep-header`. After adding the
   header, `cpuThreads=4` completes with exact parity but is slower than
   single-thread for this GRU decoder (about 14.2x vs 17.7x warmed RTFx). The
   runner now omits `cpuThreads` by default and keeps `--cpu-threads=N` as the
   explicit diagnostic.
4. Decoder time remains the largest measured Parakeet phase. Next optimization
   work should profile WASM SIMD/thread viability, decoder graph/provider
   placement, and tensor allocation/state reuse using the same warm-up and
   audio contract; do not infer a graph regression from a short cold run.

## Reproduction

Start the sibling server from `N:\github\asrjs\webgpu-agent-test`:

```powershell
npm run dev
```

Parakeet v3 labeled control:

```powershell
node scripts/run-parakeet-tdt-webgpu.mjs --model=v3 --mode=wasm `
  --encoder=fp16 --encoder-backend=webgpu --preprocessor=onnx `
  --decoder-quant=fp32 --repeat=3 --warmup=1 `
  --audio=/parakeet-audio/librivox.org.wav --oracle=fixed
```

Parakeet v2 comparison:

```powershell
node scripts/run-parakeet-tdt-webgpu.mjs --model=v2 --mode=wasm `
  --encoder=fp16 --encoder-backend=webgpu --preprocessor=js `
  --decoder-quant=int8 --repeat=3 --warmup=1 `
  --audio=/parakeet-audio/librivox.org.wav --oracle=fixed
```

Whisper CPU-KV control on the same clip:

```powershell
node scripts/run-test.mjs fp16io-fp32-webgpu `
  --audio=/parakeet-audio/librivox.org.wav --oracle=none `
  --audio-strategy=native --wait-role=measurement
```

The raw browser JSON files are retained in the sibling harness `_results`
directory. The checked-in JSON manifest binds the selected result paths,
audio hashes, model/backend settings, and measured values.

## Remaining work

- Multi-threaded WASM worker path is repaired at the harness level: the
  `/ort-dist/*` static responses now send `Cross-Origin-Embedder-Policy:
  require-corp`. Threaded Parakeet decode is slower than single-thread for
  this workload, so re-run v2 with fp32 encoder/int8 decoder and 4–12 threads
  only as a larger-workload comparison, not as an assumed win.
- Add the shared audio/role/warm-up parameters to the remaining family pages
  (GigaAM, SenseVoice, X-ASR, Qwen) and collect the same-window matrix.
- Compare Parakeet v3 decoder WebGPU and hybrid placements on this contract,
  preserving exact 91-token parity and disposal safety.
- Add a verified windowed long-audio runner for source recordings longer than
  a graph's static frame limit; aggregate effective RTFx separately from
  per-window RTFx.

# Parakeet TDT GPU-state second-browser validation (2026-08-29)

## Context

The opt-in decoderStateOutputLocation='gpu-buffer' placement previously
measured a 17.3% latency reduction / 21% RTFx gain on Chrome (NVIDIA
Blackwell, D3D11) with exact 91-token parity and clean disposal. The
documented promotion gates were a second browser/adapter and a longer
lifecycle soak.

## Second-adapter attempt (blocked, documented)

Chrome ANGLE backends vulkan and gl were both attempted on this host and
failed with WEBGPU_NO_ADAPTER (no available backend found). Artifacts:
tools/data/results/nemo-tdt/parakeet-tdt-v3-webgpu-dec-fp32-cpustate-vulkan-librivox-18s.json
and ...-angle-gl-librivox-18s.json. D3D11 is the only working WebGPU
backend here, so the second-engine requirement is met via a second
browser (Microsoft Edge, Chromium) on the same NVIDIA Blackwell adapter.

## Edge A/B (library loadSpeechModel path, fp16/WebGPU encoder + fp32/WebGPU decoder,
ONNX preprocessor, native-rate audio, 1 warm-up + 3 measured runs)

| Decoder state | Median ms | RTFx | Parity | Teardown |
|---|---|---|---|---|
| cpu (control) | 2512.5 | 7.46x | exact 91 tokens | clean |
| gpu-buffer | 2214.5 | 8.47x | exact 91 tokens | clean |

GPU-state reproduced a 298 ms (11.9%) median latency reduction and 13.5%
RTFx gain on the second browser with zero disposal errors. Notably, Edge's
WebGPU decoder path is itself faster than Chrome's earlier ~3.4 s controls
on the same machine (Chromium/Dawn version difference); the A/B is
same-browser, so the relative win stands.

## Lifecycle soak (Edge, GPU-state, 1 warm-up + 8 same-session runs)

8/8 runs passed with the exact transcript. Transcribe times stayed in a
tight 1860-2164 ms band with no upward drift. JS-heap snapshots ran
107 -> 116 MB, dropped to 32 MB after GC, then stabilized at 32-54 MB -
normal GC behavior, no monotonic leak signature. Model and runtime
disposal completed with no errors.

Artifact: tools/data/results/nemo-tdt/parakeet-tdt-v3-webgpu-dec-fp32-gpustate-edge-soak8-librivox-18s.json

## Verdict

Both stated promotion gates (second browser, longer soak) are now passed,
and the win is real on every measured browser. The placement remains
opt-in for one honest reason: all evidence is from a single GPU vendor
(NVIDIA Blackwell via D3D11); AMD/Intel adapters and non-Chromium engines
are untested, and ORT gpu-buffer outputs are a relatively new capability.
Given that the much faster hybrid path (fp16/WebGPU encoder + int8/WASM
decoder, ~36x) is the practical default for this model, flipping the
WebGPU-decoder state default has limited real-world upside today.
Promotion should follow a non-NVIDIA adapter pass or an explicit decision
to accept single-vendor evidence.

## Reproduction

node scripts/run-parakeet-tdt-webgpu.mjs --browser=edge --mode=webgpu
  --model=v3 --encoder=fp16 --encoder-backend=webgpu --preprocessor=onnx
  --decoder-quant=fp32 [--gpu-state] --warmup=1 --repeat=3


# X-ASR-zh-en porting workflow

Use this playbook for the official Gilgamesh-J / GilgameshWind X-ASR-zh-en
chain. Do not start from an unrelated Zipformer ONNX as the oracle.

## Oracle

The published runtime is **sherpa-onnx**, not a third-party converter.

- Repo: https://github.com/Gilgamesh-J/X-ASR (Apache-2.0)
- Weights/ONNX: HuggingFace `GilgameshWind/X-ASR-zh-en`
- Official inference: `sherpa_onnx.OnlineRecognizer.from_transducer(..., model_type="zipformer2")`
- Official graphs: `encoder-*.onnx` + `decoder-*.onnx` + `joiner-*.onnx` + `tokens.txt`
  under `deployment/models/chunk-{160,480,960,1920}ms-model/`

This is **true stateful streaming**: Zipformer2 encoder caches plus a
stateless decoder/joiner. Do not describe fixed-window looping as streaming.

PyTorch `streaming_exp/pretrained.pt` exists on the HF repo. The documented
deployment path is sherpa-onnx on the exported graphs; that is the transcript
oracle for this family.

## Bounded candidate

`chunk-160ms-model` (lowest-latency streaming variant). 16 kHz mono, 80-dim
fbank inside sherpa-onnx.

## Ladder

1. Official ONNX + hashes (HF revision, git clone, license)
2. Official sherpa-onnx streaming decode on `jfk-short.wav`
3. Record encoder/decoder/joiner IO and cache-state shapes
4. Native ORT / library greedy transducer vs sherpa-onnx text
5. WASM, then Chrome WebGPU
6. Preset only after gates pass

## Failure classes

- `EXPORT_BLOCKED` — no official ONNX / icefall export failed
- `LICENSE_BLOCKED`
- `ARCHITECTURE_NOT_BROWSER_SUITABLE`
- `PREPROCESSING_MISMATCH` — JS fbank vs sherpa-onnx fbank
- `ENCODER_MISMATCH` / `TOKENIZER_MISMATCH`
- `ORT_WEB_UNSUPPORTED_OP` / `WASM_MEMORY_LIMIT`
- `WEBGPU_NO_ADAPTER` / `WEBGPU_UNSUPPORTED_OP` / `WEBGPU_UNSUPPORTED_DTYPE` / `WEBGPU_MEMORY_LIMIT`

The family stays experimental. No weights in Git. Qwen stays intact unless
this ladder is classified blocked.

## Chrome WebGPU

Use `N:\github\asrjs\webgpu-agent-test` (`npm run dev` on :8765) and
`node scripts/run-xasr-webgpu.mjs`. Create encoder/decoder/joiner sessions
sequentially; parallel WebGPU EP create fails with
`another WebGPU EP inference session is being created.`

## Feature extractor

JS `XAsrJsFrontend` follows sherpa-onnx / knf defaults: 80-bin Kaldi fbank,
dither 0, snip_edges false, high_freq -400 (7600 Hz), within-window
preemphasis, float32 log floor. Dump with
`tools/model-debugging/reference/x-asr/dump_xasr_fbank.py`.

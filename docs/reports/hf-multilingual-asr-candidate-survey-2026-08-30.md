# Hugging Face multilingual ASR candidate survey (2026-08-30)

This is a dated discovery snapshot, not a quality benchmark. Hugging Face
download counters and model repositories change over time; the raw API results
are kept in:

- `tools/data/results/model-candidates/hf-asr-candidates-2026-08-30.json`
  (ranked `automatic-speech-recognition` index)
- `tools/data/results/model-candidates/hf-asr-search-asr-2026-08-30.json`
  (ranked index with `search=asr`)
- `tools/data/results/model-candidates/hf-asr-candidate-review-2026-08-30.json`
  (the search snapshot plus selected repositories that may not appear in the
  first page)

## Method

The reusable command is:

```text
node tools/scripts/survey-hf-asr-candidates.mjs --limit 50 --search asr --include nvidia/nemotron-3.5-asr-streaming-0.6b,FunAudioLLM/Fun-ASR-MLT-Nano-2512,ibm-granite/granite-speech-4.1-2b,ibm-granite/granite-speech-4.1-2b-nar,zai-org/GLM-ASR-Nano-2512,mistralai/Voxtral-Mini-4B-Realtime-2602,onnx-community/Voxtral-Mini-3B-2507-ONNX,istupakov/canary-1b-v2-onnx,microsoft/VibeVoice-ASR,multimodalart/VibeVoice-ASR-BitNet-ONNX,onnx-community/granite-speech-4.1-2b-ONNX --output tools/data/results/model-candidates/hf-asr-candidate-review-2026-08-30.json
```

The script records the API's download/like counters, tags and language codes,
last-modified time, library, and file-list signals (`.onnx`, external ONNX
data, `.wasm`, GGUF, safetensors, PyTorch, and NeMo). These signals identify
what to inspect; they do not prove graph correctness, parity, or WebGPU
performance. Model cards, official repositories, sibling projects, and the
library's reference chain remain authoritative for promotion.

The ranked pages used for the review were the [multilingual ASR model
index](https://huggingface.co/models?language=multilingual&p=0&pipeline_tag=automatic-speech-recognition&sort=downloads)
and the [ASR search index](https://huggingface.co/models?search=asr&sort=downloads).

The local preflight found no Nemotron, Fun-ASR, Granite, GLM-ASR, Voxtral, or
VibeVoice weight directories under `N:\models`, `C:\Drive\hf_cache`, or this
repository's reference tree. Consequently this report makes no inference or
quality claim for those models; the next port must cross the artifact-access
and original-reference gates before any browser benchmark is recorded.

## Decisions

| Family / source | Snapshot signal | Browser artifact evidence | Local library status | Decision |
| --- | ---: | --- | --- | --- |
| [NVIDIA Nemotron 3.5 streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | 927,137 downloads; 40 language-locales | Official repo has NeMo weights only. Community [ONNX export](https://huggingface.co/codavidgarcia/nemotron-3.5-asr-streaming-0.6b-onnx) and [FP16 WebGPU export](https://huggingface.co/goryodog/tokihisu-nemotron-3.5-asr-streaming-0.6b-webgpu-fp16) exist. | No Nemotron family; NeMo RNNT executor and cache-aware streaming primitives exist. | **Priority 1: adapt/validate**, do not duplicate the existing export. Verify original NVIDIA inference, chunk/cache contracts, license, native/WASM/WebGPU parity, and streaming latency. |
| [Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) | 995 downloads in direct API snapshot; 31 languages | PyTorch `model.pt` and tokenizer/config only; no ONNX or WASM files in the selected repo. | No family. Existing SenseVoice/CTC and Qwen-style tooling may be reusable, but this is a speech-LLM topology. | **Priority 2: bounded export spike** after Nemotron. Capture official FunASR output first; stop as `EXPORT_BLOCKED` or `ARCHITECTURE_NOT_BROWSER_SUITABLE` if the Qwen3-style decoder cannot be exported without changing semantics. |
| [IBM Granite Speech 4.1 2B](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) | 274,524 downloads; 7 languages | [ONNX Community q4/q4f16 artifacts](https://huggingface.co/onnx-community/granite-speech-4.1-2b-ONNX) and an existing [Granite WebGPU Space](https://huggingface.co/spaces/ibm-granite/granite-speech-webgpu) are available. | No Granite family. | **Adapt/benchmark only** if a library-native API adds value; do not re-port a working Transformers.js/WebGPU path. |
| [IBM Granite Speech 4.1 2B NAR](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar) | 123,849 downloads; 5 languages; non-autoregressive/CTC-tagged | No ONNX/WASM files in the official repository. | No family. | **Exploratory later**: potentially valuable non-autoregressive graph, but requires original/reference capture and a dedicated export before any runtime work. |
| [GLM-ASR-Nano-2512](https://huggingface.co/zai-org/GLM-ASR-Nano-2512) | 82,285 downloads; English/Chinese; 4.52 GB safetensors | No ONNX/WASM files in the official repository. | No family. | **Defer**: model size and lack of an export/browser path make it a poor next web target. |
| [Voxtral Mini 3B ONNX](https://huggingface.co/onnx-community/Voxtral-Mini-3B-2507-ONNX) / [4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | 2.20M downloads for 4B Realtime; 8 languages for 3B | Existing ONNX files and [WebGPU Space](https://huggingface.co/spaces/webml-community/Voxtral-WebGPU); Transformers.js model contract already exists. | No Voxtral family. | **Do not duplicate**. Revisit only for a library-native headless wrapper or a measured graph/runtime improvement that is not present upstream. |
| [VibeVoice-ASR](https://huggingface.co/microsoft/VibeVoice-ASR) / [BitNet ONNX](https://huggingface.co/multimodalart/VibeVoice-ASR-BitNet-ONNX) | 699,585 downloads for the full model; multilingual BitNet variant | Existing ONNX WebGPU/WASM export and live demo; the export documents 24 kHz audio, ternary decoder, and large memory tiers. | No VibeVoice family. | **Defer/adapt later**: existing browser implementation means the differentiated work would be a headless integration plus measured memory/long-form behavior, not another exporter. |
| [Canary 1B v2 ONNX](https://huggingface.co/istupakov/canary-1b-v2-onnx) | 1,758 downloads for the ONNX repo | ONNX artifacts exist; official model is NeMo AED. | Canary AED preset/executor exists here for 180M Flash. | **Candidate extension** after current artifact parity: reuse `src/models/nemo-aed`, prove 1B prompt/tokenizer and memory behavior, then add a thin manifest only if it earns promotion. |
| Qwen3-ASR 0.6B / 1.7B | 1.52M / 4.59M downloads in the review indexes | Official repositories are safetensors; this library already has the 0.6B experimental family and measured GPU-KV path. | Qwen 0.6B integrated; INT4 explicitly paused because it hangs. | **No new port now**. Treat 1.7B as a later size/quality comparison, preserving the no-INT4-hang boundary. |
| Parakeet, Whisper, GigaAM, SenseVoice | High download signals and/or existing ONNX ecosystems | Existing library families and benchmark/evidence paths. | Implemented and measured here. | **Regression/reference set**, not new candidates. |

## Chosen next action

Nemotron 3.5 is the highest-value next bounded objective because it combines
multilingual coverage, true cache-aware RNNT streaming, a close fit to the
existing NeMo RNNT state machinery, and independently published ONNX/WebGPU
artifacts. The first implementation slice must remain an adaptation/validation
slice:

1. verify the NVIDIA original checkpoint and official inference output on a
   fixed short clip and a speech/silence streaming fixture;
2. inspect the community export manifest, graph inputs/outputs, external-data
   hashes, chunk sizes, language prompt, and LSTM/RNNT state layout;
3. compare the earliest divergence through native ORT, WASM, WebGPU, and the
   library executor; keep the ONNX/WebGPU export clearly third-party until
   provenance and license checks pass;
4. measure warm-up, first-partial latency, per-chunk latency, steady-state
   RTFx, memory, and transcript/token parity; benchmark fp32/fp16/int8 only
   when each variant is available and semantics remain exact.

If the existing Nemotron browser export is already complete and performant,
the result should be a **reuse/adapt** decision and a reusable compatibility
adapter or benchmark—not a redundant model port. Fun-ASR MLT remains the next
genuinely new export candidate, subject to the mandatory original-model chain.

## Reusable lessons

- Popularity is a prioritization signal, not a correctness or quality claim.
- A model card, HF Space, or `.onnx` file does not prove the graph contract or
  browser parity; inspect exact files and run the same artifact through the
  validation ladder.
- Existing browser ports change the engineering question from “can we port
  it?” to “does a library-native, model-specific integration add measured
  value?”
- Keep discovery snapshots, source/reference outputs, quality labels, and
  throughput evidence as separate artifacts. Never infer a WER or RTFx claim
  from download counts or a demo.
- The existing porting playbooks and automated scripts remain the execution
  system: original weights → official inference → reference captures →
  optimized ONNX → native ORT → WASM/WebGPU → library executor → sibling
  browser integration.

# SenseVoiceSmall porting workflow

Use this playbook for the official FunAudioLLM SenseVoiceSmall chain. Do not
start from `OpenVoiceOS/sensevoice-small-onnx` or another public ONNX file as
the oracle.

## Ranking note

SenseVoice is the second family after GigaAM multilingual CTC: same
encoder+CTC shape, FunASR license, CJK-first language set. X-ASR and Qwen
stay intact until this official ladder is recorded.

## Official provenance

- Repo: https://github.com/FunAudioLLM/SenseVoice
- Local clone: `N:\github\FunAudioLLM\SenseVoice`
- Weights: HuggingFace `FunAudioLLM/SenseVoiceSmall`
- License: FunASR Model Open Source License (`model-license`), not Apache-2.0
- Official inference: FunASR `AutoModel` / `SenseVoiceSmall.inference` with
  `vad_model=None` for clip-level oracle
- Official ONNX: `model.export(type="onnx")` then `utils/export_utils.py`
  (`quantize=False` for the first native/WASM gate)

Official ONNX IO (not the OpenVoiceOS folded graph):

| Name | Role |
| --- | --- |
| `speech` | float32 `[B, T, 560]` LFR+CMVN features (`lfr_m=7`) |
| `speech_lengths` | int32 `[B]` |
| `language` | int32 `[B]` (`auto=0`, `zh=3`, `en=4`, `yue=7`, `ja=11`, `ko=12`) |
| `textnorm` | int32 `[B]` (`withitn=14`, `woitn=15`) |
| `ctc_logits` | float32 `[B, T', V]` |
| `encoder_out_lens` | int32 `[B]` |

The official runtime applies Kaldi fbank + LFR + CMVN **outside** the graph.
The library executor detects `speech` vs folded OpenVoiceOS `features`.

## Ladder

1. Official weights + hashes (repo revision, license)
2. Official FunASR generate on `jfk-short.wav`
3. Official unquantized ONNX export
4. Native ORT CPU vs official PyTorch text/tokens
5. JS frontend vs official LFR+CMVN features
6. WASM, then WebGPU, then library
7. Preset only after the relevant gates pass

JS LFR+CMVN, WASM, and Chrome WebGPU now match official jfk-short text.
Node remains `WEBGPU_NO_ADAPTER`. The family stays experimental. No preset.

## Failure classes

- `PREPROCESSING_MISMATCH` — JS fbank/LFR/CMVN vs FunASR `WavFrontend`
- `ENCODER_MISMATCH` — native ORT logits diverge from PyTorch
- `TOKENIZER_MISMATCH` — SentencePiece / prompt-token filtering
- `GRAPH_CONTRACT_MISMATCH` — OpenVoiceOS `features` `[B,T,80]` vs official `speech` `[B,T,560]`
- `WEBGPU_NO_ADAPTER` — Node/vitest has no WebGPU device. Chrome with
  `--enable-unsafe-webgpu` on NVIDIA Blackwell passed official fp32 JFK text.
- `WEBGPU_UNSUPPORTED_OP` / `WEBGPU_UNSUPPORTED_DTYPE` / `WEBGPU_MEMORY_LIMIT`
- `ORT_WEB_UNSUPPORTED_OP` / `WASM_MEMORY_LIMIT`

## Proven results (jfk-short)

- Official FunASR 1.4.4 (`vad_model=None`, `language=en`, `use_itn=false`):
  `<|en|><|EMO_UNKNOWN|><|Speech|><|woitn|>and so my fellow americans ask not what your country can do for you ask what you can do for your country`
- Official unquantized ONNX (`FunASR AutoModel.export`, opset 14):
  937,615,562 bytes, SHA-256
  `8fc794f08c390ce26f0c8878904a2e0a63214faaa53e2354245a50c0d2b65700`
- Native ORT CPU: exact tagged text match vs FunASR
- WASM: exact JFK body; total ~4.07s, RTF 0.370
- Chrome WebGPU: exact JFK body; load 17.03s, transcribe 2.26s, RTF 0.206

The family stays experimental. No preset. No weights in Git. X-ASR is next.

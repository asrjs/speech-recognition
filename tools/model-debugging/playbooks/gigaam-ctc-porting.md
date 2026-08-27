# GigaAM Multilingual CTC porting workflow

Use this playbook for the official GigaAM multilingual CTC chain. Do not start
from `istupakov/gigaam-multilingual-ctc-onnx` or another public ONNX file as
the oracle.

## Ranking note

GigaAM multilingual CTC (220M) is the first of the four new families because
it is a single encoder+CTC graph, MIT-licensed, and has an official
`model.to_onnx` path. SenseVoice is similar topologically but uses the FunASR
license and a CJK-first language set. X-ASR is the streaming Zipformer track.
Qwen3-ASR-0.6B has the widest language coverage, including Turkish, but is a
large speech-LLM with KV cache and no official ONNX.

## Ladder

1. Official checkpoint MD5 `5379d887c53ccd9cb95981e2a1832720`
2. Official `gigaam.load_model('multilingual_ctc').transcribe`
3. Official `model.to_onnx`
4. Native ONNX Runtime CPU vs PyTorch logits/text
5. JS frontend vs official features
6. WASM, then WebGPU, then library browser smoke
7. Preset only after the relevant gates pass

## Commands

See `tools/model-debugging/reference/gigaam-multilingual-ctc/README.md`.

## Failure classes

- `PREPROCESSING_MISMATCH` — JS mel vs torchaudio MelSpectrogram (onset log-floor blow-up is closed; remaining STFT max-abs ~0.007)
- `ENCODER_MISMATCH` — native ORT logits diverge from PyTorch
- `TOKENIZER_MISMATCH` — blank is `len(vocab)`, not a spoken token
- `WEBGPU_NO_ADAPTER` — Node/vitest has no WebGPU device. Chrome with
  `--enable-unsafe-webgpu` on NVIDIA Blackwell passed official fp16 JFK text.
- `WEBGPU_UNSUPPORTED_OP` / `WEBGPU_UNSUPPORTED_DTYPE` / `WEBGPU_MEMORY_LIMIT`
- `ORT_WEB_UNSUPPORTED_OP` / `WASM_MEMORY_LIMIT`

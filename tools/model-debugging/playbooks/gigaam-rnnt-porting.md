# GigaAM v3 E2E RNN-T porting workflow

Use this playbook for official `v3_e2e_rnnt`. Do not start from a public ONNX
file as the oracle.

## Bounded variant

`v3_e2e_rnnt` is the official ONNX-tested RNN-T (`tests/test_onnx.py`).
It is Russian with punctuation/normalization, not multilingual CTC English.

## Ladder

1. Official checkpoint MD5 `2730de7545ac43ad256485a462b0a27a`
2. Official `gigaam.load_model('v3_e2e_rnnt').transcribe` on `example.wav`
3. Official `model.to_onnx` → encoder / decoder / joint
4. Native ONNX Runtime CPU vs PyTorch encoded + greedy text
5. JS frontend + official greedy (blank does not update predictor state)
6. WASM, then Chrome WebGPU
7. No preset until a public English/multilingual claim is justified

## Graph contract

- Encoder: `audio_signal`, `length` → `encoded`, `encoded_len`
- Decoder: `x`, `hi`, `ci` → `dec`, `ho`, `co` (`hi`/`ci` are `[layers, batch, hidden]`)
- Joint: `enc` `[B, enc_hidden, 1]`, `dec` `[B, pred_hidden, 1]` → `joint`
- Blank id = SentencePiece vocab length (1024). Embedding `padding_idx` is blank.
- Official ONNX greedy uses at most 3 letters/frame; PyTorch default is 10. Both match on `example.wav`.

## Failure classes

- `PREPROCESSING_MISMATCH` — JS mel vs official FeatureExtractor
- `ENCODER_MISMATCH` — native ORT encoded diverges from PyTorch
- `TOKENIZER_MISMATCH` — piece-join vs SentencePiece decode
- `WEBGPU_NO_ADAPTER` — Node/vitest has no WebGPU device
- `WEBGPU_UNSUPPORTED_OP` / `WEBGPU_UNSUPPORTED_DTYPE` / `WEBGPU_MEMORY_LIMIT`
- `ORT_WEB_UNSUPPORTED_OP` / `WASM_MEMORY_LIMIT`

See `tools/model-debugging/reference/gigaam-v3-e2e-rnnt/README.md`.

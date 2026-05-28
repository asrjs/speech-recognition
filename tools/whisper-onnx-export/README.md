# Whisper ONNX Export Tool

Produces ASR.js-compatible Whisper ONNX artifacts with cross-attention support.

## Usage

```bash
# Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install torch transformers optimum onnx onnxconverter-common huggingface_hub

# Export whisper-tiny
python export_whisper.py openai/whisper-tiny ./output/whisper-tiny

# Export with fp16 variants
python export_whisper.py openai/whisper-base ./output/whisper-base --fp16
```

## Output Structure

```
output/whisper-tiny/
  manifest.json              — model metadata
  tokenizer.json             — BPE tokenizer
  generation_config.json     — alignment_heads, suppress tokens
  config.json                — model config
  preprocessor_config.json   — audio preprocessing
  encoder_model.onnx         — mel features → encoder hidden states
  decoder_init_model.onnx    — full prompt decode (no cache)
  decoder_step_model.onnx    — single-token decode with KV cache
  decoder_align_model.onnx   — forced alignment with selected cross-attention
```

## Graph Details

### encoder_model.onnx
- Input: `mel` [batch, n_mels, num_frames]
- Output: `encoder_hidden_states` [batch, audio_ctx, d_model]

### decoder_init_model.onnx
- Inputs: `input_ids` [batch, seq], `encoder_hidden_states` [batch, audio_ctx, d_model]
- Output: `logits` [batch, seq, vocab]
- Use: first decode pass, language detection, forced alignment prompt

### decoder_step_model.onnx
- Inputs: `input_ids` [batch, 1], `encoder_hidden_states`, `past_key_values.*`
- Outputs: `logits` [batch, 1, vocab], `present.*` (KV cache)
- Use: fast autoregressive decode

### decoder_align_model.onnx
- Inputs: `input_ids` [batch, alignment_seq], `encoder_hidden_states`
- Outputs: `logits` [batch, alignment_seq, vocab], `selected_cross_attentions` [batch, n_selected, alignment_seq, audio_ctx]
- Use: word-level timestamps via cross-attention DTW

## Model Variants

| Model | Params | n_mels | alignment_heads |
|-------|--------|--------|-----------------|
| whisper-tiny | ~39M | 80 | [[2,2],[3,0],[3,2],[3,3],[3,4],[3,5]] |
| whisper-base | ~74M | 80 | [[3,3],[4,7],[5,1],[5,5],[5,7]] |
| whisper-small | ~244M | 80 | [[5,3],[5,9],[8,0],[8,4],[8,7],[8,8],[9,0],[9,7],[9,9],[10,5]] |
| whisper-medium | ~769M | 80 | [[13,8],[16,11],[17,14],[19,14],[21,1],[23,2]] |
| whisper-large-v3 | ~1.6B | 128 | [[10,17],[13,19],[15,18],[19,7],[20,18],[20,19],[22,16],[23,17]] |

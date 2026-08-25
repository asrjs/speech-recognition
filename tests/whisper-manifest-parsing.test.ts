import { describe, expect, it } from 'vitest';
import { parseWhisperManifest } from '../src/models/whisper-seq2seq/manifest.js';

const tinyManifest = {
  model_id: 'openai/whisper-tiny',
  format: 'whisper-browser-self-export-v1',
  opset: 17,
  num_mel_bins: 80,
  max_source_positions: 3000,
  max_target_positions: 448,
  d_model: 384,
  decoder_layers: 4,
  decoder_attention_heads: 6,
  head_dim: 64,
  vocab_size: 51865,
  alignment_heads: [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]],
  alignment_heads_source: 'generation_config_or_config',
  special_tokens: {
    eos_token_id: 50257,
    bos_token_id: 50257,
    pad_token_id: 50257,
    decoder_start_token_id: 50258,
    no_timestamps_token_id: 50363,
    suppress_tokens: [1, 2, 7],
    timestamp_begin: 50364,
  },
  artifacts: {
    encoder: 'encoder_model.onnx',
    decoder_init: 'decoder_init.onnx',
    decoder_step: 'decoder_step.onnx',
    decoder_align: 'decoder_align.onnx',
  },
};

const baseManifest = {
  ...tinyManifest,
  model_id: 'openai/whisper-base',
  d_model: 512,
  decoder_layers: 6,
  decoder_attention_heads: 8,
  head_dim: 64,
  alignment_heads: [[3, 1], [4, 2], [4, 5], [5, 0], [5, 3]],
};

describe('Whisper manifest parsing (whisper-browser-self-export-v1)', () => {
  it('parses generation config from manifest', () => {
    const parsed = parseWhisperManifest(tinyManifest);
    const genConfig = parsed.generationConfig;
    expect(genConfig.alignmentHeads).toEqual([
      { layer: 2, head: 2 },
      { layer: 3, head: 0 },
      { layer: 3, head: 2 },
      { layer: 3, head: 3 },
      { layer: 3, head: 4 },
      { layer: 3, head: 5 },
    ]);
    expect(genConfig.noTimestampsTokenId).toBe(50363);
  });

  it('parses the causal alignment export marker', () => {
    const parsed = parseWhisperManifest({
      ...baseManifest,
      alignment_export: {
        causal_self_attention: true,
        encoder_hidden_state_dtype: 'float32',
        attention_implementation: 'eager',
        attention_values: 'logits',
        attention_layout: 'selected_heads',
      },
    });

    expect(parsed.alignmentExport).toEqual({
      causalSelfAttention: true,
      encoderHiddenStateDtype: 'float32',
      attentionImplementation: 'eager',
      attentionValues: 'logits',
      attentionLayout: 'selected_heads',
    });
  });

  it('leaves legacy manifests unverified', () => {
    expect(parseWhisperManifest(baseManifest).alignmentExport).toBeUndefined();
  });

  it('parses model config from manifest', () => {
    const parsed = parseWhisperManifest(tinyManifest);
    const modelConfig = parsed.modelConfig;
    expect(modelConfig.decoderLayers).toBe(4);
    expect(modelConfig.decoderAttentionHeads).toBe(6);
    expect(modelConfig.dModel).toBe(384);
    expect(modelConfig.headDim).toBe(64);
    expect(modelConfig.medianFilterWidth).toBe(7); // default
  });

  it('computes headDim from d_model when head_dim not in manifest', () => {
    // Some manifests might omit head_dim
    const raw = { ...tinyManifest };
    delete (raw as Record<string, unknown>).head_dim;
    const parsed = parseWhisperManifest(raw);
    expect(parsed.modelConfig.headDim).toBe(64); // 384 / 6
  });

  it('parses whisper-base dimensions correctly', () => {
    const parsed = parseWhisperManifest(baseManifest);
    expect(parsed.modelConfig.decoderLayers).toBe(6);
    expect(parsed.modelConfig.decoderAttentionHeads).toBe(8);
    expect(parsed.modelConfig.dModel).toBe(512);
    expect(parsed.modelConfig.headDim).toBe(64);
    expect(parsed.generationConfig.alignmentHeads).toEqual([
      { layer: 3, head: 1 },
      { layer: 4, head: 2 },
      { layer: 4, head: 5 },
      { layer: 5, head: 0 },
      { layer: 5, head: 3 },
    ]);
  });

  it('validates manifest format', () => {
    expect(() => parseWhisperManifest({ format: 'unknown' })).toThrow(/Unsupported/);
    expect(() => parseWhisperManifest({})).toThrow(/Unsupported/);
  });

  it('returns empty alignment heads when field missing', () => {
    const raw = { ...tinyManifest };
    delete (raw as Record<string, unknown>).alignment_heads;
    const parsed = parseWhisperManifest(raw);
    expect(parsed.generationConfig.alignmentHeads).toEqual([]);
  });

  it('parses manifest even when artifacts are object-format with externalData (backward compat)', () => {
    // New-style artifacts: { file, externalData? } instead of plain string
    const newStyle = {
      ...tinyManifest,
      artifacts: {
        encoder: { file: 'encoder_model.onnx', externalData: [{ path: './encoder_model.onnx.data', file: 'encoder_model.onnx.data', sizeBytes: 123456 }] },
        decoder_init: { file: 'decoder_init.onnx', externalData: [{ path: './decoder_init.onnx.data', file: 'decoder_init.onnx.data', sizeBytes: 456789 }] },
        decoder_step: { file: 'decoder_step.onnx' },
        decoder_align: { file: 'decoder_align.onnx' },
      },
    };
    const parsed = parseWhisperManifest(newStyle);
    // Model config should still parse correctly
    expect(parsed.modelConfig.decoderLayers).toBe(4);
    expect(parsed.modelConfig.decoderAttentionHeads).toBe(6);
    expect(parsed.modelConfig.dModel).toBe(384);
  });
});

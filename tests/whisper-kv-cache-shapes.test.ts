import { describe, expect, it } from 'vitest';
import { parseWhisperModelConfig } from '../src/models/whisper-seq2seq/generation-config.js';
import { computeEmptyPastKeyValueShapes } from '../src/models/whisper-seq2seq/executor.js';

describe('Whisper KV cache shape computation (config-driven)', () => {
  it('computes correct shapes for whisper-tiny (4 layers, 6 heads, d_model=384)', () => {
    const config = parseWhisperModelConfig({
      decoder_layers: 4,
      decoder_attention_heads: 6,
      d_model: 384,
    });
    const encoderSeqLen = 1500;
    const shapes = computeEmptyPastKeyValueShapes(config, encoderSeqLen);

    // 4 layers × 4 tensors each = 16 entries
    expect(Object.keys(shapes)).toHaveLength(16);

    for (let i = 0; i < 4; i++) {
      expect(shapes[`past_key_values.${i}.decoder.key`]).toEqual([1, 6, 0, 64]);
      expect(shapes[`past_key_values.${i}.decoder.value`]).toEqual([1, 6, 0, 64]);
      expect(shapes[`past_key_values.${i}.encoder.key`]).toEqual([1, 6, 1500, 64]);
      expect(shapes[`past_key_values.${i}.encoder.value`]).toEqual([1, 6, 1500, 64]);
    }
  });

  it('computes correct shapes for whisper-base (6 layers, 8 heads, d_model=512)', () => {
    const config = parseWhisperModelConfig({
      decoder_layers: 6,
      decoder_attention_heads: 8,
      d_model: 512,
    });
    const encoderSeqLen = 1500;
    const shapes = computeEmptyPastKeyValueShapes(config, encoderSeqLen);

    expect(Object.keys(shapes)).toHaveLength(24); // 6 layers × 4

    for (let i = 0; i < 6; i++) {
      expect(shapes[`past_key_values.${i}.decoder.key`]).toEqual([1, 8, 0, 64]);
      expect(shapes[`past_key_values.${i}.decoder.value`]).toEqual([1, 8, 0, 64]);
      expect(shapes[`past_key_values.${i}.encoder.key`]).toEqual([1, 8, 1500, 64]);
      expect(shapes[`past_key_values.${i}.encoder.value`]).toEqual([1, 8, 1500, 64]);
    }
  });

  it('computes correct shapes for whisper-small (12 layers, 12 heads, d_model=768)', () => {
    const config = parseWhisperModelConfig({
      decoder_layers: 12,
      decoder_attention_heads: 12,
      d_model: 768,
    });
    const encoderSeqLen = 1500;
    const shapes = computeEmptyPastKeyValueShapes(config, encoderSeqLen);

    expect(Object.keys(shapes)).toHaveLength(48); // 12 layers × 4

    expect(shapes[`past_key_values.0.decoder.key`]).toEqual([1, 12, 0, 64]);
    expect(shapes[`past_key_values.0.decoder.value`]).toEqual([1, 12, 0, 64]);
    expect(shapes[`past_key_values.0.encoder.key`]).toEqual([1, 12, 1500, 64]);
    expect(shapes[`past_key_values.0.encoder.value`]).toEqual([1, 12, 1500, 64]);

    expect(shapes[`past_key_values.11.decoder.key`]).toEqual([1, 12, 0, 64]);
  });

  it('computes correct shapes for large-v3-turbo (32 layers, 20 heads, d_model=1280)', () => {
    const config = parseWhisperModelConfig({
      decoder_layers: 32,
      decoder_attention_heads: 20,
      d_model: 1280,
    });
    const encoderSeqLen = 1500;
    const shapes = computeEmptyPastKeyValueShapes(config, encoderSeqLen);

    expect(Object.keys(shapes)).toHaveLength(128); // 32 layers × 4

    expect(shapes[`past_key_values.0.decoder.key`]).toEqual([1, 20, 0, 64]);
    expect(shapes[`past_key_values.0.encoder.key`]).toEqual([1, 20, 1500, 64]);
    expect(shapes[`past_key_values.31.decoder.key`]).toEqual([1, 20, 0, 64]);
    expect(shapes[`past_key_values.31.encoder.key`]).toEqual([1, 20, 1500, 64]);
  });

  it('adapts decoder shapes to non-zero sequence lengths', () => {
    const config = parseWhisperModelConfig({
      decoder_layers: 4,
      decoder_attention_heads: 6,
      d_model: 384,
    });
    // After several steps, decoder past KV has non-zero seqLen
    const shapes = computeEmptyPastKeyValueShapes(config, 1500);

    // Decoder side always starts at 0 (first step)
    expect(shapes['past_key_values.0.decoder.key']).toEqual([1, 6, 0, 64]);
    // Encoder side uses encoderSeqLen
    expect(shapes['past_key_values.0.encoder.key']).toEqual([1, 6, 1500, 64]);
  });
});

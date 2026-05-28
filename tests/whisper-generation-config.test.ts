import { describe, expect, it } from 'vitest';
import { parseWhisperGenerationConfig, parseWhisperModelConfig } from '../src/models/whisper-seq2seq/index.js';

describe('Whisper generation config parsing', () => {
  const tinyGenConfig = {
    alignment_heads: [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]],
    no_timestamps_token_id: 50363,
    begin_suppress_tokens: [220, 50257],
    suppress_tokens: [1, 2, 50257],
    max_length: 448,
    bos_token_id: 50257,
    eos_token_id: 50257,
    decoder_start_token_id: 50258,
    is_multilingual: true,
    lang_to_id: { '<|tr|>': 50268, '<|en|>': 50259 },
    task_to_id: { transcribe: 50359, translate: 50358 },
  };

  it('parses alignment_heads into [layer, head] pairs', () => {
    const parsed = parseWhisperGenerationConfig(tinyGenConfig);
    expect(parsed.alignmentHeads).toEqual([
      { layer: 2, head: 2 },
      { layer: 3, head: 0 },
      { layer: 3, head: 2 },
      { layer: 3, head: 3 },
      { layer: 3, head: 4 },
      { layer: 3, head: 5 },
    ]);
  });

  it('exposes noTimestampsTokenId', () => {
    const parsed = parseWhisperGenerationConfig(tinyGenConfig);
    expect(parsed.noTimestampsTokenId).toBe(50363);
  });

  it('handles missing optional fields gracefully', () => {
    const parsed = parseWhisperGenerationConfig({});
    expect(parsed.alignmentHeads).toEqual([]);
    expect(parsed.noTimestampsTokenId).toBeUndefined();
  });

  it('filters out non-pair alignment_heads entries', () => {
    const parsed = parseWhisperGenerationConfig({
      alignment_heads: [[2, 2], [99], [-1, -1, -1], [3, 0]],
    });
    expect(parsed.alignmentHeads).toEqual([
      { layer: 2, head: 2 },
      { layer: 3, head: 0 },
    ]);
  });
});

describe('Whisper model config parsing', () => {
  it('reads median_filter_width from config.json', () => {
    const parsed = parseWhisperModelConfig({ median_filter_width: 7 });
    expect(parsed.medianFilterWidth).toBe(7);
  });

  it('defaults medianFilterWidth to 7 when not present', () => {
    const parsed = parseWhisperModelConfig({});
    expect(parsed.medianFilterWidth).toBe(7);
  });

  it('reads decoder_layers and decoder_attention_heads', () => {
    const parsed = parseWhisperModelConfig({
      decoder_layers: 4,
      decoder_attention_heads: 6,
    });
    expect(parsed.decoderLayers).toBe(4);
    expect(parsed.decoderAttentionHeads).toBe(6);
  });

  it('defaults decoderLayers to 4 when absent', () => {
    const parsed = parseWhisperModelConfig({});
    expect(parsed.decoderLayers).toBe(4);
  });

  it('reads num_mel_bins', () => {
    const tiny = parseWhisperModelConfig({ num_mel_bins: 80 });
    expect(tiny.numMelBins).toBe(80);

    const large = parseWhisperModelConfig({ num_mel_bins: 128 });
    expect(large.numMelBins).toBe(128);
  });
});

import { describe, expect, it } from 'vitest';
import {
  resolveWhisperArtifacts,
  type ResolvedWhisperArtifacts,
} from '../src/models/whisper-seq2seq/ort.js';
import type { WhisperSplitGraphArtifactSource } from '../src/models/whisper-seq2seq/types.js';

const sampleSplitGraphSource: WhisperSplitGraphArtifactSource = {
  kind: 'splitgraph',
  artifacts: {
    encoderUrl: 'https://example.com/models/tiny/encoder_model.onnx',
    decoderInitUrl: 'https://example.com/models/tiny/decoder_init.onnx',
    decoderStepUrl: 'https://example.com/models/tiny/decoder_step.onnx',
    decoderAlignUrl: 'https://example.com/models/tiny/decoder_align.onnx',
    tokenizerUrl: 'https://example.com/models/tiny/tokenizer.json',
    manifestUrl: 'https://example.com/models/tiny/manifest.json',
  },
};

describe('Whisper splitgraph artifact resolution', () => {
  it('resolves splitgraph source with all 4 graph URLs', () => {
    const resolved: ResolvedWhisperArtifacts = resolveWhisperArtifacts(
      sampleSplitGraphSource,
      'wasm',
    );

    // Standard artifacts still present (for backward compat)
    expect(resolved.artifacts.encoderUrl).toBe(sampleSplitGraphSource.artifacts.encoderUrl);
    expect(resolved.artifacts.decoderUrl).toBe(sampleSplitGraphSource.artifacts.decoderInitUrl);
    expect(resolved.artifacts.tokenizerUrl).toBe(sampleSplitGraphSource.artifacts.tokenizerUrl);

    // Split-graph specific URLs
    expect(resolved.decoderInitUrl).toBe(sampleSplitGraphSource.artifacts.decoderInitUrl);
    expect(resolved.decoderStepUrl).toBe(sampleSplitGraphSource.artifacts.decoderStepUrl);
    expect(resolved.decoderAlignUrl).toBe(sampleSplitGraphSource.artifacts.decoderAlignUrl);
    expect(resolved.manifestUrl).toBe(sampleSplitGraphSource.artifacts.manifestUrl);
    expect(resolved.isSplitGraph).toBe(true);
  });

  it('marks merged-decoder sources as non-splitgraph', () => {
    const resolved = resolveWhisperArtifacts(
      {
        kind: 'direct',
        artifacts: {
          encoderUrl: 'https://example.com/encoder.onnx',
          decoderUrl: 'https://example.com/decoder.onnx',
          tokenizerUrl: 'https://example.com/tokenizer.json',
        },
      },
      'wasm',
    );
    expect(resolved.isSplitGraph).toBe(false);
    expect(resolved.decoderInitUrl).toBeUndefined();
    expect(resolved.decoderStepUrl).toBeUndefined();
    expect(resolved.decoderAlignUrl).toBeUndefined();
    expect(resolved.manifestUrl).toBeUndefined();
  });

  it('resolves splitgraph backends correctly', () => {
    const resolved = resolveWhisperArtifacts(sampleSplitGraphSource, 'webgpu');
    expect(resolved.ortBackend).toBe('webgpu');
    expect(resolved.encoderBackendForOrt).toBe('webgpu');
    // decoder defaults to wasm even for webgpu
    expect(resolved.decoderBackendForOrt).toBe('wasm');
  });
});

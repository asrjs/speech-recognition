import { describe, expect, it } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  resolveWhisperArtifacts,
  type ResolvedWhisperArtifacts,
} from '../src/models/whisper-seq2seq/ort.js';
import { loadSplitGraphLocalModel } from '../src/models/whisper-seq2seq/local-file.js';
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

function withExternalData(
  externalDataUrls: NonNullable<WhisperSplitGraphArtifactSource['artifacts']['externalDataUrls']>,
): WhisperSplitGraphArtifactSource {
  return {
    ...sampleSplitGraphSource,
    artifacts: {
      ...sampleSplitGraphSource.artifacts,
      externalDataUrls,
    },
  };
}

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

describe('Whisper splitgraph external data resolution', () => {
  it('does not invent externalData entries when the manifest/source has none', () => {
    const resolved = resolveWhisperArtifacts(sampleSplitGraphSource, 'wasm');

    expect(resolved.externalData).toBeUndefined();
  });

  it('populates externalData only from splitgraph source manifest metadata', () => {
    const resolved = resolveWhisperArtifacts(
      withExternalData({
        encoder: [
          {
            path: './encoder_model.onnx.data',
            file: 'encoder_model.onnx.data',
          },
        ],
        decoder_init: [
          {
            path: './decoder_init.onnx.data',
            file: 'decoder_init.onnx.data',
          },
        ],
      }),
      'wasm',
    );

    expect(resolved.externalData).toBeDefined();
    const ext = resolved.externalData!;
    expect(ext.encoder).toBeDefined();
    expect(ext.encoder![0]!.dataUrl).toBe(
      'https://example.com/models/tiny/encoder_model.onnx.data',
    );
    expect(ext.encoder![0]!.path).toBe('./encoder_model.onnx.data');
    expect(ext.decoder_init).toBeDefined();
    expect(ext.decoder_init![0]!.dataUrl).toBe(
      'https://example.com/models/tiny/decoder_init.onnx.data',
    );
    expect(ext.decoder_init![0]!.path).toBe('./decoder_init.onnx.data');
    expect(ext.decoder_step).toBeUndefined();
    expect(ext.decoder_align).toBeUndefined();
  });

  it('does not include externalData for decoder_align when align URL is absent', () => {
    const sourceNoAlign: WhisperSplitGraphArtifactSource = {
      ...sampleSplitGraphSource,
      artifacts: {
        ...sampleSplitGraphSource.artifacts,
        decoderAlignUrl: undefined,
      },
    };
    const resolved = resolveWhisperArtifacts(sourceNoAlign, 'wasm');

    expect(resolved.externalData?.encoder).toBeUndefined();
    expect(resolved.externalData?.decoder_init).toBeUndefined();
    expect(resolved.externalData?.decoder_step).toBeUndefined();
    expect(resolved.externalData?.decoder_align).toBeUndefined();
  });

  it('externalData is absent for non-splitgraph sources', () => {
    const resolved = resolveWhisperArtifacts(
      {
        kind: 'huggingface',
        repoId: 'openai/whisper-tiny',
      },
      'wasm',
    );
    expect(resolved.externalData).toBeUndefined();
    expect(resolved.isSplitGraph).toBe(false);
  });
});

describe('Browser externalData path matching (ONNX internal location)', () => {
  it('preserves manifest externalData path because ORT matches it against ONNX internal location', () => {
    const resolved = resolveWhisperArtifacts(
      withExternalData({
        encoder: [{ path: './encoder_model.onnx.data', file: 'encoder_model.onnx.data' }],
        decoder_init: [{ path: './decoder_init.onnx.data', file: 'decoder_init.onnx.data' }],
        decoder_step: [{ path: './decoder_step.onnx.data', file: 'decoder_step.onnx.data' }],
        decoder_align: [{ path: './decoder_align.onnx.data', file: 'decoder_align.onnx.data' }],
      }),
      'wasm',
    );

    expect(resolved.externalData?.decoder_init?.[0]?.path).toBe('./decoder_init.onnx.data');
    expect(resolved.externalData?.decoder_step?.[0]?.path).toBe('./decoder_step.onnx.data');
    expect(resolved.externalData?.decoder_align?.[0]?.path).toBe('./decoder_align.onnx.data');
    expect(resolved.externalData?.encoder?.[0]?.path).toBe('./encoder_model.onnx.data');
  });

  it('externalData URL is resolved relative to the graph URL directory', () => {
    const resolved = resolveWhisperArtifacts(
      withExternalData({
        decoder_init: [{ path: './decoder_init.onnx.data', file: 'decoder_init.onnx.data' }],
        decoder_step: [{ path: './nested/decoder_step.onnx.data', file: 'nested/decoder_step.onnx.data' }],
      }),
      'wasm',
    );

    expect(resolved.externalData?.decoder_init?.[0]?.dataUrl).toBe(
      'https://example.com/models/tiny/decoder_init.onnx.data',
    );
    expect(resolved.externalData?.decoder_step?.[0]?.dataUrl).toBe(
      'https://example.com/models/tiny/nested/decoder_step.onnx.data',
    );
  });
});

describe('Local splitgraph manifest external data', () => {
  it('propagates per-graph externalData entries and omits inline graphs', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'asrjs-whisper-manifest-'));
    try {
      for (const name of [
        'encoder_model.onnx',
        'decoder_init.onnx',
        'decoder_step.onnx',
        'decoder_align.onnx',
        'tokenizer.json',
      ]) {
        fs.writeFileSync(path.join(dir, name), name === 'tokenizer.json' ? '{}' : 'onnx');
      }
      fs.writeFileSync(
        path.join(dir, 'manifest.json'),
        JSON.stringify({
          format: 'whisper-browser-self-export-v1',
          model_id: 'openai/whisper-large-v3-turbo',
          decoder_layers: 4,
          decoder_attention_heads: 20,
          d_model: 1280,
          max_source_positions: 1500,
          artifacts: {
            encoder: { file: 'encoder_model.onnx' },
            decoder_init: {
              file: 'decoder_init.onnx',
              externalData: [
                { path: './decoder_init.onnx.data', file: 'decoder_init.onnx.data' },
              ],
            },
            decoder_step: { file: 'decoder_step.onnx' },
            decoder_align: {
              file: 'decoder_align.onnx',
              externalData: [
                { path: './decoder_align.onnx.data', file: 'decoder_align.onnx.data' },
              ],
            },
          },
        }),
      );

      const loaded = loadSplitGraphLocalModel(dir, { variant: null });
      expect(loaded.source.kind).toBe('splitgraph');
      if (loaded.source.kind !== 'splitgraph') throw new Error('expected splitgraph source');
      expect(loaded.config.maxSourcePositions).toBe(3000);
      expect(loaded.source.artifacts.externalDataUrls?.encoder).toBeUndefined();
      expect(loaded.source.artifacts.externalDataUrls?.decoder_step).toBeUndefined();
      expect(loaded.source.artifacts.externalDataUrls?.decoder_init?.[0]?.path).toBe('./decoder_init.onnx.data');

      const resolved = resolveWhisperArtifacts(loaded.source, 'webgpu');
      expect(resolved.externalData?.encoder).toBeUndefined();
      expect(resolved.externalData?.decoder_step).toBeUndefined();
      expect(resolved.externalData?.decoder_init?.[0]?.dataUrl).toBe(
        `file://${path.join(dir, 'decoder_init.onnx.data')}`,
      );
      expect(resolved.externalData?.decoder_align?.[0]?.dataUrl).toBe(
        `file://${path.join(dir, 'decoder_align.onnx.data')}`,
      );
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});

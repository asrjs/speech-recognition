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

describe('Whisper splitgraph external data resolution', () => {
  it('populates externalData for all 4 graphs from splitgraph source', () => {
    const resolved = resolveWhisperArtifacts(sampleSplitGraphSource, 'wasm');

    expect(resolved.externalData).toBeDefined();
    const ext = resolved.externalData!;

    // encoder
    expect(ext.encoder).toBeDefined();
    expect(ext.encoder![0]!.dataUrl).toBe(
      'https://example.com/models/tiny/encoder_model.onnx.data',
    );
    expect(ext.encoder![0]!.path).toBe('encoder_model.onnx.data');

    // decoder_init
    expect(ext.decoder_init).toBeDefined();
    expect(ext.decoder_init![0]!.dataUrl).toBe(
      'https://example.com/models/tiny/decoder_init.onnx.data',
    );
    expect(ext.decoder_init![0]!.path).toBe('decoder_init.onnx.data');

    // decoder_step
    expect(ext.decoder_step).toBeDefined();
    expect(ext.decoder_step![0]!.dataUrl).toBe(
      'https://example.com/models/tiny/decoder_step.onnx.data',
    );
    expect(ext.decoder_step![0]!.path).toBe('decoder_step.onnx.data');

    // decoder_align
    expect(ext.decoder_align).toBeDefined();
    expect(ext.decoder_align![0]!.dataUrl).toBe(
      'https://example.com/models/tiny/decoder_align.onnx.data',
    );
    expect(ext.decoder_align![0]!.path).toBe('decoder_align.onnx.data');
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

    expect(resolved.externalData?.encoder).toBeDefined();
    expect(resolved.externalData?.decoder_init).toBeDefined();
    expect(resolved.externalData?.decoder_step).toBeDefined();
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
  it('decoder externalData path matches ONNX internal .data file name', () => {
    // For consolidated external data (save_as_external_data with
    // all_tensors_to_one_file=True), the ONNX internal location is
    // "<graph>.onnx.data" and the path in sessionOptions.externalData
    // must match exactly. ORT uses this to map the data URL to the
    // internal reference.
    const resolved = resolveWhisperArtifacts(sampleSplitGraphSource, 'wasm');

    // decoder_init: ONNX internal location is "decoder_init.onnx.data"
    expect(resolved.externalData?.decoder_init?.[0]?.path).toBe('decoder_init.onnx.data');

    // decoder_step: ONNX internal location is "decoder_step.onnx.data"
    expect(resolved.externalData?.decoder_step?.[0]?.path).toBe('decoder_step.onnx.data');

    // decoder_align: ONNX internal location is "decoder_align.onnx.data"
    expect(resolved.externalData?.decoder_align?.[0]?.path).toBe('decoder_align.onnx.data');

    // encoder: ONNX internal location is "encoder_model.onnx.data"
    expect(resolved.externalData?.encoder?.[0]?.path).toBe('encoder_model.onnx.data');
  });

  it('externalData URL is derived from graph URL (not a random string)', () => {
    const resolved = resolveWhisperArtifacts(sampleSplitGraphSource, 'wasm');

    // The data URL should be graph_url + ".data"
    expect(resolved.externalData?.decoder_init?.[0]?.dataUrl).toBe(
      'https://example.com/models/tiny/decoder_init.onnx.data',
    );
    expect(resolved.externalData?.decoder_step?.[0]?.dataUrl).toBe(
      'https://example.com/models/tiny/decoder_step.onnx.data',
    );

    // Verify the data URL matches the path: path = basename of dataUrl
    const initDataUrl = resolved.externalData?.decoder_init?.[0]?.dataUrl ?? '';
    const initPath = resolved.externalData?.decoder_init?.[0]?.path ?? '';
    expect(initDataUrl.endsWith(initPath)).toBe(true);
  });
});

import { describe, expect, it } from 'vitest';
import { resolveWhisperArtifacts } from '../src/models/whisper-seq2seq/ort.js';
import { resolveWhisperArtifactSource } from '../src/presets/whisper/manifest.js';
import { WhisperOnnxExecutor } from '../src/models/whisper-seq2seq/executor.js';
import type { AssetProvider, ResolvedAssetHandle } from '../src/types/index.js';

describe('Whisper Hugging Face artifact layout', () => {
  it('keeps the ONNX subfolder and quantization filenames for the default Base preset', () => {
    const source = resolveWhisperArtifactSource('whisper-base');
    expect(source?.kind).toBe('huggingface');
    if (source?.kind !== 'huggingface') throw new Error('Expected a Hugging Face source');

    const resolved = resolveWhisperArtifacts(source, 'wasm');
    expect(resolved.artifacts).toMatchObject({
      encoderUrl: 'https://huggingface.co/onnx-community/whisper-base_timestamped/resolve/main/onnx/encoder_model_int8.onnx',
      decoderUrl: 'https://huggingface.co/onnx-community/whisper-base_timestamped/resolve/main/onnx/decoder_model_merged_int8.onnx',
      tokenizerUrl: 'https://huggingface.co/onnx-community/whisper-base_timestamped/resolve/main/tokenizer.json',
    });
  });

  it('resolves the WebGPU encoder and WASM decoder variants in the same ONNX subfolder', () => {
    const resolved = resolveWhisperArtifacts({
      kind: 'huggingface',
      repoId: 'onnx-community/whisper-base_timestamped',
      artifactSubfolder: 'onnx',
      encoderBackend: 'webgpu',
      decoderBackend: 'wasm',
    }, 'webgpu');

    expect(resolved.artifacts.encoderUrl).toBe(
      'https://huggingface.co/onnx-community/whisper-base_timestamped/resolve/main/onnx/encoder_model_fp16.onnx',
    );
    expect(resolved.artifacts.decoderUrl).toBe(
      'https://huggingface.co/onnx-community/whisper-base_timestamped/resolve/main/onnx/decoder_model_merged_int8.onnx',
    );
  });

  it('keeps the existing ONNX folder default for unannotated Hugging Face sources', () => {
    const resolved = resolveWhisperArtifacts({
      kind: 'huggingface',
      repoId: 'example/custom-whisper',
    }, 'wasm');

    expect(resolved.artifacts.encoderUrl).toBe(
      'https://huggingface.co/example/custom-whisper/resolve/main/onnx/encoder_model_int8.onnx',
    );
    expect(resolved.artifacts.decoderUrl).toBe(
      'https://huggingface.co/example/custom-whisper/resolve/main/onnx/decoder_model_merged_int8.onnx',
    );
  });

  it('supports an explicit repository-root layout for custom Hugging Face sources', () => {
    const resolved = resolveWhisperArtifacts({
      kind: 'huggingface',
      repoId: 'example/root-whisper',
      artifactSubfolder: '',
    }, 'wasm');

    expect(resolved.artifacts.encoderUrl).toBe(
      'https://huggingface.co/example/root-whisper/resolve/main/encoder_model_int8.onnx',
    );
    expect(resolved.artifacts.decoderUrl).toBe(
      'https://huggingface.co/example/root-whisper/resolve/main/decoder_model_merged_int8.onnx',
    );
  });

  it('preserves repo-relative graph paths when materializing through the asset provider', async () => {
    const source = resolveWhisperArtifactSource('whisper-base');
    if (source?.kind !== 'huggingface') throw new Error('Expected a Hugging Face source');
    const requested: string[] = [];
    const provider: AssetProvider = {
      canResolve: () => true,
      async resolve(request) {
        requested.push(request.filename ?? '');
        return {
          request,
          async *openStream() { yield new Uint8Array(); },
          async readBytes() { return new Uint8Array(); },
          async readText() { return ''; },
          async readJson<T>() { return {} as T; },
          async getLocator() { return `blob:test/${request.filename}`; },
          dispose() {},
        } satisfies ResolvedAssetHandle;
      },
    };
    const executor = Object.create(WhisperOnnxExecutor.prototype) as WhisperOnnxExecutor;
    const internals = executor as unknown as {
      sourceOptions: typeof source;
      assetProvider: AssetProvider;
      assetHandles: ResolvedAssetHandle[];
      modelId: string;
      materializeResolvedArtifacts(
        resolved: ReturnType<typeof resolveWhisperArtifacts>,
      ): Promise<ReturnType<typeof resolveWhisperArtifacts>>;
    };
    Object.assign(internals, {
      sourceOptions: source,
      assetProvider: provider,
      assetHandles: [],
      modelId: 'whisper-base',
    });

    const resolved = await internals.materializeResolvedArtifacts(resolveWhisperArtifacts(source, 'wasm'));

    expect(requested).toEqual([
      'onnx/encoder_model_int8.onnx',
      'onnx/decoder_model_merged_int8.onnx',
    ]);
    expect(resolved.artifacts.encoderUrl).toBe('blob:test/onnx/encoder_model_int8.onnx');
    expect(resolved.artifacts.decoderUrl).toBe('blob:test/onnx/decoder_model_merged_int8.onnx');
  });
});

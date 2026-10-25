import { describe, expect, it } from 'vitest';
import { whisperBuiltInLocalModelAdapter } from '../src/presets/whisper/local-adapter.js';

const WHISPER_4GRAPH_FILES = [
  'encoder_model.onnx',
  'decoder_init.onnx',
  'decoder_step.onnx',
  'decoder_align.onnx',
  'tokenizer.json',
  'manifest.json',
  'config.json',
  'generation_config.json',
  'encoder_model.onnx.data',
  'decoder_init.onnx.data',
  'decoder_step.onnx.data',
  'decoder_align.onnx.data',
] as const;

function whisperEntries(root = 'bundle', inlineEncoder = false, decoderLayers = 4) {
  const manifest = JSON.stringify({
    format: 'whisper-browser-self-export-v1',
    num_mel_bins: 128,
    max_source_positions: 1500,
    max_target_positions: 448,
    d_model: 1280,
    decoder_attention_heads: 20,
    head_dim: 64,
    decoder_layers: decoderLayers,
    vocab_size: 51866,
    alignment_export: {
      causal_self_attention: true,
      encoder_hidden_state_dtype: 'float16',
      attention_values: 'logits',
      attention_layout: 'selected_heads',
    },
    artifacts: {
      encoder: {
        file: 'encoder_model.onnx',
        externalData: inlineEncoder
          ? []
          : [{ path: './encoder_model.onnx.data', file: 'encoder_model.onnx.data' }],
      },
      decoder_init: {
        file: 'decoder_init.onnx',
        externalData: [{ path: './decoder_init.onnx.data', file: 'decoder_init.onnx.data' }],
      },
      decoder_step: {
        file: 'decoder_step.onnx',
        externalData: [{ path: './decoder_step.onnx.data', file: 'decoder_step.onnx.data' }],
      },
      decoder_align: {
        file: 'decoder_align.onnx',
        externalData: [{ path: './decoder_align.onnx.data', file: 'decoder_align.onnx.data' }],
      },
    },
  });
  const config = JSON.stringify({
    model_type: 'whisper',
    num_mel_bins: 128,
    max_source_positions: 1500,
    max_target_positions: 448,
    d_model: 1280,
    decoder_attention_heads: 20,
    decoder_layers: decoderLayers,
    head_dim: 64,
    vocab_size: 51866,
  });
  return WHISPER_4GRAPH_FILES.filter(
    (name) => !(inlineEncoder && name === 'encoder_model.onnx.data'),
  ).map((name) => ({
    path: `${root}/${name}`,
    basename: name,
    file: new File(
      [name === 'manifest.json' ? manifest : name === 'config.json' ? config : name],
      name,
    ),
  }));
}

const MODEL_ID = 'ysdede/whisper-large-v3-turbo-onnx-4graph';

describe('Whisper Turbo local adapter', () => {
  it('resolves original Large v3 geometry and keeps it distinct from Turbo', async () => {
    const entries = whisperEntries('large-v3', false, 32);
    const resolved = await whisperBuiltInLocalModelAdapter.resolveEntries({
      modelId: 'openai/whisper-large-v3',
      entries,
      encoderQuant: 'fp16',
      decoderQuant: 'fp16',
    } as never);
    try {
      expect(resolved.modelId).toBe('openai/whisper-large-v3');
      expect((resolved.builtInLoadOptions as { source: any }).source.kind).toBe('splitgraph');
    } finally {
      await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
    }
    await expect(
      whisperBuiltInLocalModelAdapter.resolveEntries({ modelId: MODEL_ID, entries } as never),
    ).rejects.toThrow(/does not match the registered preset configuration/);
  });

  it('resolves the graph set, configs, tokenizer, manifest, and declared sidecars', async () => {
    const entries = whisperEntries();
    expect(whisperBuiltInLocalModelAdapter.inspectEntries(entries)).toEqual({
      encoderQuantizations: ['fp16'],
      decoderQuantizations: ['fp16'],
      tokenizerNames: ['tokenizer.json'],
      preprocessorNames: [],
    });

    const resolved = await whisperBuiltInLocalModelAdapter.resolveEntries({
      modelId: MODEL_ID,
      entries,
      encoderQuant: 'fp16',
      decoderQuant: 'fp16',
    } as never);
    try {
      expect(resolved.selection).toMatchObject({
        encoderName: 'encoder_model.onnx',
        decoderName: 'decoder_step.onnx',
        tokenizerName: 'tokenizer.json',
        encoderQuant: 'fp16',
        decoderQuant: 'fp16',
      });
      const source = (resolved.builtInLoadOptions as { source: any }).source;
      expect(source).toMatchObject({
        kind: 'splitgraph',
        encoderBackend: 'webgpu',
        decoderBackend: 'webgpu',
        artifacts: {
          encoderUrl: expect.stringMatching(/^blob:/),
          decoderInitUrl: expect.stringMatching(/^blob:/),
          decoderStepUrl: expect.stringMatching(/^blob:/),
          decoderAlignUrl: expect.stringMatching(/^blob:/),
          tokenizerUrl: expect.stringMatching(/^blob:/),
          manifestUrl: expect.stringMatching(/^blob:/),
          configUrl: expect.stringMatching(/^blob:/),
          generationConfigUrl: expect.stringMatching(/^blob:/),
        },
      });
      expect(Object.values(source.artifacts.externalDataUrls)).toHaveLength(4);
      for (const sidecar of Object.values(source.artifacts.externalDataUrls)) {
        expect(sidecar[0]).toMatchObject({ file: expect.stringMatching(/^blob:/) });
      }
    } finally {
      await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
    }
  });

  it('accepts an inline encoder when the manifest declares no encoder sidecar', async () => {
    const resolved = await whisperBuiltInLocalModelAdapter.resolveEntries({
      modelId: MODEL_ID,
      entries: whisperEntries('inline', true),
    } as never);
    try {
      const artifacts = (resolved.builtInLoadOptions as { source: any }).source.artifacts;
      expect(artifacts.externalDataUrls).not.toHaveProperty('encoder');
      expect(Object.keys(artifacts.externalDataUrls)).toEqual([
        'decoder_init',
        'decoder_step',
        'decoder_align',
      ]);
    } finally {
      await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
    }
  });

  it('rejects missing declared sidecars, wrong dtype, duplicate bundles, and other model IDs', async () => {
    const missingSidecar = whisperEntries().filter(
      (entry) => entry.basename !== 'decoder_step.onnx.data',
    );
    await expect(
      whisperBuiltInLocalModelAdapter.resolveEntries({
        modelId: MODEL_ID,
        entries: missingSidecar,
      } as never),
    ).rejects.toThrow(/Missing declared external data for decoder_step/);

    const entries = whisperEntries();
    const manifestEntry = entries.find((entry) => entry.basename === 'manifest.json')!;
    const wrongManifestText = JSON.stringify({
      ...JSON.parse(await manifestEntry.file!.text()),
      alignment_export: {
        causal_self_attention: true,
        encoder_hidden_state_dtype: 'float32',
      },
    });
    const wrongDtype = entries.map((entry) =>
      entry.basename === 'manifest.json'
        ? { ...entry, file: new File([wrongManifestText], 'manifest.json') }
        : entry,
    );
    await expect(
      whisperBuiltInLocalModelAdapter.resolveEntries({
        modelId: MODEL_ID,
        entries: wrongDtype,
      } as never),
    ).rejects.toThrow(/must declare float16 encoder hidden states/);

    await expect(
      whisperBuiltInLocalModelAdapter.resolveEntries({
        modelId: MODEL_ID,
        entries: [...whisperEntries('first'), ...whisperEntries('second')],
      } as never),
    ).rejects.toThrow(/Multiple complete Whisper 4-graph bundles/);
    await expect(
      whisperBuiltInLocalModelAdapter.resolveEntries({
        modelId: 'openai/whisper-large-v3',
        entries,
      } as never),
    ).rejects.toThrow(/does not match the registered preset configuration/);
    await expect(
      whisperBuiltInLocalModelAdapter.resolveEntries({
        modelId: 'openai/whisper-base',
        entries,
      } as never),
    ).rejects.toThrow(/does not match the registered preset configuration/);
  });
});

import { describe, expect, it } from 'vitest';
import { medAsrBuiltInLocalModelAdapter } from '../src/presets/medasr/local-adapter.js';
import {
  listBuiltInLocalModelAdapters,
  resolveBuiltInLocalModelAdapter,
} from '../src/runtime/local-adapter-registry.js';
import type { SpeechModelLocalEntry } from '../src/runtime/local-types.js';
import { resolveLasrCtcArtifacts } from '../src/models/lasr-ctc/ort.js';

function entries(folder = 'medasr') {
  const tokenLines = Array.from(
    { length: 513 },
    (_, id) => `${id === 0 ? '<epsilon>' : `p${id}`} ${id}`,
  );
  return [
    ['model.onnx', 'graph'],
    ['model.onnx.data', 'external weights'],
    ['tokens.txt', tokenLines.join('\n')],
    ['config.json', JSON.stringify({ model_type: 'lasr_ctc', vocab_size: 512 })],
    [
      'preprocessor_config.json',
      JSON.stringify({
        feature_size: 128,
        sampling_rate: 16000,
        n_fft: 512,
        win_length: 400,
        hop_length: 160,
      }),
    ],
  ].map(([name, contents]) => ({
    path: `${folder}/${name}`,
    basename: name,
    file: new File([contents], name),
  }));
}

describe('MedASR local model adapter', () => {
  it('registers local loading for the canonical MedASR model', () => {
    expect(listBuiltInLocalModelAdapters()).toContain(medAsrBuiltInLocalModelAdapter);
    expect(resolveBuiltInLocalModelAdapter('google/medasr')).toEqual({
      modelId: 'google/medasr',
      preset: 'medasr',
      adapter: medAsrBuiltInLocalModelAdapter,
    });
    expect(medAsrBuiltInLocalModelAdapter.inspectEntries(entries())).toMatchObject({
      encoderQuantizations: ['fp32'],
      decoderQuantizations: ['fp32'],
      tokenizerNames: ['tokens.txt'],
    });
  });

  it('resolves the matching FP32 graph, tokenizer, and external data', async () => {
    const resolved = await medAsrBuiltInLocalModelAdapter.resolveEntries({
      modelId: 'google/medasr',
      entries: entries(),
    });

    expect(resolved.builtInLoadOptions).toMatchObject({
      source: {
        kind: 'direct',
        artifacts: {
          modelDataFilename: 'model.onnx.data',
        },
      },
    });
    const loadOptions = resolved.builtInLoadOptions as {
      readonly source: Parameters<typeof resolveLasrCtcArtifacts>[0];
    };
    expect(resolveLasrCtcArtifacts(loadOptions.source, 'wasm').artifacts.modelDataFilename).toBe(
      'model.onnx.data',
    );
    expect(resolved.assetHandles).toHaveLength(3);
    expect(resolved.selection).toMatchObject({
      encoderName: 'model.onnx',
      tokenizerName: 'tokens.txt',
      encoderQuant: 'fp32',
      decoderQuant: 'fp32',
    });
    await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
  });

  it('rejects duplicate, missing, or cross-folder artifacts instead of guessing', async () => {
    const valid = entries() as SpeechModelLocalEntry[];
    await expect(
      medAsrBuiltInLocalModelAdapter.resolveEntries({
        modelId: 'google/medasr',
        entries: [...valid, { ...valid[0]!, path: 'other/model.onnx' }],
      }),
    ).rejects.toThrow('Multiple local MedASR artifacts named model.onnx');
    await expect(
      medAsrBuiltInLocalModelAdapter.resolveEntries({
        modelId: 'google/medasr',
        entries: valid.filter((entry) => entry.basename !== 'model.onnx.data'),
      }),
    ).rejects.toThrow('Missing MedASR external graph data');
    await expect(
      medAsrBuiltInLocalModelAdapter.resolveEntries({
        modelId: 'google/medasr',
        entries: [
          ...valid.filter((entry) => entry.basename !== 'tokens.txt'),
          {
            path: 'other/tokens.txt',
            basename: 'tokens.txt',
            file: new File(['tokens'], 'tokens.txt'),
          },
        ],
      }),
    ).rejects.toThrow('MedASR model.onnx and tokens.txt must be in the same folder');
  });

  it('validates the source config, frontend, token IDs, model ID, and FP32-only selection before allocating assets', async () => {
    const valid = entries() as SpeechModelLocalEntry[];
    const replace = (basename: string, contents: string): SpeechModelLocalEntry[] =>
      valid.map((entry) =>
        entry.basename === basename ? { ...entry, file: new File([contents], basename) } : entry,
      );

    await expect(
      medAsrBuiltInLocalModelAdapter.resolveEntries({
        modelId: 'unknown/medasr',
        entries: valid,
      }),
    ).rejects.toThrow('Unsupported MedASR model');
    await expect(
      medAsrBuiltInLocalModelAdapter.resolveEntries({
        modelId: 'google/medasr',
        entries: valid,
        encoderQuant: 'int8',
      }),
    ).rejects.toThrow('only FP32');
    await expect(
      medAsrBuiltInLocalModelAdapter.resolveEntries({
        modelId: 'google/medasr',
        entries: replace('config.json', JSON.stringify({ model_type: 'lasr_ctc', vocab_size: 32 })),
      }),
    ).rejects.toThrow('vocab_size 512');
    await expect(
      medAsrBuiltInLocalModelAdapter.resolveEntries({
        modelId: 'google/medasr',
        entries: replace(
          'preprocessor_config.json',
          JSON.stringify({
            feature_size: 80,
            sampling_rate: 16000,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
          }),
        ),
      }),
    ).rejects.toThrow('128 bins');
    await expect(
      medAsrBuiltInLocalModelAdapter.resolveEntries({
        modelId: 'google/medasr',
        entries: replace('tokens.txt', '<epsilon> 0\np1 2\n'),
      }),
    ).rejects.toThrow('contiguous IDs');
  });
});

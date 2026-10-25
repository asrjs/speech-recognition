import { describe, expect, it } from 'vitest';
import { gigaAmCtcBuiltInLocalModelAdapter } from '../src/presets/gigaam-ctc-local-adapter.js';
import { gigaAmRnntBuiltInLocalModelAdapter } from '../src/presets/gigaam-rnnt-local-adapter.js';
import { senseVoiceBuiltInLocalModelAdapter } from '../src/presets/sensevoice-local-adapter.js';
import { xAsrBuiltInLocalModelAdapter } from '../src/presets/x-asr-local-adapter.js';
import { listBuiltInLocalModelAdapters } from '../src/runtime/local-adapter-registry.js';
import type { SpeechModelLocalEntry } from '../src/runtime/local-types.js';

function entries(names: readonly string[]): SpeechModelLocalEntry[] {
  return names.map((name) => ({
    path: `model/${name}`,
    basename: name,
    file: new File([`fixture:${name}`], name),
  }));
}

describe('official experimental local model adapters', () => {
  it('registers official CTC artifact sets and resolves direct sources', async () => {
    expect(listBuiltInLocalModelAdapters()).toEqual(expect.arrayContaining([
      gigaAmCtcBuiltInLocalModelAdapter, senseVoiceBuiltInLocalModelAdapter,
    ]));
    const giga = await gigaAmCtcBuiltInLocalModelAdapter.resolveEntries({
      modelId: 'gigaam-multilingual-ctc',
      entries: entries(['multilingual_ctc.onnx', 'multilingual_vocab.txt']),
      encoderBackend: 'webgpu',
    });
    expect(giga.builtInLoadOptions).toMatchObject({
      source: { kind: 'direct', artifacts: { modelUrl: expect.stringMatching(/^blob:/), tokenizerUrl: expect.stringMatching(/^blob:/) } },
    });
    expect(giga.selection).toMatchObject({ encoderQuant: 'fp32', decoderQuant: 'fp32' });
    await Promise.all(giga.assetHandles.map((handle) => handle.dispose()));

    const sense = await senseVoiceBuiltInLocalModelAdapter.resolveEntries({
      modelId: 'FunAudioLLM/SenseVoiceSmall',
      entries: entries(['model.onnx', 'vocab.txt', 'am.mvn']),
      enableProfiling: true,
    });
    expect(sense.builtInLoadOptions).toMatchObject({
      source: { kind: 'direct', enableProfiling: true, artifacts: {
        modelUrl: expect.stringMatching(/^blob:/),
        tokenizerUrl: expect.stringMatching(/^blob:/),
        cmvnUrl: expect.stringMatching(/^blob:/),
      } },
    });
    await Promise.all(sense.assetHandles.map((handle) => handle.dispose()));
  });

  it('resolves all four Russian GigaAM RNN-T direct graphs and keeps component selection explicit', async () => {
    expect(listBuiltInLocalModelAdapters()).toContain(gigaAmRnntBuiltInLocalModelAdapter);
    const resolved = await gigaAmRnntBuiltInLocalModelAdapter.resolveEntries({
      modelId: 'gigaam-v3-e2e-rnnt',
      entries: entries([
        'v3_e2e_rnnt_encoder.onnx', 'v3_e2e_rnnt_decoder.onnx',
        'v3_e2e_rnnt_joint.onnx', 'v3_e2e_rnnt_vocab.txt',
      ]),
      encoderBackend: 'webgpu',
      decoderBackend: 'wasm',
    });
    expect(resolved.builtInLoadOptions).toMatchObject({
      source: {
        kind: 'direct', encoderBackend: 'webgpu', decoderBackend: 'wasm', jointBackend: 'wasm',
        artifacts: {
          encoderUrl: expect.stringMatching(/^blob:/),
          decoderUrl: expect.stringMatching(/^blob:/),
          jointUrl: expect.stringMatching(/^blob:/),
          tokenizerUrl: expect.stringMatching(/^blob:/),
        },
      },
    });
    await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
  });

  it('resolves X-ASR 160 ms streaming graphs through a direct local source', async () => {
    expect(listBuiltInLocalModelAdapters()).toContain(xAsrBuiltInLocalModelAdapter);
    const resolved = await xAsrBuiltInLocalModelAdapter.resolveEntries({
      modelId: 'x-asr-zh-en-160ms',
      entries: entries(['encoder-160ms.onnx', 'decoder-160ms.onnx', 'joiner-160ms.onnx', 'tokens.txt']),
      enableProfiling: true,
    });
    expect(resolved.builtInLoadOptions).toMatchObject({
      source: {
        kind: 'direct',
        enableProfiling: true,
        artifacts: {
          encoderUrl: expect.stringMatching(/^blob:/),
          decoderUrl: expect.stringMatching(/^blob:/),
          joinerUrl: expect.stringMatching(/^blob:/),
          tokenizerUrl: expect.stringMatching(/^blob:/),
        },
      },
      config: { chunkMs: 160 },
    });
    await Promise.all(resolved.assetHandles.map((handle) => handle.dispose()));
  });

  it('rejects incomplete and duplicate local graph sets before loading', async () => {
    expect(gigaAmCtcBuiltInLocalModelAdapter.inspectEntries(entries(['multilingual_ctc.onnx'])).encoderQuantizations).toEqual([]);
    await expect(gigaAmRnntBuiltInLocalModelAdapter.resolveEntries({
      modelId: 'gigaam-v3-e2e-rnnt', entries: entries(['v3_e2e_rnnt_encoder.onnx']),
    })).rejects.toThrow('Missing local artifact');
    await expect(senseVoiceBuiltInLocalModelAdapter.resolveEntries({
      modelId: 'FunAudioLLM/SenseVoiceSmall',
      entries: [...entries(['model.onnx', 'vocab.txt', 'am.mvn']), ...entries(['model.onnx'])],
    })).rejects.toThrow('Multiple local artifacts named model.onnx');
  });
});

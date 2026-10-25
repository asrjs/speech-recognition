import { beforeEach, describe, expect, it, vi } from 'vitest';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

const state = vi.hoisted(() => ({ captured: [] as Float32Array[] }));

vi.mock('../src/models/lasr-ctc/ort.js', () => ({
  initOrt: vi.fn(async () => ({
    Tensor: class {
      constructor(
        readonly type: string,
        readonly data: Float32Array,
        readonly dims: number[],
      ) {}
      dispose() {}
    },
  })),
  createOrtSession: vi.fn(async () => ({
    run: vi.fn(async (feeds: Record<string, { data: Float32Array }>) => {
      state.captured.push(feeds.x!.data.slice());
      throw new Error('stop after capturing model input');
    }),
  })),
  disposeOrtOutputs: vi.fn(),
  releaseOrtSession: vi.fn(),
}));

vi.mock('../src/models/omnilingual-ctc/tokenizer.js', () => ({
  OmnilingualCtcTokenizer: {
    fromUrl: vi.fn(async () => ({ vocabSize: 2, blankId: 0 })),
  },
}));

import { loadOmnilingualCtc } from '../src/models/omnilingual-ctc/model.js';
import { normalizeWav2Vec2Waveform } from '../src/models/wav2vec2/executor.js';

function readJfkPcm(): { bytes: Buffer; pcm: Float32Array } {
  const bytes = readFileSync(new URL('./fixtures/jfk2.en.wav', import.meta.url));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let format: { encoding: number; channels: number; sampleRate: number; bits: number } | undefined;
  let dataOffset = -1;
  let dataLength = 0;
  for (let offset = 12; offset + 8 <= bytes.byteLength; ) {
    const name = bytes.toString('ascii', offset, offset + 4);
    const size = view.getUint32(offset + 4, true);
    if (name === 'fmt ') {
      format = {
        encoding: view.getUint16(offset + 8, true),
        channels: view.getUint16(offset + 10, true),
        sampleRate: view.getUint32(offset + 12, true),
        bits: view.getUint16(offset + 22, true),
      };
    }
    if (name === 'data') {
      dataOffset = offset + 8;
      dataLength = size;
      break;
    }
    offset += 8 + size + (size % 2);
  }
  expect(format).toEqual({ encoding: 1, channels: 1, sampleRate: 16000, bits: 16 });
  expect(dataOffset).toBeGreaterThanOrEqual(0);
  expect(dataLength % 2).toBe(0);
  const pcm = new Float32Array(dataLength / 2);
  for (let index = 0; index < pcm.length; index += 1) {
    pcm[index] = view.getInt16(dataOffset + index * 2, true) / 32768;
  }
  return { bytes, pcm };
}

describe('Omnilingual CTC waveform normalization', () => {
  beforeEach(() => {
    state.captured = [];
  });

  it('matches the official Meta v2 JFK preprocessor summary at epsilon 1e-5', () => {
    const { bytes, pcm } = readJfkPcm();
    expect(
      createHash('sha256').update(bytes).digest('hex'),
    ).toBe('59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e');

    const normalized = normalizeWav2Vec2Waveform(pcm, 1e-5);
    let sum = 0;
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (const value of normalized) {
      sum += value;
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
    const mean = sum / normalized.length;
    let variance = 0;
    for (const value of normalized) variance += (value - mean) ** 2;
    const populationStd = Math.sqrt(variance / normalized.length);

    // Frozen official Meta v2 native CPU preprocessing receipt, not an ASR oracle.
    expect(normalized).toHaveLength(176000);
    expect(Math.abs(mean - 7.521022027390245e-9)).toBeLessThan(1e-7);
    expect(Math.abs(populationStd - 0.9997525215148926)).toBeLessThan(1e-7);
    expect(Math.abs(min - -5.090773105621338)).toBeLessThan(1e-6);
    expect(Math.abs(max - 5.506680011749268)).toBeLessThan(1e-6);
  });

  it('uses the configured epsilon before the raw-waveform graph', async () => {
    const pcm = new Float32Array([0, 0.25, 0.5, 0.75]);
    const model = await loadOmnilingualCtc({ modelUrl: 'model', tokensUrl: 'tokens' });

    await expect(model.transcribe(pcm)).rejects.toThrow('stop after capturing model input');

    expect(state.captured).toHaveLength(1);
    const expected = new Float32Array([
      -1.3415549,
      -0.447185,
      0.447185,
      1.3415549,
    ]);
    state.captured[0]!.forEach((value, index) => {
      expect(value).toBeCloseTo(expected[index]!, 6);
    });
  });

  it('allows an artifact to override its normalization epsilon', async () => {
    const pcm = new Float32Array([0, 0.25, 0.5, 0.75]);
    const model = await loadOmnilingualCtc({
      modelUrl: 'model',
      tokensUrl: 'tokens',
      normalizationEpsilon: 1e-4,
    });

    await expect(model.transcribe(pcm)).rejects.toThrow('stop after capturing model input');

    expect(state.captured).toHaveLength(1);
    expect(state.captured[0]![0]).toBeCloseTo(-1.340783, 6);
  });
});

import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import {
  describeModelInputPcm,
  prepareInspectorAudio,
} from '../tools/model-debugging/scripts/node-asrjs-nemo-inspect.mjs';

const pcm = [0, 0.5, -1, 0.25];
const sha256 = createHash('sha256')
  .update(Buffer.from('000000000000003f000080bf0000803e', 'hex'))
  .digest('hex');

function source(sampleRate = 16000) {
  return { audio: Float32Array.from(pcm), sampleRate, numberOfChannels: 1 };
}

function oracle() {
  return {
    sample: {
      model_input: {
        boundary: 'preprocessor_input_signal',
        sample_rate: 16000,
        channels: 1,
        dtype: 'float32',
        encoding: 'f32le',
        shape: [1, 4],
        samples: 4,
        valid_samples: 4,
        right_padding_samples: 0,
        sha256,
      },
      stages: {
        waveform: { shape: [1, 4], dtype: 'float32', data: [...pcm] },
        waveform_length: { shape: [1], dtype: 'int64', data: [4] },
      },
    },
  };
}

describe('NeMo inspector effective PCM contract (not ASR parity)', () => {
  it('hashes exact float32 little-endian samples including typed-array view offsets', () => {
    const padded = Float32Array.from([99, ...pcm, 99]);
    expect(describeModelInputPcm(padded.subarray(1, 5), 16000)).toEqual({
      sample_rate: 16000,
      channels: 1,
      dtype: 'float32',
      encoding: 'f32le',
      shape: [1, 4],
      samples: 4,
      valid_samples: 4,
      right_padding_samples: 0,
      sha256,
    });
  });

  it('covers samples after the bounded hash buffer, not just a prefix', () => {
    const samples = new Float32Array(20000);
    const before = describeModelInputPcm(samples, 16000);
    samples[19999] = 0.25;
    const expectedBytes = Buffer.alloc(samples.length * 4);
    expectedBytes.writeFloatLE(0.25, 19999 * 4);
    const after = describeModelInputPcm(samples, 16000);
    expect(after.sha256).not.toBe(before.sha256);
    expect(after.sha256).toBe(createHash('sha256').update(expectedBytes).digest('hex'));
  });

  it('rejects non-float32, empty, nonfinite and invalid-rate inputs', () => {
    for (const samples of [[], new Float64Array(pcm), new Float32Array(), Float32Array.of(NaN)]) {
      expect(() => describeModelInputPcm(samples, 16000)).toThrow('model-input');
    }
    for (const rate of [0, NaN, 16000.5]) {
      expect(() => describeModelInputPcm(Float32Array.from(pcm), rate)).toThrow('model-input');
    }
  });

  it('checks source PCM against captured effective PCM before model loading', () => {
    const result = prepareInspectorAudio(source(), 16000, oracle(), 'source-wav');
    expect([...result.audio]).toEqual(pcm);
    expect(result.modelInput.sha256).toBe(sha256);
    expect(result.waveformComparison).toMatchObject({
      scope: 'EFFECTIVE_MODEL_INPUT_PCM',
      status: 'PASS',
      pass: true,
    });
    const changed = source();
    changed.audio[2] = -0.5;
    expect(() => prepareInspectorAudio(changed, 16000, oracle(), 'source-wav')).toThrow(
      'INPUT_CONTRACT_MISMATCH',
    );
  });

  it('feeds captured official PCM unchanged without claiming library resampling parity', () => {
    const original = source(22050);
    original.audio.fill(0.75);
    const expected = oracle();
    const result = prepareInspectorAudio(original, 16000, expected, 'oracle-pcm');
    expect([...result.audio]).toEqual(pcm);
    expect(result.resampler).toBe('official-captured-pcm');
    expect(result.waveformComparison).toMatchObject({ pass: true });
    result.audio[0] = 99;
    expect(expected.sample.stages.waveform.data).toEqual(pcm);
  });

  it('keeps legacy source-oracle input identity unverified instead of inventing a waveform PASS', () => {
    const legacy = { sample: {} };
    expect(
      prepareInspectorAudio(source(), 16000, legacy, 'source-wav').waveformComparison,
    ).toMatchObject({ status: 'UNVERIFIED', pass: null });
    expect(() => prepareInspectorAudio(source(22050), 16000, legacy, 'source-wav')).toThrow(
      'same-rate mono',
    );
    expect(() => prepareInspectorAudio(source(), 16000, legacy, 'oracle-pcm')).toThrow(
      'INPUT_CONTRACT_MISMATCH',
    );
  });

  it('does not invent oracle evidence when no reference was requested', () => {
    expect(
      prepareInspectorAudio(source(), 16000, null, 'source-wav').waveformComparison,
    ).toBeNull();
    expect(() => prepareInspectorAudio(source(), 16000, null, 'oracle-pcm')).toThrow('--oracle');
  });

  it.each([
    'metadata-rate',
    'metadata-channels',
    'metadata-encoding',
    'metadata-count',
    'metadata-shape',
    'metadata-hash',
    'padding',
    'stage-dtype',
    'stage-shape',
    'stage-count',
    'stage-value',
    'length-value',
    'length-dtype',
  ])('fails closed on corrupt %s without silently coercing or trimming PCM', (corruption) => {
    const expected = oracle();
    const contract = expected.sample.model_input;
    const waveform = expected.sample.stages.waveform;
    const length = expected.sample.stages.waveform_length;
    if (corruption === 'metadata-rate') contract.sample_rate = 22050;
    if (corruption === 'metadata-channels') contract.channels = 2;
    if (corruption === 'metadata-encoding') contract.encoding = 'f32be';
    if (corruption === 'metadata-count') contract.samples = 3;
    if (corruption === 'metadata-shape') contract.shape = [4];
    if (corruption === 'metadata-hash') contract.sha256 = '0'.repeat(64);
    if (corruption === 'padding') {
      contract.valid_samples = 2;
      contract.right_padding_samples = 2;
      length.data = [2];
    }
    if (corruption === 'stage-dtype') waveform.dtype = 'float64';
    if (corruption === 'stage-shape') waveform.shape = [2, 2];
    if (corruption === 'stage-count') waveform.data.pop();
    if (corruption === 'stage-value') waveform.data[1] = 0.1;
    if (corruption === 'length-value') length.data = [3];
    if (corruption === 'length-dtype') length.dtype = 'float32';
    expect(() => prepareInspectorAudio(source(), 16000, expected, 'oracle-pcm')).toThrow(
      'INPUT_CONTRACT_MISMATCH',
    );
  });
});

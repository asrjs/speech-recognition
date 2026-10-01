import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  compareNativeExamplePcm,
  nativeExampleArtifactPaths,
  nativeExampleBuildFiles,
} from './smoke/nemo-tdt-native-example-smoke.mjs';
import { describeModelInputPcm } from '../tools/model-debugging/scripts/node-asrjs-nemo-inspect.mjs';

function oracle(pcm: Float32Array) {
  return {
    sample: {
      model_input: {
        boundary: 'preprocessor_input_signal',
        ...describeModelInputPcm(pcm, 16000),
      },
      stages: {
        waveform: { dtype: 'float32', shape: [1, pcm.length], data: [...pcm] },
        waveform_length: { dtype: 'int64', shape: [1], data: [pcm.length] },
      },
    },
  };
}

describe('native consumer receipt artifact selection (not model acceptance)', () => {
  it('includes nested compiled runtime and model files in a deterministic build snapshot', (context) => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'asr-native-build-'));
    context.onTestFinished(() => fs.rmSync(directory, { recursive: true, force: true }));
    const names = ['runtime/session.js', 'models/nemo-tdt/executor.js', 'index.js'];
    for (const name of names) {
      const filename = path.join(directory, name);
      fs.mkdirSync(path.dirname(filename), { recursive: true });
      fs.writeFileSync(filename, 'simulation only');
    }
    expect(nativeExampleBuildFiles(directory)).toEqual(
      names.map((name) => path.join(directory, name)).sort(),
    );
  });

  it('fails rather than inventing an empty compiled-build receipt', (context) => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'asr-native-build-'));
    context.onTestFinished(() => fs.rmSync(directory, { recursive: true, force: true }));
    expect(() => nativeExampleBuildFiles(directory)).toThrow('nonempty compiled build');
  });

  it('records both graph sidecars when present', (context) => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'asr-native-receipt-'));
    context.onTestFinished(() => fs.rmSync(directory, { recursive: true, force: true }));
    const names = [
      'encoder-model.onnx',
      'encoder-model.onnx.data',
      'decoder_joint-model.onnx',
      'decoder_joint-model.onnx.data',
      'vocab.txt',
    ];
    for (const name of names) fs.writeFileSync(path.join(directory, name), 'simulation only');
    expect(nativeExampleArtifactPaths(directory)).toEqual(
      names.map((name) => path.join(directory, name)),
    );
  });

  it('does not fabricate absent optional sidecars', (context) => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'asr-native-receipt-'));
    context.onTestFinished(() => fs.rmSync(directory, { recursive: true, force: true }));
    const names = ['encoder-model.onnx', 'decoder_joint-model.onnx', 'vocab.txt'];
    for (const name of names) fs.writeFileSync(path.join(directory, name), 'simulation only');
    expect(nativeExampleArtifactPaths(directory)).toEqual(
      names.map((name) => path.join(directory, name)),
    );
  });

  it('checks the complete captured input contract, not only its hash', () => {
    const pcm = Float32Array.of(0, 0.5, -1);
    expect(compareNativeExamplePcm(pcm, 16000, oracle(pcm))).toMatchObject({
      status: 'PASS',
      pass: true,
      candidate: { samples: 3, sample_rate: 16000 },
    });
    expect(() =>
      compareNativeExamplePcm(Float32Array.of(0, 0.5, -0.5), 16000, oracle(pcm)),
    ).toThrow('INPUT_CONTRACT_MISMATCH');
  });

  it.for(['sample_rate', 'valid_samples', 'right_padding_samples'])(
    'rejects stale %s even when the reference PCM hash matches',
    (key) => {
      const pcm = Float32Array.of(0, 0.5, -1);
      const expected = oracle(pcm);
      const contract = expected.sample.model_input as Record<string, unknown>;
      contract[key] = key === 'sample_rate' ? 22050 : 1;
      expect(() => compareNativeExamplePcm(pcm, 16000, expected)).toThrow(
        'INPUT_CONTRACT_MISMATCH',
      );
    },
  );

  it('requires captured samples when effective PCM metadata is supplied', () => {
    const pcm = Float32Array.of(0, 0.5, -1);
    const expected = oracle(pcm);
    expected.sample.stages.waveform.data = [];
    expect(() => compareNativeExamplePcm(pcm, 16000, expected)).toThrow('INPUT_CONTRACT_MISMATCH');
  });

  it('keeps legacy input unverified and rejects a missing oracle', () => {
    const pcm = Float32Array.of(0, 0.5, -1);
    expect(compareNativeExamplePcm(pcm, 16000, { sample: {} })).toMatchObject({
      status: 'UNVERIFIED',
      pass: null,
    });
    expect(() => compareNativeExamplePcm(pcm, 16000, null)).toThrow('official oracle');
  });
});

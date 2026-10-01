import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  parseArgs,
  resolveModelArtifacts,
  validateInspectorOptions,
} from '../tools/model-debugging/scripts/node-asrjs-nemo-inspect.mjs';

function fixture(context: { onTestFinished: (callback: () => void) => void }) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'asr-nemo-artifacts-'));
  context.onTestFinished(() => fs.rmSync(directory, { recursive: true, force: true }));
  for (const name of ['encoder-model.onnx', 'decoder_joint-model.onnx', 'vocab.txt']) {
    fs.writeFileSync(path.join(directory, name), 'artifact-contract simulation only');
  }
  return {
    modelDir: directory,
    encoderQuant: 'fp32',
    decoderQuant: 'fp32',
    preprocessor: 'nemo128',
    preprocessorBackend: 'js',
  };
}

describe('NeMo inspector artifact prerequisites (not numerical parity)', () => {
  it('does not require an ONNX processor when JS features are requested', (context) => {
    const options = fixture(context);
    expect(resolveModelArtifacts(options).preprocessorPath).toBeNull();
  });

  it('still fails on a missing requested ONNX processor', (context) => {
    const options = { ...fixture(context), preprocessorBackend: 'onnx' };
    expect(() => resolveModelArtifacts(options)).toThrow('Preprocessor model not found');
  });

  it('resolves a provided ONNX processor with unchanged family filenames', (context) => {
    const options = { ...fixture(context), preprocessorBackend: 'onnx' };
    const processor = path.join(options.modelDir, 'nemo128.onnx');
    fs.writeFileSync(processor, 'processor-contract simulation only');
    expect(resolveModelArtifacts(options).preprocessorPath).toBe(processor);
  });
});

describe('NeMo inspector explicit selection (not model acceptance)', () => {
  it('accepts the catalogued multilingual TDT alias without an ID pattern', () => {
    const options = parseArgs(['--model-id', 'parakeet-tdt-multilingual', '--model-dir', '.']);
    expect(() => validateInspectorOptions(options)).not.toThrow();
  });

  it('rejects a made-up TDT-like name instead of borrowing Parakeet defaults', () => {
    const options = parseArgs(['--model-id', 'test/parakeet-tdt-v3-custom', '--model-dir', '.']);
    expect(() => validateInspectorOptions(options)).toThrow('explicit --family');
  });

  it('rejects a catalogued RNNT topology in the TDT-only inspector', () => {
    const options = parseArgs(['--model-id', 'parakeet-realtime-eou-120m-v1', '--model-dir', '.']);
    expect(() => validateInspectorOptions(options)).toThrow('explicit --family');
  });

  it('accepts a candidate only with explicit family/config and native selection', () => {
    const options = parseArgs([
      '--model-id',
      'oruk/orukeet',
      '--model-dir',
      '.',
      '--family',
      'nemo-tdt',
      '--config',
      'runtime-config.json',
      '--backend',
      'native-cpu',
      '--preprocessor-backend',
      'js',
      '--no-grid-batching',
    ]);
    expect(() => validateInspectorOptions(options)).not.toThrow();
    expect(options.gridBatching).toBe(false);
    expect(options.config).toBe(path.resolve('runtime-config.json'));
  });

  it('rejects explicit family selection without an audited config', () => {
    const options = parseArgs(['--model-dir', '.', '--family', 'nemo-tdt']);
    expect(() => validateInspectorOptions(options)).toThrow('requires --config');
  });
});

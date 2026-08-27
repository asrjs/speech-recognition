import * as fs from 'node:fs';
import * as path from 'node:path';
import { describe, expect, it } from 'vitest';

import { GigaAmJsPreprocessor } from '../src/models/gigaam-ctc/frontend.js';

const REFERENCE = path.resolve(
  'tools/data/results/gigaam/multilingual-ctc-jfk-short-reference.json',
);

function loadNpyFloat32(filePath: string): { shape: number[]; data: Float32Array } {
  const buffer = fs.readFileSync(filePath);
  if (buffer[0] !== 0x93 || buffer.toString('latin1', 1, 6) !== 'NUMPY') {
    throw new Error(`Not a NumPy file: ${filePath}`);
  }
  const major = buffer[6];
  const headerLength = major === 1 ? buffer.readUInt16LE(8) : Number(buffer.readBigUInt64LE(8));
  const headerStart = major === 1 ? 10 : 16;
  const header = buffer.toString('ascii', headerStart, headerStart + headerLength);
  const descr = header.match(/'descr':\s*'([^']+)'/)?.[1];
  const shape = (header.match(/'shape':\s*\(([^)]*)\)/)?.[1] ?? '')
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean)
    .map(Number);
  const fortran = header.match(/'fortran_order':\s*(True|False)/)?.[1] === 'True';
  if (fortran) {
    throw new Error(`Fortran-order NPY is not supported: ${filePath}`);
  }
  if (descr !== '<f4' && descr !== '|f4') {
    throw new Error(`Unsupported NPY dtype ${descr} in ${filePath}`);
  }
  const dataOffset = headerStart + headerLength;
  return {
    shape,
    data: new Float32Array(
      buffer.buffer,
      buffer.byteOffset + dataOffset,
      (buffer.byteLength - dataOffset) / 4,
    ),
  };
}

describe.skipIf(!fs.existsSync(REFERENCE))('GigaAM JS frontend vs official PyTorch features', () => {
  it('matches official multilingual CTC log-mel on jfk-short', () => {
    const capture = JSON.parse(fs.readFileSync(REFERENCE, 'utf8')) as {
      preprocessor: {
        n_mels: number;
        n_fft: number;
        win_length: number;
        hop_length: number;
        center: boolean;
        f_min: number;
        f_max: number;
        norm: string;
      };
      samples: Array<{
        audio: { waveform_npy: string };
        stages: { features: { npy: string; lengths: number[] } };
      }>;
    };
    const sample = capture.samples[0];
    if (!sample?.audio.waveform_npy || !fs.existsSync(sample.audio.waveform_npy)) {
      return;
    }
    const waveform = loadNpyFloat32(sample.audio.waveform_npy);
    const reference = loadNpyFloat32(sample.stages.features.npy);
    const preprocessor = new GigaAmJsPreprocessor({
      nMels: capture.preprocessor.n_mels,
      nFft: capture.preprocessor.n_fft,
      winLength: capture.preprocessor.win_length,
      hopLength: capture.preprocessor.hop_length,
      center: capture.preprocessor.center,
      melLowHz: capture.preprocessor.f_min,
      melHighHz: capture.preprocessor.f_max || undefined,
      slaneyNorm: capture.preprocessor.norm === 'slaney',
    });
    const processed = preprocessor.process(waveform.data);
    expect(processed.featureSize).toBe(capture.preprocessor.n_mels);
    expect(processed.frameCount).toBe(sample.stages.features.lengths[0]);
    expect(processed.features.length).toBe(reference.data.length);
    let maxAbs = 0;
    let sumAbs = 0;
    for (let index = 0; index < processed.features.length; index += 1) {
      const difference = Math.abs((processed.features[index] ?? 0) - (reference.data[index] ?? 0));
      sumAbs += difference;
      maxAbs = Math.max(maxAbs, difference);
    }
    const meanAbs = sumAbs / processed.features.length;
    // Remaining error is CompositeFft vs torch.stft (~0.007 log-mel), not the
    // previous SpecScaler floor blow-up (log(x+1e-9) vs log(clamp(x, 1e-9))).
    expect(meanAbs).toBeLessThan(1e-3);
    expect(maxAbs).toBeLessThan(0.02);
  });
});

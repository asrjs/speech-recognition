import * as fs from 'node:fs';
import * as path from 'node:path';
import { describe, expect, it } from 'vitest';

import { GigaAmJsPreprocessor } from '../src/models/gigaam-ctc/frontend.js';
import { CompositeFft } from '../src/models/lasr-ctc/mel.js';

const REFERENCE = path.resolve(
  'tools/data/results/gigaam/multilingual-ctc-jfk-short-reference.json',
);
const STAGES = path.resolve('N:/models/gigaam/multilingual-ctc/captures/frontend-stages');

function loadNpyFloat32(filePath: string): { shape: number[]; data: Float32Array } {
  const buffer = fs.readFileSync(filePath);
  if (buffer[0] !== 0x93 || buffer.toString('latin1', 1, 6) !== 'NUMPY') {
    throw new Error(`Not a NumPy file: ${filePath}`);
  }
  const major = buffer[6];
  const headerLength = major === 1 ? buffer.readUInt16LE(8) : Number(buffer.readBigUInt64LE(8));
  const headerStart = major === 1 ? 10 : 16;
  const header = buffer.toString('ascii', headerStart, headerStart + headerLength);
  const shape = (header.match(/'shape':\s*\(([^)]*)\)/)?.[1] ?? '')
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean)
    .map(Number);
  const fortran = header.match(/'fortran_order':\s*(True|False)/)?.[1] === 'True';
  if (fortran) throw new Error(`Fortran-order NPY is not supported: ${filePath}`);
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

function locateMax(
  reference: Float32Array,
  candidate: Float32Array,
  rows: number,
  cols: number,
): {
  maxAbs: number;
  meanAbs: number;
  row: number;
  col: number;
  ref: number;
  cand: number;
  topFrames: Array<{ frame: number; maxAbs: number }>;
} {
  let maxAbs = 0;
  let sumAbs = 0;
  let maxIndex = 0;
  const perFrame = new Float64Array(cols);
  const count = Math.min(reference.length, candidate.length);
  for (let index = 0; index < count; index += 1) {
    const difference = Math.abs((reference[index] ?? 0) - (candidate[index] ?? 0));
    sumAbs += difference;
    if (difference > maxAbs) {
      maxAbs = difference;
      maxIndex = index;
    }
    const frame = index % cols;
    perFrame[frame] = Math.max(perFrame[frame] ?? 0, difference);
  }
  const topFrames = Array.from(perFrame)
    .map((value, frame) => ({ frame, maxAbs: value }))
    .sort((left, right) => right.maxAbs - left.maxAbs)
    .slice(0, 8);
  return {
    maxAbs,
    meanAbs: count ? sumAbs / count : 0,
    row: Math.floor(maxIndex / cols),
    col: maxIndex % cols,
    ref: reference[maxIndex] ?? 0,
    cand: candidate[maxIndex] ?? 0,
    topFrames,
  };
}

function writeNpyFloat32(filePath: string, data: Float32Array, shape: readonly number[]): void {
  const shapeStr = shape.length === 1 ? `${shape[0]},` : shape.join(', ');
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shapeStr}), }`;
  const prefixLen = 10;
  const pad = (64 - ((prefixLen + header.length + 1) % 64)) % 64;
  header = `${header}${' '.repeat(pad)}\n`;
  const payload = Buffer.alloc(prefixLen + header.length + data.byteLength);
  payload[0] = 0x93;
  payload.write('NUMPY', 1, 'ascii');
  payload[6] = 1;
  payload[7] = 0;
  payload.writeUInt16LE(header.length, 8);
  payload.write(header, prefixLen, 'ascii');
  Buffer.from(data.buffer, data.byteOffset, data.byteLength).copy(payload, prefixLen + header.length);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, payload);
}

describe.skipIf(!fs.existsSync(REFERENCE) || !fs.existsSync(path.join(STAGES, 'stft_power.npy')))(
  'GigaAM frontend stage localization',
  () => {
    it('reports the earliest diverging stage vs official torchaudio', () => {
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
      const sample = capture.samples[0]!;
      const waveform = loadNpyFloat32(sample.audio.waveform_npy);
      const officialLog = loadNpyFloat32(sample.stages.features.npy);
      const officialStft = loadNpyFloat32(path.join(STAGES, 'stft_power.npy'));
      const officialFb = loadNpyFloat32(path.join(STAGES, 'filterbank.npy'));
      const officialPreLog = loadNpyFloat32(path.join(STAGES, 'pre_log_mel.npy'));
      const officialWindow = loadNpyFloat32(path.join(STAGES, 'window.npy'));

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
      const nMels = processed.featureSize;
      const frameCount = processed.frameCount;
      const nFft = capture.preprocessor.n_fft;
      const nFreqBins = (nFft >> 1) + 1;
      const hop = capture.preprocessor.hop_length;

      const window = new Float64Array(nFft);
      for (let index = 0; index < nFft; index += 1) {
        window[index] = 0.5 * (1 - Math.cos((2 * Math.PI * index) / nFft));
      }
      let windowMax = 0;
      for (let index = 0; index < nFft; index += 1) {
        windowMax = Math.max(
          windowMax,
          Math.abs((officialWindow.data[index] ?? 0) - (window[index] ?? 0)),
        );
      }

      const fft = new CompositeFft(nFft);
      const real = new Float64Array(nFft);
      const imag = new Float64Array(nFft);
      const jsStft = new Float32Array(nFreqBins * frameCount);
      for (let frame = 0; frame < frameCount; frame += 1) {
        const offset = frame * hop;
        for (let index = 0; index < nFft; index += 1) {
          real[index] = (waveform.data[offset + index] ?? 0) * (window[index] ?? 0);
          imag[index] = 0;
        }
        fft.transform(real, imag);
        for (let bin = 0; bin < nFreqBins; bin += 1) {
          const re = real[bin] ?? 0;
          const im = imag[bin] ?? 0;
          jsStft[bin * frameCount + frame] = re * re + im * im;
        }
      }

      const jsPreLogOfficialFb = new Float32Array(nMels * frameCount);
      const jsLogAdd = new Float32Array(nMels * frameCount);
      const jsLogClamp = new Float32Array(nMels * frameCount);
      for (let frame = 0; frame < frameCount; frame += 1) {
        for (let mel = 0; mel < nMels; mel += 1) {
          let value = 0;
          for (let bin = 0; bin < nFreqBins; bin += 1) {
            value +=
              (jsStft[bin * frameCount + frame] ?? 0) *
              (officialFb.data[bin * nMels + mel] ?? 0);
          }
          jsPreLogOfficialFb[mel * frameCount + frame] = value;
          jsLogAdd[mel * frameCount + frame] = Math.log(value + 1e-9);
          jsLogClamp[mel * frameCount + frame] = Math.log(Math.min(Math.max(value, 1e-9), 1e9));
        }
      }

      const report = {
        windowMax,
        stft: locateMax(officialStft.data, jsStft, nFreqBins, frameCount),
        preLog: locateMax(officialPreLog.data, jsPreLogOfficialFb, nMels, frameCount),
        logAdd: locateMax(officialLog.data, jsLogAdd, nMels, frameCount),
        logClamp: locateMax(officialLog.data, jsLogClamp, nMels, frameCount),
        preprocessor: locateMax(officialLog.data, processed.features, nMels, frameCount),
      };
      const outPath = path.resolve(
        'tools/data/results/gigaam/multilingual-ctc-jfk-short-frontend-diagnose.json',
      );
      fs.mkdirSync(path.dirname(outPath), { recursive: true });
      fs.writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`);
      writeNpyFloat32(
        path.resolve('N:/models/gigaam/multilingual-ctc/captures/jfk-short.js-features.npy'),
        processed.features,
        [1, nMels, frameCount],
      );
      expect(report.windowMax).toBeLessThan(1e-6);
      expect(report.preprocessor.maxAbs).toBeLessThan(0.02);
      expect(report.preprocessor.meanAbs).toBeLessThan(1e-3);
    });
  },
);

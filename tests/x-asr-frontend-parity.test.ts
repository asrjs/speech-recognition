import * as fs from 'node:fs';
import { describe, expect, it } from 'vitest';

import { XAsrJsFrontend } from '../src/models/x-asr/frontend.js';

const WAVEFORM = 'N:/models/gigaam/multilingual-ctc/captures/jfk-short.waveform.npy';
const REFERENCE = 'tools/data/results/x-asr/jfk-short-sherpa-fbank.npy';

function loadNpyFloat32(filePath: string): Float32Array {
  const buffer = fs.readFileSync(filePath);
  const major = buffer[6];
  const headerLength = major === 1 ? buffer.readUInt16LE(8) : Number(buffer.readBigUInt64LE(8));
  const headerStart = major === 1 ? 10 : 16;
  const dataOffset = headerStart + headerLength;
  return new Float32Array(buffer.buffer, buffer.byteOffset + dataOffset, (buffer.byteLength - dataOffset) / 4);
}

describe.skipIf(!fs.existsSync(WAVEFORM) || !fs.existsSync(REFERENCE))(
  'X-ASR JS fbank vs sherpa knf',
  () => {
    it('matches kaldi-native-fbank snip_edges=false high_freq=-400 on jfk-short', () => {
      const waveform = loadNpyFloat32(WAVEFORM);
      const reference = loadNpyFloat32(REFERENCE);
      const features = new XAsrJsFrontend().process(waveform);
      expect(features.length).toBe(reference.length);
      let maxAbs = 0;
      let sumSq = 0;
      for (let i = 0; i < features.length; i += 1) {
        const delta = Math.abs((features[i] ?? 0) - (reference[i] ?? 0));
        maxAbs = Math.max(maxAbs, delta);
        sumSq += delta * delta;
      }
      const rms = Math.sqrt(sumSq / features.length);
      expect(features.length / 80).toBe(1100);
      expect(rms).toBeLessThan(5e-4);
      expect(maxAbs).toBeLessThan(5e-3);
    });
  },
);

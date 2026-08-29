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

describe('X-ASR incremental frontend', () => {
  it('matches full-buffer features across uneven streaming chunks', () => {
    const waveform = Float32Array.from({ length: 16000 * 2 + 137 }, (_, index) =>
      Math.sin(index / 17) * 0.2 + Math.cos(index / 43) * 0.03,
    );
    const frontend = new XAsrJsFrontend();
    const full = frontend.process(waveform);
    let previousTail = new Float32Array(0);
    let previousSamples = 0;
    let previousFrames = 0;
    const pieces: Float32Array[] = [];
    let audioOffset = 0;
    for (const requestedChunkSize of [37, 241, 1600, 79, 4096, 503, 7777, 991, 8192]) {
      if (audioOffset >= waveform.length) break;
      const chunkSize = Math.min(requestedChunkSize, waveform.length - audioOffset);
      const start = audioOffset;
      const chunk = waveform.subarray(start, Math.min(waveform.length, start + chunkSize));
      const result = frontend.processIncremental(previousTail, previousSamples, chunk, previousFrames);
      pieces.push(result.features);
      previousTail = result.tail;
      previousSamples += chunk.length;
      previousFrames += result.frameCount;
      audioOffset += chunk.length;
    }
    if (audioOffset < waveform.length) {
      const result = frontend.processIncremental(previousTail, previousSamples, waveform.subarray(audioOffset), previousFrames, true);
      pieces.push(result.features);
    } else {
      const result = frontend.processIncremental(previousTail, previousSamples, new Float32Array(0), previousFrames, true);
      pieces.push(result.features);
    }
    const streamed = new Float32Array(pieces.reduce((sum, piece) => sum + piece.length, 0));
    let offset = 0;
    for (const piece of pieces) {
      streamed.set(piece, offset);
      offset += piece.length;
    }
    expect(streamed.length).toBe(full.length);
    let maxAbs = 0;
    for (let index = 0; index < full.length; index += 1) {
      const delta = Math.abs((streamed[index] ?? 0) - (full[index] ?? 0));
      maxAbs = Math.max(maxAbs, delta);
    }
    expect(maxAbs).toBeLessThan(1e-6);
  });
});

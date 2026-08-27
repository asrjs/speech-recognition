import * as fs from 'node:fs';
import { describe, expect, it } from 'vitest';

import {
  applySenseVoiceLfr,
  parseSenseVoiceCmvn,
  applySenseVoiceCmvn,
  SENSEVOICE_LFR_DIM,
} from '../src/models/sensevoice/frontend.js';

describe('SenseVoice official LFR/CMVN', () => {
  it('matches FunASR apply_lfr on a 10x80 ramp', () => {
    const frames = new Float32Array(10 * 80);
    for (let index = 0; index < frames.length; index += 1) frames[index] = index;
    const { features, frameCount } = applySenseVoiceLfr(frames, 10, 80, 7, 6);
    expect(frameCount).toBe(2);
    expect(features.length).toBe(2 * 560);
    expect(features[0]).toBe(0);
    expect(features[features.length - 1]).toBe(799);
  });

  it('applies (x + mean) * scale per dimension', () => {
    const frames = new Float32Array([3, 4]);
    const cmvn = { means: new Float32Array([-1, -2]), scales: new Float32Array([0.5, 2]) };
    expect(Array.from(applySenseVoiceCmvn(frames, 1, 2, cmvn))).toEqual([1, 4]);
  });
});

const AM_MVN = 'N:/models/onnx/sensevoice/small/am.mvn';

describe.skipIf(!fs.existsSync(AM_MVN))('SenseVoice official am.mvn', () => {
  it('parses 560-dim AddShift/Rescale vectors', () => {
    const cmvn = parseSenseVoiceCmvn(fs.readFileSync(AM_MVN, 'utf8'));
    expect(cmvn.means.length).toBe(SENSEVOICE_LFR_DIM);
    expect(cmvn.scales.length).toBe(SENSEVOICE_LFR_DIM);
    expect(cmvn.means[0]).toBeCloseTo(-8.311879, 5);
    expect(cmvn.scales[0]).toBeCloseTo(0.155775, 5);
  });
});

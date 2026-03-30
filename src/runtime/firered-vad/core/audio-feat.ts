import type { CmvnStats } from '../types.js';
import { FireRedFbank } from './fbank.js';

export class AudioFeat {
  private readonly fbank: FireRedFbank;
  private readonly cmvn: CmvnStats;

  constructor(cmvn: CmvnStats) {
    this.cmvn = cmvn;
    this.fbank = new FireRedFbank({
      num_bins: 80,
      frame_length: 400,
      frame_shift: 160,
      sample_rate: 16000,
      stateful_pre_emphasis: false,
    });
  }

  reset(): void {
    this.fbank.reset();
  }

  extract(input: ArrayLike<number>): Float32Array[] {
    const frames = this.fbank.compute(input);
    for (const frame of frames) {
      this.applyCmvn(frame);
    }
    return frames;
  }

  private applyCmvn(frame: Float32Array): void {
    for (let i = 0; i < frame.length; i += 1) {
      frame[i] = (frame[i]! - this.cmvn.means[i]!) * this.cmvn.istd[i]!;
    }
  }
}

export class StreamingPackedAudioFeat {
  private readonly fbank: FireRedFbank;
  private readonly cmvn: CmvnStats;

  constructor(cmvn: CmvnStats) {
    this.cmvn = cmvn;
    this.fbank = new FireRedFbank({
      num_bins: 80,
      frame_length: 400,
      frame_shift: 160,
      sample_rate: 16000,
      stateful_pre_emphasis: true,
    });
  }

  reset(): void {
    this.fbank.reset();
  }

  extractSingleFrame(input: ArrayLike<number>): Float32Array | null {
    const frames = this.fbank.compute(input);
    if (frames.length === 0) {
      return null;
    }
    const frame = frames[frames.length - 1]!;
    for (let i = 0; i < frame.length; i += 1) {
      frame[i] = (frame[i]! - this.cmvn.means[i]!) * this.cmvn.istd[i]!;
    }
    return frame;
  }
}

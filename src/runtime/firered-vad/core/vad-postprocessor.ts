import { FRAME_LENGTH_S, FRAME_SHIFT_S } from './constants.js';
import { roundTo } from './util.js';

enum VadState {
  SILENCE = 0,
  POSSIBLE_SPEECH = 1,
  SPEECH = 2,
  POSSIBLE_SILENCE = 3,
}

export interface VadPostprocessorOptions {
  readonly smoothWindowSize: number;
  readonly probThreshold: number;
  readonly minSpeechFrame: number;
  readonly maxSpeechFrame: number;
  readonly minSilenceFrame: number;
  readonly mergeSilenceFrame: number;
  readonly extendSpeechFrame: number;
}

export class VadPostprocessor {
  readonly smoothWindowSize: number;
  readonly probThreshold: number;
  readonly minSpeechFrame: number;
  readonly maxSpeechFrame: number;
  readonly minSilenceFrame: number;
  readonly mergeSilenceFrame: number;
  readonly extendSpeechFrame: number;

  constructor(options: VadPostprocessorOptions) {
    this.smoothWindowSize = Math.max(1, options.smoothWindowSize);
    this.probThreshold = options.probThreshold;
    this.minSpeechFrame = options.minSpeechFrame;
    this.maxSpeechFrame = options.maxSpeechFrame;
    this.minSilenceFrame = options.minSilenceFrame;
    this.mergeSilenceFrame = options.mergeSilenceFrame;
    this.extendSpeechFrame = options.extendSpeechFrame;
  }

  process(rawProbs: number[]): number[] {
    if (rawProbs.length === 0) {
      return [];
    }
    const smoothed = this.smoothProb(rawProbs);
    const binary = this.applyThreshold(smoothed);
    const decisions = this.smoothPredsWithStateMachine(binary);
    const fixedDecisions = this.fixSmoothWindowStart(decisions);
    const merged = this.mergeShortSilenceSegments(fixedDecisions);
    const extended = this.extendSpeechSegments(merged);
    return this.splitLongSpeechSegments(extended, rawProbs);
  }

  decisionToSegment(decisions: number[], wavDur?: number): Array<[number, number]> {
    const segments: Array<[number, number]> = [];
    let speechStart: number | null = null;
    for (let t = 0; t < decisions.length; t += 1) {
      const decision = decisions[t]!;
      if (decision === 1 && speechStart === null) {
        speechStart = t;
      } else if (decision === 0 && speechStart !== null) {
        segments.push([roundTo(speechStart * FRAME_SHIFT_S, 3), roundTo(t * FRAME_SHIFT_S, 3)]);
        speechStart = null;
      }
    }
    if (speechStart !== null) {
      const end = Math.min(
        decisions.length * FRAME_SHIFT_S + FRAME_LENGTH_S,
        wavDur ?? Number.POSITIVE_INFINITY,
      );
      segments.push([roundTo(speechStart * FRAME_SHIFT_S, 3), roundTo(end, 3)]);
    }
    return segments;
  }

  private smoothProb(probs: number[]): number[] {
    if (this.smoothWindowSize <= 1) {
      return probs.slice();
    }
    const out = new Array<number>(probs.length);
    const window: number[] = [];
    let sum = 0;
    for (let i = 0; i < probs.length; i += 1) {
      window.push(probs[i]!);
      sum += probs[i]!;
      if (window.length > this.smoothWindowSize) {
        sum -= window.shift() ?? 0;
      }
      out[i] = sum / window.length;
    }
    return out;
  }

  private applyThreshold(probs: number[]): number[] {
    return probs.map((value) => Number(value >= this.probThreshold));
  }

  private smoothPredsWithStateMachine(binaryPreds: number[]): number[] {
    if (this.minSpeechFrame <= 0 && this.minSilenceFrame <= 0) {
      return binaryPreds.slice();
    }
    const decisions = new Array<number>(binaryPreds.length).fill(0);
    let state = VadState.SILENCE;
    let speechStart = -1;
    let silenceStart = -1;

    for (let t = 0; t < binaryPreds.length; t += 1) {
      const isSpeech = binaryPreds[t] === 1;
      if (state === VadState.SILENCE) {
        if (isSpeech) {
          state = VadState.POSSIBLE_SPEECH;
          speechStart = t;
        }
      } else if (state === VadState.POSSIBLE_SPEECH) {
        if (isSpeech) {
          if (t - speechStart >= this.minSpeechFrame) {
            state = VadState.SPEECH;
            for (let i = speechStart; i < t; i += 1) {
              decisions[i] = 1;
            }
          }
        } else {
          state = VadState.SILENCE;
          speechStart = -1;
        }
      } else if (state === VadState.SPEECH) {
        if (!isSpeech) {
          state = VadState.POSSIBLE_SILENCE;
          silenceStart = t;
        }
      } else if (!isSpeech) {
        if (t - silenceStart >= this.minSilenceFrame) {
          state = VadState.SILENCE;
          speechStart = -1;
        }
      } else {
        state = VadState.SPEECH;
        silenceStart = -1;
      }

      decisions[t] = state === VadState.SPEECH || state === VadState.POSSIBLE_SILENCE ? 1 : 0;
    }
    return decisions;
  }

  private fixSmoothWindowStart(decisions: number[]): number[] {
    const out = decisions.slice();
    for (let t = 1; t < decisions.length; t += 1) {
      if (decisions[t - 1] === 0 && decisions[t] === 1) {
        const start = Math.max(0, t - this.smoothWindowSize);
        for (let i = start; i < t; i += 1) {
          out[i] = 1;
        }
      }
    }
    return out;
  }

  private mergeShortSilenceSegments(decisions: number[]): number[] {
    if (this.mergeSilenceFrame <= 0) {
      return decisions;
    }
    const out = decisions.slice();
    let silenceStart: number | null = null;
    for (let t = 1; t < decisions.length; t += 1) {
      if (decisions[t - 1] === 1 && decisions[t] === 0 && silenceStart === null) {
        silenceStart = t;
      } else if (decisions[t - 1] === 0 && decisions[t] === 1 && silenceStart !== null) {
        const silenceFrames = t - silenceStart;
        if (silenceFrames < this.mergeSilenceFrame) {
          for (let i = silenceStart; i < t; i += 1) {
            out[i] = 1;
          }
        }
        silenceStart = null;
      }
    }
    return out;
  }

  private extendSpeechSegments(decisions: number[]): number[] {
    if (this.extendSpeechFrame <= 0) {
      return decisions;
    }
    const out = decisions.slice();
    for (let t = 0; t < decisions.length; t += 1) {
      if (decisions[t] === 1) {
        const start = Math.max(0, t - this.extendSpeechFrame);
        const end = Math.min(decisions.length, t + this.extendSpeechFrame + 1);
        for (let i = start; i < end; i += 1) {
          out[i] = 1;
        }
      }
    }
    return out;
  }

  private splitLongSpeechSegments(decisions: number[], probs: number[]): number[] {
    const out = decisions.slice();
    const segments = this.decisionToSegment(decisions);
    for (const [startS, endS] of segments) {
      const startFrame = Math.floor(startS / FRAME_SHIFT_S);
      const endFrame = Math.floor(endS / FRAME_SHIFT_S);
      const durationFrames = endFrame - startFrame;
      if (durationFrames <= this.maxSpeechFrame) {
        continue;
      }

      let start = 0;
      const segmentProbs = probs.slice(startFrame, endFrame);
      while (start < segmentProbs.length) {
        if (segmentProbs.length - start <= this.maxSpeechFrame) {
          break;
        }
        const windowStart = Math.floor(start + this.maxSpeechFrame / 2);
        const windowEnd = Math.min(segmentProbs.length, start + this.maxSpeechFrame);
        let minIdx = windowStart;
        let minVal = Number.POSITIVE_INFINITY;
        for (let i = windowStart; i < windowEnd; i += 1) {
          if (segmentProbs[i]! < minVal) {
            minVal = segmentProbs[i]!;
            minIdx = i;
          }
        }
        out[startFrame + minIdx] = 0;
        start = minIdx + 1;
      }
    }
    return out;
  }
}

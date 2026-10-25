export interface SpeechContextOptions {
  readonly sampleRate: number;
  readonly maxDurationSeconds: number;
  readonly retentionDurationSeconds?: number;
  readonly maxGapSeconds: number;
  readonly prerollSeconds?: number;
}

export interface SpeechContextRevision {
  readonly id: string;
  readonly revision: number;
  readonly startFrame: number;
  readonly endFrame: number;
  readonly sampleRate: number;
  readonly pcm: Float32Array;
  readonly finalized: boolean;
}

let nextContextId = 1;

/** Keeps original captured PCM available across VAD boundaries in a bounded ring. */
export class SpeechContextBuffer {
  readonly maxFrames: number;
  readonly retainedFrames: number;
  readonly maxGapFrames: number;
  readonly prerollFrames: number;
  private readonly ring: Float32Array;
  private captureStartFrame: number | null = null;
  private captureEndFrame: number | null = null;
  private contextStartFrame: number | null = null;
  private contextEndFrame: number | null = null;
  private lastLogicalEndFrame: number | null = null;
  private revision = 0;
  private contextId = '';
  private contextFinalized = false;

  constructor(private readonly options: SpeechContextOptions) {
    if (!Number.isFinite(options.sampleRate) || options.sampleRate <= 0) {
      throw new TypeError('SpeechContextBuffer requires a positive sampleRate.');
    }
    if (!Number.isFinite(options.maxDurationSeconds) || options.maxDurationSeconds <= 0) {
      throw new TypeError('SpeechContextBuffer requires a positive maxDurationSeconds.');
    }
    if (!Number.isFinite(options.maxGapSeconds) || options.maxGapSeconds < 0) {
      throw new TypeError('SpeechContextBuffer requires a non-negative maxGapSeconds.');
    }
    const prerollSeconds = options.prerollSeconds ?? 0;
    if (!Number.isFinite(prerollSeconds) || prerollSeconds < 0) {
      throw new TypeError('SpeechContextBuffer requires a non-negative prerollSeconds.');
    }
    this.maxFrames = Math.floor(options.sampleRate * options.maxDurationSeconds);
    const retentionDurationSeconds = options.retentionDurationSeconds ?? options.maxDurationSeconds;
    if (!Number.isFinite(retentionDurationSeconds) || retentionDurationSeconds < options.maxDurationSeconds) {
      throw new TypeError('Speech context retention must be at least its maximum duration.');
    }
    this.retainedFrames = Math.floor(options.sampleRate * retentionDurationSeconds);
    this.maxGapFrames = Math.floor(options.sampleRate * options.maxGapSeconds);
    this.prerollFrames = Math.floor(options.sampleRate * prerollSeconds);
    if (this.maxFrames < 1) throw new RangeError('Speech context duration is shorter than one sample.');
    if (this.retainedFrames < this.maxFrames) throw new RangeError('Speech context retention is shorter than one sample window.');
    this.ring = new Float32Array(this.retainedFrames);
  }

  /** Append every captured frame, including silence between VAD segments. */
  write(startFrame: number, input: Float32Array): void {
    if (!Number.isSafeInteger(startFrame) || startFrame < 0 || !(input instanceof Float32Array)) {
      throw new RangeError('Speech context capture requires a non-negative frame and Float32 PCM.');
    }
    if (this.captureEndFrame !== null && startFrame !== this.captureEndFrame) {
      throw new RangeError(`Speech context capture discontinuity: expected frame ${this.captureEndFrame}, got ${startFrame}.`);
    }
    if (this.captureStartFrame === null) this.captureStartFrame = startFrame;
    for (let index = 0; index < input.length; index += 1) {
      this.ring[(startFrame + index) % this.retainedFrames] = input[index]!;
    }
    this.captureEndFrame = startFrame + input.length;
  }

  /** Snapshot the original contiguous capture through a VAD boundary. */
  snapshot(
    startFrame: number,
    endFrame: number,
    logicalStartFrame = startFrame,
    logicalEndFrame = endFrame,
  ): SpeechContextRevision {
    if (this.contextFinalized) {
      throw new RangeError('Speech context has already been finalized.');
    }
    if (!Number.isSafeInteger(startFrame) || !Number.isSafeInteger(endFrame)
      || !Number.isSafeInteger(logicalStartFrame) || !Number.isSafeInteger(logicalEndFrame)
      || startFrame < 0 || endFrame <= startFrame) {
      throw new RangeError('Speech context frame range must be a positive integer interval.');
    }
    if (this.captureEndFrame === null || endFrame > this.captureEndFrame) {
      throw new RangeError('Speech context boundary extends beyond captured audio.');
    }
    if (logicalStartFrame < startFrame || logicalEndFrame < logicalStartFrame || logicalEndFrame > endFrame) {
      throw new RangeError('Speech context logical bounds must fit inside the extracted frame range.');
    }
    if (this.captureStartFrame === null || startFrame < this.captureStartFrame) {
      throw new RangeError('Speech context boundary begins before captured audio.');
    }
    const baseFrame = Math.max(this.captureStartFrame, this.captureEndFrame - this.retainedFrames);

    const beginsNew = this.contextStartFrame === null || this.contextEndFrame === null
      || logicalStartFrame - this.lastLogicalEndFrame! > this.maxGapFrames
      || endFrame - this.contextStartFrame > this.maxFrames;
    if (beginsNew) {
      const capturedStartFrame = Math.max(this.captureStartFrame, startFrame - this.prerollFrames);
      const nextStartFrame = this.contextEndFrame === null
        ? capturedStartFrame
        : Math.max(capturedStartFrame, this.contextEndFrame);
      if (nextStartFrame >= endFrame) {
        throw new RangeError('VAD segment contains no unowned frames for a new speech context.');
      }
      if (endFrame - nextStartFrame > this.maxFrames) {
        throw new RangeError('VAD segment exceeds the configured speech context duration.');
      }
      if (nextStartFrame < baseFrame) {
        throw new RangeError(`Speech context frame ${nextStartFrame} was overwritten; earliest retained frame is ${baseFrame}.`);
      }
      this.contextId = `speech-context-${nextContextId++}`;
      this.contextStartFrame = nextStartFrame;
      this.contextEndFrame = endFrame;
      this.lastLogicalEndFrame = logicalEndFrame;
      this.revision = 1;
    } else {
      // VAD extraction can overlap. Keep the first onset and include each captured
      // frame once, rather than concatenating overlapping segment buffers.
      if (this.contextStartFrame! < baseFrame) {
        throw new RangeError(`Speech context frame ${this.contextStartFrame} was overwritten; earliest retained frame is ${baseFrame}.`);
      }
      this.contextEndFrame = Math.max(this.contextEndFrame!, endFrame);
      this.lastLogicalEndFrame = Math.max(this.lastLogicalEndFrame!, logicalEndFrame);
      this.revision += 1;
    }

    const contextStart = this.contextStartFrame!;
    const contextEnd = this.contextEndFrame!;
    const pcm = new Float32Array(contextEnd - contextStart);
    for (let index = 0; index < pcm.length; index += 1) {
      pcm[index] = this.ring[(contextStart + index) % this.retainedFrames]!;
    }
    return {
      id: this.contextId,
      revision: this.revision,
      startFrame: contextStart,
      endFrame: contextEnd,
      sampleRate: this.options.sampleRate,
      pcm,
      finalized: false,
    };
  }

  /** Extend the latest context once with bounded, already-captured right context. */
  finalize(): SpeechContextRevision | null {
    if (this.contextFinalized) return null;
    if (this.contextStartFrame === null || this.contextEndFrame === null || this.captureEndFrame === null) {
      return null;
    }
    const baseFrame = Math.max(this.captureStartFrame!, this.captureEndFrame - this.retainedFrames);
    const finalEndFrame = Math.min(
      this.captureEndFrame,
      this.contextEndFrame + this.prerollFrames,
      this.contextStartFrame + this.maxFrames,
    );
    if (finalEndFrame <= this.contextEndFrame) return null;
    if (this.contextStartFrame < baseFrame) {
      throw new RangeError(`Speech context frame ${this.contextStartFrame} was overwritten; earliest retained frame is ${baseFrame}.`);
    }

    this.contextEndFrame = finalEndFrame;
    this.revision += 1;
    this.contextFinalized = true;
    const contextStart = this.contextStartFrame;
    const pcm = new Float32Array(finalEndFrame - contextStart);
    for (let index = 0; index < pcm.length; index += 1) {
      pcm[index] = this.ring[(contextStart + index) % this.retainedFrames]!;
    }
    return {
      id: this.contextId,
      revision: this.revision,
      startFrame: contextStart,
      endFrame: finalEndFrame,
      sampleRate: this.options.sampleRate,
      pcm,
      finalized: true,
    };
  }

  reset(): void {
    this.captureStartFrame = null;
    this.captureEndFrame = null;
    this.contextStartFrame = null;
    this.contextEndFrame = null;
    this.lastLogicalEndFrame = null;
    this.revision = 0;
    this.contextId = '';
    this.contextFinalized = false;
  }
}

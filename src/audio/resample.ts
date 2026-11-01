/**
 * Band-limited sample-rate conversion shared by microphone capture and
 * offline audio preparation.
 *
 * Linear interpolation does not low-pass before decimating, so content above
 * the target Nyquist folds back into the speech band (48 kHz to 16 kHz keeps
 * a 9 kHz tone at full level at 7 kHz). This resampler applies a Kaiser
 * windowed-sinc filter through a polyphase table.
 *
 * `SincResampler` must stay self-contained: the capture AudioWorklet embeds it
 * through `Function.prototype.toString()`, so it may not reference anything
 * from module scope.
 */

export interface SincResamplerOptions {
  /** Filter zero crossings on each side, at the lower of the two rates. */
  readonly zeroCrossings?: number;
  /** Interpolated filter phases between input samples. */
  readonly phases?: number;
  /** Cutoff as a fraction of the lower Nyquist frequency. */
  readonly cutoff?: number;
  /** Kaiser window shape parameter. */
  readonly beta?: number;
}

/** Streaming resampler. Equal rates pass samples through unchanged. */
export class SincResampler {
  readonly sourceSampleRate: number;
  readonly targetSampleRate: number;
  /** Source samples of look-ahead needed before an output sample is emitted. */
  readonly halfWidth: number;
  private readonly passthrough: boolean;
  private readonly step: number;
  private readonly tapCount: number;
  private readonly phaseCount: number;
  private readonly table: Float32Array;
  private buffer: Float32Array;
  private length: number;
  // Integer counters keep output positions independent of input chunking;
  // accumulating a float step per sample would drift with chunk boundaries.
  private emitted = 0;
  private dropped = 0;

  constructor(
    sourceSampleRate: number,
    targetSampleRate: number,
    options: SincResamplerOptions = {},
  ) {
    if (
      !Number.isFinite(sourceSampleRate) ||
      !Number.isFinite(targetSampleRate) ||
      sourceSampleRate <= 0 ||
      targetSampleRate <= 0
    ) {
      throw new RangeError(
        `Invalid resample rates: from=${sourceSampleRate}, to=${targetSampleRate}`,
      );
    }
    this.sourceSampleRate = sourceSampleRate;
    this.targetSampleRate = targetSampleRate;
    this.passthrough = sourceSampleRate === targetSampleRate;
    this.step = sourceSampleRate / targetSampleRate;

    const scale = Math.min(1, targetSampleRate / sourceSampleRate);
    const cutoff = (options.cutoff ?? 0.9) * scale;
    const zeroCrossings = Math.max(2, Math.floor(options.zeroCrossings ?? 16));
    const beta = options.beta ?? 8;
    const phaseCount = Math.max(1, Math.floor(options.phases ?? 256));
    const halfWidth = this.passthrough ? 1 : Math.ceil(zeroCrossings / scale);
    const tapCount = 2 * halfWidth;

    const besselI0 = (value: number): number => {
      let sum = 1;
      let term = 1;
      for (let k = 1; k < 40; k += 1) {
        term *= (value / (2 * k)) * (value / (2 * k));
        sum += term;
        if (term < sum * 1e-12) break;
      }
      return sum;
    };
    const windowNorm = besselI0(beta);

    // One extra row lets phase interpolation read row p + 1 for every p.
    const table = new Float32Array((phaseCount + 1) * tapCount);
    for (let phase = 0; phase <= phaseCount; phase += 1) {
      const fraction = phase / phaseCount;
      const row = phase * tapCount;
      let sum = 0;
      for (let tap = 0; tap < tapCount; tap += 1) {
        const distance = tap - halfWidth + 1 - fraction;
        const ratio = distance / halfWidth;
        const window =
          Math.abs(ratio) <= 1 ? besselI0(beta * Math.sqrt(1 - ratio * ratio)) / windowNorm : 0;
        const x = Math.PI * cutoff * distance;
        const sinc = x === 0 ? 1 : Math.sin(x) / x;
        const weight = cutoff * sinc * window;
        table[row + tap] = weight;
        sum += weight;
      }
      // Unity DC gain for every phase.
      if (sum !== 0) {
        for (let tap = 0; tap < tapCount; tap += 1) {
          table[row + tap] = table[row + tap]! / sum;
        }
      }
    }

    this.halfWidth = halfWidth;
    this.tapCount = tapCount;
    this.phaseCount = phaseCount;
    this.table = table;
    this.buffer = new Float32Array(Math.max(4096, tapCount * 4));
    // Zero history before the first sample keeps output 0 aligned to input 0.
    this.length = halfWidth - 1;
  }

  /** Buffer position of output sample `index`. */
  private positionOf(index: number): number {
    return this.halfWidth - 1 + index * this.step - this.dropped;
  }

  /** Consumes source samples and returns every output sample now available. */
  process(input: Float32Array): Float32Array {
    if (this.passthrough) {
      return new Float32Array(input);
    }

    const required = this.length + input.length;
    if (required > this.buffer.length) {
      let nextLength = this.buffer.length;
      while (nextLength < required) nextLength *= 2;
      const next = new Float32Array(nextLength);
      next.set(this.buffer.subarray(0, this.length));
      this.buffer = next;
    }
    this.buffer.set(input, this.length);
    this.length += input.length;

    const limit = this.length - this.halfWidth;
    let count = 0;
    while (this.positionOf(this.emitted + count) < limit) {
      count += 1;
    }
    const output = new Float32Array(count);
    const buffer = this.buffer;
    const table = this.table;
    const taps = this.tapCount;
    const phases = this.phaseCount;
    const halfWidth = this.halfWidth;

    for (let index = 0; index < count; index += 1) {
      const position = this.positionOf(this.emitted + index);
      const integer = Math.floor(position);
      const phasePosition = (position - integer) * phases;
      const phase = Math.floor(phasePosition);
      const blend = phasePosition - phase;
      const start = integer - halfWidth + 1;
      const rowA = phase * taps;
      let accA = 0;
      for (let tap = 0; tap < taps; tap += 1) {
        accA += buffer[start + tap]! * table[rowA + tap]!;
      }
      if (blend > 0) {
        const rowB = rowA + taps;
        let accB = 0;
        for (let tap = 0; tap < taps; tap += 1) {
          accB += buffer[start + tap]! * table[rowB + tap]!;
        }
        accA += (accB - accA) * blend;
      }
      output[index] = accA;
    }
    this.emitted += count;

    const consumed = Math.floor(this.positionOf(this.emitted)) - halfWidth + 1;
    if (consumed > 0) {
      buffer.copyWithin(0, consumed, this.length);
      this.length -= consumed;
      this.dropped += consumed;
    }
    return output;
  }

  /** Emits the samples held back for look-ahead by padding with silence. */
  flush(): Float32Array {
    if (this.passthrough) {
      return new Float32Array(0);
    }
    return this.process(new Float32Array(this.halfWidth));
  }
}

/**
 * Band-limited one-shot resampling with the same output length as
 * `resampleLinear`. Equal rates return the input array.
 */
export function resampleSinc(
  audio: Float32Array,
  fromRate: number,
  toRate: number,
  options?: SincResamplerOptions,
): Float32Array {
  if (fromRate === toRate) {
    return audio;
  }
  const resampler = new SincResampler(fromRate, toRate, options);
  const head = resampler.process(audio);
  const tail = resampler.flush();
  const outputLength = Math.max(1, Math.round(audio.length * (toRate / fromRate)));
  const output = new Float32Array(outputLength);
  output.set(head.subarray(0, outputLength));
  if (head.length < outputLength) {
    output.set(tail.subarray(0, outputLength - head.length), head.length);
  }
  return output;
}

/**
 * Stateful linear resampler with fractional phase accumulator.
 *
 * Maintains fractional phase and boundary samples across arbitrary chunk sizes
 * without lookahead delay, integer underflow, or floating-phase drift.
 *
 * Designed for microphone capture fallbacks (e.g. ScriptProcessorNode) where
 * incoming chunk sizes are irregular or small (e.g. 128, 256, 512 frames) and
 * fractional sample rate ratios (e.g. 44,100 Hz to 16,000 Hz: step = 2.75625)
 * require continuous sample-accurate phase tracking across millions of frames.
 */
export class StatefulLinearResampler {
  readonly sourceSampleRate: number;
  readonly targetSampleRate: number;
  readonly step: number;
  private readonly passthrough: boolean;
  private phase = 0;
  private lastSample = 0;
  private hasLastSample = false;
  private sourceOffset = 0;
  private isPaused = false;

  constructor(sourceSampleRate: number, targetSampleRate: number) {
    if (
      !Number.isFinite(sourceSampleRate) ||
      !Number.isFinite(targetSampleRate) ||
      sourceSampleRate <= 0 ||
      targetSampleRate <= 0
    ) {
      throw new RangeError(
        `Invalid resample rates: from=${sourceSampleRate}, to=${targetSampleRate}`,
      );
    }
    this.sourceSampleRate = sourceSampleRate;
    this.targetSampleRate = targetSampleRate;
    this.passthrough = sourceSampleRate === targetSampleRate;
    this.step = sourceSampleRate / targetSampleRate;
  }

  getPhase(): number {
    return this.phase;
  }

  getLastSample(): number {
    return this.lastSample;
  }

  pause(): void {
    this.isPaused = true;
  }

  resume(): void {
    this.isPaused = false;
  }

  reset(): void {
    this.phase = 0;
    this.lastSample = 0;
    this.hasLastSample = false;
    this.sourceOffset = 0;
    this.isPaused = false;
  }

  process(input: Float32Array): Float32Array {
    if (this.isPaused || input.length === 0) {
      return new Float32Array(0);
    }
    if (this.passthrough) {
      this.lastSample = input[input.length - 1]!;
      this.hasLastSample = true;
      return new Float32Array(input);
    }

    const inputLength = input.length;
    if (inputLength < 0) {
      throw new RangeError(`StatefulLinearResampler received negative input length: ${inputLength}`);
    }

    let i = this.sourceOffset;
    if (i < -1) {
      throw new RangeError(`StatefulLinearResampler negative offset invariant violation: ${i}`);
    }

    if (!this.hasLastSample) {
      this.lastSample = input[0]!;
      this.hasLastSample = true;
      this.phase = 0;
      i = 0;
    }

    const outputValues: number[] = [];

    while (true) {
      const lastNeededIndex = this.phase === 0 ? i : i + 1;
      if (lastNeededIndex >= inputLength) {
        break;
      }

      const s0 = i === -1 ? this.lastSample : input[i]!;
      let value: number;
      if (this.phase === 0) {
        value = s0;
      } else {
        const s1 = input[i + 1]!;
        value = s0 + this.phase * (s1 - s0);
      }
      outputValues.push(value);

      const nextPos = this.phase + this.step;
      let advance = Math.floor(nextPos);
      let nextPhase = nextPos - advance;

      if (nextPhase >= 1) {
        const extra = Math.floor(nextPhase);
        advance += extra;
        nextPhase -= extra;
      }
      if (nextPhase < 0 || Object.is(nextPhase, -0)) {
        nextPhase = 0;
      }
      if (nextPhase > 1 - 1e-12) {
        advance += 1;
        nextPhase = 0;
      }

      if (advance < 0) {
        throw new RangeError(`StatefulLinearResampler negative advance invariant violation: ${advance}`);
      }

      this.phase = nextPhase;
      i += advance;
    }

    const nextOffset = i - inputLength;
    if (nextOffset < -1) {
      throw new RangeError(
        `StatefulLinearResampler sourceOffset underflow invariant violation: ${nextOffset}`,
      );
    }
    this.sourceOffset = nextOffset;
    this.lastSample = input[inputLength - 1]!;

    return Float32Array.from(outputValues);
  }

  flush(): Float32Array {
    if (this.passthrough || !this.hasLastSample) {
      return new Float32Array(0);
    }
    if (this.sourceOffset === -1 && this.phase > 0) {
      const sample = (1 - this.phase) * this.lastSample;
      this.sourceOffset = 0;
      this.phase = 0;
      this.hasLastSample = false;
      return new Float32Array([sample]);
    }
    this.sourceOffset = 0;
    this.phase = 0;
    this.hasLastSample = false;
    return new Float32Array(0);
  }
}


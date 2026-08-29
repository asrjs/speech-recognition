/**
 * Exact mixed-radix FFT for transform sizes N = 5^a * 2^b.
 *
 * The Whisper model contract fixes n_fft=400 (25 * 16) and Qwen reuses that
 * frontend. Before this module the exact path used a Bluestein chirp-z
 * transform, which costs three 1024-point radix-2 FFTs per frame even though
 * 400 factors cleanly. This class applies Cooley-Tukey decomposition with
 * N1 = 5 repeatedly (five strided subsequences of length N/5) until the
 * remaining size is a power of two, where a cached radix-2 kernel finishes
 * the work. The result is the same DFT up to floating-point rounding, so it
 * is a drop-in replacement for the general Bluestein path whenever the size
 * is covered.
 *
 * The interface mirrors `RadixFivePowerOfTwoFft` in `src/models/lasr-ctc`:
 * in-place `transform(real, imaginary)` over Float64Array buffers of length
 * `size`, plus `transformRealInput` for the common mel case where each frame
 * is real-valued and only the first half of the spectrum is consumed.
 */

interface Pow2Twiddles {
  readonly cos: Float64Array;
  readonly sin: Float64Array;
  readonly bitReverse: Uint32Array;
}

const POW2_TWIDDLE_CACHE = new Map<number, Pow2Twiddles>();

function isPowerOfTwo(size: number): boolean {
  return Number.isInteger(size) && size >= 1 && (size & (size - 1)) === 0;
}

function pow2Twiddles(size: number): Pow2Twiddles {
  const cached = POW2_TWIDDLE_CACHE.get(size);
  if (cached) {
    return cached;
  }

  const bits = Math.log2(size);
  const half = size >> 1;
  const cos = new Float64Array(Math.max(half, 1));
  const sin = new Float64Array(Math.max(half, 1));
  for (let index = 0; index < half; index += 1) {
    const angle = (-2 * Math.PI * index) / size;
    cos[index] = Math.cos(angle);
    sin[index] = Math.sin(angle);
  }

  const bitReverse = new Uint32Array(size);
  for (let index = 0; index < size; index += 1) {
    let value = index;
    let reversed = 0;
    for (let bit = 0; bit < bits; bit += 1) {
      reversed = (reversed << 1) | (value & 1);
      value >>= 1;
    }
    bitReverse[index] = reversed;
  }

  const twiddles: Pow2Twiddles = { cos, sin, bitReverse };
  POW2_TWIDDLE_CACHE.set(size, twiddles);
  return twiddles;
}

function radix2Fft(real: Float64Array, imaginary: Float64Array, size: number): void {
  if (size <= 1) {
    return;
  }
  const twiddles = pow2Twiddles(size);
  const bitReverse = twiddles.bitReverse;
  for (let index = 0; index < size; index += 1) {
    const swapped = bitReverse[index] as number;
    if (index >= swapped) {
      continue;
    }
    const realValue = real[index] as number;
    real[index] = real[swapped] as number;
    real[swapped] = realValue;
    const imaginaryValue = imaginary[index] as number;
    imaginary[index] = imaginary[swapped] as number;
    imaginary[swapped] = imaginaryValue;
  }

  for (let length = 2; length <= size; length <<= 1) {
    const halfLength = length >> 1;
    const step = size / length;
    for (let segment = 0; segment < size; segment += length) {
      for (let offset = 0; offset < halfLength; offset += 1) {
        const twiddleIndex = offset * step;
        const cosine = twiddles.cos[twiddleIndex] as number;
        const sine = twiddles.sin[twiddleIndex] as number;
        const first = segment + offset;
        const second = first + halfLength;
        const oddReal = real[second] as number;
        const oddImaginary = imaginary[second] as number;
        const tReal = oddReal * cosine - oddImaginary * sine;
        const tImaginary = oddReal * sine + oddImaginary * cosine;
        const evenReal = real[first] as number;
        const evenImaginary = imaginary[first] as number;
        real[first] = evenReal + tReal;
        imaginary[first] = evenImaginary + tImaginary;
        real[second] = evenReal - tReal;
        imaginary[second] = evenImaginary - tImaginary;
      }
    }
  }
}

/** Returns whether `MixedRadixFft` supports the size exactly: N = 5^a * 2^b. */
export function isMixedRadixSize(size: number): boolean {
  if (!Number.isInteger(size) || size < 1) {
    return false;
  }
  let remaining = size;
  while (remaining % 5 === 0) {
    remaining /= 5;
  }
  return isPowerOfTwo(remaining);
}

/**
 * Exact forward DFT for sizes N = 5^a * 2^b via repeated radix-5
 * Cooley-Tukey decomposition down to a cached power-of-two kernel.
 */
export class MixedRadixFft {
  readonly size: number;
  private readonly n2: number;
  private readonly child: MixedRadixFft | null;
  private readonly scratchReal: Float64Array;
  private readonly scratchImaginary: Float64Array;
  private readonly partReal: Float64Array;
  private readonly partImaginary: Float64Array;
  private readonly outReal: Float64Array;
  private readonly outImaginary: Float64Array;
  private readonly phaseCos: Float64Array;
  private readonly phaseSin: Float64Array;
  private readonly partTwiddleCos: Float64Array;
  private readonly partTwiddleSin: Float64Array;
  private readonly frameReal: Float64Array;
  private readonly frameImaginary: Float64Array;

  constructor(size: number) {
    if (!isMixedRadixSize(size)) {
      throw new RangeError("MixedRadixFft requires size = 5^a * 2^b. Received " + String(size) + ".");
    }
    this.size = size;
    if (isPowerOfTwo(size)) {
      this.n2 = size;
      this.child = null;
      this.scratchReal = new Float64Array(0);
      this.scratchImaginary = new Float64Array(0);
      this.partReal = new Float64Array(0);
      this.partImaginary = new Float64Array(0);
      this.outReal = new Float64Array(0);
      this.outImaginary = new Float64Array(0);
      this.phaseCos = new Float64Array(0);
      this.phaseSin = new Float64Array(0);
      this.partTwiddleCos = new Float64Array(0);
      this.partTwiddleSin = new Float64Array(0);
      this.frameReal = new Float64Array(0);
      this.frameImaginary = new Float64Array(0);
      return;
    }

    const n2 = size / 5;
    this.n2 = n2;
    this.child = isPowerOfTwo(n2) ? null : new MixedRadixFft(n2);
    this.scratchReal = new Float64Array(n2);
    this.scratchImaginary = new Float64Array(n2);
    this.partReal = new Float64Array(size);
    this.partImaginary = new Float64Array(size);
    this.outReal = new Float64Array(size);
    this.outImaginary = new Float64Array(size);
    this.frameReal = new Float64Array(size);
    this.frameImaginary = new Float64Array(size);
    // Phase table for W_N^(n1*k2), n1 in 1..4 (n1 = 0 is the identity and is
    // folded directly into the combine loop). Sines are stored conjugated
    // (-sin) so both the phase factor and the 5-point factor W_5^(n1*k1)
    // keep the forward-DFT sign convention.
    this.phaseCos = new Float64Array(size);
    this.phaseSin = new Float64Array(size);
    this.partTwiddleCos = new Float64Array(25);
    this.partTwiddleSin = new Float64Array(25);
    for (let n1 = 1; n1 < 5; n1 += 1) {
      for (let k2 = 0; k2 < n2; k2 += 1) {
        const offset = n1 * n2 + k2;
        const angle = (2 * Math.PI * n1 * k2) / size;
        this.phaseCos[offset] = Math.cos(angle);
        this.phaseSin[offset] = -Math.sin(angle);
      }
    }
    for (let n1 = 0; n1 < 5; n1 += 1) {
      for (let k1 = 0; k1 < 5; k1 += 1) {
        const fiveAngle = (2 * Math.PI * n1 * k1) / 5;
        this.partTwiddleCos[n1 * 5 + k1] = Math.cos(fiveAngle);
        this.partTwiddleSin[n1 * 5 + k1] = -Math.sin(fiveAngle);
      }
    }
  }

  /**
   * In-place forward DFT over buffers of length `this.size`. Buffers must
   * not alias internal scratch state. When `zeroImaginary` is true the
   * imaginary buffer is cleared and the gather loop skips imaginary reads
   * (the strided subsequences of a real sequence are themselves real).
   */
  transform(real: Float64Array, imaginary: Float64Array, zeroImaginary = false): void {
    if (zeroImaginary) {
      imaginary.fill(0);
    }
    if (isPowerOfTwo(this.size)) {
      radix2Fft(real, imaginary, this.size);
      return;
    }

    const n2 = this.n2;
    for (let n1 = 0; n1 < 5; n1 += 1) {
      for (let j = 0; j < n2; j += 1) {
        const source = n1 + 5 * j;
        this.scratchReal[j] = real[source] as number;
        if (!zeroImaginary) {
          this.scratchImaginary[j] = imaginary[source] as number;
        }
      }
      if (zeroImaginary) {
        // Clear stale imaginary outputs from the previous subsequence before
        // the FFT consumes them.
        this.scratchImaginary.fill(0);
      }
      if (this.child) {
        this.child.transform(this.scratchReal, this.scratchImaginary);
      } else {
        radix2Fft(this.scratchReal, this.scratchImaginary, n2);
      }
      this.partReal.set(this.scratchReal, n1 * n2);
      this.partImaginary.set(this.scratchImaginary, n1 * n2);
    }

    for (let k2 = 0; k2 < n2; k2 += 1) {
      for (let k1 = 0; k1 < 5; k1 += 1) {
        // n1 = 0 term: W_N^0 * W_5^0 = 1.
        let sumReal = this.partReal[k2] as number;
        let sumImaginary = this.partImaginary[k2] as number;
        for (let n1 = 1; n1 < 5; n1 += 1) {
          const phaseOffset = n1 * n2 + k2;
          const fiveOffset = n1 * 5 + k1;
          const pc = this.phaseCos[phaseOffset] as number;
          const ps = this.phaseSin[phaseOffset] as number;
          const wc = this.partTwiddleCos[fiveOffset] as number;
          const ws = this.partTwiddleSin[fiveOffset] as number;
          const twiddleReal = pc * wc - ps * ws;
          const twiddleImaginary = pc * ws + ps * wc;
          const partIndex = n1 * n2 + k2;
          const partValueReal = this.partReal[partIndex] as number;
          const partValueImaginary = this.partImaginary[partIndex] as number;
          sumReal += partValueReal * twiddleReal - partValueImaginary * twiddleImaginary;
          sumImaginary += partValueReal * twiddleImaginary + partValueImaginary * twiddleReal;
        }
        this.outReal[k2 + n2 * k1] = sumReal;
        this.outImaginary[k2 + n2 * k1] = sumImaginary;
      }
    }

    real.set(this.outReal);
    imaginary.set(this.outImaginary);
  }

  /**
   * Forward DFT of a real-valued input frame. Writes the first
   * `outputBinCount` bins of the full spectrum into `outReal` and
   * `outImaginary`; the remaining bins are the conjugate mirror that
   * power-spectrum consumers never read.
   */
  transformRealInput(
    input: Float32Array,
    outReal: Float64Array,
    outImaginary: Float64Array,
    outputBinCount: number,
  ): void {
    const size = this.size;
    const frame = this.frameReal;
    for (let index = 0; index < size; index += 1) {
      frame[index] = input[index] as number;
    }
    // Dedicated zeroed imaginary buffer: transform() writes its final
    // spectrum back into both buffers it is given, so the input frame and
    // any internal live table must not be passed as those arguments.
    const imag = this.frameImaginary;
    imag.fill(0);
    this.transform(frame, imag, false);
    outReal.set(frame.subarray(0, outputBinCount));
    outImaginary.set(imag.subarray(0, outputBinCount));
  }
}

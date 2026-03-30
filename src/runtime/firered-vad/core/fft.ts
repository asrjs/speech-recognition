const TWO_PI = Math.PI * 2;

export function upperPowerOfTwo(value: number): number {
  return 2 ** Math.ceil(Math.log2(value));
}

export function makeBitrev(n: number): Int32Array {
  const bitrev = new Int32Array(n);
  const n2 = n / 2;
  let i = 0;
  let j = 0;
  while (true) {
    bitrev[i] = j;
    i += 1;
    if (i >= n) {
      break;
    }
    let k = n2;
    while (k <= j) {
      j -= k;
      k /= 2;
    }
    j += k;
  }
  return bitrev;
}

export function makeSintbl(n: number): Float32Array {
  const table = new Float32Array(n + n / 4);
  const n2 = n / 2;
  const n4 = n / 4;
  const n8 = n / 8;
  let t = Math.sin(Math.PI / n);
  let dc = 2 * t * t;
  let ds = Math.sqrt(dc * (2 - dc));
  t = 2 * dc;
  let c = 1;
  let s = 0;
  table[n4] = 1;
  table[0] = 0;

  for (let i = 1; i < n8; i += 1) {
    c -= dc;
    dc += t * c;
    s += ds;
    ds -= t * s;
    table[i] = s;
    table[n4 - i] = c;
  }
  if (n8 !== 0) {
    table[n8] = Math.sqrt(0.5);
  }
  for (let i = 0; i < n4; i += 1) {
    table[n2 - i] = table[i]!;
  }
  for (let i = 0; i < n2 + n4; i += 1) {
    table[i + n2] = -table[i]!;
  }
  return table;
}

export function fft(
  bitrev: Int32Array,
  sintbl: Float32Array,
  real: Float32Array,
  imag: Float32Array,
): void {
  const n = real.length;
  const n4 = n / 4;

  for (let i = 0; i < n; i += 1) {
    const j = bitrev[i]!;
    if (i < j) {
      const tr = real[i]!;
      real[i] = real[j]!;
      real[j] = tr;
      const ti = imag[i]!;
      imag[i] = imag[j]!;
      imag[j] = ti;
    }
  }

  for (let k = 1; k < n; k *= 2) {
    const k2 = k * 2;
    const d = n / k2;
    let h = 0;
    for (let j = 0; j < k; j += 1) {
      const c = sintbl[h + n4]!;
      const s = sintbl[h]!;
      for (let i = j; i < n; i += k2) {
        const ik = i + k;
        const dx = s * imag[ik]! + c * real[ik]!;
        const dy = c * imag[ik]! - s * real[ik]!;
        real[ik] = real[i]! - dx;
        real[i] = (real[i] ?? 0) + dx;
        imag[ik] = imag[i]! - dy;
        imag[i] = (imag[i] ?? 0) + dy;
      }
      h += d;
    }
  }
}

export function createPoveyWindow(length: number): Float32Array {
  const window = new Float32Array(length);
  const a = TWO_PI / (length - 1);
  for (let i = 0; i < length; i += 1) {
    window[i] = (0.5 - 0.5 * Math.cos(a * i)) ** 0.85;
  }
  return window;
}

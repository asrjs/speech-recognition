// Weight-free diagnostic only; not a production FFT or a claimed MKL algorithm.
// Products are JS binary64 followed by explicit f32 rounding, not hardware FMA.
export function fftRadix4F32(real, imag, options = {}) {
  if (!(real instanceof Float32Array) || !(imag instanceof Float32Array) ||
      real.length !== imag.length || ![4, 16, 64, 256].includes(real.length) ||
      !real.every(Number.isFinite) || !imag.every(Number.isFinite))
    throw new Error('Expected finite equal-length f32 diagnostic inputs of size 4/16/64/256');
  if (!options || typeof options !== 'object' || Array.isArray(options) ||
      Object.keys(options).some(key => key !== 'products')) throw new Error('Invalid radix4 options');
  const { products = 'single-round' } = options;
  if (!['separate', 'single-round'].includes(products)) throw new Error('Invalid radix4 product mode');
  const n = real.length, digits = Math.log(n) / Math.log(4), f = Math.fround;
  const re = new Float32Array(n), im = new Float32Array(n);
  for (let index = 0; index < n; index++) {
    let value = index, reversed = 0;
    for (let digit = 0; digit < digits; digit++) {
      reversed = reversed * 4 + value % 4;
      value = Math.floor(value / 4);
    }
    re[reversed] = real[index]; im[reversed] = imag[index];
  }
  function multiply(r, i, branch, j, length) {
    if (j === 0) return [r, i];
    // Cardinal roots are exact; trig approximations retain spurious residues.
    const phase = (branch * j) % length;
    if (phase === 0) return [r, i];
    if (phase === length / 4) return [i, -r];
    if (phase === length / 2) return [-r, -i];
    if (phase === 3 * length / 4) return [-i, r];
    const angle = -2 * Math.PI * branch * j / length;
    const c = f(Math.cos(angle)), s = f(Math.sin(angle));
    return products === 'single-round'
      ? [f(r * c - i * s), f(r * s + i * c)]
      : [f(f(r * c) - f(i * s)), f(f(r * s) + f(i * c))];
  }
  for (let length = 4; length <= n; length *= 4) {
    const quarter = length / 4;
    for (let offset = 0; offset < n; offset += length) for (let j = 0; j < quarter; j++) {
      const a = offset + j, b = a + quarter, c = b + quarter, d = c + quarter;
      const ar = re[a], ai = im[a];
      const [br, bi] = multiply(re[b], im[b], 1, j, length);
      const [cr, ci] = multiply(re[c], im[c], 2, j, length);
      const [dr, di] = multiply(re[d], im[d], 3, j, length);
      const t0r = f(ar + cr), t0i = f(ai + ci);
      const t1r = f(ar - cr), t1i = f(ai - ci);
      const t2r = f(br + dr), t2i = f(bi + di);
      const t3r = f(bi - di), t3i = f(dr - br);
      re[a] = f(t0r + t2r); im[a] = f(t0i + t2i);
      re[b] = f(t1r + t3r); im[b] = f(t1i + t3i);
      re[c] = f(t0r - t2r); im[c] = f(t0i - t2i);
      re[d] = f(t1r - t3r); im[d] = f(t1i - t3i);
    }
  }
  return { real: re, imag: im };
}

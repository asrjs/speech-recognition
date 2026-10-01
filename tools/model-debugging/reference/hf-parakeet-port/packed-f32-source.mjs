// Diagnostic source transform only. Never writes or imports published runtime code.
export function packedF32Source(source) {
  function once(text, before, after) {
    if (text.split(before).length !== 2) throw new Error('Packed f32 source contract changed: ' + before);
    return text.replace(before, after);
  }
  const begin = 'export function fft(re, im, size, twiddles) {';
  const end = 'export class JSMelProcessor';
  if (source.split(begin).length !== 2 || source.split(end).length !== 2)
    throw new Error('Packed f32 FFT boundaries changed');
  const start = source.indexOf(begin), stop = source.indexOf(end);
  if (stop <= start) throw new Error('Packed f32 FFT boundary order changed');
  let body = source.slice(start, stop);
  function rounded(pattern, count, replacement) {
    if ([...body.matchAll(pattern)].length !== count) throw new Error('Packed f32 arithmetic contract changed');
    body = body.replace(pattern, replacement);
  }
  rounded(/const (wCos|wSin) = ([^;]+);/g, 6, (_, name, expression) => `const ${name} = Math.fround(${expression});`);
  rounded(/const (tRe|tIm) = (re\[q\]|im\[q\]) \* (wCos|wSin) ([+-]) (re\[q\]|im\[q\]) \* (wCos|wSin);/g,
    6, (_, name, a, b, op, c, d) => `const ${name} = Math.fround(Math.fround(${a} * ${b}) ${op} Math.fround(${c} * ${d}));`);
  rounded(/((?:re|im)\[[^\]]+\]) = ([^;]+);/g, 36,
    (_, target, expression) => `${target} = Math.fround(${expression});`);
  source = source.slice(0, start) + body + source.slice(stop);
  for (const [name, expression] of [
    ['xeR', '0.5 * (rk + rnk)'], ['xeI', '0.5 * (ik - ink)'],
    ['xoR', '0.5 * (ik + ink)'], ['xoI', '-0.5 * (rk - rnk)'],
  ]) source = once(source, `const ${name} = ${expression};`,
    `const ${name} = Math.fround(${expression.replace(/\(([^)]+)\)/, 'Math.fround($1)')});`);
  for (const name of ['wc', 'ws']) {
    const expression = name === 'wc' ? 'this.twiddles.cos[k]' : 'this.twiddles.sin[k]';
    source = once(source, `const ${name} = ${expression};`, `const ${name} = Math.fround(${expression});`);
  }
  for (const [name, a, b, op, c, d] of [
    ['tr', 'xoR', 'wc', '-', 'xoI', 'ws'], ['ti', 'xoR', 'ws', '+', 'xoI', 'wc'],
  ]) source = once(source, `const ${name} = ${a} * ${b} ${op} ${c} * ${d};`,
    `const ${name} = Math.fround(Math.fround(${a} * ${b}) ${op} Math.fround(${c} * ${d}));`);
  for (const [name, expression] of [['xkR', 'xeR + tr'], ['xkI', 'xeI + ti'],
    ['xnkR', 'xeR - tr'], ['xnkI', 'xeI - ti']])
    source = once(source, `const ${name} = ${expression};`, `const ${name} = Math.fround(${expression});`);
  // DC/Nyquist addition is rounded at diagnosticComplexPower's component input;
  // keep the source boundary intact for guarded capture instrumentation.
  return source;
}

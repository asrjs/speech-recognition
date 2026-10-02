
import { JSMelProcessor, MEL_CONSTANTS } from './dist/audio/js-mel.js';
import { readFileSync } from 'node:fs';
import { normalizeCandidate } from './tools/model-debugging/reference/hf-parakeet-port/normalization-reduction.mjs';
import { packedF32Source } from './tools/model-debugging/reference/hf-parakeet-port/packed-f32-source.mjs';
let input = ''; for await (const chunk of process.stdin) input += chunk;
const data = JSON.parse(input);
if (data.powerOverrides !== undefined && !Array.isArray(data.powerOverrides))
  throw new Error('Selected power overrides must be an array');
const pcm = Float32Array.from(data.pcm);
const options = { nMels: data.bins, validLengthMode: 'onnx' };
const raw = new JSMelProcessor({ ...options, normalization: 'none' }).process(pcm);
let officialLogInputControl = null;
if (Object.hasOwn(data, 'referenceLogInput')) {
  const values = data.referenceLogInput;
  if (!Array.isArray(values) || values.length !== data.bins * raw.frameCount ||
      values.some(value => typeof value !== 'number' || !Number.isFinite(value) || value <= 0 || Math.fround(value) !== value))
    throw new Error('Official log input must be a full positive finite float32 mel boundary');
  const logRaw = Float32Array.from(values, value => Math.log(value));
  const result = normalizeCandidate(logRaw, raw.frameCount, raw.length, 8);
  officialLogInputControl = { input: values, raw: Array.from(logRaw),
    features: Array.from(result.features), mean: Array.from(result.mean), std: Array.from(result.std) };
}
const normalized = new JSMelProcessor({ ...options, normalization: 'per_feature' }).process(pcm);
const processor = new JSMelProcessor(options);
const sameRawNormalized = processor.normalizeFeatures(Float32Array.from(data.referenceRaw),
  raw.frameCount, raw.length);
function readReferenceComplex(boundary, expectedShape) {
  if (!boundary || boundary.dtype !== 'complex64' || boundary.layout !== 'batch-bin-frame' ||
    boundary.encoding !== 'base64-f32le-components' || !Array.isArray(boundary.shape) ||
    boundary.shape.length !== expectedShape.length ||
    boundary.shape.some((value, index) => value !== expectedShape[index]))
    throw new Error('Reference complex metadata contract differs');
  const count = expectedShape.reduce((product, value) => product * value, 1);
  function readComponent(encoded) {
    if (typeof encoded !== 'string') throw new Error('Reference complex component bytes missing');
    const bytes = Buffer.from(encoded, 'base64');
    if (bytes.length !== count * 4 || bytes.toString('base64') !== encoded)
      throw new Error('Reference complex component byte contract differs');
    const values = new Float32Array(count);
    for (let index = 0; index < count; index++) values[index] = bytes.readFloatLE(index * 4);
    if (values.some(value => !Number.isFinite(value))) throw new Error('Reference complex component is nonfinite');
    return values;
  }
  return { real: readComponent(boundary.realF32LE), imag: readComponent(boundary.imagF32LE) };
}
// An isolated arithmetic experiment, not a mutation of the library/oracle.
let source = readFileSync('./dist/audio/js-mel.js', 'utf8');
const productionSource = source;
for (const [before, after] of [
  ['const PREEMPH = MEL_CONSTANTS.PREEMPH;', 'const PREEMPH = Math.fround(MEL_CONSTANTS.PREEMPH);'],
  ['audio[index] - PREEMPH * audio[index - 1]', 'audio[index] - Math.fround(PREEMPH * audio[index - 1])'],
]) {
  if (source.split(before).length !== 2) throw new Error('Re-inspect actual processor before this experiment');
  source = source.replace(before, after);
}
async function experiment(name, { window = false, windowProduct = false, filterbank = false,
  referencePower = false, magnitudeRoundTrip = false, capturePower = false, complexPower = false,
  fullFft = null, referenceComplexPower = null, production = false, capturePacked = false, packedF32 = false,
  powerOverrides = null, melArithmetic = null, melSpectrum = null, roundLogGuard = false } = {}) {
  if (melSpectrum !== null && (!['production', 'official-complex-rounded', 'aligned-packed'].includes(melSpectrum) ||
      melArithmetic !== 'f64-product-f32-sum' || !capturePower || referencePower || production ||
      magnitudeRoundTrip || fullFft || packedF32 || capturePacked || powerOverrides !== null ||
      (melSpectrum === 'production' ? (window || windowProduct || filterbank || referenceComplexPower || complexPower) :
        (!window || !windowProduct || !filterbank || (melSpectrum === 'aligned-packed'
          ? (!complexPower || referenceComplexPower) : (complexPower || referenceComplexPower !== 'fp32-rounded'))))))
    throw new Error('Mel spectrum control must preserve its named upstream boundary');
  if (roundLogGuard && melSpectrum === null) throw new Error('Guard rounding requires an isolated mel spectrum control');
  if (melArithmetic !== null && melSpectrum === null && (!['f32-product-f64-sum', 'f32-product-f32-sum', 'f64-product-f32-sum'].includes(melArithmetic) ||
      !referencePower || !capturePower || !window || !windowProduct || !filterbank || production ||
      complexPower || magnitudeRoundTrip || fullFft || referenceComplexPower || powerOverrides !== null))
    throw new Error('Mel arithmetic requires an isolated official-power capture');
  if (powerOverrides !== null) {
    if (!window || !windowProduct || !filterbank || !complexPower || !capturePower || packedF32 || fullFft || production || referencePower ||
        referenceComplexPower || !Array.isArray(powerOverrides) || powerOverrides.length === 0)
      throw new Error('Selected power overrides require the aligned complex-power control');
    const seen = new Set(), count = ((MEL_CONSTANTS.N_FFT >> 1) + 1) * raw.frameCount;
    if (!Array.isArray(data.referencePower) || data.referencePower.length !== count ||
        data.referencePower.some(value => !Number.isFinite(value) || value < 0 || Math.fround(value) !== value))
      throw new Error('Selected power reference must be finite nonnegative full float32 spectrum');
    for (const coordinate of powerOverrides) {
      if (!Array.isArray(coordinate) || coordinate.length !== 2 || !coordinate.every(Number.isSafeInteger) ||
          coordinate[0] < 0 || coordinate[0] >= raw.length || coordinate[1] < 0 ||
          coordinate[1] > (MEL_CONSTANTS.N_FFT >> 1) || seen.has(coordinate.join(':')))
        throw new Error('Selected power coordinate invalid, duplicate or out of valid range');
      seen.add(coordinate.join(':'));
    }
  }
  if (packedF32 && (!complexPower || !window || !windowProduct || !filterbank || !capturePower ||
      fullFft || referencePower || referenceComplexPower || production || capturePacked))
    throw new Error('Packed f32 control requires the identical aligned complex-power boundary');
  if (capturePacked && (!capturePower || referencePower || magnitudeRoundTrip || complexPower ||
      fullFft || referenceComplexPower || production))
    throw new Error('Packed arithmetic capture must retain original power expressions');
  if (production && (!capturePower || window || windowProduct || filterbank || referencePower ||
      magnitudeRoundTrip || complexPower || fullFft || referenceComplexPower))
    throw new Error('Production capture must preserve unmodified arithmetic');
  let candidateSource = production || melSpectrum === 'production' ? productionSource : source;
  if (melArithmetic) {
    const boundary = 'melValue +=\n                        this.powerBuf[freqIndex] * this.melFilterbank[filterbankOffset + freqIndex];';
    if (candidateSource.split(boundary).length !== 2) throw new Error('Mel accumulator contract changed');
    const product = 'this.powerBuf[freqIndex] * this.melFilterbank[filterbankOffset + freqIndex]';
    const term = melArithmetic.startsWith('f32-product') ? `Math.fround(${product})` : product;
    const replacement = melArithmetic.endsWith('f32-sum')
      ? `melValue = Math.fround(melValue + ${term});` : `melValue += ${term};`;
    candidateSource = candidateSource.replace(boundary, replacement);
  }
  if (packedF32) candidateSource = packedF32Source(candidateSource, data.packedProductRounding);
  if (referenceComplexPower !== null && !['library', 'fp32-rounded'].includes(referenceComplexPower))
    throw new Error('Invalid reference complex power control');
  if (referenceComplexPower && (fullFft || referencePower || magnitudeRoundTrip || complexPower || !capturePower))
    throw new Error('Reference complex control must have its own unmodified power boundary');
  if (fullFft !== null && !['float32', 'float64'].includes(fullFft)) throw new Error('Invalid full FFT control');
  if (fullFft && (referencePower || magnitudeRoundTrip || complexPower || !capturePower))
    throw new Error('Full FFT control must have its own unmodified power boundary');
  if (windowProduct) {
    for (const expression of [
      'padded[offset + sampleIndex] * this.hannWindow[sampleIndex]',
      'padded[offset + sampleIndex + 1] * this.hannWindow[sampleIndex + 1]',
    ]) {
      if (candidateSource.split(expression).length !== 2) throw new Error('Window multiply contract changed');
      candidateSource = candidateSource.replace(expression, `Math.fround(${expression})`);
    }
  }
  if (referencePower || magnitudeRoundTrip) {
    if (referencePower && (!Array.isArray(data.referencePower) ||
        data.referencePower.length !== ((MEL_CONSTANTS.N_FFT >> 1) + 1) * raw.frameCount ||
        data.referencePower.some(value => !Number.isFinite(value) || value < 0 || Math.fround(value) !== value)))
      throw new Error((data.powerOverrides?.length ? 'Selected power reference' : 'Reference power') +
        ' must be finite nonnegative full float32 spectrum');
    const boundary = 'this.powerBuf[quarterN] = quarterRe * quarterRe + quarterIm * quarterIm;';
    if (candidateSource.split(boundary).length !== 2) throw new Error('Power boundary contract changed');
    const injected = referencePower
      ? 'for (let bin = 0; bin < N_FREQ_BINS; bin++) this.powerBuf[bin] = data.referencePower[bin * nFrames + frameIndex];'
      : 'for (let bin = 0; bin < N_FREQ_BINS; bin++) { const magnitude = Math.fround(Math.sqrt(this.powerBuf[bin])); this.powerBuf[bin] = Math.fround(magnitude * magnitude); }';
    // The data-URL module receives only this run's immutable reference input.
    candidateSource = 'const data = ' + JSON.stringify({ referencePower: data.referencePower }) + ';\n' +
      candidateSource.replace(boundary, boundary + '\n' + injected);
  }
  if (fullFft) {
    for (const component of ['Re', 'Im']) {
      const before = `this.fft${component} = new Float64Array(N_FFT >> 1);`;
      if (candidateSource.split(before).length !== 2) throw new Error('FFT scratch contract changed');
      candidateSource = candidateSource.replace(before,
        `this.fft${component} = new ${fullFft === 'float32' ? 'Float32Array' : 'Float64Array'}(N_FFT);`);
    }
    const begin = 'for (let k = 0; k < halfN; k += 1) {';
    const end = 'this.powerBuf[quarterN] = quarterRe * quarterRe + quarterIm * quarterIm;';
    if (candidateSource.split(begin).length !== 2 || candidateSource.split(end).length !== 2)
      throw new Error('Packed real FFT boundary contract changed');
    const start = candidateSource.indexOf(begin), stop = candidateSource.indexOf(end);
    if (stop <= start) throw new Error('Packed real FFT boundary order changed');
    // Same float32 windowed input and full-size twiddles in both controls.
    // Float32 scratch rounds stored butterflies; JS intermediates/twiddles
    // remain double, so this is not an emulation of Torch/MKL's FFT algorithm.
    candidateSource = candidateSource.slice(0, start) + `
      for (let sample = 0; sample < N_FFT; sample++) {
        this.fftRe[sample] = Math.fround(padded[offset + sample] * this.hannWindow[sample]);
        capturedFrames[frameIndex * N_FFT + sample] = this.fftRe[sample];
        this.fftIm[sample] = 0;
      }
      fft(this.fftRe, this.fftIm, N_FFT, this.twiddles);
      for (let bin = 0; bin < N_FREQ_BINS; bin++) {
        this.powerBuf[bin] = diagnosticComplexPower(this.fftRe[bin],
          bin === 0 || bin === halfN ? 0 : this.fftIm[bin]);
      }
    ` + candidateSource.slice(stop + end.length);
  }
  if (complexPower || production || capturePacked) {
    const fftBoundary = 'fft(this.fftRe, this.fftIm, halfN, this.twiddlesHalf);';
    if (candidateSource.split(fftBoundary).length !== 2) throw new Error('Packed FFT input capture contract changed');
    candidateSource = candidateSource.replace(fftBoundary, `for (let sample = 0; sample < N_FFT; sample++) {
      capturedFrames[frameIndex * N_FFT + sample] = sample % 2 === 0
        ? this.fftRe[sample >> 1] : this.fftIm[sample >> 1];
    }\n` + fftBoundary);
    if (complexPower) for (const [before, after] of [
      ['this.powerBuf[0] = (z0r + z0i) * (z0r + z0i);', 'this.powerBuf[0] = diagnosticComplexPower(z0r + z0i, 0);'],
      ['this.powerBuf[halfN] = (z0r - z0i) * (z0r - z0i);', 'this.powerBuf[halfN] = diagnosticComplexPower(z0r - z0i, 0);'],
      ['this.powerBuf[k] = xkR * xkR + xkI * xkI;', 'this.powerBuf[k] = diagnosticComplexPower(xkR, xkI);'],
      ['this.powerBuf[halfN - k] = xnkR * xnkR + xnkI * xnkI;', 'this.powerBuf[halfN - k] = diagnosticComplexPower(xnkR, xnkI);'],
      ['this.powerBuf[quarterN] = quarterRe * quarterRe + quarterIm * quarterIm;', 'this.powerBuf[quarterN] = diagnosticComplexPower(quarterRe, quarterIm);'],
    ]) {
      if (candidateSource.split(before).length !== 2) throw new Error('Complex power contract changed');
      candidateSource = candidateSource.replace(before, after);
    }
    for (const [bin, re, im] of [
      ['0', 'z0r + z0i', '0'], ['halfN', 'z0r - z0i', '0'],
      ['k', 'xkR', 'xkI'], ['halfN - k', 'xnkR', '-xnkI'],
      ['quarterN', 'quarterRe', '-quarterIm'],
    ]) {
      const boundary = `this.powerBuf[${bin}] = `;
      if (candidateSource.split(boundary).length !== 2) throw new Error('Packed complex capture contract changed');
      // Paired upper/quarter bins require conjugation for the negative-exponent
      // DFT convention. The original power expression remains unchanged.
      candidateSource = candidateSource.replace(boundary,
        `capturedReal[(${bin}) * nFrames + frameIndex] = ${packedF32 ? `Math.fround(${re})` : re};
         capturedImag[(${bin}) * nFrames + frameIndex] = ${packedF32 ? `Math.fround(${im})` : im};\n` + boundary);
    }
  }
  if (complexPower || fullFft || referenceComplexPower === 'fp32-rounded') {
    candidateSource = `function diagnosticComplexPower(re, im) {
      const r = Math.fround(re), i = Math.fround(im);
      const sum = Math.fround(Math.fround(r * r) + Math.fround(i * i));
      const magnitude = Math.fround(Math.sqrt(sum));
      return Math.fround(magnitude * magnitude);
    }\n` + candidateSource;
  }
  if (capturePower) {
    const boundary = complexPower
      ? 'this.powerBuf[quarterN] = diagnosticComplexPower(quarterRe, quarterIm);'
      : 'this.powerBuf[quarterN] = quarterRe * quarterRe + quarterIm * quarterIm;';
    if (!fullFft && candidateSource.split(boundary).length !== 2) throw new Error('Power capture contract changed');
    // Capture after any arithmetic experiment, immediately before mel accumulation.
    const melBoundary = 'for (let melIndex = 0; melIndex < this.nMels; melIndex += 1) {\n                let melValue = 0;';
    if (candidateSource.split(melBoundary).length !== 2) throw new Error('Mel capture boundary contract changed');
    let complexInjection = '';
    if (referenceComplexPower) {
      const shape = [1, (MEL_CONSTANTS.N_FFT >> 1) + 1, raw.frameCount];
      // Decoder is self-contained and validates the same immutable canonical bytes
      // inside each isolated module; JSON numeric serialization never touches them.
      candidateSource = 'const referenceComplex = (' + readReferenceComplex.toString() + ')(' +
        JSON.stringify(data.referenceComplex) + ', ' + JSON.stringify(shape) + ');\n' + candidateSource;
      const expression = referenceComplexPower === 'library'
        ? 're * re + im * im' : 'diagnosticComplexPower(re, im)';
      complexInjection = `for (let bin = 0; bin < N_FREQ_BINS; bin++) {
        const index = bin * nFrames + frameIndex;
        const re = referenceComplex.real[index], im = referenceComplex.imag[index];
        this.powerBuf[bin] = ${expression};
      }\n`;
    }
    const complexCapture = fullFft ? `capturedReal[bin * nFrames + frameIndex] = this.fftRe[bin];
      capturedImag[bin * nFrames + frameIndex] = this.fftIm[bin];` : referenceComplexPower
      ? `capturedReal[bin * nFrames + frameIndex] = referenceComplex.real[bin * nFrames + frameIndex];
        capturedImag[bin * nFrames + frameIndex] = referenceComplex.imag[bin * nFrames + frameIndex];` : '';
    let selectedInjection = '';
    if (powerOverrides !== null) {
      candidateSource = 'const selectedPower = ' + JSON.stringify({ coordinates: powerOverrides,
        reference: data.referencePower }) + ';\n' + candidateSource;
      selectedInjection = `for (const [frame, bin] of selectedPower.coordinates) {
        if (frame === frameIndex) this.powerBuf[bin] = selectedPower.reference[bin * nFrames + frame];
      }
      for (let bin = 0; bin < N_FREQ_BINS; bin++) effectivePower[bin * nFrames + frameIndex] = this.powerBuf[bin];\n`;
    }
    candidateSource = 'export const capturedPower = [], capturedReal = [], capturedImag = [], capturedFrames = [], effectivePower = [];\n' + candidateSource.replace(melBoundary,
      complexInjection + `for (let bin = 0; bin < N_FREQ_BINS; bin++) {
        capturedPower[bin * nFrames + frameIndex] = this.powerBuf[bin]; ${complexCapture}
      }\n` + selectedInjection + melBoundary);
    if (referencePower || melArithmetic) {
      const logBoundary = 'rawMel[melIndex * nFrames + frameIndex] = Math.log(melValue + LOG_ZERO_GUARD);';
      if (candidateSource.split(logBoundary).length !== 2) throw new Error('Mel/log capture boundary contract changed');
      const logInput = roundLogGuard ? 'Math.fround(melValue + LOG_ZERO_GUARD)' : 'melValue + LOG_ZERO_GUARD';
      candidateSource = 'export const capturedMel = [], capturedLogInput = [];\n' + candidateSource.replace(logBoundary,
        'capturedMel[melIndex * nFrames + frameIndex] = melValue;\n' +
        `capturedLogInput[melIndex * nFrames + frameIndex] = ${logInput};\n` +
        (roundLogGuard ? `rawMel[melIndex * nFrames + frameIndex] = Math.log(${logInput});` : logBoundary));
    }
  }
  // Separate module instances isolate private caches; production code stays unchanged.
  const variant = await import('data:text/javascript;base64,' + Buffer.from(candidateSource).toString('base64') + '#' + name);
  const processor = new variant.JSMelProcessor(options);
  if (window) {
    if (data.referenceWindow.length !== variant.MEL_CONSTANTS.WIN_LENGTH ||
      processor.hannWindow.length !== variant.MEL_CONSTANTS.N_FFT) throw new Error('Window contract differs');
    processor.hannWindow.fill(0);
    processor.hannWindow.set(data.referenceWindow,
      (variant.MEL_CONSTANTS.N_FFT - variant.MEL_CONSTANTS.WIN_LENGTH) >> 1);
  }
  if (filterbank) {
    if (processor.melFilterbank.length !== data.referenceFilterbank.length) throw new Error('Filterbank contract differs');
    processor.melFilterbank.set(data.referenceFilterbank);
  }
  let capturedRaw = null;
  if (capturePower) {
    const finalize = processor.finalizeFeatures.bind(processor);
    processor.finalizeFeatures = (rawMel, ...args) => {
      capturedRaw = Array.from(rawMel);
      return finalize(rawMel, ...args);
    };
  }
  const result = processor.process(pcm);
  const unroundedFrames = production || capturePacked;
  const frameBytes = fullFft || complexPower || unroundedFrames
    ? Buffer.alloc(variant.capturedFrames.length * (unroundedFrames ? 8 : 4)) : null;
  if (frameBytes) variant.capturedFrames.forEach((value, index) => unroundedFrames
    ? frameBytes.writeDoubleLE(value, index * 8) : frameBytes.writeFloatLE(value, index * 4));
  return capturePower ? { features: Array.from(result.features), raw: capturedRaw, power: variant.capturedPower,
    filterbankBounds: Array.from(processor.fbBounds), filterbank: Array.from(processor.melFilterbank),
    ...(referencePower || melArithmetic ? { linearMel: variant.capturedMel, logInput: variant.capturedLogInput,
      f32LogInputRaw: Array.from(Float32Array.from(variant.capturedLogInput, value => Math.log(Math.fround(value)))) } : {}),
    ...(powerOverrides !== null ? { effectivePower: variant.effectivePower, powerOverrides } : {}),
    shape: [1, (variant.MEL_CONSTANTS.N_FFT >> 1) + 1, result.frameCount],
    ...(referenceComplexPower ? { real: variant.capturedReal, imag: variant.capturedImag } : {}),
    ...(unroundedFrames ? { real: variant.capturedReal, imag: variant.capturedImag,
      frames: variant.capturedFrames, frameShape: [1, result.frameCount, variant.MEL_CONSTANTS.N_FFT],
      frameDtype: 'float64', frameLayout: 'batch-frame-sample',
      frameBytesEncoding: 'base64-f64le', frameBytesF64LE: frameBytes.toString('base64') } : {}),
    ...(fullFft || complexPower ? { real: variant.capturedReal, imag: variant.capturedImag,
      frames: variant.capturedFrames, frameShape: [1, result.frameCount, variant.MEL_CONSTANTS.N_FFT],
      frameDtype: 'float32', frameLayout: 'batch-frame-sample',
      frameBytesEncoding: 'base64-f32le', frameBytesF32LE: frameBytes.toString('base64') } : {}) } : Array.from(result.features);
}

if (raw.frameCount !== data.physicalFrames || raw.length !== data.validLength)
  throw new Error('Probe physical/valid frame contract differs');
const results = {};
for (const [name, boundary] of Object.entries(data.spectra)) {
  data.referenceComplex = boundary;
  const c = await experiment(name, { window: true, windowProduct: true, filterbank: true,
    referenceComplexPower: 'fp32-rounded', melSpectrum: 'official-complex-rounded',
    melArithmetic: 'f64-product-f32-sum', roundLogGuard: true, capturePower: true });
  const n = normalizeCandidate(Float32Array.from(c.raw), raw.frameCount, raw.length, 8);
  results[name] = { features: Array.from(n.features), mean: Array.from(n.mean), std: Array.from(n.std) };
  if (data.coordinate) {
    const [mel, frame] = data.coordinate, index = mel * raw.frameCount + frame;
    results[name].point = { raw: c.raw[index], linearMel: c.linearMel[index], logInput: c.logInput[index],
      normalized: n.features[index], mean: n.mean[mel], std: n.std[mel] };
  }
}
console.log(JSON.stringify({ physicalFrames: raw.frameCount, validLength: raw.length, results }));

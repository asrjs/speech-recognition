"""Weight-free official NeMo vs library feature boundary diagnostic; no ASR loop."""

import argparse
import base64
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from tensor_parity import compare_tensors
from artifact_outputs import require_new_outputs, write_new_text
from normalization_reduction import normalize_candidate


NODE_PROBE = r'''
import { JSMelProcessor, MEL_CONSTANTS } from './dist/audio/js-mel.js';
import { readFileSync } from 'node:fs';
import { normalizeCandidate } from './tools/model-debugging/reference/hf-parakeet-port/normalization-reduction.mjs';
let input = ''; for await (const chunk of process.stdin) input += chunk;
const data = JSON.parse(input);
const pcm = Float32Array.from(data.pcm);
const options = { nMels: data.bins, validLengthMode: 'onnx' };
const raw = new JSMelProcessor({ ...options, normalization: 'none' }).process(pcm);
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
  fullFft = null, referenceComplexPower = null, production = false, capturePacked = false } = {}) {
  if (capturePacked && (!capturePower || referencePower || magnitudeRoundTrip || complexPower ||
      fullFft || referenceComplexPower || production))
    throw new Error('Packed arithmetic capture must retain original power expressions');
  if (production && (!capturePower || window || windowProduct || filterbank || referencePower ||
      magnitudeRoundTrip || complexPower || fullFft || referenceComplexPower))
    throw new Error('Production capture must preserve unmodified arithmetic');
  let candidateSource = production ? productionSource : source;
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
        `capturedReal[(${bin}) * nFrames + frameIndex] = ${re};
         capturedImag[(${bin}) * nFrames + frameIndex] = ${im};\n` + boundary);
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
    candidateSource = 'export const capturedPower = [], capturedReal = [], capturedImag = [], capturedFrames = [];\n' + candidateSource.replace(melBoundary,
      complexInjection + `for (let bin = 0; bin < N_FREQ_BINS; bin++) {
        capturedPower[bin * nFrames + frameIndex] = this.powerBuf[bin]; ${complexCapture}
      }\n` + melBoundary);
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
const experiments = {
  fp32_preemphasis: await experiment('preemphasis'),
  reference_window: await experiment('window', { window: true }),
  fp32_window_product: await experiment('window-product', { window: true, windowProduct: true }),
  reference_filterbank: await experiment('filterbank', { window: true, windowProduct: true, filterbank: true }),
  magnitude_round_trip: await experiment('magnitude', { window: true, windowProduct: true, filterbank: true, magnitudeRoundTrip: true }),
  reference_power: await experiment('power', { window: true, windowProduct: true, filterbank: true, referencePower: true }),
};
const powerCapture = await experiment('power-capture', { window: true, windowProduct: true,
  filterbank: true, magnitudeRoundTrip: true, capturePower: true });
const productionCapture = await experiment('production-capture', { production: true, capturePower: true });
const frameArithmeticCaptures = {
  preemphasis: await experiment('frame-preemphasis', { capturePacked: true, capturePower: true }),
  window: await experiment('frame-window', { window: true, capturePacked: true, capturePower: true }),
  windowProduct: await experiment('frame-window-product', {
    window: true, windowProduct: true, capturePacked: true, capturePower: true }),
};
const complexPowerCapture = await experiment('complex-power-capture', { window: true,
  windowProduct: true, filterbank: true, complexPower: true, capturePower: true });
const fullFftF32Capture = await experiment('full-fft-f32', { window: true,
  windowProduct: true, filterbank: true, fullFft: 'float32', capturePower: true });
const fullFftF64Capture = await experiment('full-fft-f64', { window: true,
  windowProduct: true, filterbank: true, fullFft: 'float64', capturePower: true });
const referenceComplexLibraryCapture = await experiment('reference-complex-library', { window: true,
  windowProduct: true, filterbank: true, referenceComplexPower: 'library', capturePower: true });
const referenceComplexRoundedCapture = await experiment('reference-complex-rounded', { window: true,
  windowProduct: true, filterbank: true, referenceComplexPower: 'fp32-rounded', capturePower: true });
const normalizationInputs = {
  officialRaw: data.referenceRaw, productionRaw: productionCapture.raw,
  alignedWindowProducts: frameArithmeticCaptures.windowProduct.raw,
  alignedComplexPower: complexPowerCapture.raw,
  magnitudeRoundTrip: powerCapture.raw, fullFftF32: fullFftF32Capture.raw,
  fullFftF64: fullFftF64Capture.raw, officialComplexLibrary: referenceComplexLibraryCapture.raw,
  officialComplexRounded: referenceComplexRoundedCapture.raw,
};
const normalizationControls = {};
for (const [name, values] of Object.entries(normalizationInputs)) {
  const result = normalizeCandidate(Float32Array.from(values), raw.frameCount, raw.length, 8);
  normalizationControls[name] = { features: Array.from(result.features), mean: Array.from(result.mean), std: Array.from(result.std) };
}
console.log(JSON.stringify({ raw: Array.from(raw.features), normalized: Array.from(normalized.features),
  sameRawNormalized: Array.from(sameRawNormalized), normalizationControls, frameCount: raw.frameCount, validLength: raw.length,
  experiments, powerCapture, productionCapture, frameArithmeticCaptures, complexPowerCapture, fullFftF32Capture, fullFftF64Capture,
  referenceComplexLibraryCapture, referenceComplexRoundedCapture,
  logZeroGuard: MEL_CONSTANTS.LOG_ZERO_GUARD, nodeVersion: process.version }));
'''


def encode_complex_boundary(values):
    """Canonical complex64 components from the real STFT, not JSON float mirrors."""
    values = np.asarray(values)
    if values.dtype != np.complex64 or values.ndim != 3 or values.shape[0] != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Official complex boundary must be finite, nonempty batch-one complex64")
    return {"shape": list(values.shape), "dtype": "complex64", "layout": "batch-bin-frame",
        "encoding": "base64-f32le-components",
        **{f"{component}F32LE": base64.b64encode(
            getattr(values, component).astype("<f4", copy=False).tobytes()).decode("ascii")
            for component in ("real", "imag")}}


def read_windowed_frames(capture, n_fft, frame_count):
    """Validate the exact captured FFT input, never silently round its values."""
    if (capture.get("frameShape") != [1, frame_count, n_fft]
            or capture.get("frameDtype") != "float32"
            or capture.get("frameLayout") != "batch-frame-sample"
            or capture.get("frameBytesEncoding") != "base64-f32le"):
        raise ValueError("JS windowed-frame metadata differs from the FFT input contract")
    values = np.asarray(capture.get("frames"), dtype=np.float64)
    if values.ndim != 1 or values.size != frame_count * n_fft or not np.isfinite(values).all():
        raise ValueError("JS windowed-frame count or finite-value contract differs")
    frames = values.astype(np.float32)
    if not np.array_equal(values, frames.astype(np.float64)):
        raise ValueError("JS windowed-frame values are not exactly representable as float32")
    encoded = capture.get("frameBytesF32LE")
    if not isinstance(encoded, str):
        raise ValueError("JS windowed-frame float32 bytes are missing")
    frame_bytes = base64.b64decode(encoded, validate=True)
    if len(frame_bytes) != frame_count * n_fft * 4:
        raise ValueError("JS windowed-frame float32 byte count differs")
    exact_frames = np.frombuffer(frame_bytes, dtype="<f4")
    if not np.isfinite(exact_frames).all() or not np.array_equal(frames, exact_frames):
        raise ValueError("JS windowed-frame numeric mirror and float32 bytes disagree")
    # JSON numbers erase -0. Read canonical bytes rather than that lossy mirror.
    return exact_frames.reshape(1, frame_count, n_fft)


def read_production_frames(capture, n_fft, frame_count):
    """Unrounded production f64 window products; never cast them to f32 here."""
    if (type(n_fft) is not int or type(frame_count) is not int
            or n_fft < 2 or n_fft % 2 or frame_count < 1):
        raise ValueError("Production frame dimensions must be positive integers with even FFT size")
    shape = capture.get("frameShape")
    if (not isinstance(shape, list) or any(type(value) is not int for value in shape)
            or shape != [1, frame_count, n_fft]
            or capture.get("frameDtype") != "float64"
            or capture.get("frameLayout") != "batch-frame-sample"
            or capture.get("frameBytesEncoding") != "base64-f64le"):
        raise ValueError("Production frame metadata must describe canonical float64 products")
    encoded = capture.get("frameBytesF64LE")
    if not isinstance(encoded, str):
        raise ValueError("Production float64 frame bytes missing")
    try:
        frame_bytes = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError("Production float64 frame bytes malformed") from error
    if len(frame_bytes) != frame_count * n_fft * 8:
        raise ValueError("Production float64 frame byte count differs")
    frames = np.frombuffer(frame_bytes, dtype="<f8")
    mirror = np.asarray(capture.get("frames"))
    if (mirror.ndim != 1 or mirror.dtype.kind not in "fi" or not np.isfinite(frames).all()
            or not np.array_equal(frames, mirror)):
        raise ValueError("Production frame bytes and finite numeric mirror disagree")
    return frames.reshape(1, frame_count, n_fft)


def production_frame_diagnostic(capture, aligned_frames):
    """Separate input arithmetic differences from packed FFT on its own input."""
    aligned_frames = np.asarray(aligned_frames)
    if (aligned_frames.dtype != np.float32 or aligned_frames.ndim != 3
            or aligned_frames.shape[0] != 1 or not aligned_frames.size
            or not np.isfinite(aligned_frames).all()):
        raise ValueError("Aligned control frames must be finite nonempty batch-one float32")
    frames = read_production_frames(capture, aligned_frames.shape[2], aligned_frames.shape[1])
    shape = (1, frames.shape[2] // 2 + 1, frames.shape[1])
    if capture.get("shape") != list(shape):
        raise ValueError("Production spectrum shape differs from captured frames")
    arrays = {}
    for component in ("real", "imag", "power"):
        values = np.asarray(capture.get(component))
        if (values.ndim != 1 or values.size != np.prod(shape)
                or values.dtype.kind not in "fi" or not np.isfinite(values).all()):
            raise ValueError("Production spectrum must have finite full bin-major components")
        arrays[component] = values.astype(np.float64).reshape(shape)
    reconstructed_power = np.float32(arrays["real"] ** 2 + arrays["imag"] ** 2)
    independent_fft = np.fft.rfft(frames, axis=-1).transpose(0, 2, 1)
    return {"status": "DIAGNOSTIC_NOT_PORT_ACCEPTANCE",
        "scope": "Different production f64 and aligned f32 inputs; NumPy FFT is a same-input diagnostic, not an official-engine substitute",
        "production_frame_sha256_f64le": hashlib.sha256(frames.tobytes()).hexdigest(),
        "aligned_frame_sha256_f32le": hashlib.sha256(aligned_frames.astype("<f4", copy=False).tobytes()).hexdigest(),
        "frame_shape": list(frames.shape),
        "original_frame_values": compare_tensors(aligned_frames.astype(np.float64), frames, atol=0, rtol=0),
        "explicit_f32_cast_frame_bits": compare_tensors(
            aligned_frames.astype("<f4", copy=False).view("<u4"),
            frames.astype("<f4").view("<u4"), atol=0, rtol=0),
        "original_power_reconstruction": compare_tensors(
            arrays["power"].astype(np.float32), reconstructed_power, atol=0, rtol=0),
        "same_production_frames_numpy_fft": compare_fft_components(
            capture, independent_fft, atol=1e-12, rtol=1e-12),
        "numpy_version": np.__version__}


def direct_dft_bins(frames, coordinates):
    """Independent selected-bin DFT on canonical f32 input, not an FFT engine.

    Float64 sin/cos and math.fsum approximate the mathematical transform;
    this is not an exact-arithmetic oracle or a production replacement.
    Modular phases keep high-bin angle reduction bounded to one turn.
    """
    frames = np.asarray(frames)
    if (frames.dtype != np.float32 or frames.ndim != 3 or frames.shape[0] != 1
            or not frames.size or frames.shape[2] < 2 or frames.shape[2] % 2
            or not np.isfinite(frames).all()):
        raise ValueError("Direct DFT requires finite nonempty batch-one float32 windowed frames and even FFT size")
    selected, seen = [], set()
    for coordinate in coordinates:
        if (not isinstance(coordinate, (tuple, list)) or len(coordinate) != 2
                or any(type(value) is not int for value in coordinate)
                or not 0 <= coordinate[0] < frames.shape[1]
                or not 0 <= coordinate[1] <= frames.shape[2] // 2
                or tuple(coordinate) in seen):
            raise ValueError("Direct DFT frame/bin coordinate is invalid, duplicate or out of range")
        seen.add(tuple(coordinate))
        selected.append(tuple(coordinate))
    values = []
    size = frames.shape[2]
    for frame, bin_index in selected:
        samples = frames[0, frame]
        angles = [2 * math.pi * ((bin_index * sample) % size) / size for sample in range(size)]
        real = math.fsum(float(value) * math.cos(angle) for value, angle in zip(samples, angles))
        imag = -math.fsum(float(value) * math.sin(angle) for value, angle in zip(samples, angles))
        values.append(complex(real, imag))
    return values


def fft_dft_diagnostic(frames, official_complex, official_power, controls, mismatch_support):
    """Compare engine spectra with a DFT at every selected failed-mel support bin.

    Exact input bytes are mandatory. Differences from the float64 DFT estimate
    are diagnostic evidence, never altered feature tolerances or port acceptance.
    """
    direct_dft_bins(frames, [])  # Validate canonical input even for empty support.
    frames = np.asarray(frames)
    shape = (1, frames.shape[2] // 2 + 1, frames.shape[1])
    official_complex, official_power = np.asarray(official_complex), np.asarray(official_power)
    if (official_complex.shape != shape or official_complex.dtype != np.complex64
            or official_power.shape != shape or official_power.dtype != np.float32
            or not np.isfinite(official_complex).all() or not np.isfinite(official_power).all()
            or np.any(official_power < 0)):
        raise ValueError("Direct DFT diagnostic requires matching finite official complex64/f32 power boundaries")
    if (not isinstance(controls, dict) or not controls or set(controls) != set(mismatch_support)
            or "official_complex64" in controls):
        raise ValueError("Direct DFT controls and mismatch support must have distinct matching names")
    spectra = {"official_complex64": (official_complex.real.astype(np.float64),
        official_complex.imag.astype(np.float64), official_power)}
    pairs = set()
    canonical_bytes = frames.astype("<f4", copy=False).tobytes()
    for name, capture in controls.items():
        captured_frames = read_windowed_frames(capture, frames.shape[2], frames.shape[1])
        if captured_frames.astype("<f4", copy=False).tobytes() != canonical_bytes:
            raise ValueError("Direct DFT comparisons require identical canonical windowed frame bytes")
        if capture.get("shape") != list(shape):
            raise ValueError("Direct DFT candidate complex layout/shape differs")
        arrays = []
        for component in ("real", "imag", "power"):
            values = np.asarray(capture.get(component))
            if (values.ndim != 1 or values.size != np.prod(shape) or values.dtype.kind not in "fi"
                    or not np.isfinite(values).all()):
                raise ValueError("Direct DFT candidate components must be finite full bin-major numeric arrays")
            arrays.append(values.astype(np.float64).reshape(shape))
        if np.any(arrays[2] < 0):
            raise ValueError("Direct DFT captured powers must be nonnegative")
        spectra[name] = tuple(arrays)
        seen = set()
        for point in mismatch_support[name]:
            coordinate, active = point.get("coordinate"), point.get("active_bins")
            if (not isinstance(coordinate, (list, tuple)) or len(coordinate) != 3
                    or any(type(value) is not int for value in coordinate)
                    or coordinate[0] != 0 or coordinate[1] < 0
                    or not 0 <= coordinate[2] < frames.shape[1] or tuple(coordinate) in seen
                    or not isinstance(active, list)
                    or any(type(value) is not int or not 0 <= value < shape[1] for value in active)
                    or len(set(active)) != len(active)):
                raise ValueError("Direct DFT mismatch support coordinate/bins are invalid or duplicate")
            seen.add(tuple(coordinate))
            pairs.update((coordinate[2], bin_index) for bin_index in active)
    pairs = sorted(pairs)
    dft = direct_dft_bins(frames, pairs)
    rows = []
    errors = {name: [] for name in spectra}
    closest_counts = {name: 0 for name in spectra}
    rounded_counts = {name: 0 for name in spectra}
    for (frame, bin_index), estimate in zip(pairs, dft):
        dft_power = estimate.real ** 2 + estimate.imag ** 2
        rounded = np.asarray([estimate.real, estimate.imag], dtype="<f4")
        if not np.isfinite(rounded).all():
            raise ValueError("Direct DFT estimate cannot be represented by finite reference float32 components")
        engine_values = {}
        for name, (real, imag, boundary_power) in spectra.items():
            re, im = float(real[0, bin_index, frame]), float(imag[0, bin_index, frame])
            real_error, imag_error = re - estimate.real, im - estimate.imag
            complex_error = math.hypot(real_error, imag_error)
            errors[name].extend((abs(real_error), abs(imag_error)))
            rounded_equal = np.asarray([re, im], dtype="<f4").tobytes() == rounded.tobytes()
            rounded_counts[name] += int(rounded_equal)
            engine_values[name] = {"real": re, "imag": im, "real_error": real_error,
                "imag_error": imag_error, "complex_abs_error": complex_error,
                "component_squared_power": re ** 2 + im ** 2,
                "component_squared_power_error": re ** 2 + im ** 2 - dft_power,
                "boundary_power": float(boundary_power[0, bin_index, frame]),
                "boundary_power_error": float(boundary_power[0, bin_index, frame]) - dft_power,
                "rounded_dft_components_equal": rounded_equal}
        nearest_error = min(value["complex_abs_error"] for value in engine_values.values())
        nearest = [name for name, value in engine_values.items() if value["complex_abs_error"] == nearest_error]
        for name in nearest:
            closest_counts[name] += 1
        rows.append({"coordinate": [0, bin_index, frame],
            "dft": {"real": estimate.real, "imag": estimate.imag, "power": dft_power},
            "engines": engine_values, "closest_to_dft": nearest})
    return {"status": "DIAGNOSTIC_NOT_PORT_ACCEPTANCE",
        "method": "Selected-bin float64 sin/cos with modular phases and math.fsum; not exact arithmetic",
        "scope": "Complex accuracy estimate on identical f32 frames; no production FFT replacement or feature acceptance",
        "frame_sha256_f32le": hashlib.sha256(canonical_bytes).hexdigest(),
        "frame_shape": list(frames.shape), "complex_shape": list(shape),
        "selected_frame_bin_count": len(pairs), "failure_support": mismatch_support,
        "engine_summaries": {name: {"component_count": len(values),
            "max_abs_component_error": max(values) if values else None,
            "mean_abs_component_error": math.fsum(values) / len(values) if values else None,
            "rms_component_error": math.sqrt(math.fsum(value ** 2 for value in values) / len(values)) if values else None,
            "closest_to_dft_bins": closest_counts[name],
            "rounded_dft_equal_bins": rounded_counts[name]} for name, values in errors.items()},
        "bins": rows}


def compare_fft_components(capture, reference, *, atol, rtol):
    """Compare raw bin-major complex components without discarding f64 scratch bits."""
    reference = np.asarray(reference)
    if not np.iscomplexobj(reference) or list(reference.shape) != capture.get("shape"):
        raise ValueError("Complex FFT reference shape/type differs from the capture")
    metrics = {}
    for component in ("real", "imag"):
        candidate = np.asarray(capture[component], dtype=np.float64).reshape(reference.shape)
        expected = getattr(reference, component).astype(np.float64)
        result = compare_tensors(expected, candidate, atol=atol, rtol=rtol)
        index = result["first_divergent_index"]
        result["first_divergent_coordinate"] = (
            [int(value) for value in np.unravel_index(index, reference.shape)] if index is not None else None)
        largest = int(np.argmax(np.abs(expected - candidate)))
        result["max_error_location"] = {
            "coordinate": [int(value) for value in np.unravel_index(largest, reference.shape)],
            "reference": float(expected.ravel()[largest]), "candidate": float(candidate.ravel()[largest]),
        }
        metrics[component] = result
    return metrics


def normalization_decomposition(reference_raw, candidate_raw, reference_features,
        candidate_features, reference_mean, reference_std, *, valid_length, atol, rtol):
    """Diagnostic attribution, not a new normalizer or a relaxed parity gate.

    Official statistics are captured float32 (std includes NeMo's 1e-5).
    Counterfactuals use float64 arithmetic; actual float32 boundaries stay separate.
    The statistics term includes both raw-dependent stats and engine arithmetic.
    """
    arrays = [np.asarray(value) for value in (reference_raw, candidate_raw,
        reference_features, candidate_features, reference_mean, reference_std)]
    raw, candidate, expected, actual, mean, std = arrays
    if (raw.ndim != 3 or raw.shape[0] != 1 or not raw.size
            or any(value.dtype != np.float32 or not np.isfinite(value).all() for value in arrays)
            or any(value.shape != raw.shape for value in arrays[1:4])
            or mean.shape != raw.shape[:2] or std.shape != mean.shape or np.any(std <= 0)):
        raise ValueError("Normalization diagnostic requires matching finite batch-one float32 boundaries/statistics")
    if type(valid_length) is not int or not 1 <= valid_length <= raw.shape[2]:
        raise ValueError("Normalization diagnostic valid length is outside the captured frame contract")
    if np.any(expected[:, :, valid_length:] != 0) or np.any(actual[:, :, valid_length:] != 0):
        raise ValueError("Normalized feature tails must be masked to zero")
    feature_metrics = compare_tensors(expected, actual, atol=atol, rtol=rtol)
    r, c = raw.astype(np.float64), candidate.astype(np.float64)
    reference_reconstruction = np.zeros_like(r)
    counterfactual = np.zeros_like(r)
    own = np.zeros_like(r)
    candidate_means, candidate_stds = [], []
    for mel in range(raw.shape[1]):
        values = c[0, mel, :valid_length]
        # Same scalar accumulation order as JS, not NumPy's reduction algorithm.
        candidate_mean = sum(float(value) for value in values) / valid_length
        variance = sum((float(value) - candidate_mean) ** 2 for value in values)
        candidate_std = math.sqrt(variance / (valid_length - 1)) + 1e-5 if valid_length > 1 else 1e-5
        inv_std = 1 / candidate_std if valid_length > 1 else 0
        candidate_means.append(candidate_mean)
        candidate_stds.append(candidate_std)
        own[0, mel, :valid_length] = (values - candidate_mean) * inv_std
        reference_reconstruction[0, mel, :valid_length] = (
            r[0, mel, :valid_length] - float(mean[0, mel])) / float(std[0, mel])
        counterfactual[0, mel, :valid_length] = (values - float(mean[0, mel])) / float(std[0, mel])
    pointwise = np.zeros_like(r)
    pointwise[:, :, :valid_length] = (c[:, :, :valid_length] - r[:, :, :valid_length]) / std[:, :, None]
    terms = {"pointwise_raw": pointwise, "statistics": own - counterfactual,
        "reference_rounding": reference_reconstruction - expected.astype(np.float64),
        "candidate_rounding": actual.astype(np.float64) - own}
    terms["closure_residual"] = actual.astype(np.float64) - expected.astype(np.float64) - sum(terms.values())
    mismatched = ~np.isclose(actual.astype(np.float64), expected.astype(np.float64), atol=atol, rtol=rtol)
    points = []
    for coordinate in np.argwhere(mismatched):
        index = tuple(coordinate)
        points.append({"coordinate": coordinate.tolist(), "reference_feature": float(expected[index]),
            "candidate_feature": float(actual[index]), "reference_raw": float(raw[index]),
            "candidate_raw": float(candidate[index]),
            **{key: float(value[index]) for key, value in terms.items()}})
    per_mel = []
    for mel in range(raw.shape[1]):
        delta = c[0, mel, :valid_length] - r[0, mel, :valid_length]
        per_mel.append({"mel": mel, "reference_mean": float(mean[0, mel]),
            "reference_std_with_guard": float(std[0, mel]),
            "reference_inverse_std": 1 / float(std[0, mel]),
            "candidate_mean": candidate_means[mel], "candidate_std_with_guard": candidate_stds[mel],
            "candidate_inverse_std": 1 / candidate_stds[mel] if valid_length > 1 else 0,
            "reference_raw_min": float(r[0, mel, :valid_length].min()),
            "raw_max_abs_error": float(np.abs(delta).max()), "raw_rms_error": float(np.sqrt(np.mean(delta ** 2))),
            "mismatch_frames": np.flatnonzero(mismatched[0, mel]).tolist(),
            "normalized_max_abs_error": float(np.abs(actual[0, mel].astype(np.float64) - expected[0, mel]).max()),
            **{f"{key}_max_abs": float(np.abs(value[0, mel]).max()) for key, value in terms.items()}})
    return {"shape": list(raw.shape), "valid_length": valid_length,
        "scope": "Diagnostic f64 counterfactuals only; not production normalization or port acceptance",
        "statistics_term_scope": "Raw-dependent statistics plus JS/official statistic arithmetic differences",
        "reference_rounding_scope": "F64 reconstruction minus official feature boundary includes reduction/operation rounding",
        "actual_features": feature_metrics,
        "candidate_with_official_statistics": compare_tensors(expected, counterfactual.astype(np.float32), atol=atol, rtol=rtol),
        "reference_reconstruction": compare_tensors(expected, reference_reconstruction.astype(np.float32), atol=atol, rtol=rtol),
        "candidate_reconstruction": compare_tensors(actual, own.astype(np.float32), atol=atol, rtol=rtol),
        "raw_sha256_f32le": {name: hashlib.sha256(value.astype("<f4", copy=False).tobytes()).hexdigest()
            for name, value in (("reference", raw), ("candidate", candidate))},
        "terms_max_abs": {key: float(np.abs(value).max()) for key, value in terms.items()},
        "per_mel": per_mel, "mismatches": points}


def power_error_attribution(reference_power, candidate_power, filterbank,
        reference_raw, candidate_raw, *, mismatch_coordinates, valid_length, log_zero_guard):
    """Signed FFT-bin support at failed mel/frames; diagnostic, not a processor fix.

    Both projections use scalar-order f64 accumulation and a final f32 log.
    Official projection/reduction differences remain separate from power error.
    """
    arrays = [np.asarray(value) for value in (reference_power, candidate_power,
        filterbank, reference_raw, candidate_raw)]
    reference, candidate, weights, raw, actual = arrays
    if (reference.ndim != 3 or reference.shape[0] != 1 or not reference.size
            or candidate.shape != reference.shape or raw.ndim != 3 or raw.shape[0] != 1
            or not raw.size or actual.shape != raw.shape or raw.shape[2] != reference.shape[2]
            or weights.shape != (raw.shape[1], reference.shape[1])
            or any(value.dtype != np.float32 or not np.isfinite(value).all() for value in arrays)
            or any(np.any(value < 0) for value in (reference, candidate, weights))):
        raise ValueError("Power attribution requires matching finite batch-one f32 powers/raw and nonnegative filterbank")
    if (type(valid_length) is not int or not 1 <= valid_length <= raw.shape[2]
            or type(log_zero_guard) not in (int, float) or not math.isfinite(log_zero_guard)
            or log_zero_guard <= 0):
        raise ValueError("Power attribution valid length/log guard differs from the captured contract")
    coordinates = []
    for coordinate in mismatch_coordinates:
        if (not isinstance(coordinate, (list, tuple)) or len(coordinate) != 3
                or any(type(value) is not int for value in coordinate)
                or coordinate[0] != 0 or not 0 <= coordinate[1] < raw.shape[1]
                or not 0 <= coordinate[2] < valid_length or tuple(coordinate) in coordinates):
            raise ValueError("Power attribution mismatch coordinate is invalid, padded or duplicate")
        coordinates.append(tuple(coordinate))
    reference_projection = np.empty((1, raw.shape[1], valid_length), dtype=np.float32)
    candidate_projection = np.empty_like(reference_projection)
    points, per_mel = [], []
    for mel in range(raw.shape[1]):
        active = np.flatnonzero(weights[mel] > 0).tolist()
        r_sum, c_sum = np.zeros(valid_length, dtype=np.float64), np.zeros(valid_length, dtype=np.float64)
        # Preserve ascending-bin scalar accumulation order, not BLAS/reduction order.
        for bin_index in active:
            weight = float(weights[mel, bin_index])
            r_sum += reference[0, bin_index, :valid_length].astype(np.float64) * weight
            c_sum += candidate[0, bin_index, :valid_length].astype(np.float64) * weight
        reference_projection[0, mel] = [math.log(float(value) + log_zero_guard) for value in r_sum]
        candidate_projection[0, mel] = [math.log(float(value) + log_zero_guard) for value in c_sum]
        frames = [frame for _, band, frame in coordinates if band == mel]
        per_mel.append({"mel": mel, "active_bins": active, "mismatch_frames": frames,
            "reference_linear_min": float(r_sum.min()),
            "weighted_linear_max_abs_error": float(np.abs(c_sum - r_sum).max()),
            "power_log_max_abs_term": float(np.abs(candidate_projection[0, mel].astype(np.float64)
                - reference_projection[0, mel]).max()),
            "reference_projection_max_abs_term": float(np.abs(reference_projection[0, mel].astype(np.float64)
                - raw[0, mel, :valid_length]).max())})
        for frame in frames:
            contributions = [{"bin": bin_index, "weight": float(weights[mel, bin_index]),
                "reference_power": float(reference[0, bin_index, frame]),
                "candidate_power": float(candidate[0, bin_index, frame]),
                "weighted_error": float(weights[mel, bin_index]) * (
                    float(candidate[0, bin_index, frame]) - float(reference[0, bin_index, frame]))}
                for bin_index in active]
            positive = sum(point["weighted_error"] for point in contributions if point["weighted_error"] > 0)
            negative = sum(point["weighted_error"] for point in contributions if point["weighted_error"] < 0)
            absolute = positive - negative
            power_term = float(candidate_projection[0, mel, frame]) - float(reference_projection[0, mel, frame])
            reference_term = float(reference_projection[0, mel, frame]) - float(raw[0, mel, frame])
            candidate_term = float(actual[0, mel, frame]) - float(candidate_projection[0, mel, frame])
            observed = float(actual[0, mel, frame]) - float(raw[0, mel, frame])
            points.append({"coordinate": [0, mel, frame], "active_bins": active,
                "reference_linear": float(r_sum[frame]), "candidate_linear": float(c_sum[frame]),
                "weighted_positive_error": positive, "weighted_negative_error": negative,
                "weighted_signed_error": positive + negative, "weighted_absolute_error": absolute,
                "signed_to_absolute_ratio": (positive + negative) / absolute if absolute else None,
                "largest_contributions": sorted(contributions, key=lambda point: abs(point["weighted_error"]), reverse=True)[:5],
                "observed_raw_delta": observed, "power_log_term": power_term,
                "reference_projection_term": reference_term, "candidate_projection_term": candidate_term,
                "closure_residual": observed - power_term - reference_term - candidate_term})
    return {"scope": "Diagnostic f64 ordered projections; no FFT replacement or port acceptance",
        "shape": list(raw.shape), "valid_length": valid_length, "log_zero_guard": log_zero_guard,
        "reference_projection_scope": "Official mel reduction/log arithmetic versus JS-order reference-power projection",
        "candidate_raw_reconstruction": compare_tensors(actual[:, :, :valid_length], candidate_projection, atol=0, rtol=0),
        "per_mel": per_mel, "mismatches": points}


def normalized_failure_coordinates(reference, candidate, valid_length):
    """Select actual f32 control failures, never the older normalizer's support."""
    arrays = [np.asarray(value) for value in (reference, candidate)]
    left, right = arrays
    if (left.ndim != 3 or left.shape[0] != 1 or not left.size or right.shape != left.shape
            or any(value.dtype != np.float32 or not np.isfinite(value).all() for value in arrays)
            or type(valid_length) is not int or not 1 <= valid_length <= left.shape[2]
            or any(np.any(value[:, :, valid_length:] != 0) for value in arrays)):
        raise ValueError("Normalized failure support requires finite batch-one f32 features and masked tails")
    return np.argwhere(~np.isclose(right.astype(np.float64), left.astype(np.float64),
        atol=1e-5, rtol=1e-4)).tolist()


def compare_raw_valid_prefix(reference, candidate, valid_length):
    """Compare equivalent raw samples, excluding differently masked physical tails."""
    if (type(valid_length) is not int or reference.ndim != 2 or candidate.shape != reference.shape
            or not 1 <= valid_length <= reference.shape[1]):
        raise ValueError("Raw-mel valid-prefix shape/length contract differs")
    left, right = reference[:, :valid_length], candidate[:, :valid_length]
    return {"tolerant": compare_tensors(left, right, atol=1e-5, rtol=1e-4),
        "exact": compare_tensors(left, right, atol=0, rtol=0)}


def write_diagnostic_report(path, report):
    """Preserve full evidence with optional gzip, never overwrite an earlier run."""
    if path.suffix == ".gz":
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "xt", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
            file.write("\n")
    else:
        write_new_text(path, json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--official-capture", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-only", action="store_true", help="Print only receipt path/status; full evidence remains in output")
    args = parser.parse_args()
    require_new_outputs([args.output], protected_inputs=[args.audio, args.source_config, args.official_capture])
    config = json.loads(args.source_config.read_text(encoding="utf-8"))["preprocessor"]
    with wave.open(str(args.audio), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getframerate() != config["sample_rate"]:
            raise ValueError("This diagnostic requires same-rate mono PCM16; no hidden resampling")
        pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
    captured_bytes = args.official_capture.read_bytes()
    captured = json.loads(gzip.decompress(captured_bytes) if args.official_capture.suffix == ".gz" else captured_bytes)
    matches = [sample for sample in captured["samples"] if sample["sample_id"] == args.sample_id]
    if len(matches) != 1:
        raise ValueError("Exactly one official sample with the requested ID is required")
    official = matches[0]
    audio_sha = hashlib.sha256(args.audio.read_bytes()).hexdigest()
    if official["audio"]["sha256"] != audio_sha:
        raise ValueError("Official/candidate audio identity differs")
    cache = ROOT / ".cache" / "nemo-reference" / "numba"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache)
    import torch
    from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures, normalize_batch
    import numba
    numba.config.reload_config()
    torch.set_num_threads(4)
    processor = FilterbankFeatures(
        sample_rate=config["sample_rate"],
        n_window_size=round(config["window_size"] * config["sample_rate"]),
        n_window_stride=round(config["window_stride"] * config["sample_rate"]),
        window=config["window"], normalize=None, n_fft=config["n_fft"], nfilt=config["features"],
        log=config["log"], dither=config["dither"], pad_to=config["pad_to"], pad_value=config["pad_value"],
        frame_splicing=config["frame_splicing"],
    ).eval()
    waveform = torch.from_numpy(pcm).unsqueeze(0)
    lengths = torch.tensor([pcm.size], dtype=torch.int64)
    with torch.no_grad():
        raw, valid_length = processor(waveform, lengths)
        if config["normalize"] != "per_feature":
            raise ValueError("Raw/statistics attribution currently requires per_feature normalization")
        normalized, reference_mean, reference_std = normalize_batch(raw, valid_length, config["normalize"])
        power, power_length = processor(waveform, lengths, linear_spec=True)
        # Inspect the actual pre-magnitude STFT boundary on the same full-length input.
        if processor.exact_pad or processor.stft_pad_amount is not None or processor.frame_splicing != 1:
            raise ValueError("Re-inspect non-centered/spliced STFT before using this framed-input diagnostic")
        preemphasized = torch.cat((waveform[:, :1],
            waveform[:, 1:] - processor.preemph * waveform[:, :-1]), dim=1)
        official_complex = processor.stft(preemphasized)
        window_left = (processor.n_fft - processor.win_length) // 2
        padded_window = torch.nn.functional.pad(processor.window,
            (window_left, processor.n_fft - processor.win_length - window_left))
        official_frames = torch.nn.functional.pad(preemphasized,
            (processor.n_fft // 2, processor.n_fft // 2)).unfold(1, processor.n_fft,
            processor.hop_length) * padded_window
        official_frame_fft = torch.fft.rfft(official_frames, n=processor.n_fft, dim=-1).transpose(1, 2)
    if power.shape != (1, config["n_fft"] // 2 + 1, raw.shape[2]) or not torch.equal(power_length, valid_length):
        raise ValueError("Official power/feature boundary contract differs")
    expected = official["stages"]["features"]
    expected_tensor = np.asarray(expected["data"], dtype=np.float32).reshape(expected["shape"])
    control = compare_tensors(expected_tensor, normalized.numpy(), atol=1e-5, rtol=1e-4)
    if not control["allclose"] or int(valid_length[0]) != official["stages"]["features_length"]["data"][0]:
        raise ValueError(f"Official processor reproduction failed: {control}")
    normalization_candidates = {}
    for width in (8, 16):
        candidate, mean, std = normalize_candidate(raw.numpy()[0], int(valid_length[0]), width)
        normalization_candidates[str(width)] = {
            "scope": "Diagnostic contiguous f32 cascade reduction on identical official raw mel; not production or active-dispatch proof",
            "normalized_features": compare_tensors(normalized.numpy()[0], candidate, atol=1e-5, rtol=1e-4),
            "mean_exact": compare_tensors(reference_mean.numpy()[0], mean, atol=0, rtol=0),
            "guarded_std_exact": compare_tensors(reference_std.numpy()[0], std, atol=0, rtol=0)}
    complex_boundary = encode_complex_boundary(official_complex.numpy())
    probe = subprocess.run(["node", "--input-type=module", "-e", NODE_PROBE], cwd=ROOT,
        input=json.dumps({"pcm": pcm.tolist(), "bins": config["features"], "referenceRaw": raw.flatten().tolist(),
            "referenceWindow": processor.window.tolist(), "referenceFilterbank": processor.fb.flatten().tolist(),
            "referencePower": power.flatten().tolist(), "referenceComplex": complex_boundary}),
        text=True, encoding="utf-8", capture_output=True, check=True, timeout=60)
    js = json.loads(probe.stdout)
    if js["frameCount"] != raw.shape[2] or js["validLength"] != int(valid_length[0]):
        raise ValueError("JS/reference frame or valid-length mismatch")
    reshape = lambda values: np.asarray(values, dtype=np.float32).reshape(raw.shape)
    normalization_input_keys = {
        "officialRaw": raw.numpy()[0], "productionRaw": reshape(js["productionCapture"]["raw"])[0],
        "alignedWindowProducts": reshape(js["frameArithmeticCaptures"]["windowProduct"]["raw"])[0],
        "alignedComplexPower": reshape(js["complexPowerCapture"]["raw"])[0],
        "magnitudeRoundTrip": reshape(js["powerCapture"]["raw"])[0],
        "fullFftF32": reshape(js["fullFftF32Capture"]["raw"])[0],
        "fullFftF64": reshape(js["fullFftF64Capture"]["raw"])[0],
        "officialComplexLibrary": reshape(js["referenceComplexLibraryCapture"]["raw"])[0],
        "officialComplexRounded": reshape(js["referenceComplexRoundedCapture"]["raw"])[0]}
    if set(js["normalizationControls"]) != set(normalization_input_keys):
        raise ValueError("JS normalization control boundaries differ")
    js_normalization_controls = {}
    for name, input_values in normalization_input_keys.items():
        candidate, mean, std = normalize_candidate(input_values, int(valid_length[0]), 8)
        observation = js["normalizationControls"][name]
        observed_features = reshape(observation["features"])[0]
        js_normalization_controls[name] = {
            "scope": "Temporary JS f32 normalization on named raw-mel boundary; not changed production",
            "width": 8, "valid_length": int(valid_length[0]),
            "raw_sha256_f32le": hashlib.sha256(input_values.astype("<f4").tobytes()).hexdigest(),
            "raw_valid_prefix_vs_official": compare_raw_valid_prefix(raw.numpy()[0], input_values, int(valid_length[0])),
            "features_vs_official": compare_tensors(normalized.numpy()[0], observed_features, atol=1e-5, rtol=1e-4),
            "features_vs_python_exact": compare_tensors(candidate, observed_features, atol=0, rtol=0),
            "mean_vs_python_exact": compare_tensors(mean, np.asarray(observation["mean"], dtype=np.float32), atol=0, rtol=0),
            "std_vs_python_exact": compare_tensors(std, np.asarray(observation["std"], dtype=np.float32), atol=0, rtol=0)}
    metrics = {
        "official_processor_vs_capture": control,
        "raw_log_mel": compare_tensors(raw.numpy(), reshape(js["raw"]), atol=1e-5, rtol=1e-4),
        "normalization_of_same_official_raw_mel": compare_tensors(normalized.numpy(), reshape(js["sameRawNormalized"]), atol=1e-5, rtol=1e-4),
        "complete_js_features": compare_tensors(expected_tensor, reshape(js["normalized"]), atol=1e-5, rtol=1e-4),
    }
    for name, values in js["experiments"].items():
        metrics[f"experiment_{name}_features"] = compare_tensors(expected_tensor, reshape(values), atol=1e-5, rtol=1e-4)
    for power_key, capture_key, feature_key in [
        ("aligned_power_spectrum", "powerCapture", "captured_power_features"),
        ("fp32_complex_power_spectrum", "complexPowerCapture", "fp32_complex_power_features"),
        ("full_fft_f32_power_spectrum", "fullFftF32Capture", "full_fft_f32_features"),
        ("full_fft_f64_power_spectrum", "fullFftF64Capture", "full_fft_f64_features"),
        ("official_complex_library_power_spectrum", "referenceComplexLibraryCapture", "official_complex_library_features"),
        ("official_complex_rounded_power_spectrum", "referenceComplexRoundedCapture", "official_complex_rounded_features"),
    ]:
        captured_power = js[capture_key]
        if captured_power["shape"] != list(power.shape):
            raise ValueError("JS power capture/reference shape differs")
        candidate_power = np.asarray(captured_power["power"], dtype=np.float32).reshape(power.shape)
        power_metrics = compare_tensors(power.numpy(), candidate_power, atol=1e-5, rtol=1e-4)
        index = power_metrics["first_divergent_index"]
        power_metrics["first_divergent_coordinate"] = (
            [int(value) for value in np.unravel_index(index, power.shape)] if index is not None else None)
        largest = int(np.argmax(np.abs(power.numpy().astype(np.float64) - candidate_power)))
        power_metrics["max_error_location"] = {
            "coordinate": [int(value) for value in np.unravel_index(largest, power.shape)],
            "reference": float(power.numpy().ravel()[largest]),
            "candidate": float(candidate_power.ravel()[largest]),
        }
        metrics[power_key] = power_metrics
        metrics[feature_key] = compare_tensors(expected_tensor,
            reshape(captured_power["features"]), atol=1e-5, rtol=1e-4)
        if capture_key.startswith("referenceComplex"):
            metrics[f"{power_key}_exact"] = compare_tensors(power.numpy(), candidate_power, atol=0, rtol=0)
            for component, result in compare_fft_components(captured_power, official_complex.numpy(),
                    atol=0, rtol=0).items():
                metrics[f"{feature_key}_input_{component}"] = result
    frame_evidence = {}
    official_frames_array = official_frames.numpy()
    official_complex_array = official_complex.numpy()
    production_capture = js["productionCapture"]
    production_feature_control = compare_tensors(reshape(js["normalized"]),
        reshape(production_capture["features"]), atol=0, rtol=0)
    production_raw_control = compare_tensors(reshape(js["raw"])[:, :, :int(valid_length[0])],
        reshape(production_capture["raw"])[:, :, :int(valid_length[0])], atol=0, rtol=0)
    if not production_feature_control["allclose"] or not production_raw_control["allclose"]:
        raise ValueError("Production instrumentation changed raw/normalized features")
    production_diagnostic = production_frame_diagnostic(production_capture, official_frames_array)
    production_diagnostic["unchanged_normalized_features"] = production_feature_control
    production_diagnostic["unchanged_valid_raw_mel"] = production_raw_control
    production_diagnostic["official_stft_components"] = compare_fft_components(
        production_capture, official_complex_array, atol=1e-5, rtol=1e-4)
    frame_arithmetic_controls = {}
    previous_frames = read_production_frames(production_capture, processor.n_fft, raw.shape[2])
    for name, capture in js["frameArithmeticCaptures"].items():
        frames = read_production_frames(capture, processor.n_fft, raw.shape[2])
        phase_features = compare_tensors(expected_tensor, reshape(capture["features"]), atol=1e-5, rtol=1e-4)
        experiment_key = {"preemphasis": "fp32_preemphasis", "window": "reference_window",
            "windowProduct": "fp32_window_product"}[name]
        invariant = compare_tensors(reshape(js["experiments"][experiment_key]),
            reshape(capture["features"]), atol=0, rtol=0)
        if not invariant["allclose"]:
            raise ValueError("Packed frame observations changed the existing arithmetic experiment")
        frame_arithmetic_controls[name] = {
            "scope": "Cumulative isolated arithmetic control, original packed FFT and power; not changed production",
            "frame_sha256_f64le": hashlib.sha256(frames.tobytes()).hexdigest(),
            "frame_values_vs_official": compare_tensors(official_frames_array.astype(np.float64), frames, atol=0, rtol=0),
            "explicit_f32_cast_frame_bits_vs_official": compare_tensors(
                official_frames_array.astype("<f4", copy=False).view("<u4"), frames.astype("<f4").view("<u4"), atol=0, rtol=0),
            "frame_delta_from_previous_control": compare_tensors(previous_frames, frames, atol=0, rtol=0),
            "features_vs_official": phase_features, "unchanged_existing_experiment": invariant}
        previous_frames = frames
    # Separately cross-check the independent unfold/window reconstruction against
    # NeMo's actual STFT call; matching JS frames alone does not certify this seam.
    stft_control = {"shape": list(official_complex.shape),
        "real": official_frame_fft.real.numpy().ravel(),
        "imag": official_frame_fft.imag.numpy().ravel()}
    for component, result in compare_fft_components(stft_control, official_complex_array,
            atol=1e-5, rtol=1e-4).items():
        metrics[f"official_framed_rfft_vs_stft_{component}"] = result
    for name, key in (("f32", "fullFftF32Capture"), ("f64", "fullFftF64Capture")):
        capture = js[key]
        frames = read_windowed_frames(capture, processor.n_fft, raw.shape[2])
        metrics[f"full_fft_{name}_windowed_input"] = compare_tensors(
            official_frames_array, frames, atol=0, rtol=0)
        metrics[f"full_fft_{name}_windowed_input_bits"] = compare_tensors(
            official_frames_array.astype("<f4", copy=False).view("<u4"), frames.view("<u4"), atol=0, rtol=0)
        with torch.no_grad():
            same_frame_fft = torch.fft.rfft(torch.from_numpy(frames.copy()),
                n=processor.n_fft, dim=-1).transpose(1, 2).numpy()
        for boundary, reference in (("same_frames", same_frame_fft),
                ("official_stft", official_complex_array)):
            for component, result in compare_fft_components(capture, reference,
                    atol=1e-5, rtol=1e-4).items():
                metrics[f"full_fft_{name}_{boundary}_{component}"] = result
        frame_evidence[name] = {"shape": list(frames.shape), "dtype": "float32",
            "layout": "batch-frame-sample", "encoding": "base64-f32le", "sha256_f32le": hashlib.sha256(
                frames.astype("<f4", copy=False).tobytes()).hexdigest(),
            "scratch_dtype": "float32" if name == "f32" else "float64"}
    fft_control_keys = ("complexPowerCapture", "fullFftF32Capture", "fullFftF64Capture")
    attribution_keys = (*fft_control_keys, "referenceComplexLibraryCapture", "referenceComplexRoundedCapture")
    decomposition = {}
    for name, candidate_raw, candidate_features in [
        ("production", js["raw"], js["normalized"]),
        *[(key, js[key]["raw"], js[key]["features"]) for key in attribution_keys],
    ]:
        decomposition[name] = normalization_decomposition(raw.numpy(), reshape(candidate_raw),
            normalized.numpy(), reshape(candidate_features), reference_mean.numpy(), reference_std.numpy(),
            valid_length=int(valid_length[0]), atol=1e-5, rtol=1e-4)
    attribution = {}
    for key in attribution_keys:
        capture = js[key]
        attribution[key] = power_error_attribution(power.numpy(),
            np.asarray(capture["power"], dtype=np.float32).reshape(power.shape),
            processor.fb.numpy().reshape(config["features"], power.shape[1]),
            raw.numpy(), reshape(capture["raw"]),
            mismatch_coordinates=[point["coordinate"] for point in decomposition[key]["mismatches"]],
            valid_length=int(valid_length[0]), log_zero_guard=js["logZeroGuard"])
        metrics[f"{key}_raw_power_projection"] = attribution[key]["candidate_raw_reconstruction"]
    fft_controls = {key: js[key] for key in fft_control_keys}
    dft_diagnostic = fft_dft_diagnostic(official_frames_array, official_complex_array,
        power.numpy(), fft_controls, {key: [{"coordinate": point["coordinate"],
            "active_bins": point["active_bins"]} for point in attribution[key]["mismatches"]]
            for key in fft_controls})
    cascade_attribution = {}
    for key, name in (("complexPowerCapture", "alignedComplexPower"),
            ("fullFftF32Capture", "fullFftF32"), ("fullFftF64Capture", "fullFftF64")):
        capture = js[key]
        coordinates = normalized_failure_coordinates(normalized.numpy(),
            reshape(js["normalizationControls"][name]["features"]), int(valid_length[0]))
        cascade_attribution[key] = power_error_attribution(power.numpy(),
            np.asarray(capture["power"], dtype=np.float32).reshape(power.shape),
            processor.fb.numpy().reshape(config["features"], power.shape[1]),
            raw.numpy(), reshape(capture["raw"]), mismatch_coordinates=coordinates,
            valid_length=int(valid_length[0]), log_zero_guard=js["logZeroGuard"])
    cascade_dft = fft_dft_diagnostic(official_frames_array, official_complex_array,
        power.numpy(), fft_controls, {key: [{"coordinate": point["coordinate"],
            "active_bins": point["active_bins"]} for point in cascade_attribution[key]["mismatches"]]
            for key in fft_controls})
    report = {"schema_version": 1, "status": "DIAGNOSTIC_NOT_PORT_ACCEPTANCE", "sample_id": args.sample_id,
        "audio_sha256": audio_sha, "source_config_sha256": hashlib.sha256(args.source_config.read_bytes()).hexdigest(),
        "official_capture_sha256": hashlib.sha256(captured_bytes).hexdigest(),
        "library_processor_sha256": hashlib.sha256((ROOT / "dist/audio/js-mel.js").read_bytes()).hexdigest(),
        "environment": {"python": sys.version, "torch": torch.__version__, "node": js["nodeVersion"]},
        "fft_input_evidence": {"controls": frame_evidence,
            "official_sha256_f32le": hashlib.sha256(
                official_frames_array.astype("<f4", copy=False).tobytes()).hexdigest(),
            "window": "Official processor.window float32, centrally padded to n_fft",
            "stft": "Official FilterbankFeatures.stft on float32 preemphasis; center=True, constant padding",
            "comparison_dtype": "float64 preserves both raw scratch components; Torch reference is complex64",
            "limitation": "Complex/power allclose does not establish downstream feature parity or model acceptance"},
        "complex_injection_evidence": {"shape": complex_boundary["shape"], "dtype": "complex64",
            "layout": "batch-bin-frame", "encoding": complex_boundary["encoding"],
            "components_sha256_f32le": {component: hashlib.sha256(
                base64.b64decode(complex_boundary[f"{component}F32LE"], validate=True)).hexdigest()
                for component in ("real", "imag")},
            "source": "Actual FilterbankFeatures.stft(preemphasized), independent of linear_spec power",
            "library_control": "JS double re*re+im*im, one Float32Array power assignment",
            "rounded_control": "fp32 component squares and sum, fp32 sqrt, fp32 square",
            "scope": "Only isolated diagnostic module power buffer is overwritten; production is unchanged"},
        "normalization_decomposition": decomposition,
        "mel_power_attribution": attribution,
        "selected_bin_dft": dft_diagnostic,
        "cascade_normalized_failure_power_attribution": cascade_attribution,
        "cascade_normalized_failure_dft": cascade_dft,
        "production_input_diagnostic": production_diagnostic,
        "frame_arithmetic_controls": frame_arithmetic_controls,
        "normalization_reduction_candidates": normalization_candidates,
        "js_normalization_reduction_controls": js_normalization_controls,
        "js_normalization_candidate_sha256": hashlib.sha256(Path(__file__).with_name("normalization-reduction.mjs").read_bytes()).hexdigest(),
        "normalization_candidate_sha256": hashlib.sha256(Path(__file__).with_name("normalization_reduction.py").read_bytes()).hexdigest(),
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparisons": metrics}
    write_diagnostic_report(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output),
        "all_comparisons_pass": all(metric["allclose"] for metric in metrics.values())}
        if args.summary_only else report, indent=2))
    return 0 if all(metric["allclose"] for metric in metrics.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

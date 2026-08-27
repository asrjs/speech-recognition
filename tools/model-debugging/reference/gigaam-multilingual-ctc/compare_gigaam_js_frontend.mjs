#!/usr/bin/env node

/**
 * Compare GigaAmJsPreprocessor features against the official PyTorch capture.
 * Loads waveform/features NPY sidecars written by capture_gigaam_reference.py.
 */

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';

function parseArgs(argv) {
  const options = {
    reference: null,
    output: null,
    absTolerance: 1e-4,
    relTolerance: 1e-3,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--reference') options.reference = path.resolve(argv[++index]);
    else if (arg === '--output') options.output = path.resolve(argv[++index]);
    else if (arg === '--abs-tolerance') options.absTolerance = Number(argv[++index]);
    else if (arg === '--rel-tolerance') options.relTolerance = Number(argv[++index]);
    else if (arg === '--help' || arg === '-h') {
      console.log(
        'Usage: node compare_gigaam_js_frontend.mjs --reference <capture.json> [--output <json>]',
      );
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!options.reference) throw new Error('--reference is required');
  return options;
}

function loadNpy(filePath) {
  const buffer = fs.readFileSync(filePath);
  if (buffer[0] !== 0x93 || buffer.toString('latin1', 1, 6) !== 'NUMPY') {
    throw new Error(`Not a NumPy file: ${filePath}`);
  }
  const major = buffer[6];
  const headerLength = major === 1 ? buffer.readUInt16LE(8) : Number(buffer.readBigUInt64LE(8));
  const headerStart = major === 1 ? 10 : 16;
  const header = buffer.toString('ascii', headerStart, headerStart + headerLength);
  const descrMatch = header.match(/'descr':\s*'([^']+)'/);
  const fortranMatch = header.match(/'fortran_order':\s*(True|False)/);
  const shapeMatch = header.match(/'shape':\s*\(([^)]*)\)/);
  if (!descrMatch || !fortranMatch || !shapeMatch) {
    throw new Error(`Unsupported NPY header in ${filePath}: ${header}`);
  }
  const dtype = descrMatch[1];
  if (fortranMatch[1] === 'True') {
    throw new Error(`Fortran-order NPY is not supported: ${filePath}`);
  }
  const shape = shapeMatch[1]
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => Number(part));
  const dataOffset = headerStart + headerLength;
  let values;
  if (dtype === '<f4' || dtype === '|f4') {
    values = new Float32Array(
      buffer.buffer,
      buffer.byteOffset + dataOffset,
      (buffer.byteLength - dataOffset) / 4,
    );
  } else if (dtype === '<f8') {
    values = Float32Array.from(
      new Float64Array(
        buffer.buffer,
        buffer.byteOffset + dataOffset,
        (buffer.byteLength - dataOffset) / 8,
      ),
    );
  } else {
    throw new Error(`Unsupported NPY dtype ${dtype} in ${filePath}`);
  }
  return { shape, data: values };
}

function compare(reference, candidate, absTolerance, relTolerance) {
  const count = Math.min(reference.length, candidate.length);
  let maxAbs = 0;
  let sumAbs = 0;
  let sumSq = 0;
  let mismatches = 0;
  let firstMismatch = null;
  for (let index = 0; index < count; index += 1) {
    const left = reference[index] ?? 0;
    const right = candidate[index] ?? 0;
    const difference = Math.abs(left - right);
    const allowed = absTolerance + relTolerance * Math.max(Math.abs(left), Math.abs(right));
    maxAbs = Math.max(maxAbs, difference);
    sumAbs += difference;
    sumSq += difference * difference;
    if (difference > allowed) {
      mismatches += 1;
      firstMismatch ??= { index, reference: left, candidate: right, absDiff: difference, allowed };
    }
  }
  return {
    count: { reference: reference.length, candidate: candidate.length, compared: count },
    maxAbs,
    meanAbs: count ? sumAbs / count : 0,
    rmse: count ? Math.sqrt(sumSq / count) : 0,
    mismatches,
    firstMismatch,
    pass: mismatches === 0 && reference.length === candidate.length,
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const { GigaAmJsPreprocessor } = await import(
    pathToFileURL(
      path.resolve(process.cwd(), 'src/models/gigaam-ctc/frontend.ts'),
    ).href
  );
  const capture = JSON.parse(fs.readFileSync(options.reference, 'utf8'));
  const preprocessorCfg = capture.preprocessor ?? {};
  const preprocessor = new GigaAmJsPreprocessor({
    nMels: preprocessorCfg.n_mels ?? 64,
    nFft: preprocessorCfg.n_fft ?? 320,
    winLength: preprocessorCfg.win_length ?? 320,
    hopLength: preprocessorCfg.hop_length ?? 160,
    center: preprocessorCfg.center ?? false,
    melLowHz: preprocessorCfg.f_min,
    melHighHz: preprocessorCfg.f_max || undefined,
    melScale: preprocessorCfg.mel_scale === 'slaney' ? 'slaney' : 'htk',
    slaneyNorm: preprocessorCfg.norm === 'slaney',
  });
  const rows = [];
  for (const sample of capture.samples ?? []) {
    const waveformPath = sample.audio?.waveform_npy;
    const featuresPath = sample.stages?.features?.npy;
    if (!waveformPath || !featuresPath) {
      throw new Error(`Sample ${sample.sample_id} is missing waveform/features NPY sidecars.`);
    }
    const waveform = loadNpy(waveformPath);
    const referenceFeatures = loadNpy(featuresPath);
    const processed = preprocessor.process(Float32Array.from(waveform.data));
    const referenceFlat = Float32Array.from(referenceFeatures.data);
    const comparison = compare(
      referenceFlat,
      processed.features,
      options.absTolerance,
      options.relTolerance,
    );
    rows.push({
      sample_id: sample.sample_id,
      audio_sha256: sample.audio?.sha256 ?? null,
      reference_shape: referenceFeatures.shape,
      js_shape: [1, processed.featureSize, processed.frameCount],
      official_lengths: sample.stages.features.lengths,
      js_frame_count: processed.frameCount,
      features: comparison,
    });
  }
  const payload = {
    schema_version: 1,
    engine: 'asrjs-gigaam-js-frontend',
    reference: options.reference,
    preprocessor: preprocessorCfg,
    samples: rows,
    pass: rows.every((row) => row.features.pass),
  };
  const text = `${JSON.stringify(payload, null, 2)}\n`;
  if (options.output) {
    fs.mkdirSync(path.dirname(options.output), { recursive: true });
    fs.writeFileSync(options.output, text);
    console.log(`Wrote ${options.output}`);
  } else {
    process.stdout.write(text);
  }
  console.log(`JS frontend pass: ${payload.pass}`);
  process.exitCode = payload.pass ? 0 : 1;
}

await main();

#!/usr/bin/env node

/**
 * Compare two model-capture JSON files by stable sample and stage identity.
 * Never aligns rows by array position or transcript text.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DEFAULT_ABS_TOLERANCE = 1e-5;
const DEFAULT_REL_TOLERANCE = 1e-4;

function parseArgs(argv) {
  const options = {
    reference: null,
    candidate: null,
    output: null,
    absTolerance: DEFAULT_ABS_TOLERANCE,
    relTolerance: DEFAULT_REL_TOLERANCE,
    topK: 5,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--reference') options.reference = path.resolve(argv[++index]);
    else if (arg === '--candidate') options.candidate = path.resolve(argv[++index]);
    else if (arg === '--output') options.output = path.resolve(argv[++index]);
    else if (arg === '--abs-tolerance') options.absTolerance = Number(argv[++index]);
    else if (arg === '--rel-tolerance') options.relTolerance = Number(argv[++index]);
    else if (arg === '--top-k') options.topK = Number(argv[++index]);
    else if (arg === '--help' || arg === '-h') {
      printUsage();
      process.exit(0);
    } else throw new Error(`Unknown argument: ${arg}`);
  }
  if (!options.reference || !options.candidate)
    throw new Error('--reference and --candidate are required');
  if (!Number.isFinite(options.absTolerance) || options.absTolerance < 0)
    throw new Error('--abs-tolerance must be non-negative');
  if (!Number.isFinite(options.relTolerance) || options.relTolerance < 0)
    throw new Error('--rel-tolerance must be non-negative');
  if (!Number.isInteger(options.topK) || options.topK < 1)
    throw new Error('--top-k must be a positive integer');
  return options;
}

function printUsage() {
  console.log(
    [
      'Usage:',
      '  node node-compare-stage-captures.mjs --reference <json> --candidate <json> [options]',
      '',
      'Capture shape:',
      '  { schema_version: 1, samples: [{ sample_id, audio: { sha256 }, stages: { name: { data, shape, dtype } } }] }',
      '',
      'Options:',
      '  --output <file>       Write JSON to a file instead of stdout',
      '  --abs-tolerance <n>   Absolute element tolerance (default 1e-5)',
      '  --rel-tolerance <n>   Relative element tolerance (default 1e-4)',
      '  --top-k <n>           Number of top reference values to report (default 5)',
    ].join('\n'),
  );
}

function loadJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function sampleList(capture, label) {
  if (!capture || !Array.isArray(capture.samples))
    throw new Error(`${label} must contain a samples array`);
  const byId = new Map();
  for (const sample of capture.samples) {
    const id = String(sample?.sample_id ?? '').trim();
    if (!id) throw new Error(`${label} contains a sample without sample_id`);
    if (byId.has(id)) throw new Error(`${label} contains duplicate sample_id: ${id}`);
    byId.set(id, sample);
  }
  return byId;
}

function stageMap(sample, label, sampleId) {
  if (!sample?.stages || typeof sample.stages !== 'object' || Array.isArray(sample.stages))
    throw new Error(`${label} sample ${sampleId} must contain a stages object`);
  return sample.stages;
}

function numericData(stage, label) {
  if (!Array.isArray(stage?.data))
    throw new Error(`${label} stage must contain a flat numeric data array`);
  if (stage.data.some((value) => typeof value !== 'number' || !Number.isFinite(value)))
    throw new Error(`${label} stage contains non-finite or non-numeric data`);
  return stage.data;
}

function shapeOf(stage) {
  return Array.isArray(stage?.shape) ? stage.shape.map(Number) : null;
}
function argmax(values) {
  if (!values.length) return null;
  let best = 0;
  for (let index = 1; index < values.length; index += 1)
    if (values[index] > values[best]) best = index;
  return best;
}
function topKIndices(values, count) {
  return values
    .map((value, index) => ({ value, index }))
    .sort((left, right) => Math.abs(right.value) - Math.abs(left.value))
    .slice(0, count);
}

function compareStage(referenceStage, candidateStage, options) {
  const reference = numericData(referenceStage, 'reference');
  const candidate = numericData(candidateStage, 'candidate');
  const count = Math.min(reference.length, candidate.length);
  let sumAbs = 0;
  let sumSquared = 0;
  let dot = 0;
  let refNorm = 0;
  let candidateNorm = 0;
  let mismatches = 0;
  let firstMismatch = null;
  let maxAbs = 0;
  for (let index = 0; index < count; index += 1) {
    const left = reference[index];
    const right = candidate[index];
    const difference = Math.abs(left - right);
    const allowed =
      options.absTolerance + options.relTolerance * Math.max(Math.abs(left), Math.abs(right));
    sumAbs += difference;
    sumSquared += difference * difference;
    dot += left * right;
    refNorm += left * left;
    candidateNorm += right * right;
    maxAbs = Math.max(maxAbs, difference);
    if (difference > allowed) {
      mismatches += 1;
      if (!firstMismatch)
        firstMismatch = { index, reference: left, candidate: right, absDiff: difference, allowed };
    }
  }
  const lengthMatch = reference.length === candidate.length;
  const shapeMatch =
    JSON.stringify(shapeOf(referenceStage)) === JSON.stringify(shapeOf(candidateStage));
  const cosine =
    refNorm && candidateNorm
      ? dot / Math.sqrt(refNorm * candidateNorm)
      : refNorm === candidateNorm
        ? 1
        : 0;
  const argmaxReference = argmax(reference);
  const argmaxCandidate = argmax(candidate);
  return {
    pass: lengthMatch && shapeMatch && mismatches === 0,
    shape: {
      reference: shapeOf(referenceStage),
      candidate: shapeOf(candidateStage),
      match: shapeMatch,
    },
    dtype: { reference: referenceStage?.dtype ?? null, candidate: candidateStage?.dtype ?? null },
    count: {
      reference: reference.length,
      candidate: candidate.length,
      compared: count,
      match: lengthMatch,
    },
    stats: {
      maxAbs,
      meanAbs: count ? sumAbs / count : 0,
      rmse: count ? Math.sqrt(sumSquared / count) : 0,
      cosine,
      mismatches,
      mismatchRate: count ? mismatches / count : 0,
      firstMismatch,
      argmax: {
        reference: argmaxReference,
        candidate: argmaxCandidate,
        match: argmaxReference === argmaxCandidate,
      },
      referenceTopK: topKIndices(reference.slice(0, count), options.topK),
      candidateTopK: topKIndices(candidate.slice(0, count), options.topK),
    },
  };
}

function audioIdentity(sample) {
  return sample?.audio?.sha256 ?? sample?.audio?.audio_sha256 ?? sample?.audio?.identity ?? null;
}

function optionalTokenIds(sample, label) {
  const value = sample?.tokens ?? sample?.token_ids;
  if (value == null) return null;
  const ids = Array.isArray(value) ? value : value?.ids;
  if (!Array.isArray(ids) || ids.some((id) => !Number.isInteger(id))) {
    throw new Error(`${label} tokens must be an integer array or an object with an ids array`);
  }
  return ids;
}

function compareTokenIds(referenceSample, candidateSample, sampleId) {
  const reference = optionalTokenIds(referenceSample, `reference sample ${sampleId}`);
  const candidate = optionalTokenIds(candidateSample, `candidate sample ${sampleId}`);
  if (reference == null && candidate == null) return { pass: true, present: false };
  if (reference == null || candidate == null) {
    return {
      pass: false,
      present: true,
      failure: reference == null ? 'missing_reference_tokens' : 'missing_candidate_tokens',
    };
  }
  const compared = Math.min(reference.length, candidate.length);
  let firstMismatch = null;
  for (let index = 0; index < compared; index += 1) {
    if (reference[index] !== candidate[index]) {
      firstMismatch = { index, reference: reference[index], candidate: candidate[index] };
      break;
    }
  }
  return {
    pass: reference.length === candidate.length && firstMismatch == null,
    present: true,
    count: { reference: reference.length, candidate: candidate.length, compared },
    firstMismatch,
  };
}

function compareOptionalText(referenceSample, candidateSample) {
  const reference = referenceSample?.transcript;
  const candidate = candidateSample?.transcript;
  if (reference == null && candidate == null) return { pass: true, present: false };
  if (typeof reference !== 'string' || typeof candidate !== 'string') {
    return { pass: false, present: true, failure: 'missing_transcript' };
  }
  return {
    pass: reference === candidate,
    present: true,
    reference,
    candidate,
    match: reference === candidate,
  };
}

function optionalEos(sample) {
  return sample?.eos ?? sample?.generation?.eos ?? sample?.generation?.eos_id ?? null;
}

export function compareStageCaptures(referenceCapture, candidateCapture, options = {}) {
  const resolved = {
    absTolerance: options.absTolerance ?? DEFAULT_ABS_TOLERANCE,
    relTolerance: options.relTolerance ?? DEFAULT_REL_TOLERANCE,
    topK: options.topK ?? 5,
  };
  const referenceSamples = sampleList(referenceCapture, 'reference');
  const candidateSamples = sampleList(candidateCapture, 'candidate');
  const referenceIds = [...referenceSamples.keys()].sort();
  const referenceOnly = referenceIds.filter((id) => !candidateSamples.has(id));
  const candidateOnly = [...candidateSamples.keys()]
    .filter((id) => !referenceSamples.has(id))
    .sort();
  const samples = [];
  for (const sampleId of referenceIds) {
    const referenceSample = referenceSamples.get(sampleId);
    const candidateSample = candidateSamples.get(sampleId);
    if (!candidateSample) {
      samples.push({ sample_id: sampleId, pass: false, failure: 'missing_candidate_sample' });
      continue;
    }
    const referenceAudio = audioIdentity(referenceSample);
    const candidateAudio = audioIdentity(candidateSample);
    const audioMatch =
      referenceAudio == null || candidateAudio == null || referenceAudio === candidateAudio;
    const referenceStages = stageMap(referenceSample, 'reference', sampleId);
    const candidateStages = stageMap(candidateSample, 'candidate', sampleId);
    const stageNames = [
      ...new Set([...Object.keys(referenceStages), ...Object.keys(candidateStages)]),
    ].sort();
    const stages = {};
    for (const stageName of stageNames) {
      if (!(stageName in referenceStages))
        stages[stageName] = { pass: false, failure: 'missing_reference_stage' };
      else if (!(stageName in candidateStages))
        stages[stageName] = { pass: false, failure: 'missing_candidate_stage' };
      else
        stages[stageName] = compareStage(
          referenceStages[stageName],
          candidateStages[stageName],
          resolved,
        );
    }
    const failedStage = stageNames.find((stageName) => stages[stageName].pass !== true);
    const tokens = compareTokenIds(referenceSample, candidateSample, sampleId);
    const transcript = compareOptionalText(referenceSample, candidateSample);
    const referenceEos = optionalEos(referenceSample);
    const candidateEos = optionalEos(candidateSample);
    const eos = {
      present: referenceEos != null || candidateEos != null,
      reference: referenceEos,
      candidate: candidateEos,
      match: referenceEos === candidateEos,
    };
    const firstFailedStage = !audioMatch
      ? 'audio_identity'
      : (failedStage ??
        (!tokens.pass ? 'tokens' : !transcript.pass ? 'transcript' : !eos.match ? 'eos' : null));
    samples.push({
      sample_id: sampleId,
      pass: audioMatch && firstFailedStage == null,
      failure: audioMatch ? null : 'audio_identity_mismatch',
      audio: { reference: referenceAudio, candidate: candidateAudio, match: audioMatch },
      first_failed_stage: firstFailedStage,
      stages,
      outputs: { tokens, transcript, eos },
    });
  }
  return {
    schema_version: 1,
    comparison: {
      pass:
        referenceOnly.length === 0 &&
        candidateOnly.length === 0 &&
        samples.every((sample) => sample.pass),
      sample_count: samples.length,
      reference_only_sample_ids: referenceOnly,
      candidate_only_sample_ids: candidateOnly,
      tolerances: resolved,
    },
    samples,
  };
}

export function runCli(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  const report = compareStageCaptures(
    loadJson(options.reference),
    loadJson(options.candidate),
    options,
  );
  const serialized = `${JSON.stringify(report, null, 2)}\n`;
  if (options.output) fs.writeFileSync(options.output, serialized);
  else process.stdout.write(serialized);
  return report.comparison.pass ? 0 : 1;
}

if (
  process.argv[1] &&
  path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))
) {
  try {
    process.exitCode = runCli();
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 2;
  }
}

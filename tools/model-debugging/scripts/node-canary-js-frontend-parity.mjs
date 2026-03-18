import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';
import zlib from 'node:zlib';

const DEFAULT_REFERENCE = path.resolve(
  process.cwd(),
  'tools/data/results/canary/canary-180m-flash-reference.json',
);
const DEFAULT_MELJS_ROOT = 'N:\\github\\ysdede\\meljs';
const DEFAULT_PARAKEET_JS_ROOT = 'N:\\github\\ysdede\\parakeet.js';
const DEFAULT_ASRJS_DIST = path.resolve(process.cwd(), 'dist', 'audio', 'js-mel.js');

function parseNormalizationOption(value) {
  if (value !== 'per_feature' && value !== 'none') {
    throw new Error(`Unsupported normalization "${value}". Use "per_feature" or "none".`);
  }

  return value;
}

function normalizeModelId(value) {
  return String(value ?? '').trim().toLowerCase().replaceAll('-', '_');
}

function isParakeetRealtimeEouModel(value) {
  return normalizeModelId(value).includes('parakeet_realtime_eou_120m');
}

function parseArgs(argv) {
  const options = {
    frontend: 'meljs',
    reference: DEFAULT_REFERENCE,
    output: null,
    meljsRoot: DEFAULT_MELJS_ROOT,
    parakeetJsRoot: DEFAULT_PARAKEET_JS_ROOT,
    asrjsDist: DEFAULT_ASRJS_DIST,
    maxAbsThreshold: null,
    meanAbsThreshold: null,
    rmseThreshold: null,
    lengthTolerance: null,
    validLengthMode: null,
    normalization: null,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--frontend') options.frontend = argv[++index];
    else if (arg === '--reference') options.reference = path.resolve(argv[++index]);
    else if (arg === '--output') options.output = path.resolve(argv[++index]);
    else if (arg === '--meljs-root') options.meljsRoot = path.resolve(argv[++index]);
    else if (arg === '--parakeet-js-root') options.parakeetJsRoot = path.resolve(argv[++index]);
    else if (arg === '--asrjs-dist') options.asrjsDist = path.resolve(argv[++index]);
    else if (arg === '--max-abs-threshold') options.maxAbsThreshold = Number(argv[++index]);
    else if (arg === '--mean-abs-threshold') options.meanAbsThreshold = Number(argv[++index]);
    else if (arg === '--rmse-threshold') options.rmseThreshold = Number(argv[++index]);
    else if (arg === '--length-tolerance') options.lengthTolerance = Number(argv[++index]);
    else if (arg === '--valid-length-mode') options.validLengthMode = argv[++index];
    else if (arg === '--normalization') options.normalization = parseNormalizationOption(argv[++index]);
  }

  return options;
}

function ensureFile(filePath, label) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`${label} not found: ${filePath}`);
  }
  return filePath;
}

function resolveJsonArtifact(filePath) {
  if (fs.existsSync(filePath)) {
    return filePath;
  }
  const gzipPath = `${filePath}.gz`;
  if (fs.existsSync(gzipPath)) {
    return gzipPath;
  }
  throw new Error(`Reference JSON not found: ${filePath}`);
}

function loadReference(referencePath) {
  const resolvedPath = resolveJsonArtifact(referencePath);
  const raw = fs.readFileSync(resolvedPath);
  const decoded = JSON.parse(
    resolvedPath.endsWith('.gz') ? zlib.gunzipSync(raw).toString('utf8') : raw.toString('utf8'),
  );
  const waveform = decoded?.audio?.waveform;
  const preprocessor = decoded?.preprocessor;
  const features = preprocessor?.features;
  const featureLengths = preprocessor?.feature_lengths;
  const melBins = Number(decoded?.runtime_config?.mel_bins ?? 0);
  const sampleRate = Number(decoded?.runtime_config?.sample_rate ?? decoded?.meta?.input_sample_rate ?? 0);

  if (!waveform?.data || !Array.isArray(waveform.data)) {
    throw new Error(`Reference JSON does not contain audio.waveform.data: ${referencePath}`);
  }
  if (!features?.data || !Array.isArray(features.data)) {
    throw new Error(`Reference JSON does not contain preprocessor.features.data: ${referencePath}`);
  }
  if (!Array.isArray(featureLengths) || featureLengths.length === 0) {
    throw new Error(`Reference JSON does not contain preprocessor.feature_lengths: ${referencePath}`);
  }
  if (!Number.isFinite(melBins) || melBins <= 0) {
    throw new Error(`Reference JSON does not contain a valid runtime_config.mel_bins: ${referencePath}`);
  }
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new Error(`Reference JSON does not contain a valid sample rate: ${referencePath}`);
  }

  const refTotalFrames = Number(features.dims?.[2] ?? 0);
  const refValidLength = Number(featureLengths[0] ?? 0);

  return {
    modelId: decoded?.runtime_config?.model_id ?? decoded?.meta?.model_id ?? 'unknown',
    referencePath: resolvedPath,
    sampleRate,
    melBins,
    waveform: Float32Array.from(waveform.data),
    waveformSamples: Number(waveform.dims?.[1] ?? waveform.data.length),
    referenceFeatures: Float32Array.from(features.data),
    referenceTotalFrames: refTotalFrames,
    referenceValidLength: refValidLength,
  };
}

function resolveAsrjsValidLengthMode(options, reference) {
  if (options.validLengthMode === 'onnx' || options.validLengthMode === 'centered') {
    return options.validLengthMode;
  }

  const modelId = normalizeModelId(reference.modelId);
  if (modelId.includes('canary')) {
    return 'centered';
  }
  if (isParakeetRealtimeEouModel(modelId)) {
    return 'centered';
  }

  return 'onnx';
}

function resolveAsrjsNormalization(options, reference) {
  if (options.normalization != null) {
    return parseNormalizationOption(options.normalization);
  }

  if (isParakeetRealtimeEouModel(reference.modelId)) {
    return 'none';
  }

  return 'per_feature';
}

async function loadFrontendAdapter(options, reference) {
  const requested = String(options.frontend ?? '').toLowerCase();
  const melBins = reference.melBins;

  if (requested === 'meljs') {
    const modulePath = ensureFile(path.join(options.meljsRoot, 'src', 'mel.js'), 'meljs frontend');
    const module = await import(pathToFileURL(modulePath).href);
    return {
      label: 'meljs',
      modulePath,
      process(waveform) {
        const processor = new module.MelSpectrogram({ nMels: melBins });
        return processor.process(waveform);
      },
    };
  }

  if (requested === 'parakeet.js' || requested === 'parakeet-js') {
    const modulePath = ensureFile(
      path.join(options.parakeetJsRoot, 'src', 'mel.js'),
      'parakeet.js frontend',
    );
    const module = await import(pathToFileURL(modulePath).href);
    return {
      label: 'parakeet.js',
      modulePath,
      process(waveform) {
        const processor = new module.JsPreprocessor({ nMels: melBins });
        return processor.process(waveform);
      },
    };
  }

  if (requested === 'asrjs') {
    const modulePath = ensureFile(options.asrjsDist, 'asrjs frontend');
    const module = await import(pathToFileURL(modulePath).href);
    const validLengthMode = resolveAsrjsValidLengthMode(options, reference);
    const normalization = resolveAsrjsNormalization(options, reference);
    return {
      label: 'asrjs',
      modulePath,
      validLengthMode,
      normalization,
      process(waveform) {
        const processor = new module.JSMelProcessor({
          nMels: melBins,
          validLengthMode,
          normalization,
        });
        return processor.process(waveform);
      },
    };
  }

  throw new Error(
    `Unsupported frontend "${options.frontend}". Use "asrjs", "meljs", or "parakeet.js".`,
  );
}

function updateLargestMismatches(rows, entry, limit = 8) {
  rows.push(entry);
  rows.sort((left, right) => right.absDiff - left.absDiff);
  if (rows.length > limit) {
    rows.length = limit;
  }
}

function compareFeatures(reference, frontendResult) {
  const frontendFeatures = frontendResult.features;
  const frontendValidLength = Number(frontendResult.length ?? 0);
  const frontendTotalFrames = Math.floor(frontendFeatures.length / reference.melBins);
  const compareFrames = Math.min(
    reference.referenceValidLength,
    reference.referenceTotalFrames,
    frontendValidLength,
    frontendTotalFrames,
  );
  const compareValues = compareFrames * reference.melBins;

  let maxAbs = 0;
  let sumAbs = 0;
  let sumSquared = 0;
  const largestMismatches = [];

  for (let melIndex = 0; melIndex < reference.melBins; melIndex += 1) {
    const referenceBase = melIndex * reference.referenceTotalFrames;
    const frontendBase = melIndex * frontendTotalFrames;
    for (let frameIndex = 0; frameIndex < compareFrames; frameIndex += 1) {
      const referenceValue = reference.referenceFeatures[referenceBase + frameIndex] ?? 0;
      const frontendValue = frontendFeatures[frontendBase + frameIndex] ?? 0;
      const absDiff = Math.abs(referenceValue - frontendValue);
      if (absDiff > maxAbs) {
        maxAbs = absDiff;
      }
      sumAbs += absDiff;
      sumSquared += absDiff * absDiff;
      updateLargestMismatches(largestMismatches, {
        melIndex,
        frameIndex,
        referenceValue,
        frontendValue,
        absDiff,
      });
    }
  }

  return {
    frontendValidLength,
    frontendTotalFrames,
    compareFrames,
    compareValues,
    maxAbs,
    meanAbs: compareValues > 0 ? sumAbs / compareValues : 0,
    rmse: compareValues > 0 ? Math.sqrt(sumSquared / compareValues) : 0,
    largestMismatches,
  };
}

function evaluateThresholds(options, comparison) {
  const checks = {
    lengthMatch:
      options.lengthTolerance == null
        ? null
        : Math.abs(comparison.frontendValidLength - comparison.referenceValidLength) <=
            options.lengthTolerance,
    maxAbs:
      options.maxAbsThreshold == null ? null : comparison.maxAbs <= options.maxAbsThreshold,
    meanAbs:
      options.meanAbsThreshold == null ? null : comparison.meanAbs <= options.meanAbsThreshold,
    rmse: options.rmseThreshold == null ? null : comparison.rmse <= options.rmseThreshold,
  };

  const enforced = Object.values(checks).filter((value) => value !== null);
  return {
    checks,
    hasThresholds: enforced.length > 0,
    passed: enforced.every(Boolean),
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const reference = loadReference(options.reference);
  const frontend = await loadFrontendAdapter(options, reference);
  const frontendResult = frontend.process(reference.waveform);
  const comparison = compareFeatures(reference, frontendResult);

  const payload = {
    comparedAt: new Date().toISOString(),
    frontend: {
      name: frontend.label,
      modulePath: frontend.modulePath,
      validLengthMode: frontend.validLengthMode ?? null,
      normalization: frontend.normalization ?? null,
    },
    reference: {
      modelId: reference.modelId,
      referencePath: reference.referencePath,
      sampleRate: reference.sampleRate,
      waveformSamples: reference.waveformSamples,
      melBins: reference.melBins,
      validLength: reference.referenceValidLength,
      totalFrames: reference.referenceTotalFrames,
    },
    comparison: {
      featureLayoutCompatible: frontendResult.features.length % reference.melBins === 0,
      referenceValidLength: reference.referenceValidLength,
      frontendValidLength: comparison.frontendValidLength,
      validLengthDelta: comparison.frontendValidLength - reference.referenceValidLength,
      referenceTotalFrames: reference.referenceTotalFrames,
      frontendTotalFrames: comparison.frontendTotalFrames,
      totalFrameDelta: comparison.frontendTotalFrames - reference.referenceTotalFrames,
      comparedFrames: comparison.compareFrames,
      comparedValues: comparison.compareValues,
      maxAbs: comparison.maxAbs,
      meanAbs: comparison.meanAbs,
      rmse: comparison.rmse,
      largestMismatches: comparison.largestMismatches,
    },
  };

  const thresholdEvaluation = evaluateThresholds(options, {
    ...comparison,
    referenceValidLength: reference.referenceValidLength,
  });
  if (thresholdEvaluation.hasThresholds) {
    payload.thresholds = {
      lengthTolerance: options.lengthTolerance,
      maxAbsThreshold: options.maxAbsThreshold,
      meanAbsThreshold: options.meanAbsThreshold,
      rmseThreshold: options.rmseThreshold,
      checks: thresholdEvaluation.checks,
      passed: thresholdEvaluation.passed,
    };
  }

  const encoded = JSON.stringify(payload, null, 2);
  if (options.output) {
    fs.mkdirSync(path.dirname(options.output), { recursive: true });
    fs.writeFileSync(options.output, encoded, 'utf8');
  } else {
    console.log(encoded);
  }

  if (thresholdEvaluation.hasThresholds && !thresholdEvaluation.passed) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error('[node-canary-js-frontend-parity] failed:', error);
  process.exitCode = 1;
});

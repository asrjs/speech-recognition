#!/usr/bin/env node

/**
 * Audit a local ONNX bundle before it becomes a runtime artifact.
 *
 * The command is intentionally local-only. It never resolves model IDs,
 * contacts a model hub, or fills in missing external-data files.
 */

import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { inspectWhisperArtifactContract } from './whisper-artifact-contract.mjs';
import { inspectXAsrArtifactContract } from './x-asr-artifact-contract.mjs';

function parseArgs(argv) {
  const options = {
    modelDir: undefined,
    output: undefined,
    recursive: false,
    allowLoadFailures: false,
    whisperContract: false,
    requireCausalAlignment: false,
    requireMergedCrossAttention: false,
    xAsrContract: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--model-dir' && argv[index + 1]) {
      options.modelDir = argv[++index];
    } else if (arg === '--output' && argv[index + 1]) {
      options.output = argv[++index];
    } else if (arg === '--recursive') {
      options.recursive = true;
    } else if (arg === '--allow-load-failures') {
      options.allowLoadFailures = true;
    } else if (arg === '--whisper-contract') {
      options.whisperContract = true;
    } else if (arg === '--require-causal-alignment') {
      options.whisperContract = true;
      options.requireCausalAlignment = true;
    } else if (arg === '--require-merged-cross-attention') {
      options.whisperContract = true;
      options.requireMergedCrossAttention = true;
    } else if (arg === '--x-asr-contract') {
      options.xAsrContract = true;
    } else if (arg === '--help' || arg === '-h') {
      printUsage();
      process.exit(0);
    } else {
      throw new Error('Unknown argument: ' + arg);
    }
  }

  if (!options.modelDir) {
    throw new Error('--model-dir is required');
  }
  return options;
}

function printUsage() {
  console.log(
    [
      'Usage:',
      '  node node-audit-onnx-artifact.mjs --model-dir <directory> [options]',
      '',
      'Options:',
      '  --output <file>           Write JSON to a file instead of stdout',
      '  --recursive               Include ONNX files below nested directories',
      '  --allow-load-failures     Report CPU ORT failures without a non-zero exit',
      '  --whisper-contract        Report splitgraph/merged timestamp capabilities',
      '  --require-causal-alignment  Fail unless split decoder_align is explicitly causal',
      '  --require-merged-cross-attention  Fail unless every merged decoder exports cross_attentions.*',
      '  --x-asr-contract         Validate the four-variant X-ASR streaming artifact layout',
    ].join('\n'),
  );
}

async function walkFiles(directory, recursive) {
  const entries = await fs.readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      if (recursive) {
        files.push(...(await walkFiles(entryPath, true)));
      }
    } else if (entry.isFile()) {
      files.push(entryPath);
    }
  }
  return files;
}

function sha256File(filePath) {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = createReadStream(filePath);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

function relativeArtifactPath(modelDir, filePath) {
  return path.relative(modelDir, filePath).split(path.sep).join('/');
}

async function readJsonArtifact(modelDir, files, basenameToFind) {
  const relativeFiles = files.map((filePath) => ({
    filePath,
    relativePath: relativeArtifactPath(modelDir, filePath),
  }));
  const match =
    relativeFiles.find(({ relativePath }) => relativePath === basenameToFind) ??
    relativeFiles.find(({ relativePath }) => path.posix.basename(relativePath) === basenameToFind);
  if (!match) return undefined;
  try {
    return JSON.parse(await fs.readFile(match.filePath, 'utf8'));
  } catch {
    return undefined;
  }
}

function sidecarCandidates(graphPath) {
  const candidates = [
    graphPath + '.data',
    path.join(path.dirname(graphPath), path.basename(graphPath, '.onnx') + '.data'),
  ];
  return [...new Set(candidates)];
}

function sessionNames(session, key) {
  const value = session?.[key];
  return Array.isArray(value) ? [...value] : [];
}

function sessionMetadata(session, key) {
  const metadata = session?.[key];
  if (!metadata || typeof metadata !== 'object') return [];
  return Object.entries(metadata).map(([name, value]) => ({
    name,
    type: value?.type,
    dimensions: value?.dimensions ?? value?.shape,
  }));
}

async function auditGraph(ort, modelDir, graphPath) {
  const relativePath = relativeArtifactPath(modelDir, graphPath);
  const started = performance.now();
  const sidecars = [];
  for (const candidate of sidecarCandidates(graphPath)) {
    try {
      const stat = await fs.stat(candidate);
      sidecars.push({
        path: relativeArtifactPath(modelDir, candidate),
        exists: true,
        size_bytes: stat.size,
        sha256: await sha256File(candidate),
      });
    } catch (error) {
      if (error?.code !== 'ENOENT') {
        sidecars.push({
          path: relativeArtifactPath(modelDir, candidate),
          exists: false,
          error: String(error?.message ?? error),
        });
      }
    }
  }

  try {
    const session = await ort.InferenceSession.create(graphPath, {
      executionProviders: ['cpu'],
    });
    return {
      path: relativePath,
      loaded: true,
      execution_provider: 'cpu',
      load_ms: Number((performance.now() - started).toFixed(3)),
      input_names: sessionNames(session, 'inputNames'),
      output_names: sessionNames(session, 'outputNames'),
      input_metadata: sessionMetadata(session, 'inputMetadata'),
      output_metadata: sessionMetadata(session, 'outputMetadata'),
      external_data_candidates: sidecars,
    };
  } catch (error) {
    return {
      path: relativePath,
      loaded: false,
      execution_provider: 'cpu',
      load_ms: Number((performance.now() - started).toFixed(3)),
      external_data_candidates: sidecars,
      error: String(error?.message ?? error),
    };
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const modelDir = path.resolve(options.modelDir);
  const stat = await fs.stat(modelDir).catch(() => undefined);
  if (!stat?.isDirectory()) {
    throw new Error('Model directory not found: ' + modelDir);
  }

  const files = await walkFiles(modelDir, options.recursive);
  const onnxFiles = files.filter((filePath) => filePath.toLowerCase().endsWith('.onnx'));
  if (onnxFiles.length === 0) {
    throw new Error(
      'No ONNX files found in ' + modelDir + (options.recursive ? ' (recursive)' : ''),
    );
  }

  let ort;
  let ortImportError;
  try {
    const module = await import('onnxruntime-node');
    ort = module.default ?? module;
  } catch (error) {
    ortImportError = String(error?.message ?? error);
  }

  const inventory = [];
  for (const filePath of files) {
    const fileStat = await fs.stat(filePath);
    inventory.push({
      path: relativeArtifactPath(modelDir, filePath),
      size_bytes: fileStat.size,
      sha256: await sha256File(filePath),
    });
  }

  const graphs = ort
    ? await Promise.all(onnxFiles.map((filePath) => auditGraph(ort, modelDir, filePath)))
    : onnxFiles.map((filePath) => ({
        path: relativeArtifactPath(modelDir, filePath),
        loaded: false,
        execution_provider: 'cpu',
        load_ms: 0,
        external_data_candidates: [],
        error: 'onnxruntime-node import failed: ' + ortImportError,
      }));

  const failedGraphs = graphs.filter((graph) => !graph.loaded);
  const whisperContract = options.whisperContract
    ? inspectWhisperArtifactContract({
        config: await readJsonArtifact(modelDir, files, 'config.json'),
        manifest: await readJsonArtifact(modelDir, files, 'manifest.json'),
        graphs,
      })
    : undefined;
  const xAsrContract = options.xAsrContract
    ? inspectXAsrArtifactContract({ modelDir, files, graphs })
    : undefined;
  const contractFailures = [];
  if (
    options.requireCausalAlignment &&
    !whisperContract?.alignment.causal_self_attention_verified
  ) {
    contractFailures.push(
      'Whisper causal alignment is not verified: decoder_align requires alignment_export.causal_self_attention=true.',
    );
  }
  if (options.requireMergedCrossAttention && !whisperContract?.merged_cross_attention_verified) {
    contractFailures.push(
      'Whisper merged cross-attention is not verified: every merged decoder must export cross_attentions.*.',
    );
  }
  const report = {
    schema_version: 1,
    generated_at: new Date().toISOString(),
    model_dir: modelDir,
    recursive: options.recursive,
    local_only: true,
    inventory,
    graphs,
    ...(whisperContract ? { whisper_contract: whisperContract } : {}),
    ...(xAsrContract ? { x_asr_contract: xAsrContract } : {}),
    summary: {
      file_count: inventory.length,
      onnx_count: graphs.length,
      loaded_onnx_count: graphs.length - failedGraphs.length,
      failed_onnx_count: failedGraphs.length,
      ok: failedGraphs.length === 0,
      ...(options.whisperContract ? { whisper_contract_ok: contractFailures.length === 0 } : {}),
      ...(xAsrContract ? { x_asr_contract_ok: xAsrContract.ok } : {}),
    },
  };

  const encoded = JSON.stringify(report, null, 2);
  if (options.output) {
    const outputPath = path.resolve(options.output);
    const outputDirectory = path.dirname(outputPath);
    const outputDirectoryStat = await fs.stat(outputDirectory).catch(() => undefined);
    if (!outputDirectoryStat?.isDirectory()) {
      await fs.mkdir(outputDirectory, { recursive: true });
    }
    await fs.writeFile(outputPath, encoded + '\n', 'utf8');
    console.log('Wrote ONNX artifact audit to ' + outputPath);
  } else {
    console.log(encoded);
  }

  if (failedGraphs.length > 0 && !options.allowLoadFailures) {
    process.exitCode = 1;
  }
  if (contractFailures.length > 0) {
    for (const failure of contractFailures) console.error('Whisper contract failure: ' + failure);
    process.exitCode = 1;
  }
  if (xAsrContract && !xAsrContract.ok) {
    for (const failure of xAsrContract.failures) console.error('X-ASR contract failure: ' + failure);
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(error?.stack ?? String(error));
  process.exitCode = 1;
});

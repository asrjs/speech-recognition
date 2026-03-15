import fs from 'fs';
import path from 'path';
import process from 'process';

function parseArgs() {
  const args = process.argv.slice(2);
  const out = {
    python: null,
    node: null,
    pythonCase: null,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--python') out.python = path.resolve(process.cwd(), args[++i]);
    else if (arg === '--node') out.node = path.resolve(process.cwd(), args[++i]);
    else if (arg === '--python-case') out.pythonCase = args[++i];
  }

  if (!out.python || !out.node || !out.pythonCase) {
    throw new Error(
      'Usage: node ./scripts/compare-whisper-python-node.mjs ' +
      '--python <python-json> --python-case <case-name> --node <node-json>',
    );
  }

  return out;
}

function normalizeText(text) {
  return String(text ?? '')
    .replace(/\s+/g, ' ')
    .trim();
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function equalTimestamp(a, b, tolerance = 0.05) {
  if (a == null || b == null) return a === b;
  return Math.abs(Number(a) - Number(b)) <= tolerance;
}

function firstChunkMismatch(leftChunks, rightChunks) {
  const maxLength = Math.max(leftChunks.length, rightChunks.length);
  for (let i = 0; i < maxLength; i++) {
    const left = leftChunks[i] ?? null;
    const right = rightChunks[i] ?? null;
    if (!left || !right) {
      return { index: i, left, right };
    }

    const textMatches = normalizeText(left.text) === normalizeText(right.text);
    const leftTs = left.timestamp ?? [];
    const rightTs = right.timestamp ?? [];
    const tsMatches =
      leftTs.length === rightTs.length &&
      leftTs.every((value, idx) => equalTimestamp(value, rightTs[idx]));

    if (!textMatches || !tsMatches) {
      return { index: i, left, right };
    }
  }
  return null;
}

function main() {
  const opts = parseArgs();
  const python = readJson(opts.python);
  const node = readJson(opts.node);

  const model = python.models?.[0];
  if (!model) {
    throw new Error(`No models found in python JSON: ${opts.python}`);
  }

  const pythonCase = model.results?.find((entry) => entry.name === opts.pythonCase);
  if (!pythonCase) {
    throw new Error(`Case '${opts.pythonCase}' not found in python JSON: ${opts.python}`);
  }
  if (pythonCase.error) {
    throw new Error(`Python case '${opts.pythonCase}' failed: ${pythonCase.error.message}`);
  }

  const pythonOutput = pythonCase.output ?? {};
  const nodeOutput = node.output ?? {};
  const pythonChunks = Array.isArray(pythonOutput.chunks) ? pythonOutput.chunks : [];
  const nodeChunks = Array.isArray(nodeOutput.chunks) ? nodeOutput.chunks : [];

  const report = {
    python: {
      file: opts.python,
      case: opts.pythonCase,
      model: model.model,
      chunk_count: pythonChunks.length,
    },
    node: {
      file: opts.node,
      model: node.model,
      mode: node.mode,
      chunk_count: nodeChunks.length,
    },
    normalized_text_equal: normalizeText(pythonOutput.text) === normalizeText(nodeOutput.text),
    first_chunk_mismatch: firstChunkMismatch(pythonChunks, nodeChunks),
    python_first_chunk: pythonChunks[0] ?? null,
    node_first_chunk: nodeChunks[0] ?? null,
    python_last_chunk: pythonChunks.at(-1) ?? null,
    node_last_chunk: nodeChunks.at(-1) ?? null,
  };

  console.log(JSON.stringify(report, null, 2));
}

main();

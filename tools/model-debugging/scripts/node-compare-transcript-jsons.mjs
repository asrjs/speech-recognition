import fs from 'node:fs';
import path from 'node:path';

function parseArgs(argv) {
  const options = {
    left: '',
    right: '',
    leftPath: '',
    rightPath: '',
    leftLabel: 'left',
    rightLabel: 'right',
    outJson: '',
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--left' && argv[index + 1]) {
      options.left = argv[index + 1];
      index += 1;
      continue;
    }
    if (arg === '--right' && argv[index + 1]) {
      options.right = argv[index + 1];
      index += 1;
      continue;
    }
    if (arg === '--left-path' && argv[index + 1]) {
      options.leftPath = argv[index + 1];
      index += 1;
      continue;
    }
    if (arg === '--right-path' && argv[index + 1]) {
      options.rightPath = argv[index + 1];
      index += 1;
      continue;
    }
    if (arg === '--left-label' && argv[index + 1]) {
      options.leftLabel = argv[index + 1];
      index += 1;
      continue;
    }
    if (arg === '--right-label' && argv[index + 1]) {
      options.rightLabel = argv[index + 1];
      index += 1;
      continue;
    }
    if (arg === '--out-json' && argv[index + 1]) {
      options.outJson = argv[index + 1];
      index += 1;
    }
  }

  if (!options.left || !options.right) {
    throw new Error(
      'Usage: node node-compare-transcript-jsons.mjs --left <file> --right <file> ' +
        '[--left-path dotted.path] [--right-path dotted.path] [--out-json output.json]',
    );
  }

  return options;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
}

function getNestedValue(value, dottedPath) {
  if (!dottedPath) {
    return value;
  }

  return dottedPath.split('.').reduce((current, segment) => {
    if (current === null || current === undefined) {
      return undefined;
    }
    return current[segment];
  }, value);
}

function normalize(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function levenshtein(a, b) {
  const m = a.length;
  const n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Uint32Array(n + 1));
  for (let i = 0; i <= m; i += 1) {
    dp[i][0] = i;
  }
  for (let j = 0; j <= n; j += 1) {
    dp[0][j] = j;
  }
  for (let i = 1; i <= m; i += 1) {
    for (let j = 1; j <= n; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return dp[m][n];
}

function average(values) {
  return values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : 0;
}

function resolveFileKey(row) {
  if (typeof row?.file === 'string' && row.file.length > 0) {
    return row.file;
  }
  if (typeof row?.audio_path === 'string' && row.audio_path.length > 0) {
    return path.basename(row.audio_path);
  }
  if (typeof row?.wav_path === 'string' && row.wav_path.length > 0) {
    return path.basename(row.wav_path);
  }
  return undefined;
}

function resolveText(row, explicitPath) {
  const candidates = explicitPath
    ? [explicitPath]
    : ['prediction', 'text', 'transcript', 'asrjs.text', 'medasrjs.text', 'canonical.text'];

  for (const candidate of candidates) {
    const value = getNestedValue(row, candidate);
    if (typeof value === 'string') {
      return value;
    }
  }

  return undefined;
}

function extractRows(document, explicitPath) {
  const rows = Array.isArray(document) ? document : Array.isArray(document?.rows) ? document.rows : null;
  if (!rows) {
    throw new Error('Expected either a JSON array or an object with a top-level "rows" array.');
  }

  const extracted = new Map();
  for (const row of rows) {
    const file = resolveFileKey(row);
    const text = resolveText(row, explicitPath);
    if (!file || typeof text !== 'string') {
      continue;
    }
    extracted.set(file, text);
  }
  return extracted;
}

const options = parseArgs(process.argv.slice(2));
const leftRows = extractRows(readJson(options.left), options.leftPath);
const rightRows = extractRows(readJson(options.right), options.rightPath);

const sharedFiles = [...leftRows.keys()].filter((file) => rightRows.has(file)).sort();
const leftOnly = [...leftRows.keys()].filter((file) => !rightRows.has(file)).sort();
const rightOnly = [...rightRows.keys()].filter((file) => !leftRows.has(file)).sort();

const comparisons = sharedFiles.map((file) => {
  const leftText = leftRows.get(file) ?? '';
  const rightText = rightRows.get(file) ?? '';
  const normalizedLeft = normalize(leftText);
  const normalizedRight = normalize(rightText);
  const normalizedExact = normalizedLeft === normalizedRight;
  const rawExact = leftText === rightText;
  const charDistance = levenshtein(normalizedLeft, normalizedRight);
  const leftWords = normalizedLeft.split(' ').filter(Boolean);
  const rightWords = normalizedRight.split(' ').filter(Boolean);
  const wordDistance = levenshtein(leftWords, rightWords);

  return {
    file,
    rawExact,
    normalizedExact,
    charErrorRate: normalizedRight.length ? charDistance / normalizedRight.length : 0,
    wordErrorRate: rightWords.length ? wordDistance / rightWords.length : 0,
    [options.leftLabel]: leftText,
    [options.rightLabel]: rightText,
  };
});

comparisons.sort((left, right) => right.charErrorRate - left.charErrorRate);

const summary = {
  comparedAt: new Date().toISOString(),
  left: options.left,
  right: options.right,
  leftLabel: options.leftLabel,
  rightLabel: options.rightLabel,
  leftPath: options.leftPath || 'auto',
  rightPath: options.rightPath || 'auto',
  comparedFiles: comparisons.length,
  rawExactMatches: comparisons.filter((row) => row.rawExact).length,
  normalizedExactMatches: comparisons.filter((row) => row.normalizedExact).length,
  averageCharErrorRate: average(comparisons.map((row) => row.charErrorRate)),
  averageWordErrorRate: average(comparisons.map((row) => row.wordErrorRate)),
  leftOnlyFiles: leftOnly,
  rightOnlyFiles: rightOnly,
};

const output = {
  summary,
  worstMismatches: comparisons.slice(0, 10),
};

if (options.outJson) {
  fs.mkdirSync(path.dirname(options.outJson), { recursive: true });
  fs.writeFileSync(options.outJson, JSON.stringify(output, null, 2), 'utf-8');
}

console.log('--- transcript json comparison ---');
console.log(`Compared files: ${summary.comparedFiles}`);
console.log(`Raw exact matches: ${summary.rawExactMatches}/${summary.comparedFiles}`);
console.log(`Normalized exact matches: ${summary.normalizedExactMatches}/${summary.comparedFiles}`);
console.log(`Average CER: ${(summary.averageCharErrorRate * 100).toFixed(2)}%`);
console.log(`Average WER: ${(summary.averageWordErrorRate * 100).toFixed(2)}%`);
if (leftOnly.length > 0) {
  console.log(`${options.leftLabel}-only files: ${leftOnly.length}`);
}
if (rightOnly.length > 0) {
  console.log(`${options.rightLabel}-only files: ${rightOnly.length}`);
}
if (options.outJson) {
  console.log(`Results: ${options.outJson}`);
}

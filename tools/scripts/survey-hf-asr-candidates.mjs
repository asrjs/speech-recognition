#!/usr/bin/env node

/**
 * Capture a reproducible popularity and artifact-signal snapshot for ASR
 * candidates on Hugging Face.
 *
 * This is discovery evidence only. It does not claim model quality, ONNX
 * parity, or browser support. Those claims require the reference and browser
 * gates in docs/GOAL_PROMPT.md.
 *
 * Examples:
 *   node tools/scripts/survey-hf-asr-candidates.mjs --output tools/data/results/model-candidates/hf-asr-candidates.json
 *   node tools/scripts/survey-hf-asr-candidates.mjs --limit 100 --search asr
 */

import { mkdir, writeFile } from 'node:fs/promises';

const API_ROOT = 'https://huggingface.co/api/models';
const USER_AGENT = 'asrjs-speech-recognition-candidate-survey/1.0';
const DEFAULT_LIMIT = 50;
const DEFAULT_OUTPUT = 'tools/data/results/model-candidates/hf-asr-candidates.json';
const LANGUAGE_CODES = new Set(
  'af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr fy ga gl gu ha he hi hr hu hy id is it ja jv ka kk km kn ko ky la lb ln lo lt lv mg mi mk ml mn mr ms mt my ne nl no oc pa pl ps pt ro ru sa sd si sk sl sn so sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi yi yo zh zu'.split(
    ' ',
  ),
);

function parseArgs(argv) {
  const args = {
    limit: DEFAULT_LIMIT,
    search: '',
    include: [],
    output: DEFAULT_OUTPUT,
    concurrency: 4,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === '--limit') args.limit = Number(argv[++index]);
    else if (value === '--search') args.search = argv[++index] ?? '';
    else if (value === '--include') {
      args.include.push(
        ...(argv[++index] ?? '')
          .split(',')
          .map((id) => id.trim())
          .filter(Boolean),
      );
    } else if (value === '--output') args.output = argv[++index] ?? DEFAULT_OUTPUT;
    else if (value === '--concurrency') args.concurrency = Number(argv[++index]);
    else if (value === '--help' || value === '-h') {
      console.log('Usage: node tools/scripts/survey-hf-asr-candidates.mjs [options]');
      console.log('  --limit N          ranked model count (default 50)');
      console.log('  --search TEXT      Hugging Face search term');
      console.log('  --include IDS      comma-separated IDs to inspect even if not ranked');
      console.log('  --output PATH      JSON destination');
      console.log('  --concurrency N    metadata request concurrency (default 4)');
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${value}`);
    }
  }
  if (!Number.isInteger(args.limit) || args.limit < 1 || args.limit > 500) {
    throw new Error('--limit must be an integer between 1 and 500');
  }
  if (!Number.isInteger(args.concurrency) || args.concurrency < 1 || args.concurrency > 16) {
    throw new Error('--concurrency must be an integer between 1 and 16');
  }
  return args;
}

async function fetchJson(url) {
  const response = await fetch(url, {
    headers: { accept: 'application/json', 'user-agent': USER_AGENT },
    signal: AbortSignal.timeout(30_000),
  });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} (${url})`);
  }
  return response.json();
}

function fileSignals(siblings) {
  const files = siblings.map((entry) => entry.rfilename).filter(Boolean);
  const count = (pattern) => files.filter((name) => pattern.test(name)).length;
  return {
    onnx: count(/\.onnx$/i),
    onnxExternalData: count(/\.onnx(_data|\.data)$/i),
    wasm: count(/\.wasm$/i),
    gguf: count(/\.gguf$/i),
    safetensors: count(/\.safetensors(?:\.index\.json)?$/i),
    pytorch: count(/(?:^|\/)(?:model|pytorch_model).*\.(?:bin|pt|pth)$/i),
    nemo: count(/\.nemo$/i),
    fileCount: files.length,
  };
}

function multilingualSignal(model) {
  const tags = Array.isArray(model.tags) ? model.tags : [];
  const languages = Array.isArray(model.languages) ? model.languages : [];
  const normalizedTags = tags.map((tag) => String(tag).toLowerCase());
  const languageTags = normalizedTags.filter((tag) => LANGUAGE_CODES.has(tag));
  const text = [...normalizedTags, ...languages].join(' ');
  return {
    tagged:
      text.includes('multilingual') || text.includes('multi-lingual') || languageTags.length >= 2,
    languages: languages.slice(0, 64),
    languageTags,
  };
}

function normalizeModel(model, detail) {
  const siblings = Array.isArray(detail?.siblings) ? detail.siblings : [];
  const tags = Array.isArray(detail?.tags) ? detail.tags : (model.tags ?? []);
  const cardData = detail?.cardData ?? {};
  const languages = Array.isArray(cardData.language)
    ? cardData.language
    : Array.isArray(detail?.languages)
      ? detail.languages
      : [];
  const licenseTag = tags.find((tag) => String(tag).startsWith('license:'));
  return {
    id: model.id,
    url: `https://huggingface.co/${model.id}`,
    downloads: model.downloads ?? null,
    likes: model.likes ?? null,
    lastModified: model.lastModified ?? null,
    pipelineTag: detail?.pipeline_tag ?? model.pipeline_tag ?? null,
    library: detail?.library_name ?? model.library_name ?? null,
    license: cardData.license ?? (licenseTag ? String(licenseTag).slice(8) : null),
    licenseName: cardData.license_name ?? null,
    languages,
    parameterCount: detail?.safetensors?.total ?? null,
    usedStorageBytes: detail?.usedStorage ?? null,
    tags,
    multilingual: multilingualSignal({ tags, languages }),
    artifactSignals: fileSignals(siblings),
    representativeFiles: siblings
      .map((entry) => entry.rfilename)
      .filter(Boolean)
      .slice(0, 40),
    metadataError: null,
  };
}

async function mapWithConcurrency(items, concurrency, mapper) {
  const results = new Array(items.length);
  let next = 0;
  async function worker() {
    while (true) {
      const index = next;
      next += 1;
      if (index >= items.length) return;
      results[index] = await mapper(items[index], index);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, () => worker()));
  return results;
}

const args = parseArgs(process.argv.slice(2));
const rankedUrl = new URL(API_ROOT);
rankedUrl.searchParams.set('pipeline_tag', 'automatic-speech-recognition');
rankedUrl.searchParams.set('sort', 'downloads');
rankedUrl.searchParams.set('direction', '-1');
rankedUrl.searchParams.set('limit', String(args.limit));
if (args.search) rankedUrl.searchParams.set('search', args.search);

const ranked = await fetchJson(rankedUrl);
const included = await mapWithConcurrency(args.include, args.concurrency, async (id) => {
  try {
    const detail = await fetchJson(`${API_ROOT}/${id}`);
    return normalizeModel(detail, detail);
  } catch (error) {
    return {
      id,
      url: `https://huggingface.co/${id}`,
      downloads: null,
      likes: null,
      lastModified: null,
      pipelineTag: null,
      library: null,
      license: null,
      licenseName: null,
      languages: [],
      parameterCount: null,
      usedStorageBytes: null,
      tags: [],
      multilingual: { tagged: false, languages: [], languageTags: [] },
      artifactSignals: fileSignals([]),
      representativeFiles: [],
      metadataError: error instanceof Error ? error.message : String(error),
    };
  }
});
const rankedIds = new Set(ranked.map((model) => model.id));
const records = [...ranked, ...included.filter((model) => !rankedIds.has(model.id))];
const models = await mapWithConcurrency(records, args.concurrency, async (model) => {
  try {
    return normalizeModel(model, await fetchJson(`${API_ROOT}/${model.id}`));
  } catch (error) {
    const normalized = normalizeModel(model, null);
    normalized.metadataError = error instanceof Error ? error.message : String(error);
    return normalized;
  }
});

const output = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  query: {
    endpoint: rankedUrl.toString(),
    includedIds: args.include,
    sort: 'downloads descending',
    purpose: 'candidate discovery; not quality or parity evidence',
  },
  models,
};

await mkdir(args.output.split(/[\\/]/).slice(0, -1).join('/') || '.', {
  recursive: true,
});
await writeFile(args.output, `${JSON.stringify(output, null, 2)}\n`, 'utf8');
console.log(`Wrote ${models.length} model records to ${args.output}`);

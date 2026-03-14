const HF_DATASET_API = 'https://datasets-server.huggingface.co';
const MIN_REQUEST_GAP_MS = 200;

let lastRequestAt = 0;
let requestQueue: Promise<void> = Promise.resolve();

export interface DatasetSplitInfo {
  readonly dataset?: string;
  readonly config?: string;
  readonly split?: string;
  readonly num_examples?: number;
}

export interface DatasetRowsRequest {
  readonly dataset: string;
  readonly config: string;
  readonly split: string;
  readonly offset?: number;
  readonly length?: number;
}

export interface DatasetRowWrapper<T = Record<string, unknown>> {
  readonly row?: T;
  readonly row_idx?: number;
}

export interface NormalizedDatasetRow<T = Record<string, unknown>> {
  readonly rowIndex: number;
  readonly audioUrl: string | null;
  readonly referenceText: string;
  readonly speaker: string;
  readonly gender: string;
  readonly speed?: number;
  readonly volume?: number;
  readonly sampleRate: number;
  readonly raw: T;
}

function buildUrl(path: string, params: Record<string, string | number | undefined | null>): string {
  const url = new URL(`${HF_DATASET_API}${path}`);
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== '') {
      url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function enqueueRequest<T>(task: () => Promise<T>): Promise<T> {
  const run = requestQueue
    .catch(() => {})
    .then(async () => {
      const now = Date.now();
      const elapsed = now - lastRequestAt;
      if (elapsed < MIN_REQUEST_GAP_MS) {
        await sleep(MIN_REQUEST_GAP_MS - elapsed);
      }
      const result = await task();
      lastRequestAt = Date.now();
      return result;
    });
  requestQueue = run.then(() => undefined).catch(() => undefined);
  return run;
}

function parseRetryAfterMs(response: Response, fallbackMs: number): number {
  const retryAfter = response.headers.get('retry-after');
  if (!retryAfter) return fallbackMs;
  const asSeconds = Number(retryAfter);
  if (Number.isFinite(asSeconds)) {
    return Math.max(fallbackMs, asSeconds * 1000);
  }
  const asDate = Date.parse(retryAfter);
  if (Number.isFinite(asDate)) {
    return Math.max(fallbackMs, asDate - Date.now());
  }
  return fallbackMs;
}

async function fetchJson<T>(url: string, options: { retries?: number; retryDelayMs?: number } = {}): Promise<T> {
  const retries = options.retries ?? 2;
  const retryDelayMs = options.retryDelayMs ?? 700;
  let lastError: unknown = null;

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const response = await enqueueRequest(() => fetch(url));
      if (!response.ok) {
        const bodyText = await response.text();
        const shouldRetry = response.status >= 500 || response.status === 429;
        if (shouldRetry && attempt < retries) {
          await sleep(parseRetryAfterMs(response, retryDelayMs * (attempt + 1)));
          continue;
        }
        throw new Error(`Request failed (${response.status}): ${bodyText}`);
      }
      return await response.json() as T;
    } catch (error) {
      lastError = error;
      if (attempt >= retries) {
        break;
      }
      const isNetwork = /Failed to fetch|NetworkError|Load failed|CORS|ERR_FAILED/i.test(String((error as Error)?.message || error));
      const backoff = isNetwork
        ? Math.max(1200, retryDelayMs * (attempt + 1) * 2)
        : retryDelayMs * (attempt + 1);
      await sleep(backoff);
    }
  }

  throw (lastError instanceof Error ? lastError : new Error('Request failed'));
}

export async function fetchDatasetSplits(dataset: string): Promise<readonly DatasetSplitInfo[]> {
  const payload = await fetchJson<{ splits?: DatasetSplitInfo[] }>(buildUrl('/splits', { dataset }));
  return payload.splits || [];
}

export async function fetchDatasetInfo(dataset: string, config: string): Promise<Record<string, unknown>> {
  return await fetchJson<Record<string, unknown>>(buildUrl('/info', { dataset, config }));
}

export async function fetchDatasetRows(request: DatasetRowsRequest): Promise<Record<string, unknown>> {
  const safeLength = Math.max(1, Math.min(100, Number(request.length) || 1));
  const safeOffset = Math.max(0, Number(request.offset) || 0);
  return await fetchJson<Record<string, unknown>>(buildUrl('/rows', {
    dataset: request.dataset,
    config: request.config,
    split: request.split,
    offset: safeOffset,
    length: safeLength,
  }));
}

export function extractAudioUrl(audioField: unknown): string | null {
  if (!audioField) return null;
  if (typeof audioField === 'string') return audioField;

  if (Array.isArray(audioField)) {
    for (const item of audioField) {
      const extracted = extractAudioUrl(item);
      if (extracted) return extracted;
    }
    return null;
  }

  if (typeof audioField === 'object') {
    const candidate = audioField as { src?: string; url?: string; path?: string };
    return candidate.src ?? candidate.url ?? candidate.path ?? null;
  }

  return null;
}

export function normalizeReferenceText(value: unknown): string {
  return String(value || '')
    .replace(/PARAGRAPH/g, '\n\n')
    .replace(/NEWLINE/g, '\n')
    .replace(/\s+\n/g, '\n')
    .replace(/\n\s+/g, '\n')
    .trim();
}

export function normalizeDatasetRow<T extends Record<string, unknown>>(rowWrapper: DatasetRowWrapper<T> | T, fallbackIndex = 0): NormalizedDatasetRow<T> {
  const wrapper = rowWrapper as DatasetRowWrapper<T>;
  const row = (wrapper.row || rowWrapper) as T;
  const rowIndex = wrapper.row_idx ?? fallbackIndex;
  return {
    rowIndex,
    audioUrl: extractAudioUrl((row as Record<string, unknown>).audio),
    referenceText: normalizeReferenceText(
      (row as Record<string, unknown>).transcription
      || (row as Record<string, unknown>).text
      || (row as Record<string, unknown>).transcript
      || ''
    ),
    speaker: String((row as Record<string, unknown>).speaker || ''),
    gender: String((row as Record<string, unknown>).gender || ''),
    speed: Number((row as Record<string, unknown>).speed),
    volume: Number((row as Record<string, unknown>).volume),
    sampleRate: Number((row as Record<string, unknown>).sample_rate) || 16000,
    raw: row,
  };
}

export async function fetchSequentialRows(request: DatasetRowsRequest & { readonly startOffset?: number; readonly limit?: number }): Promise<{
  readonly rows: readonly Record<string, unknown>[];
  readonly totalRows: number | null;
}> {
  const rows: Record<string, unknown>[] = [];
  let cursor = Math.max(0, Number(request.startOffset) || 0);
  let totalRows: number | null = null;
  const limit = Math.max(1, Number(request.limit) || 1);

  while (rows.length < limit) {
    const pageLength = Math.min(100, limit - rows.length);
    const page = await fetchDatasetRows({ ...request, offset: cursor, length: pageLength }) as {
      rows?: Record<string, unknown>[];
      num_rows_total?: number;
    };
    const pageRows = page.rows || [];
    totalRows = page.num_rows_total ?? totalRows;
    if (pageRows.length === 0) break;
    rows.push(...pageRows);
    cursor += pageRows.length;
    if (pageRows.length < pageLength) break;
  }

  return { rows, totalRows };
}

function normalizeSeed(seed: unknown): number | null {
  if (seed === undefined || seed === null || seed === '') return null;
  const text = String(seed);
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function createMulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let value = Math.imul(state ^ (state >>> 15), state | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export async function fetchRandomRows(request: DatasetRowsRequest & {
  readonly totalRows: number;
  readonly sampleCount: number;
  readonly seed?: string | number;
}): Promise<{
  readonly rows: readonly Record<string, unknown>[];
  readonly offsets: readonly number[];
  readonly failedOffsets: readonly number[];
  readonly totalRows: number;
  readonly seedUsed: number | null;
  readonly requestedCount: number;
}> {
  const targetCount = Math.max(1, Number(request.sampleCount) || 1);
  const maxRows = Math.max(1, Number(request.totalRows) || 1);
  const wanted = Math.min(targetCount, maxRows);
  const seedValue = normalizeSeed(request.seed);
  const rand = seedValue === null ? Math.random : createMulberry32(seedValue);
  const requestedOffsets: number[] = [];
  const selectedOffsets = new Set<number>();
  const successfulOffsets: number[] = [];
  const failedOffsets: number[] = [];
  const rows: Record<string, unknown>[] = [];

  const maxOffsetAttempts = Math.min(maxRows * 3, maxRows + wanted * 20);
  let offsetAttempts = 0;
  while (selectedOffsets.size < wanted && offsetAttempts < maxOffsetAttempts && selectedOffsets.size < maxRows) {
    offsetAttempts += 1;
    const offset = Math.floor(rand() * maxRows);
    if (selectedOffsets.has(offset)) continue;
    selectedOffsets.add(offset);
    requestedOffsets.push(offset);
  }

  const pageMap = new Map<number, number[]>();
  for (const offset of requestedOffsets) {
    const pageStart = Math.floor(offset / 100) * 100;
    const offsets = pageMap.get(pageStart) || [];
    offsets.push(offset);
    pageMap.set(pageStart, offsets);
  }

  const pageStarts = [...pageMap.keys()].sort((a, b) => a - b);
  for (const pageStart of pageStarts) {
    if (rows.length >= wanted) break;
    try {
      const page = await fetchDatasetRows({ ...request, offset: pageStart, length: 100 }) as { rows?: Record<string, unknown>[] };
      const pageRows = page.rows || [];
      const offsetsInPage = pageMap.get(pageStart) || [];
      for (const absoluteOffset of offsetsInPage) {
        if (rows.length >= wanted) break;
        const row = pageRows[absoluteOffset - pageStart];
        if (!row) {
          failedOffsets.push(absoluteOffset);
          continue;
        }
        rows.push(row);
        successfulOffsets.push(absoluteOffset);
      }
    } catch {
      failedOffsets.push(...(pageMap.get(pageStart) || []));
    }
  }

  return {
    rows,
    offsets: successfulOffsets,
    failedOffsets,
    totalRows: maxRows,
    seedUsed: seedValue,
    requestedCount: wanted,
  };
}

export function getConfigsAndSplits(splits: readonly DatasetSplitInfo[]): ReadonlyMap<string, readonly string[]> {
  const byConfig = new Map<string, string[]>();
  for (const item of splits) {
    if (!item.config || !item.split) continue;
    const list = byConfig.get(item.config) || [];
    if (!list.includes(item.split)) {
      list.push(item.split);
    }
    byConfig.set(item.config, list);
  }
  return byConfig;
}

export { HF_DATASET_API };

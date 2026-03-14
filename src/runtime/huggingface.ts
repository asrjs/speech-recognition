export const DEFAULT_MODEL_REVISIONS = ['main'] as const;
export const QUANTIZATION_ORDER = ['fp16', 'int8', 'fp32'] as const;

const MODEL_REVISIONS_CACHE = new Map<string, readonly string[]>();
const MODEL_FILES_CACHE = new Map<string, readonly string[]>();

export type QuantizationMode = 'fp16' | 'int8' | 'fp32';

export interface ModelFileProgress {
  readonly loaded: number;
  readonly total: number;
  readonly file: string;
}

export function formatRepoPath(repoId: string): string {
  return String(repoId || '')
    .split('/')
    .map((part) => encodeURIComponent(part))
    .join('/');
}

function normalizePath(path: string): string {
  return String(path || '').replace(/^\.\/+/, '').replace(/\\/g, '/');
}

function parseModelFiles(payload: unknown): string[] {
  if (Array.isArray(payload)) {
    return payload
      .filter((entry) => entry && typeof entry === 'object' && (entry as { type?: unknown }).type === 'file')
      .map((entry) => normalizePath((entry as { path?: string }).path ?? ''))
      .filter(Boolean);
  }

  if (payload && typeof payload === 'object' && Array.isArray((payload as { siblings?: unknown[] }).siblings)) {
    return (payload as { siblings: Array<{ rfilename?: string }> }).siblings
      .map((entry) => normalizePath(entry?.rfilename ?? ''))
      .filter(Boolean);
  }

  return [];
}

function hasFile(files: readonly string[], filename: string): boolean {
  const target = normalizePath(filename);
  return files.some((path) => path === target || path.endsWith(`/${target}`));
}

interface IdbDatabaseLike {
  readonly objectStoreNames: {
    contains(name: string): boolean;
  };
  createObjectStore(name: string): unknown;
  transaction(names: string[], mode: 'readonly' | 'readwrite'): IdbTransactionLike;
  close?(): void;
}

interface IdbTransactionLike {
  objectStore(name: string): IdbObjectStoreLike;
}

interface IdbObjectStoreLike {
  get(key: string): { onsuccess: (() => void) | null; onerror: (() => void) | null; result?: Blob };
  put(blob: Blob, key: string): { onsuccess: (() => void) | null; onerror: (() => void) | null; result?: IDBValidKey };
}

let dbPromise: Promise<IdbDatabaseLike> | null = null;

function hasIndexedDb(): boolean {
  return typeof indexedDB !== 'undefined';
}

async function getDb(): Promise<IdbDatabaseLike> {
  if (!hasIndexedDb()) {
    throw new Error('IndexedDB is unavailable in this environment.');
  }

  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open('asrjs-cache-db', 1);
      request.onerror = () => reject(new Error('Error opening IndexedDB.'));
      request.onsuccess = () => resolve(request.result as unknown as IdbDatabaseLike);
      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result as unknown as IdbDatabaseLike;
        if (!db.objectStoreNames.contains('file-store')) {
          db.createObjectStore('file-store');
        }
      };
    });
  }

  return dbPromise;
}

async function getFileFromDb(key: string): Promise<Blob | undefined> {
  const db = await getDb();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['file-store'], 'readonly');
    const store = transaction.objectStore('file-store');
    const request = store.get(key);
    request.onerror = () => reject(new Error('Error reading from IndexedDB.'));
    request.onsuccess = () => resolve(request.result);
  });
}

async function saveFileToDb(key: string, blob: Blob): Promise<void> {
  const db = await getDb();
  await new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(['file-store'], 'readwrite');
    const store = transaction.objectStore('file-store');
    const request = store.put(blob, key);
    request.onerror = () => reject(new Error('Error writing to IndexedDB.'));
    request.onsuccess = () => resolve();
  });
}

export async function fetchModelRevisions(repoId: string): Promise<readonly string[]> {
  if (!repoId) {
    return DEFAULT_MODEL_REVISIONS;
  }
  if (MODEL_REVISIONS_CACHE.has(repoId)) {
    return MODEL_REVISIONS_CACHE.get(repoId)!;
  }

  try {
    const repoPath = formatRepoPath(repoId);
    const response = await fetch(`https://huggingface.co/api/models/${repoPath}/refs`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json() as { branches?: Array<{ name?: string }> };
    const branches = Array.isArray(payload?.branches)
      ? payload.branches.map((branch) => branch?.name).filter(Boolean) as string[]
      : [];
    const revisions = branches.length > 0 ? branches : [...DEFAULT_MODEL_REVISIONS];
    MODEL_REVISIONS_CACHE.set(repoId, revisions);
    return revisions;
  } catch (error) {
    console.warn(`[huggingface] Failed to fetch revisions for ${repoId}; using defaults.`, error);
    return DEFAULT_MODEL_REVISIONS;
  }
}

export async function fetchModelFiles(repoId: string, revision = 'main'): Promise<readonly string[]> {
  if (!repoId) {
    return [];
  }
  const cacheKey = `${repoId}@${revision}`;
  if (MODEL_FILES_CACHE.has(cacheKey)) {
    return MODEL_FILES_CACHE.get(cacheKey)!;
  }

  const repoPath = formatRepoPath(repoId);
  const encodedRevision = encodeURIComponent(revision);
  const treeUrl = `https://huggingface.co/api/models/${repoPath}/tree/${encodedRevision}?recursive=1`;
  const metadataUrl = `https://huggingface.co/api/models/${repoPath}?revision=${encodedRevision}`;

  try {
    const response = await fetch(treeUrl);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const files = parseModelFiles(await response.json());
    MODEL_FILES_CACHE.set(cacheKey, files);
    return files;
  } catch (treeError) {
    console.warn(`[huggingface] Tree listing failed for ${repoId}@${revision}; trying metadata.`, treeError);
  }

  try {
    const response = await fetch(metadataUrl);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const files = parseModelFiles(await response.json());
    MODEL_FILES_CACHE.set(cacheKey, files);
    return files;
  } catch (metadataError) {
    console.warn(`[huggingface] Metadata listing failed for ${repoId}@${revision}.`, metadataError);
    return [];
  }
}

export function getAvailableQuantModes(files: readonly string[], baseName: string): QuantizationMode[] {
  const options = QUANTIZATION_ORDER.filter((quant) => {
    if (quant === 'fp32') {
      return hasFile(files, `${baseName}.onnx`);
    }
    if (quant === 'fp16') {
      return hasFile(files, `${baseName}.fp16.onnx`);
    }
    return hasFile(files, `${baseName}.int8.onnx`);
  });

  return options.length > 0 ? [...options] : ['fp32'];
}

export function pickPreferredQuant(
  available: readonly QuantizationMode[],
  currentBackend: string,
  component: 'encoder' | 'decoder' = 'encoder'
): QuantizationMode {
  const preferred = component === 'decoder'
    ? ['int8', 'fp32', 'fp16']
    : String(currentBackend || '').startsWith('webgpu')
      ? ['fp16', 'fp32', 'int8']
      : ['int8', 'fp32', 'fp16'];

  return (preferred.find((quant) => available.includes(quant as QuantizationMode)) ?? available[0] ?? 'fp32') as QuantizationMode;
}

export async function getModelFile(
  repoId: string,
  filename: string,
  options: {
    readonly revision?: string;
    readonly subfolder?: string;
    readonly progress?: (progress: ModelFileProgress) => void;
  } = {}
): Promise<string> {
  const { revision = 'main', subfolder = '', progress } = options;
  const encodedRevision = encodeURIComponent(revision);
  const encodedSubfolder = subfolder
    ? subfolder.split('/').map((part) => encodeURIComponent(part)).join('/')
    : '';
  const encodedFilename = filename
    .split('/')
    .map((part) => encodeURIComponent(part))
    .join('/');

  const pathParts = [formatRepoPath(repoId), 'resolve', encodedRevision];
  if (encodedSubfolder) {
    pathParts.push(encodedSubfolder);
  }
  pathParts.push(encodedFilename);
  const url = `https://huggingface.co/${pathParts.join('/')}`;
  const cacheKey = `hf-${repoId}-${revision}-${subfolder}-${filename}`;

  if (hasIndexedDb()) {
    try {
      const cachedBlob = await getFileFromDb(cacheKey);
      if (cachedBlob) {
        return URL.createObjectURL(cachedBlob);
      }
    } catch (error) {
      console.warn('[huggingface] IndexedDB cache check failed.', error);
    }
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download ${filename}: ${response.status} ${response.statusText}`);
  }

  const body = response.body;
  if (!body || typeof body.getReader !== 'function') {
    const blob = await response.blob();
    if (hasIndexedDb()) {
      try {
        await saveFileToDb(cacheKey, blob);
      } catch (error) {
        console.warn('[huggingface] Failed to cache file in IndexedDB.', error);
      }
    }
    return URL.createObjectURL(blob);
  }

  const contentLength = response.headers.get('content-length');
  const total = contentLength ? Number.parseInt(contentLength, 10) : 0;
  let loaded = 0;
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    if (value) {
      chunks.push(value);
      loaded += value.length;
      if (progress && total > 0) {
        progress({ loaded, total, file: filename });
      }
    }
  }

  const blobParts: BlobPart[] = chunks.map((chunk) => new Uint8Array(chunk));
  const blob = new Blob(blobParts, {
    type: response.headers.get('content-type') || 'application/octet-stream'
  });

  if (hasIndexedDb()) {
    try {
      await saveFileToDb(cacheKey, blob);
    } catch (error) {
      console.warn('[huggingface] Failed to cache file in IndexedDB.', error);
    }
  }

  return URL.createObjectURL(blob);
}

export async function getModelText(
  repoId: string,
  filename: string,
  options: {
    readonly revision?: string;
    readonly subfolder?: string;
    readonly progress?: (progress: ModelFileProgress) => void;
  } = {}
): Promise<string> {
  const blobUrl = await getModelFile(repoId, filename, options);
  const response = await fetch(blobUrl);
  const text = await response.text();
  URL.revokeObjectURL(blobUrl);
  return text;
}

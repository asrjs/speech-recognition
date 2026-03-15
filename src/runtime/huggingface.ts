import { createHuggingFaceAssetProvider, getDefaultIndexedDbAssetCache } from '../io/index.js';
import type { AssetRequest, ResolvedAssetHandle, RuntimeLogger } from '../types/index.js';

export const DEFAULT_MODEL_REVISIONS = ['main'] as const;
export const QUANTIZATION_ORDER = ['fp16', 'int8', 'fp32'] as const;

const MODEL_REVISIONS_CACHE = new Map<string, readonly string[]>();
const MODEL_FILES_CACHE = new Map<string, readonly string[]>();

export type QuantizationMode = 'fp16' | 'int8' | 'fp32';

export interface ModelFileProgress {
  readonly loaded: number;
  readonly total: number;
  readonly file: string;
  readonly percent?: number;
  readonly loadedMiB: number;
  readonly totalMiB?: number;
  readonly isComplete?: boolean;
}

const MEBIBYTE = 1024 * 1024;

function bytesToMiB(bytes: number | undefined): number | undefined {
  return Number.isFinite(bytes) ? (bytes as number) / MEBIBYTE : undefined;
}

function createProgressMilestoneReporter(
  filename: string,
  progress?: (progress: ModelFileProgress) => void,
):
  | ((event: { readonly loaded: number; readonly total?: number; readonly done?: boolean }) => void)
  | undefined {
  if (!progress) {
    return undefined;
  }

  let lastPercent = -1;
  let lastUnknownMiBFloor = -1;
  let completed = false;

  return (event) => {
    const loaded = event.loaded;
    const total = event.total ?? 0;
    const percent = total > 0 ? Math.floor((loaded / total) * 100) : undefined;
    const done = Boolean(event.done) || (total > 0 && loaded >= total);

    if (done && completed) {
      return;
    }

    if (percent !== undefined) {
      if (!done && percent <= lastPercent) {
        return;
      }
      lastPercent = Math.max(lastPercent, percent);
    } else {
      const currentMiBFloor = Math.floor(loaded / MEBIBYTE);
      if (!done && currentMiBFloor <= lastUnknownMiBFloor) {
        return;
      }
      lastUnknownMiBFloor = Math.max(lastUnknownMiBFloor, currentMiBFloor);
    }

    if (done) {
      completed = true;
    }

    progress({
      loaded,
      total,
      file: filename,
      percent: done ? 100 : percent,
      loadedMiB: bytesToMiB(loaded) ?? 0,
      totalMiB: bytesToMiB(total),
      isComplete: done,
    });
  };
}

export function formatRepoPath(repoId: string): string {
  return String(repoId || '')
    .split('/')
    .map((part) => encodeURIComponent(part))
    .join('/');
}

function normalizePath(path: string): string {
  return String(path || '')
    .replace(/^\.\/+/, '')
    .replace(/\\/g, '/');
}

function parseModelFiles(payload: unknown): string[] {
  if (Array.isArray(payload)) {
    return payload
      .filter(
        (entry) =>
          entry && typeof entry === 'object' && (entry as { type?: unknown }).type === 'file',
      )
      .map((entry) => normalizePath((entry as { path?: string }).path ?? ''))
      .filter(Boolean);
  }

  if (
    payload &&
    typeof payload === 'object' &&
    Array.isArray((payload as { siblings?: unknown[] }).siblings)
  ) {
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

function createModelAssetRequest(
  repoId: string,
  filename: string,
  options: {
    readonly revision?: string;
    readonly subfolder?: string;
    readonly progress?: (progress: ModelFileProgress) => void;
  } = {},
): AssetRequest {
  const { revision = 'main', subfolder = '', progress } = options;
  const reportProgress = createProgressMilestoneReporter(filename, progress);

  return {
    id: `${repoId}:${revision}:${subfolder}:${filename}`,
    provider: 'huggingface',
    repoId,
    revision,
    filename,
    subfolder,
    cacheKey: `hf-${repoId}-${revision}-${subfolder}-${filename}`,
    onProgress: reportProgress,
  };
}

function createDefaultHuggingFaceProvider() {
  return createHuggingFaceAssetProvider({
    cache: getDefaultIndexedDbAssetCache() ?? undefined,
  });
}

export async function fetchModelRevisions(
  repoId: string,
  options: { readonly logger?: RuntimeLogger } = {},
): Promise<readonly string[]> {
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
    const payload = (await response.json()) as { branches?: Array<{ name?: string }> };
    const branches = Array.isArray(payload?.branches)
      ? (payload.branches.map((branch) => branch?.name).filter(Boolean) as string[])
      : [];
    const revisions = branches.length > 0 ? branches : [...DEFAULT_MODEL_REVISIONS];
    MODEL_REVISIONS_CACHE.set(repoId, revisions);
    return revisions;
  } catch (error) {
    options.logger?.warn?.(`[huggingface] Failed to fetch revisions for ${repoId}; using defaults.`, {
      error,
    });
    return DEFAULT_MODEL_REVISIONS;
  }
}

export async function fetchModelFiles(
  repoId: string,
  revision = 'main',
  options: { readonly logger?: RuntimeLogger } = {},
): Promise<readonly string[]> {
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
    options.logger?.warn?.(
      `[huggingface] Tree listing failed for ${repoId}@${revision}; trying metadata.`,
      { error: treeError },
    );
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
    options.logger?.warn?.(`[huggingface] Metadata listing failed for ${repoId}@${revision}.`, {
      error: metadataError,
    });
    return [];
  }
}

export function getAvailableQuantModes(
  files: readonly string[],
  baseName: string,
): QuantizationMode[] {
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
  component: 'encoder' | 'decoder' = 'encoder',
): QuantizationMode {
  const preferred =
    component === 'decoder'
      ? ['int8', 'fp32', 'fp16']
      : String(currentBackend || '').startsWith('webgpu')
        ? ['fp16', 'fp32', 'int8']
        : ['int8', 'fp32', 'fp16'];

  return (preferred.find((quant) => available.includes(quant as QuantizationMode)) ??
    available[0] ??
    'fp32') as QuantizationMode;
}

export async function getModelAssetHandle(
  repoId: string,
  filename: string,
  options: {
    readonly revision?: string;
    readonly subfolder?: string;
    readonly progress?: (progress: ModelFileProgress) => void;
  } = {},
): Promise<ResolvedAssetHandle> {
  const provider = createDefaultHuggingFaceProvider();
  return provider.resolve(createModelAssetRequest(repoId, filename, options));
}

export async function getModelFile(
  repoId: string,
  filename: string,
  options: {
    readonly revision?: string;
    readonly subfolder?: string;
    readonly progress?: (progress: ModelFileProgress) => void;
  } = {},
): Promise<string> {
  const handle = await getModelAssetHandle(repoId, filename, options);
  const locator = await handle.getLocator('url');
  if (!locator) {
    await handle.dispose();
    throw new Error(`Could not create a URL locator for ${filename}.`);
  }
  return locator;
}

export async function getModelText(
  repoId: string,
  filename: string,
  options: {
    readonly revision?: string;
    readonly subfolder?: string;
    readonly progress?: (progress: ModelFileProgress) => void;
  } = {},
): Promise<string> {
  const handle = await getModelAssetHandle(repoId, filename, options);
  try {
    return await handle.readText();
  } finally {
    await handle.dispose();
  }
}

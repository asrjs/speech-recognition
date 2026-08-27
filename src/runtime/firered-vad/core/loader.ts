import type { FireRedAssetCache } from '../types.js';
import { getNodeBuiltin } from '../../../io/node-builtin.js';
import { isLikelyHttpUrl, isNodeRuntime, looksLikeFileUrl } from './util.js';
import {
  fetchBytesHonoringAbort,
  rethrowIfAssetAborted,
  throwIfAssetAborted,
  type AssetAbortSignalLike,
} from '../../../io/abort.js';

function toCacheKey(source: string): string {
  return `firered:${source}`;
}

async function readNodeFile(pathLike: string): Promise<Uint8Array> {
  const fs = getNodeBuiltin<{
    readFile(path: string | URL): Promise<Uint8Array>;
  }>('fs/promises');
  const normalized = pathLike.startsWith('file://') ? new URL(pathLike) : pathLike;
  const bytes = await fs.readFile(normalized);
  return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

async function fetchBytes(url: string, signal?: AssetAbortSignalLike | null): Promise<Uint8Array> {
  return fetchBytesHonoringAbort(url, signal);
}

export async function loadBinaryResource(
  source: string | Uint8Array,
  cache?: FireRedAssetCache,
  signal?: AssetAbortSignalLike | null,
): Promise<Uint8Array> {
  throwIfAssetAborted(signal, 'download');
  if (source instanceof Uint8Array) {
    return source;
  }
  const key = toCacheKey(source);
  if (cache) {
    try {
      const cached = await cache.get(key);
      if (cached) {
        return cached.bytes;
      }
    } catch (error) {
      rethrowIfAssetAborted(error, 'download');
      // Cache issues should not block model loading.
    }
  }

  let bytes: Uint8Array;
  if (
    isLikelyHttpUrl(source) ||
    looksLikeFileUrl(source) ||
    (!isNodeRuntime() && !source.startsWith('.'))
  ) {
    bytes = await fetchBytes(source, signal);
  } else if (isNodeRuntime()) {
    throwIfAssetAborted(signal, 'download');
    bytes = await readNodeFile(source);
    throwIfAssetAborted(signal, 'download');
  } else {
    bytes = await fetchBytes(source, signal);
  }

  throwIfAssetAborted(signal, 'download');
  if (cache) {
    try {
      await cache.set(key, { bytes });
    } catch (error) {
      rethrowIfAssetAborted(error, 'download');
      // Cache issues should not block model loading.
    }
  }
  return bytes;
}

export function resolveModelUrl(modelDir: string, filename: string): string {
  if (isLikelyHttpUrl(modelDir) || looksLikeFileUrl(modelDir)) {
    const base = modelDir.endsWith('/') ? modelDir : `${modelDir}/`;
    return new URL(filename, base).href;
  }
  if (isNodeRuntime()) {
    return `${modelDir}${modelDir.endsWith('\\') || modelDir.endsWith('/') ? '' : '/'}${filename}`;
  }
  const base = modelDir.endsWith('/') ? modelDir : `${modelDir}/`;
  return `${base}${filename}`;
}

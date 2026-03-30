import type { FireRedAssetCache } from '../types.js';
import { isLikelyHttpUrl, isNodeRuntime, looksLikeFileUrl } from './util.js';

function toCacheKey(source: string): string {
  return `firered:${source}`;
}

async function readNodeFile(pathLike: string): Promise<Uint8Array> {
  const fs = await import('node:fs/promises');
  const normalized = pathLike.startsWith('file://') ? new URL(pathLike) : pathLike;
  const bytes = await fs.readFile(normalized);
  return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

export async function loadBinaryResource(
  source: string | Uint8Array,
  cache?: FireRedAssetCache,
): Promise<Uint8Array> {
  if (source instanceof Uint8Array) {
    return source;
  }
  const key = toCacheKey(source);
  if (cache) {
    const cached = await cache.get(key);
    if (cached) {
      return cached.bytes;
    }
  }

  let bytes: Uint8Array;
  if (isLikelyHttpUrl(source) || looksLikeFileUrl(source) || (!isNodeRuntime() && !source.startsWith('.'))) {
    bytes = await fetchBytes(source);
  } else if (isNodeRuntime()) {
    bytes = await readNodeFile(source);
  } else {
    bytes = await fetchBytes(source);
  }

  if (cache) {
    await cache.set(key, { bytes });
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

import type { AssetRequest } from '../types/index.js';

export interface AssetCacheIdentity {
  readonly primaryKey?: string;
  readonly automaticAliases: readonly string[];
}

interface HuggingFaceAsset {
  readonly repoId: string;
  readonly revision: string;
  readonly assetPath: string;
}

function canonicalKey(asset: HuggingFaceAsset): string {
  return `hf-asset-v1:${JSON.stringify([asset.repoId, asset.revision, asset.assetPath])}`;
}

function familyAliases(asset: HuggingFaceAsset): string[] {
  const parts = asset.assetPath.split('/').filter(Boolean);
  const keys = [`huggingface:${asset.repoId}:${asset.revision}:${asset.assetPath}`];
  for (let split = 0; split < parts.length; split += 1) {
    const subfolder = parts.slice(0, split).join('/');
    const filename = parts.slice(split).join('/');
    keys.push(`huggingface:${asset.repoId}:${asset.revision}:${subfolder}:${filename}`);
  }
  return keys;
}

function helperAliases(asset: HuggingFaceAsset): string[] {
  const parts = asset.assetPath.split('/').filter(Boolean);
  const keys: string[] = [];
  for (let split = 0; split < parts.length; split += 1) {
    const subfolder = parts.slice(0, split).join('/');
    const filename = parts.slice(split).join('/');
    keys.push(`hf-${asset.repoId}-${asset.revision}-${subfolder}-${filename}`);
  }
  return keys;
}

function trustedMainUrlAsset(url: string): HuggingFaceAsset | null {
  try {
    const parsed = new URL(url);
    if (
      parsed.protocol !== 'https:' ||
      parsed.hostname !== 'huggingface.co' ||
      parsed.port ||
      parsed.username ||
      parsed.password ||
      parsed.search ||
      parsed.hash
    ) {
      return null;
    }
    const parts = parsed.pathname.split('/').filter(Boolean).map(decodeURIComponent);
    const resolveIndex = parts.indexOf('resolve');
    if (
      resolveIndex !== 2 ||
      parts[resolveIndex + 1] !== 'main' ||
      parts.length <= resolveIndex + 2
    ) {
      return null;
    }
    const repoId = `${parts[0]}/${parts[1]}`;
    const assetPath = parts
      .slice(resolveIndex + 2)
      .filter(Boolean)
      .join('/');
    if (!assetPath) return null;
    return {
      repoId,
      revision: 'main',
      assetPath,
    };
  } catch {
    return null;
  }
}

function requestAsset(request: AssetRequest): HuggingFaceAsset | null {
  if (request.provider !== 'huggingface' || !request.repoId || !request.filename) {
    return null;
  }
  const revision = request.revision ?? 'main';
  if (revision !== 'main') return null;
  const assetPath = [request.subfolder, request.filename].filter(Boolean).join('/');
  if (!assetPath) return null;
  return {
    repoId: request.repoId,
    revision,
    assetPath: assetPath.split('/').filter(Boolean).join('/'),
  };
}

function isGeneratedHuggingFaceKey(key: string, asset: HuggingFaceAsset): boolean {
  return [...familyAliases(asset), ...helperAliases(asset)].includes(key);
}

/** Unifies only exact, automatically generated Hugging Face `main` artifact identities. */
export function getAssetCacheIdentity(
  request: AssetRequest,
  resolvedUrl: string,
): AssetCacheIdentity {
  const tuple = requestAsset(request);
  let asset = tuple;

  if (!asset && request.provider === 'url' && request.url === resolvedUrl) {
    asset = trustedMainUrlAsset(resolvedUrl);
  }
  if (!asset) return { primaryKey: request.cacheKey, automaticAliases: [] };

  // Unknown caller keys are opaque by contract. Only recognized built-in key
  // shapes (or an absent key) may opt into tuple normalization.
  if (
    request.cacheKey &&
    !isGeneratedHuggingFaceKey(request.cacheKey, asset) &&
    request.cacheKey !== `url:${resolvedUrl}`
  ) {
    return { primaryKey: request.cacheKey, automaticAliases: [] };
  }

  const aliases = [...familyAliases(asset), ...helperAliases(asset), `url:${resolvedUrl}`].filter(
    (key) => key !== canonicalKey(asset!),
  );
  return {
    primaryKey: canonicalKey(asset),
    automaticAliases: [...new Set(aliases)],
  };
}

import { getDefaultIndexedDbAssetCache } from './cache.js';
import { BlobAssetHandle, UrlAssetHandle } from './handles.js';
import type {
  AssetCache,
  AssetProvider,
  AssetRequest,
  ResolvedAssetHandle,
} from '../types/index.js';

function isHttpUrl(value: string | undefined): boolean {
  return typeof value === 'string' && /^https?:\/\//i.test(value);
}

function formatRepoPath(repoId: string): string {
  return String(repoId || '')
    .split('/')
    .map((part) => encodeURIComponent(part))
    .join('/');
}

function normalizeRevision(revision: string | undefined): string {
  return revision ?? 'main';
}

function buildHuggingFaceResolveUrl(request: AssetRequest): string {
  if (!request.repoId || !request.filename) {
    throw new Error('Hugging Face asset requests require repoId and filename.');
  }

  const encodedRevision = encodeURIComponent(normalizeRevision(request.revision));
  const encodedSubfolder = request.subfolder
    ? request.subfolder
        .split('/')
        .map((part) => encodeURIComponent(part))
        .join('/')
    : '';
  const encodedFilename = request.filename
    .split('/')
    .map((part) => encodeURIComponent(part))
    .join('/');

  const pathParts = [formatRepoPath(request.repoId), 'resolve', encodedRevision];
  if (encodedSubfolder) {
    pathParts.push(encodedSubfolder);
  }
  pathParts.push(encodedFilename);
  return `https://huggingface.co/${pathParts.join('/')}`;
}

class CompositeAssetProvider implements AssetProvider {
  constructor(private readonly providers: readonly AssetProvider[]) {}

  canResolve(request: AssetRequest): boolean {
    return this.providers.some((provider) => provider.canResolve(request));
  }

  async resolve(request: AssetRequest): Promise<ResolvedAssetHandle> {
    for (const provider of this.providers) {
      if (provider.canResolve(request)) {
        return provider.resolve(request);
      }
    }

    throw new Error(`No asset provider can resolve "${request.id}".`);
  }
}

class BlobAssetProvider implements AssetProvider {
  canResolve(request: AssetRequest): boolean {
    return !!request.blob || !!request.bytes || !!request.fileHandle;
  }

  async resolve(request: AssetRequest): Promise<ResolvedAssetHandle> {
    if (request.blob) {
      return new BlobAssetHandle(request, request.blob);
    }
    if (request.bytes) {
      return new BlobAssetHandle(
        request,
        new Blob([request.bytes.slice().buffer], {
          type: request.contentType ?? 'application/octet-stream',
        }),
      );
    }
    if (request.fileHandle) {
      const file = await request.fileHandle.getFile();
      return new BlobAssetHandle(
        request,
        file instanceof Blob
          ? file
          : new Blob([file], {
              type: request.contentType ?? 'application/octet-stream',
            }),
      );
    }

    throw new Error(`Blob asset provider cannot resolve "${request.id}".`);
  }
}

class UrlAssetProvider implements AssetProvider {
  constructor(private readonly cache?: AssetCache) {}

  canResolve(request: AssetRequest): boolean {
    return isHttpUrl(request.url) || request.provider === 'url';
  }

  async resolve(request: AssetRequest): Promise<ResolvedAssetHandle> {
    if (!request.url) {
      throw new Error(`URL asset request "${request.id}" is missing a URL.`);
    }

    return new UrlAssetHandle(request, request.url, this.cache);
  }
}

class HuggingFaceAssetProvider implements AssetProvider {
  constructor(private readonly cache?: AssetCache) {}

  canResolve(request: AssetRequest): boolean {
    return request.provider === 'huggingface';
  }

  async resolve(request: AssetRequest): Promise<ResolvedAssetHandle> {
    return new UrlAssetHandle(request, buildHuggingFaceResolveUrl(request), this.cache);
  }
}

export function createCompositeAssetProvider(providers: readonly AssetProvider[]): AssetProvider {
  return new CompositeAssetProvider(providers);
}

export function createBlobAssetProvider(): AssetProvider {
  return new BlobAssetProvider();
}

export function createBrowserFetchAssetProvider(
  options: { readonly cache?: AssetCache } = {},
): AssetProvider {
  return new UrlAssetProvider(options.cache);
}

export function createHuggingFaceAssetProvider(
  options: { readonly cache?: AssetCache } = {},
): AssetProvider {
  return new HuggingFaceAssetProvider(options.cache);
}

export function createDefaultAssetProvider(
  options: { readonly cache?: AssetCache } = {},
): AssetProvider {
  const cache = options.cache ?? getDefaultIndexedDbAssetCache() ?? undefined;

  return createCompositeAssetProvider([
    createBlobAssetProvider(),
    createHuggingFaceAssetProvider({ cache }),
    createBrowserFetchAssetProvider({ cache }),
  ]);
}

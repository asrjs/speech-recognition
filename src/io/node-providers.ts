import { getDefaultIndexedDbAssetCache } from './cache.js';
import { NodePathAssetHandle } from './node-handles.js';
import { importNodeModule } from './node.js';
import {
  createBlobAssetProvider,
  createBrowserFetchAssetProvider,
  createCompositeAssetProvider,
  createHuggingFaceAssetProvider,
} from './providers.js';
import type {
  AssetCache,
  AssetProvider,
  AssetRequest,
  ResolvedAssetHandle,
} from '../types/index.js';

function isFileUrl(value: string | undefined): boolean {
  return typeof value === 'string' && /^file:/i.test(value);
}

class NodeFileSystemAssetProvider implements AssetProvider {
  canResolve(request: AssetRequest): boolean {
    return (
      typeof request.path === 'string' || request.provider === 'path' || isFileUrl(request.url)
    );
  }

  async resolve(request: AssetRequest): Promise<ResolvedAssetHandle> {
    let path = request.path;
    if (!path && isFileUrl(request.url)) {
      const { fileURLToPath } = await importNodeModule<typeof import('node:url')>('node:url');
      path = fileURLToPath(request.url!);
    }

    if (!path) {
      throw new Error(`Path asset request "${request.id}" is missing a path.`);
    }

    return new NodePathAssetHandle(request, path);
  }
}

export function createNodeFileSystemAssetProvider(): AssetProvider {
  return new NodeFileSystemAssetProvider();
}

export function createDefaultNodeAssetProvider(
  options: { readonly cache?: AssetCache } = {},
): AssetProvider {
  const cache = options.cache ?? getDefaultIndexedDbAssetCache() ?? undefined;

  return createCompositeAssetProvider([
    createBlobAssetProvider(),
    createNodeFileSystemAssetProvider(),
    createHuggingFaceAssetProvider({ cache }),
    createBrowserFetchAssetProvider({ cache }),
  ]);
}

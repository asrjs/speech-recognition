import { getNodeBuiltin } from './node-builtin.js';

interface NodeProcessLike {
  readonly versions?: { readonly node?: string };
}

function nodeProcess(): NodeProcessLike | undefined {
  return (globalThis as typeof globalThis & { process?: NodeProcessLike }).process;
}

/**
 * Browser-safe replacement for the Node-only module bridge.
 *
 * This module deliberately contains no `node:` imports. Node-only operations
 * resolve built-ins lazily through process.getBuiltinModule(), while browser
 * callers can still import root-reachable model families safely.
 */
export function isNodeLikeRuntime(): boolean {
  return typeof nodeProcess()?.versions?.node === 'string';
}

export async function importNodeModule<T = unknown>(specifier: string): Promise<T> {
  if (!isNodeLikeRuntime()) {
    throw new Error(`Node module imports are unavailable outside Node.js: ${specifier}`);
  }

  switch (specifier) {
    case 'node:fs':
    case 'node:fs/promises':
    case 'node:module':
    case 'node:path':
    case 'node:url':
      return getNodeBuiltin<T>(specifier.slice('node:'.length));
    default:
      throw new Error(`Unsupported Node module import: ${specifier}`);
  }
}

export async function resolveNodePackageSubpathUrl(
  packageName: string,
  subpath: string,
): Promise<string> {
  const nodeModule = getNodeBuiltin<{
    createRequire(url: string): {
      resolve(id: string): string;
      (id: string): unknown;
    };
  }>('module');
  const nodePath = getNodeBuiltin<typeof import('node:path')>('path');
  const { pathToFileURL } = getNodeBuiltin<typeof import('node:url')>('url');
  const require = nodeModule.createRequire(import.meta.url);
  const packageEntryPath = require.resolve(packageName);
  let currentDir = nodePath.dirname(packageEntryPath);

  while (true) {
    const packageJsonPath = nodePath.join(currentDir, 'package.json');
    try {
      require(packageJsonPath);
      const absoluteSubpath = nodePath.resolve(currentDir, subpath);
      const url = pathToFileURL(absoluteSubpath).href;
      return url.endsWith('/') ? url : `${url}/`;
    } catch {
      const parentDir = nodePath.dirname(currentDir);
      if (parentDir === currentDir) {
        throw new Error(`Unable to locate package root for ${packageName}.`);
      }
      currentDir = parentDir;
    }
  }
}

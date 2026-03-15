import { createRequire } from 'node:module';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);

export function isNodeLikeRuntime(): boolean {
  return typeof process !== 'undefined' && !!process.versions?.node;
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
      return require(specifier) as T;
    default:
      throw new Error(`Unsupported Node module import: ${specifier}`);
  }
}

export async function resolveNodePackageSubpathUrl(
  packageName: string,
  subpath: string,
): Promise<string> {
  const packageEntryPath = require.resolve(packageName);
  let currentDir = path.dirname(packageEntryPath);

  while (true) {
    const packageJsonPath = path.join(currentDir, 'package.json');
    try {
      require(packageJsonPath);
      const absoluteSubpath = path.resolve(currentDir, subpath);
      const url = pathToFileURL(absoluteSubpath).href;
      return url.endsWith('/') ? url : `${url}/`;
    } catch {
      const parentDir = path.dirname(currentDir);
      if (parentDir === currentDir) {
        throw new Error(`Unable to locate package root for ${packageName}.`);
      }
      currentDir = parentDir;
    }
  }
}

import { importNodeModule, isNodeLikeRuntime } from './node-compat.js';

export type OrtExternalDataMount = {
  readonly data: string | Uint8Array;
  readonly path: string;
};

export type ResolveOrtExternalDataOptions = {
  readonly backendId: 'webgpu' | 'wasm';
  readonly sessionModelUrl: string;
  readonly externalDataUrl: string;
  readonly externalDataPath: string;
};

export type NativeOrtExternalDataFile = {
  readonly dataUrl: string;
  readonly path: string;
};

/** Validate native ORT sidecars without reading their contents into JavaScript. */
export async function validateNativeOrtExternalDataFiles(
  sessionModelUrl: string,
  externalData: readonly NativeOrtExternalDataFile[],
): Promise<void> {
  const [nodePath, { fileURLToPath }] = await Promise.all([
    importNodeModule<typeof import('node:path')>('node:path'),
    importNodeModule<typeof import('node:url')>('node:url'),
  ]);
  const modelPath = /^file:/i.test(sessionModelUrl)
    ? fileURLToPath(sessionModelUrl)
    : sessionModelUrl;
  if (!nodePath.isAbsolute(modelPath)) {
    throw new Error('Native CPU inference requires a local absolute model path or file URL.');
  }

  const fs = await importNodeModule<typeof import('node:fs/promises')>('node:fs/promises');
  for (const entry of externalData) {
    const dataPath = /^file:/i.test(entry.dataUrl) ? fileURLToPath(entry.dataUrl) : entry.dataUrl;
    const expectedPath = nodePath.resolve(nodePath.dirname(modelPath), entry.path);
    if (nodePath.isAbsolute(entry.path) || nodePath.resolve(dataPath) !== expectedPath) {
      throw new Error('Native external data must be colocated at its graph-relative location.');
    }
    if (!(await fs.stat(expectedPath)).isFile()) {
      throw new Error('Native external data locator is not a regular file.');
    }
  }
}

/**
 * Builds ORT `externalData` session options for colocated `.onnx.data` files.
 *
 * Browser callers pass URL locators. Node callers pass byte buffers to ORT Web
 * (including its WebGPU execution provider). Native ORT callers must handle
 * colocated files themselves and skip this resolver so large sidecars are not
 * read into JavaScript memory.
 */
export async function resolveOrtExternalDataMounts(
  options: ResolveOrtExternalDataOptions,
): Promise<readonly OrtExternalDataMount[] | undefined> {
  const { externalDataUrl, externalDataPath } = options;

  if (!isNodeLikeRuntime()) {
    return [{ data: externalDataUrl, path: externalDataPath }];
  }

  if (/^(?:https?|blob|data):/i.test(externalDataUrl)) {
    return [{ data: externalDataUrl, path: externalDataPath }];
  }

  const { fileURLToPath } = await importNodeModule<typeof import('node:url')>('node:url');
  const localDataPath = /^file:/i.test(externalDataUrl)
    ? fileURLToPath(externalDataUrl)
    : externalDataUrl;
  const promises = await importNodeModule<typeof import('node:fs/promises')>('node:fs/promises');
  return [
    {
      data: await promises.readFile(localDataPath),
      path: externalDataPath,
    },
  ];
}

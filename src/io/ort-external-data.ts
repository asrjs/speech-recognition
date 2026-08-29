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

/**
 * Builds ORT `externalData` session options for colocated `.onnx.data` files.
 *
 * Browser callers pass URL locators. Node callers pass byte buffers for ORT Web
 * WASM and omit the option for native ORT WebGPU when the data file is
 * colocated with the model (native ORT resolves it automatically).
 */
export async function resolveOrtExternalDataMounts(
  options: ResolveOrtExternalDataOptions,
): Promise<readonly OrtExternalDataMount[] | undefined> {
  const { backendId, sessionModelUrl, externalDataUrl, externalDataPath } = options;

  if (!isNodeLikeRuntime()) {
    return [{ data: externalDataUrl, path: externalDataPath }];
  }

  const nodePath = await importNodeModule<typeof import('node:path')>('node:path');
  const colocatedPath = nodePath.join(
    nodePath.dirname(sessionModelUrl),
    nodePath.basename(externalDataPath),
  );
  const fsModule = await importNodeModule<typeof import('node:fs')>('node:fs');
  if (backendId === 'webgpu' && fsModule.existsSync(colocatedPath)) {
    // Native ORT loads colocated external data automatically. Node-hosted ORT
    // Web WASM still needs the bytes mounted explicitly.
    return undefined;
  }

  const promises = await importNodeModule<typeof import('node:fs/promises')>('node:fs/promises');
  return [
    {
      data: await promises.readFile(externalDataUrl),
      path: externalDataPath,
    },
  ];
}

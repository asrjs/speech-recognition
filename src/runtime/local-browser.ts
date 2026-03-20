import { loadBuiltInSpeechModel, type BuiltInSpeechModelHandle } from './builtins.js';
import { createLoadedSpeechModelHandle, type LoadedSpeechModel } from './load.js';
import {
  listBuiltInLocalModelAdapters,
  resolveBuiltInLocalModelAdapter,
  type LoadedLocalSpeechModelSelection,
  type LoadSpeechModelFromLocalEntriesOptions,
  type SpeechModelLocalInspection,
} from './local-adapter-registry.js';
import type {
  BaseTranscriptionOptions,
  ResolvedAssetHandle,
} from '../types/index.js';
import type {
  SpeechModelLocalDirectoryHandleLike,
  SpeechModelLocalEntry,
  SpeechModelLocalFileHandleLike,
} from './local-types.js';

export type {
  LoadedLocalSpeechModelSelection,
  LoadSpeechModelFromLocalEntriesOptions,
  SpeechModelLocalDirectoryHandleLike,
  SpeechModelLocalEntry,
  SpeechModelLocalFileHandleLike,
  SpeechModelLocalInspection,
};

export interface LoadedLocalSpeechModel<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
> extends LoadedSpeechModel<TLoadOptions, TTranscriptionOptions, TNative> {
  readonly selection: LoadedLocalSpeechModelSelection;
}

async function disposeResolvedLocalArtifacts(
  assetHandles: readonly ResolvedAssetHandle[],
): Promise<void> {
  await Promise.all(assetHandles.map(async (handle) => await handle.dispose()));
}

function resolveLocalModelLoadBackend(
  backend: LoadSpeechModelFromLocalEntriesOptions['backend'],
): LoadSpeechModelFromLocalEntriesOptions['backend'] {
  // Direct local-folder loads resolve concrete backend-specific artifacts up front,
  // so the hybrid preference has to collapse to a concrete execution backend.
  return backend === 'webgpu-hybrid' ? 'webgpu' : backend;
}

function getDefaultLocalModelAdapter() {
  const adapter = listBuiltInLocalModelAdapters()[0];
  if (!adapter) {
    throw new Error('No browser local-folder adapters are registered.');
  }
  return adapter;
}

/** Converts flat browser File objects into normalized local entries for built-in speech models. */
export function createSpeechModelLocalEntries(files: readonly File[]): SpeechModelLocalEntry[] {
  return getDefaultLocalModelAdapter().createEntries(files);
}

/** Recursively collects file entries from a browser directory handle for built-in speech models. */
export async function collectSpeechModelLocalEntries(
  dirHandle: SpeechModelLocalDirectoryHandleLike,
  prefix = '',
): Promise<SpeechModelLocalEntry[]> {
  return await getDefaultLocalModelAdapter().collectEntries(dirHandle, prefix);
}

/** Inspects local entries for a built-in model and returns selectable local artifact metadata. */
export function inspectSpeechModelLocalEntries(
  modelId: string,
  entries: readonly SpeechModelLocalEntry[],
): SpeechModelLocalInspection {
  return resolveBuiltInLocalModelAdapter(modelId).adapter.inspectEntries(entries);
}

/**
 * Loads a built-in speech model directly from previously collected local browser entries.
 *
 * This returns the same ready-session handle shape as `loadSpeechModel()`, plus
 * the concrete artifact selection used for local loading.
 */
export async function loadSpeechModelFromLocalEntries(
  options: LoadSpeechModelFromLocalEntriesOptions,
): Promise<LoadedLocalSpeechModel> {
  const resolvedModel = resolveBuiltInLocalModelAdapter(options.modelId);
  const resolved = await resolvedModel.adapter.resolveEntries({
    ...options,
    modelId: resolvedModel.modelId,
  });

  let loaded: BuiltInSpeechModelHandle | null = null;
  let disposed = false;

  try {
    loaded = await loadBuiltInSpeechModel({
      runtime: options.runtime,
      hooks: options.hooks,
      useManifestSources: options.useManifestSources,
      modelId: resolved.modelId,
      preset: resolved.preset,
      backend: resolveLocalModelLoadBackend(options.backend),
      options: resolved.builtInLoadOptions,
      sessionOptions: options.sessionOptions,
      onProgress: options.onProgress,
    });

    const handle = createLoadedSpeechModelHandle(loaded);

    return {
      ...handle,
      selection: resolved.selection,
      async dispose(): Promise<void> {
        if (disposed) {
          return;
        }
        disposed = true;
        try {
          await loaded?.dispose();
        } finally {
          await disposeResolvedLocalArtifacts(resolved.assetHandles);
        }
      },
    };
  } catch (error) {
    await disposeResolvedLocalArtifacts(resolved.assetHandles);
    throw error;
  }
}

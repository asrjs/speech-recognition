import { loadBuiltInSpeechModel, type BuiltInSpeechModelHandle } from './builtins.js';
import { createLoadedSpeechModelHandle, type LoadedSpeechModel } from './load.js';
import {
  listBuiltInLocalModelAdapters,
  resolveBuiltInLocalModelAdapter,
  type LoadedLocalSpeechModelSelection,
  type LoadSpeechModelFromLocalEntriesOptions,
  type SpeechModelLocalInspection,
} from './local-adapter-registry.js';
import { isAssetLoadAbortedError, isDomAbortError } from '../io/abort.js';
import { PipelineAbortedError } from '../pipeline/index.js';
import type {
  AbortSignalLike,
  BaseTranscriptionOptions,
  ResolvedAssetHandle,
  RuntimeProgressEvent,
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
  await Promise.all(assetHandles.map((handle) => handle.dispose()));
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
  signal?: AbortSignalLike | null,
): Promise<SpeechModelLocalEntry[]> {
  return await getDefaultLocalModelAdapter().collectEntries(dirHandle, prefix, signal);
}

/** Inspects local entries for a built-in model and returns selectable local artifact metadata. */
export function inspectSpeechModelLocalEntries(
  modelId: string,
  entries: readonly SpeechModelLocalEntry[],
): SpeechModelLocalInspection {
  return resolveBuiltInLocalModelAdapter(modelId).adapter.inspectEntries(entries);
}

function emitLocalLoadCancelled(
  options: LoadSpeechModelFromLocalEntriesOptions,
  modelId?: string,
): void {
  const event: RuntimeProgressEvent = {
    phase: 'cancelled',
    modelId,
    isComplete: false,
    aborted: true,
    message: 'Load cancelled.',
  };
  options.hooks?.onProgress?.(event);
  options.onProgress?.(event);
}

function throwIfLocalLoadAborted(signal: AbortSignalLike | null | undefined): void {
  if (signal?.aborted) {
    throw new PipelineAbortedError('load');
  }
}

function isLocalLoadAbortError(error: unknown): boolean {
  return (
    error instanceof PipelineAbortedError ||
    isAssetLoadAbortedError(error) ||
    isDomAbortError(error)
  );
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
  if (options.signal?.aborted) {
    emitLocalLoadCancelled(options, options.modelId);
    throw new PipelineAbortedError('load');
  }

  const resolvedModel = resolveBuiltInLocalModelAdapter(options.modelId);
  let resolved: Awaited<ReturnType<typeof resolvedModel.adapter.resolveEntries>> | undefined;
  let loaded: BuiltInSpeechModelHandle | null = null;
  let forwardedToBuiltInLoad = false;
  let disposed = false;

  try {
    throwIfLocalLoadAborted(options.signal);
    resolved = await resolvedModel.adapter.resolveEntries({
      ...options,
      modelId: resolvedModel.modelId,
    });
    throwIfLocalLoadAborted(options.signal);

    forwardedToBuiltInLoad = true;
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
      signal: options.signal,
    });
    throwIfLocalLoadAborted(options.signal);

    const handle = createLoadedSpeechModelHandle(loaded);
    const assetHandles = resolved.assetHandles;

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
          await disposeResolvedLocalArtifacts(assetHandles);
        }
      },
    };
  } catch (error) {
    if (loaded) {
      await loaded.dispose().catch(() => undefined);
    }
    if (resolved) {
      await disposeResolvedLocalArtifacts(resolved.assetHandles);
    }
    if (isLocalLoadAbortError(error)) {
      if (!forwardedToBuiltInLoad || loaded) {
        emitLocalLoadCancelled(options, resolved?.modelId ?? options.modelId);
      }
      throw error instanceof PipelineAbortedError ? error : new PipelineAbortedError('load');
    }
    throw error;
  }
}

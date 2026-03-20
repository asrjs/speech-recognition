import { loadBuiltInSpeechModel, type BuiltInSpeechModelHandle } from './builtins.js';
import { getBuiltInModelDescriptor } from '../presets/descriptors.js';
import {
  collectParakeetLocalEntries,
  createParakeetLocalEntries,
  inspectParakeetLocalEntries,
  resolveParakeetLocalEntries,
  type ParakeetExecutionBackend,
  type ParakeetLocalDirectoryHandleLike,
  type ParakeetLocalEntry,
  type ParakeetLocalFileHandleLike,
  type ResolvedParakeetLocalArtifacts,
} from '../presets/parakeet/compat.js';
import type { QuantizationMode } from './huggingface.js';
import type {
  BaseSessionOptions,
  BaseTranscriptionOptions,
  RuntimeProgressEvent,
} from '../types/index.js';
import type { DefaultSpeechRuntime } from './session.js';
import type { CreateBuiltInSpeechRuntimeOptions } from './builtins.js';
import type {
  NemoTdtModelOptions,
  NemoTdtNativeTranscript,
  NemoTdtTranscriptionOptions,
} from '../models/nemo-tdt/index.js';

export type SpeechModelLocalFileHandleLike = ParakeetLocalFileHandleLike;
export type SpeechModelLocalDirectoryHandleLike = ParakeetLocalDirectoryHandleLike;
export type SpeechModelLocalEntry = ParakeetLocalEntry;

export interface SpeechModelLocalInspection {
  readonly encoderQuantizations: readonly QuantizationMode[];
  readonly decoderQuantizations: readonly QuantizationMode[];
  readonly tokenizerNames: readonly string[];
  readonly preprocessorNames: readonly string[];
}

export interface LoadedLocalSpeechModelSelection {
  readonly encoderName: string;
  readonly decoderName: string;
  readonly tokenizerName: string;
  readonly preprocessorName?: string;
  readonly encoderQuant: QuantizationMode;
  readonly decoderQuant: QuantizationMode;
}

export interface LoadSpeechModelFromLocalEntriesOptions
  extends CreateBuiltInSpeechRuntimeOptions {
  readonly runtime?: DefaultSpeechRuntime;
  readonly modelId: string;
  readonly entries: readonly SpeechModelLocalEntry[];
  readonly backend?: 'wasm' | 'webgpu' | 'webgpu-hybrid' | 'webgpu-strict';
  readonly encoderBackend?: ParakeetExecutionBackend;
  readonly decoderBackend?: ParakeetExecutionBackend;
  readonly encoderQuant?: QuantizationMode;
  readonly decoderQuant?: QuantizationMode;
  readonly tokenizerName?: string;
  readonly preprocessorName?: 'nemo80' | 'nemo128';
  readonly preprocessorBackend?: 'js' | 'onnx';
  readonly verbose?: boolean;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
  readonly sessionOptions?: BaseSessionOptions;
  readonly onProgress?: (event: RuntimeProgressEvent) => void;
}

export interface LoadedLocalSpeechModel<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
> extends BuiltInSpeechModelHandle<TLoadOptions, TTranscriptionOptions, TNative> {
  readonly selection: LoadedLocalSpeechModelSelection;
}

function resolveSupportedLocalBuiltInModel(modelId: string): {
  readonly modelId: string;
  readonly preset: 'parakeet';
} {
  const descriptor = getBuiltInModelDescriptor(modelId);
  if (!descriptor) {
    throw new Error(`Unknown built-in model "${modelId}".`);
  }
  if (!descriptor.loading.supportsLocalSource) {
    throw new Error(`Built-in model "${descriptor.modelId}" does not support local folder loading.`);
  }
  if (descriptor.preset !== 'parakeet') {
    throw new Error(
      `Built-in model "${descriptor.modelId}" does not yet have a browser local-folder loader.`,
    );
  }
  return {
    modelId: descriptor.modelId,
    preset: 'parakeet',
  };
}

function normalizeSpeechModelLocalInspection(
  inspection: ReturnType<typeof inspectParakeetLocalEntries>,
): SpeechModelLocalInspection {
  return {
    encoderQuantizations: inspection.encoderQuantizations,
    decoderQuantizations: inspection.decoderQuantizations,
    tokenizerNames: inspection.tokenizerNames,
    preprocessorNames: inspection.preprocessorNames,
  };
}

function toBuiltInDirectLoadOptions(
  resolved: ResolvedParakeetLocalArtifacts,
): NemoTdtModelOptions {
  return {
    source: {
      kind: 'direct',
      encoderBackend: resolved.config.encoderBackend,
      decoderBackend: resolved.config.decoderBackend,
      artifacts: {
        encoderUrl: resolved.config.encoderUrl,
        decoderUrl: resolved.config.decoderUrl,
        tokenizerUrl: resolved.config.tokenizerUrl,
        preprocessorUrl: resolved.config.preprocessorUrl,
        encoderDataUrl: resolved.config.encoderDataUrl ?? undefined,
        decoderDataUrl: resolved.config.decoderDataUrl ?? undefined,
        encoderFilename: resolved.config.filenames?.encoder,
        decoderFilename: resolved.config.filenames?.decoder,
      },
      preprocessorBackend: resolved.config.preprocessorBackend,
      cpuThreads: resolved.config.cpuThreads,
      enableProfiling: resolved.config.enableProfiling,
    },
  };
}

async function disposeResolvedLocalArtifacts(
  resolved: ResolvedParakeetLocalArtifacts,
): Promise<void> {
  await Promise.all(resolved.assetHandles.map(async (handle) => await handle.dispose()));
}

/** Converts flat browser File objects into normalized local entries for built-in speech models. */
export function createSpeechModelLocalEntries(files: readonly File[]): SpeechModelLocalEntry[] {
  return createParakeetLocalEntries(files);
}

/** Recursively collects file entries from a browser directory handle for built-in speech models. */
export async function collectSpeechModelLocalEntries(
  dirHandle: SpeechModelLocalDirectoryHandleLike,
  prefix = '',
): Promise<SpeechModelLocalEntry[]> {
  return await collectParakeetLocalEntries(dirHandle, prefix);
}

/** Inspects local entries for a built-in model and returns selectable local artifact metadata. */
export function inspectSpeechModelLocalEntries(
  modelId: string,
  entries: readonly SpeechModelLocalEntry[],
): SpeechModelLocalInspection {
  const resolved = resolveSupportedLocalBuiltInModel(modelId);
  switch (resolved.preset) {
    case 'parakeet':
      return normalizeSpeechModelLocalInspection(inspectParakeetLocalEntries(entries));
    default:
      throw new Error(`Unsupported local inspection preset "${resolved.preset}".`);
  }
}

/**
 * Loads a built-in speech model directly from previously collected local browser entries.
 *
 * This returns the same ready-session handle shape as `loadSpeechModel()`, plus
 * the concrete artifact selection used for local loading.
 */
export async function loadSpeechModelFromLocalEntries(
  options: LoadSpeechModelFromLocalEntriesOptions,
): Promise<
  LoadedLocalSpeechModel<NemoTdtModelOptions, NemoTdtTranscriptionOptions, NemoTdtNativeTranscript>
> {
  const resolvedModel = resolveSupportedLocalBuiltInModel(options.modelId);
  const resolved = await resolveParakeetLocalEntries(options.entries, {
    modelId: resolvedModel.modelId,
    encoderBackend: options.encoderBackend,
    decoderBackend: options.decoderBackend,
    encoderQuant: options.encoderQuant,
    decoderQuant: options.decoderQuant,
    tokenizerName: options.tokenizerName,
    preprocessorName: options.preprocessorName,
    preprocessorBackend: options.preprocessorBackend,
    backend: options.backend,
    verbose: options.verbose,
    cpuThreads: options.cpuThreads,
    enableProfiling: options.enableProfiling,
  });

  let loaded:
    | BuiltInSpeechModelHandle<
        NemoTdtModelOptions,
        NemoTdtTranscriptionOptions,
        NemoTdtNativeTranscript
      >
    | null = null;
  let disposed = false;

  try {
    loaded = await loadBuiltInSpeechModel<
      NemoTdtModelOptions,
      NemoTdtTranscriptionOptions,
      NemoTdtNativeTranscript
    >({
      runtime: options.runtime,
      hooks: options.hooks,
      useManifestSources: options.useManifestSources,
      modelId: resolvedModel.modelId,
      preset: resolvedModel.preset,
      backend: options.backend,
      options: toBuiltInDirectLoadOptions(resolved),
      sessionOptions: options.sessionOptions,
      onProgress: options.onProgress,
    });

    return {
      ...loaded,
      selection: resolved.selection,
      async dispose(): Promise<void> {
        if (disposed) {
          return;
        }
        disposed = true;
        try {
          await loaded?.dispose();
        } finally {
          await disposeResolvedLocalArtifacts(resolved);
        }
      },
    };
  } catch (error) {
    await disposeResolvedLocalArtifacts(resolved);
    throw error;
  }
}

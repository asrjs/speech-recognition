import type { QuantizationMode } from './huggingface.js';
import type {
  BaseSessionOptions,
  ResolvedAssetHandle,
  RuntimeProgressEvent,
} from '../types/index.js';
import type { DefaultSpeechRuntime } from './session.js';
import type { CreateBuiltInSpeechRuntimeOptions } from './builtins.js';
import type { SpeechModelLocalDirectoryHandleLike, SpeechModelLocalEntry } from './local-types.js';
import { getBuiltInModelDescriptor } from '../presets/descriptors.js';
import { parakeetBuiltInLocalModelAdapter } from '../presets/parakeet/local-adapter.js';

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

export interface LoadSpeechModelFromLocalEntriesOptions extends CreateBuiltInSpeechRuntimeOptions {
  readonly runtime?: DefaultSpeechRuntime;
  readonly modelId: string;
  readonly entries: readonly SpeechModelLocalEntry[];
  readonly backend?: 'wasm' | 'webgpu' | 'webgpu-hybrid' | 'webgpu-strict';
  readonly encoderBackend?: 'wasm' | 'webgpu';
  readonly decoderBackend?: 'wasm' | 'webgpu';
  readonly encoderQuant?: QuantizationMode;
  readonly decoderQuant?: QuantizationMode;
  readonly tokenizerName?: string;
  readonly preprocessorName?: string;
  readonly preprocessorBackend?: 'js' | 'onnx';
  readonly verbose?: boolean;
  readonly cpuThreads?: number;
  readonly enableProfiling?: boolean;
  readonly sessionOptions?: BaseSessionOptions;
  readonly onProgress?: (event: RuntimeProgressEvent) => void;
}

export interface ResolvedBuiltInLocalArtifacts {
  readonly modelId: string;
  readonly preset: string;
  readonly builtInLoadOptions: unknown;
  readonly assetHandles: readonly ResolvedAssetHandle[];
  readonly selection: LoadedLocalSpeechModelSelection;
}

export interface BuiltInLocalModelAdapter {
  readonly preset: string;
  createEntries(files: readonly File[]): SpeechModelLocalEntry[];
  collectEntries(
    dirHandle: SpeechModelLocalDirectoryHandleLike,
    prefix?: string,
  ): Promise<SpeechModelLocalEntry[]>;
  inspectEntries(entries: readonly SpeechModelLocalEntry[]): SpeechModelLocalInspection;
  resolveEntries(
    options: LoadSpeechModelFromLocalEntriesOptions & { readonly modelId: string },
  ): Promise<ResolvedBuiltInLocalArtifacts>;
}

const BUILT_IN_LOCAL_MODEL_ADAPTERS = new Map<string, BuiltInLocalModelAdapter>([
  [parakeetBuiltInLocalModelAdapter.preset, parakeetBuiltInLocalModelAdapter],
]);

export function listBuiltInLocalModelAdapters(): readonly BuiltInLocalModelAdapter[] {
  return [...BUILT_IN_LOCAL_MODEL_ADAPTERS.values()];
}

export function resolveBuiltInLocalModelAdapter(modelId: string): {
  readonly modelId: string;
  readonly preset: string;
  readonly adapter: BuiltInLocalModelAdapter;
} {
  const descriptor = getBuiltInModelDescriptor(modelId);
  if (!descriptor) {
    throw new Error(`Unknown built-in model "${modelId}".`);
  }
  if (!descriptor.loading.supportsLocalSource) {
    throw new Error(
      `Built-in model "${descriptor.modelId}" does not support local folder loading.`,
    );
  }

  const adapter = BUILT_IN_LOCAL_MODEL_ADAPTERS.get(descriptor.preset);
  if (!adapter) {
    throw new Error(
      `Built-in model "${descriptor.modelId}" does not yet have a browser local-folder loader.`,
    );
  }

  return {
    modelId: descriptor.modelId,
    preset: descriptor.preset,
    adapter,
  };
}

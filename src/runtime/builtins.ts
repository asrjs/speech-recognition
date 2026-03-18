import {
  createWebGlBackend,
  createWebGpuBackend,
  createWebNnBackend,
  createWasmBackend,
} from '../inference/index.js';
import type { AudioInputLike } from '../types/audio.js';
import { createLasrCtcModelFamily } from '../models/lasr-ctc/index.js';
import { createNemoAedModelFamily } from '../models/nemo-aed/index.js';
import { createNemoRnntModelFamily } from '../models/nemo-rnnt/index.js';
import { createNemoTdtModelFamily } from '../models/nemo-tdt/index.js';
import { createWhisperSeq2SeqModelFamily } from '../models/whisper-seq2seq/index.js';
import { createCanaryPresetFactory } from '../presets/canary/factory.js';
import { createMedAsrPresetFactory } from '../presets/medasr/factory.js';
import { createParakeetPresetFactory } from '../presets/parakeet/factory.js';
import { createWhisperPresetFactory } from '../presets/whisper/factory.js';
import type {
  BackendSelectionCriteria,
  BaseSessionOptions,
  BaseTranscriptionOptions,
  ModelClassification,
  SpeechModel,
  SpeechRuntimeHooks,
  SpeechSession,
  TranscriptResponse,
  TranscriptResponseFlavor,
  RuntimeProgressEvent,
} from '../types/index.js';
import { createSpeechRuntime, type DefaultSpeechRuntime } from './session.js';

export interface CreateBuiltInSpeechRuntimeOptions {
  readonly hooks?: SpeechRuntimeHooks;
  readonly useManifestSources?: boolean;
}

export interface LoadBuiltInSpeechModelOptions<
  TLoadOptions = unknown,
> extends CreateBuiltInSpeechRuntimeOptions {
  readonly runtime?: DefaultSpeechRuntime;
  readonly modelId?: string;
  readonly preset?: string;
  readonly family?: string;
  readonly backend?: string;
  readonly classification?: Partial<ModelClassification>;
  readonly options?: TLoadOptions;
  readonly selectionCriteria?: BackendSelectionCriteria;
  readonly sessionOptions?: BaseSessionOptions;
  readonly onProgress?: (event: RuntimeProgressEvent) => void;
}

export interface BuiltInSpeechModelHandle<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
> {
  readonly runtime: DefaultSpeechRuntime;
  readonly model: SpeechModel<TLoadOptions, TTranscriptionOptions, TNative>;
  readonly session: SpeechSession<TTranscriptionOptions, TNative>;
  transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor },
  ): Promise<TranscriptResponse<TNative, TFlavor>>;
  dispose(): Promise<void>;
}

interface ResolvedBuiltInBackendRequest {
  readonly backend?: string;
  readonly selectionCriteria?: BackendSelectionCriteria;
}

function mergeRuntimeHooks(
  hooks: SpeechRuntimeHooks | undefined,
  onProgress: ((event: RuntimeProgressEvent) => void) | undefined,
): SpeechRuntimeHooks | undefined {
  if (!hooks && !onProgress) {
    return undefined;
  }

  return {
    logger: hooks?.logger,
    onProgress(event) {
      hooks?.onProgress?.(event);
      onProgress?.(event);
    },
  };
}

function emitProgress(
  options: Pick<CreateBuiltInSpeechRuntimeOptions, 'hooks'> & {
    readonly onProgress?: (event: RuntimeProgressEvent) => void;
  },
  event: RuntimeProgressEvent,
): void {
  options.hooks?.onProgress?.(event);
  options.onProgress?.(event);
}

function resolveBuiltInBackendRequest(
  backend: string | undefined,
  selectionCriteria: BackendSelectionCriteria | undefined,
): ResolvedBuiltInBackendRequest {
  if (!backend) {
    return {
      backend: undefined,
      selectionCriteria,
    };
  }

  if (backend === 'webgpu-hybrid') {
    return {
      backend: undefined,
      selectionCriteria: {
        ...selectionCriteria,
        preferredBackendIds: ['webgpu', 'wasm'],
      },
    };
  }

  if (backend === 'webgpu-strict') {
    return {
      backend: 'webgpu',
      selectionCriteria,
    };
  }

  return {
    backend,
    selectionCriteria,
  };
}

function resolveBuiltInModelRequest<TLoadOptions>(
  runtime: DefaultSpeechRuntime,
  options: LoadBuiltInSpeechModelOptions<TLoadOptions>,
):
  | {
      readonly family: string;
      readonly preset?: never;
      readonly modelId: string;
    }
  | {
      readonly family?: never;
      readonly preset: string;
      readonly modelId?: string;
    } {
  if (options.family && options.preset) {
    throw new Error('loadBuiltInSpeechModel accepts either `family` or `preset`, not both.');
  }

  if (options.family) {
    if (!options.modelId) {
      throw new Error('loadBuiltInSpeechModel requires `modelId` when loading by `family`.');
    }
    return {
      family: options.family,
      modelId: options.modelId,
    };
  }

  if (options.preset) {
    return {
      preset: options.preset,
      modelId: options.modelId,
    };
  }

  if (!options.modelId) {
    throw new Error(
      'loadBuiltInSpeechModel requires a `modelId` when `family` or `preset` is not provided.',
    );
  }

  const presetMatches = runtime.listPresets().filter((preset) => preset.supports(options.modelId));
  if (presetMatches.length === 1) {
    return {
      preset: presetMatches[0]!.preset,
      modelId: options.modelId,
    };
  }
  if (presetMatches.length > 1) {
    throw new Error(
      `Model "${options.modelId}" matches multiple presets (${presetMatches.map((preset) => preset.preset).join(', ')}). Pass \`preset\` explicitly.`,
    );
  }

  const familyMatches = runtime
    .listModelFamilies()
    .filter((family) => family.supports(options.modelId!));
  if (familyMatches.length === 1) {
    return {
      family: familyMatches[0]!.family,
      modelId: options.modelId,
    };
  }
  if (familyMatches.length > 1) {
    throw new Error(
      `Model "${options.modelId}" matches multiple model families (${familyMatches.map((family) => family.family).join(', ')}). Pass \`family\` explicitly.`,
    );
  }

  throw new Error(
    `Could not infer a built-in preset or model family for "${options.modelId}". Pass \`preset\` or \`family\` explicitly.`,
  );
}

function isFamilyBuiltInModelRequest(
  request:
    | {
        readonly family: string;
        readonly preset?: never;
        readonly modelId: string;
      }
    | {
        readonly family?: never;
        readonly preset: string;
        readonly modelId?: string;
      },
): request is {
  readonly family: string;
  readonly preset?: never;
  readonly modelId: string;
} {
  return 'family' in request;
}

/** Registers the built-in browser and local execution backends on an existing runtime. */
export function registerBuiltInBackends(runtime: DefaultSpeechRuntime): DefaultSpeechRuntime {
  runtime.registerBackend(createWebGpuBackend());
  runtime.registerBackend(createWasmBackend());
  runtime.registerBackend(createWebNnBackend());
  runtime.registerBackend(createWebGlBackend());
  return runtime;
}

/** Registers the built-in technical model families on an existing runtime. */
export function registerBuiltInModelFamilies(runtime: DefaultSpeechRuntime): DefaultSpeechRuntime {
  runtime.registerModelFamily(createNemoAedModelFamily());
  runtime.registerModelFamily(createNemoRnntModelFamily());
  runtime.registerModelFamily(createNemoTdtModelFamily());
  runtime.registerModelFamily(createLasrCtcModelFamily());
  runtime.registerModelFamily(createWhisperSeq2SeqModelFamily());
  return runtime;
}

/** Registers the built-in branded presets on an existing runtime. */
export function registerBuiltInPresets(
  runtime: DefaultSpeechRuntime,
  options: CreateBuiltInSpeechRuntimeOptions = {},
): DefaultSpeechRuntime {
  runtime.registerPreset(
    createCanaryPresetFactory({
      useManifestSource: options.useManifestSources ?? true,
    }),
  );
  runtime.registerPreset(
    createParakeetPresetFactory({
      useManifestSource: options.useManifestSources ?? true,
    }),
  );
  runtime.registerPreset(createMedAsrPresetFactory());
  runtime.registerPreset(createWhisperPresetFactory());
  return runtime;
}

/**
 * Convenience composition helper that wires the library's default backends,
 * model families, and presets into a single runtime instance.
 */
export function createBuiltInSpeechRuntime(
  options: CreateBuiltInSpeechRuntimeOptions = {},
): DefaultSpeechRuntime {
  const runtime = createSpeechRuntime({
    hooks: options.hooks,
  });

  registerBuiltInBackends(runtime);
  registerBuiltInModelFamilies(runtime);
  registerBuiltInPresets(runtime, options);

  return runtime;
}

/**
 * High-level convenience loader for app code.
 *
 * It creates or reuses a built-in runtime, resolves a preset/model-family
 * request, loads the model, creates a ready session, and returns a small
 * session-backed handle with a single `transcribe()` entrypoint.
 */
export async function loadBuiltInSpeechModel<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
>(
  options: LoadBuiltInSpeechModelOptions<TLoadOptions>,
): Promise<BuiltInSpeechModelHandle<TLoadOptions, TTranscriptionOptions, TNative>> {
  const ownsRuntime = !options.runtime;
  const runtime =
    options.runtime ??
    createBuiltInSpeechRuntime({
      hooks: mergeRuntimeHooks(options.hooks, options.onProgress),
      useManifestSources: options.useManifestSources,
    });

  emitProgress(options, {
    phase: 'resolve:start',
    modelId: options.modelId,
    message: 'Resolving built-in model request.',
  });
  const resolved = resolveBuiltInModelRequest(runtime, options);
  emitProgress(options, {
    phase: 'resolve:complete',
    modelId: resolved.modelId,
    message: resolved.preset
      ? `Resolved preset "${resolved.preset}".`
      : `Resolved model family "${resolved.family}".`,
  });

  let model: SpeechModel<TLoadOptions, TTranscriptionOptions, TNative> | undefined;
  const resolvedBackend = resolveBuiltInBackendRequest(options.backend, options.selectionCriteria);

  try {
    if (options.runtime) {
      emitProgress(options, {
        phase: 'model-load:start',
        modelId: resolved.modelId,
        backendId: options.backend,
        message: `Loading ${resolved.modelId ?? resolved.preset ?? resolved.family}.`,
      });
    }

    if (isFamilyBuiltInModelRequest(resolved)) {
      model = (await runtime.loadModel<TLoadOptions, TNative>({
        family: resolved.family,
        modelId: resolved.modelId,
        backend: resolvedBackend.backend,
        classification: options.classification,
        options: options.options,
        selectionCriteria: resolvedBackend.selectionCriteria,
      })) as unknown as SpeechModel<TLoadOptions, TTranscriptionOptions, TNative>;
    } else {
      model = (await runtime.loadModel<TLoadOptions, TNative>({
        preset: resolved.preset,
        modelId: resolved.modelId,
        backend: resolvedBackend.backend,
        classification: options.classification,
        options: options.options,
        selectionCriteria: resolvedBackend.selectionCriteria,
      })) as unknown as SpeechModel<TLoadOptions, TTranscriptionOptions, TNative>;
    }

    if (options.runtime) {
      emitProgress(options, {
        phase: 'model-load:complete',
        modelId: model.info.modelId,
        backendId: model.backend.id,
        message: `Loaded ${model.info.modelId} with ${model.backend.id}.`,
      });
    }

    emitProgress(options, {
      phase: 'session-create:start',
      modelId: model.info.modelId,
      backendId: model.backend.id,
      message: `Creating a ready session for ${model.info.modelId}.`,
    });
    const session = await model.createSession(options.sessionOptions);
    emitProgress(options, {
      phase: 'session-create:complete',
      modelId: model.info.modelId,
      backendId: model.backend.id,
      message: `Ready session created for ${model.info.modelId}.`,
    });
    emitProgress(options, {
      phase: 'ready',
      modelId: model.info.modelId,
      backendId: model.backend.id,
      message: `${model.info.modelId} is ready for transcription.`,
    });
    const loadedModel = model;

    return {
      runtime,
      model: loadedModel,
      session,
      async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
        input: AudioInputLike,
        transcribeOptions?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor },
      ): Promise<TranscriptResponse<TNative, TFlavor>> {
        return session.transcribe(input, transcribeOptions);
      },
      async dispose(): Promise<void> {
        if (ownsRuntime) {
          await runtime.dispose();
          return;
        }
        await loadedModel.dispose();
      },
    };
  } catch (error) {
    if (ownsRuntime) {
      await runtime.dispose();
    } else if (model) {
      await model.dispose();
    }
    throw error;
  }
}

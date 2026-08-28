import {
  createWebGlBackend,
  createWebGpuBackend,
  createWebNnBackend,
  createWasmBackend,
} from '../inference/index.js';
import type { AudioInputLike } from '../types/audio.js';
import {
  createDefaultModelInferenceLimits,
  planWindowedTranscription,
  transcribeWithWindowing,
  withResolvedTranscriptDetail,
  PipelineAbortedError,
} from '../pipeline/index.js';
import { isAssetLoadAbortedError } from '../io/abort.js';
import { createLasrCtcModelFamily } from '../models/lasr-ctc/index.js';
import { createGigaAmCtcModelFamily } from '../models/gigaam-ctc/index.js';
import { createGigaAmRnntModelFamily } from '../models/gigaam-rnnt/index.js';
import { createNemoAedModelFamily } from '../models/nemo-aed/index.js';
import { createNemoRnntModelFamily } from '../models/nemo-rnnt/index.js';
import { createNemoTdtModelFamily } from '../models/nemo-tdt/index.js';
import { createWhisperSeq2SeqModelFamily } from '../models/whisper-seq2seq/index.js';
import { createQwen3AsrModelFamily } from '../models/qwen-asr/index.js';
import { createSenseVoiceModelFamily } from '../models/sensevoice/index.js';
import { createXAsrModelFamily } from '../models/x-asr/index.js';
import { createWav2Vec2ModelFamily } from '../models/wav2vec2/index.js';
import { createCanaryPresetFactory } from '../presets/canary/factory.js';
import { createMedAsrPresetFactory } from '../presets/medasr/factory.js';
import { createParakeetPresetFactory } from '../presets/parakeet/factory.js';
import { createWhisperPresetFactory } from '../presets/whisper/factory.js';
import { createWav2Vec2PresetFactory } from '../presets/wav2vec2/factory.js';
import { getBuiltInModelDescriptor } from '../presets/descriptors.js';
import {
  createExperimentalArtifactMissingError,
  getExperimentalSpeechFamily,
  hasExperimentalArtifactSource,
} from './experimental-families.js';
import type {
  AbortSignalLike,
  BackendSelectionCriteria,
  BaseSessionOptions,
  BaseTranscriptionOptions,
  ModelClassification,
  SpeechModel,
  SpeechRuntimeHooks,
  SpeechSession,
  StreamingSessionOptions,
  StreamingTranscriber,
  TranscriptResponse,
  TranscriptResponseFlavor,
  RuntimeProgressEvent,
} from '../types/index.js';
import { createSpeechRuntime, type DefaultSpeechRuntime } from './session.js';
import { NotImplementedSpeechFeatureError } from './errors.js';
import {
  createActiveOperationLease,
  createGuardedSpeechSession,
  createTrackedStreamingTranscriberFactory,
} from './operation-lease.js';

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
  readonly signal?: AbortSignalLike | null;
}

export interface BuiltInSpeechModelHandle<
  TLoadOptions = unknown,
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
> {
  readonly runtime: DefaultSpeechRuntime;
  readonly model: SpeechModel<TLoadOptions, TTranscriptionOptions, TNative>;
  readonly session: SpeechSession<TTranscriptionOptions, TNative>;
  /** True when the loaded model exposes a stateful streaming transcriber. */
  readonly supportsStreaming?: boolean;
  transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor },
  ): Promise<TranscriptResponse<TNative, TFlavor>>;
  createStreamingTranscriber?(options?: StreamingSessionOptions): Promise<StreamingTranscriber>;
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

function emitLoadCancelled(
  options: Pick<CreateBuiltInSpeechRuntimeOptions, 'hooks'> & {
    readonly onProgress?: (event: RuntimeProgressEvent) => void;
  },
  modelId?: string,
  backendId?: string,
): void {
  emitProgress(options, {
    phase: 'cancelled',
    modelId,
    backendId,
    isComplete: false,
    aborted: true,
    message: 'Load cancelled.',
  });
}

function throwIfLoadAborted(signal: AbortSignalLike | null | undefined): void {
  if (signal?.aborted) {
    throw new PipelineAbortedError('load');
  }
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

  const experimental = getExperimentalSpeechFamily(options.modelId);
  if (experimental && !hasExperimentalArtifactSource(options.options)) {
    throw createExperimentalArtifactMissingError(experimental.family, options.modelId);
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
  runtime.registerModelFamily(createGigaAmCtcModelFamily());
  runtime.registerModelFamily(createGigaAmRnntModelFamily());
  runtime.registerModelFamily(createLasrCtcModelFamily());
  runtime.registerModelFamily(createWhisperSeq2SeqModelFamily());
  runtime.registerModelFamily(createQwen3AsrModelFamily());
  runtime.registerModelFamily(createSenseVoiceModelFamily());
  runtime.registerModelFamily(createXAsrModelFamily());
  runtime.registerModelFamily(createWav2Vec2ModelFamily());
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
  runtime.registerPreset(
    createWav2Vec2PresetFactory({
      useManifestSource: options.useManifestSources ?? false,
    }),
  );
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
  const operationLease = createActiveOperationLease();
  let runtime = options.runtime;
  let model: SpeechModel<TLoadOptions, TTranscriptionOptions, TNative> | undefined;

  try {
    throwIfLoadAborted(options.signal);
    runtime =
      runtime ??
      createBuiltInSpeechRuntime({
        hooks: mergeRuntimeHooks(options.hooks, options.onProgress),
        useManifestSources: options.useManifestSources,
      });
    if (!runtime) {
      throw new Error('Speech runtime is unavailable.');
    }
    const activeRuntime = runtime;

    emitProgress(options, {
      phase: 'resolve:start',
      modelId: options.modelId,
      message: 'Resolving model request.',
    });
    const resolved = resolveBuiltInModelRequest(activeRuntime, options);
  throwIfLoadAborted(options.signal);
  const resolvedDescriptor = resolved.modelId ? getBuiltInModelDescriptor(resolved.modelId) : undefined;
  emitProgress(options, {
    phase: 'resolve:complete',
    modelId: resolved.modelId,
    message: resolved.preset
      ? `Resolved preset "${resolved.preset}".`
      : `Resolved model family "${resolved.family}".`,
  });

  const experimentalFamily = isFamilyBuiltInModelRequest(resolved)
    ? getExperimentalSpeechFamily(resolved.family) ?? getExperimentalSpeechFamily(resolved.modelId)
    : resolved.modelId
      ? getExperimentalSpeechFamily(resolved.modelId)
      : null;
  if (experimentalFamily && !hasExperimentalArtifactSource(options.options)) {
    throw createExperimentalArtifactMissingError(experimentalFamily.family, resolved.modelId);
  }

  const resolvedBackend = resolveBuiltInBackendRequest(options.backend, options.selectionCriteria);

    if (options.runtime) {
      emitProgress(options, {
        phase: 'model-load:start',
        modelId: resolved.modelId,
        backendId: options.backend,
        message: `Loading ${resolved.modelId ?? resolved.preset ?? resolved.family}.`,
      });
    }

    if (isFamilyBuiltInModelRequest(resolved)) {
      model = (await activeRuntime.loadModel<TLoadOptions, TNative>({
        family: resolved.family,
        modelId: resolved.modelId,
        backend: resolvedBackend.backend,
        classification: options.classification,
        options: options.options,
        selectionCriteria: resolvedBackend.selectionCriteria,
        signal: options.signal,
      })) as unknown as SpeechModel<TLoadOptions, TTranscriptionOptions, TNative>;
    } else {
      model = (await activeRuntime.loadModel<TLoadOptions, TNative>({
        preset: resolved.preset,
        modelId: resolved.modelId,
        backend: resolvedBackend.backend,
        classification: options.classification,
        options: options.options,
        selectionCriteria: resolvedBackend.selectionCriteria,
        signal: options.signal,
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

    throwIfLoadAborted(options.signal);
    emitProgress(options, {
      phase: 'session-create:start',
      modelId: model.info.modelId,
      backendId: model.backend.id,
      message: `Creating a ready session for ${model.info.modelId}.`,
    });
    const session = await model.createSession(options.sessionOptions);
    throwIfLoadAborted(options.signal);
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
    const guardedSession = createGuardedSpeechSession(session, operationLease);
    const streamingTranscribers = createTrackedStreamingTranscriberFactory(
      operationLease,
      async (streamingOptions) => {
        if (!loadedModel.createStreamingTranscriber) {
          throw new NotImplementedSpeechFeatureError(
            `Model "${loadedModel.info.modelId}" does not expose a streaming transcription capability.`,
            { feature: 'createStreamingTranscriber', modelId: loadedModel.info.modelId },
          );
        }
        return loadedModel.createStreamingTranscriber(streamingOptions);
      },
    );
    let disposePromise: Promise<void> | null = null;

    return {
      runtime: activeRuntime,
      model: loadedModel,
      session: guardedSession,
      supportsStreaming: typeof loadedModel.createStreamingTranscriber === 'function',
      async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
        input: AudioInputLike,
        transcribeOptions?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor },
      ): Promise<TranscriptResponse<TNative, TFlavor>> {
        const releaseOperation = operationLease.enter();
        if (!releaseOperation) {
          throw new Error('Speech model handle is disposed.');
        }
        try {
          const resolvedOptions = withResolvedTranscriptDetail(transcribeOptions);
          if (resolvedOptions?.responseFlavor === 'native') {
            return await guardedSession.transcribe(input, resolvedOptions);
          }

          const inference =
            loadedModel.info.inference ??
            resolvedDescriptor?.inference ??
            createDefaultModelInferenceLimits({
              family: loadedModel.info.family,
              modelId: loadedModel.info.modelId,
            });
          const decision = planWindowedTranscription(input, resolvedOptions, inference);
          if (!decision.shouldWindow) {
            return await guardedSession.transcribe(input, resolvedOptions);
          }

          const canonical = await transcribeWithWindowing({
            input: decision.audio,
            options: { ...(resolvedOptions ?? {}), responseFlavor: 'canonical' } as TTranscriptionOptions,
            inference,
            transcribeWindow: async (windowInput, windowOptions) =>
              (await guardedSession.transcribe(windowInput, {
                ...windowOptions,
                responseFlavor: 'canonical',
              } as TTranscriptionOptions & { readonly responseFlavor: 'canonical' })) as TranscriptResponse<
                TNative,
                'canonical'
              >,
          });

          if (resolvedOptions?.responseFlavor === 'canonical+native') {
            return { canonical } as TranscriptResponse<TNative, TFlavor>;
          }
          return canonical as TranscriptResponse<TNative, TFlavor>;
        } finally {
          releaseOperation();
        }
      },
      createStreamingTranscriber(streamingOptions = {}): Promise<StreamingTranscriber> {
        return streamingTranscribers.create(streamingOptions);
      },
      async dispose(): Promise<void> {
        if (!disposePromise) {
          disposePromise = (async () => {
            await operationLease.closeAndWait();
            await streamingTranscribers.disposeAll();
            if (ownsRuntime) {
              await activeRuntime.dispose();
              return;
            }
            await loadedModel.dispose();
          })();
        }
        await disposePromise;
      },
    };
  } catch (error) {
    if (ownsRuntime) {
      await runtime?.dispose();
    } else if (model) {
      await model.dispose();
    }
    if (error instanceof PipelineAbortedError || isAssetLoadAbortedError(error)) {
      emitLoadCancelled(
        options,
        options.modelId ?? model?.info.modelId,
        model?.backend.id,
      );
      throw error instanceof PipelineAbortedError ? error : new PipelineAbortedError('load');
    }
    throw error;
  }
}

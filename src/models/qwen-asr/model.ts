import { normalizePcmInput } from '../../audio/index.js';
import type {
  AudioInputLike,
  BaseSessionOptions,
  FamilyModelLoadRequest,
  ModelClassification,
  SpeechModel,
  SpeechModelFactory,
  SpeechSession,
  TranscriptResponse,
  TranscriptResponseFlavor,
} from '../../types/index.js';
import { createModelArchitecture } from '../../types/index.js';
import { createExperimentalArtifactMissingError } from '../../runtime/experimental-families.js';
import { DEFAULT_QWEN3_ASR_CLASSIFICATION, describeQwen3AsrModel, parseQwen3AsrConfig } from './config.js';
import { applyOfficialQwen3AsrGraphDefaults } from './official.js';
import { mapQwen3AsrNativeToCanonical } from './mapping.js';
import { OrtQwen3AsrExecutor } from './executor.js';
import type { Qwen3AsrFeatureProcessor } from './processor.js';
import type { Qwen3AsrTokenizer } from './tokenizer.js';
import type {
  Qwen3AsrExecutor,
  Qwen3AsrModelConfig,
  Qwen3AsrModelDependencies,
  Qwen3AsrModelOptions,
  Qwen3AsrNativeTranscript,
  Qwen3AsrTranscriptionOptions,
} from './types.js';

function classificationContains(
  candidate: Partial<ModelClassification>,
  requested: Partial<ModelClassification>,
): boolean {
  return Object.entries(requested).every(
    ([key, value]) => value === undefined || candidate[key as keyof ModelClassification] === value,
  );
}

function resolveClassification(
  base: Partial<ModelClassification> = {},
  request: Partial<ModelClassification> = {},
): ModelClassification {
  return { ...DEFAULT_QWEN3_ASR_CLASSIFICATION, ...base, ...request };
}

function createExecutor(
  modelId: string,
  config: Qwen3AsrModelConfig,
  backendId: string,
  loadOptions: Qwen3AsrModelOptions | undefined,
  dependencies: Qwen3AsrModelDependencies,
): Qwen3AsrExecutor | undefined {
  if (dependencies.executor) return dependencies.executor;
  if (!loadOptions?.source) return undefined;
  return new OrtQwen3AsrExecutor(modelId, config, backendId, loadOptions, {
    assetProvider: dependencies.assetProvider,
    runtimeHooks: dependencies.runtimeHooks,
    tokenizer: dependencies.tokenizer as Qwen3AsrTokenizer | undefined,
    featureProcessor: dependencies.featureProcessor as Qwen3AsrFeatureProcessor | undefined,
    signal: dependencies.signal,
  });
}

export class Qwen3AsrSpeechSession
  implements SpeechSession<Qwen3AsrTranscriptionOptions, Qwen3AsrNativeTranscript> {
  private readonly executor?: Qwen3AsrExecutor;
  private disposed = false;

  constructor(
    private readonly modelId: string,
    private readonly classification: ModelClassification,
    private readonly config: Qwen3AsrModelConfig,
    private readonly backendId: string,
    loadOptions: Qwen3AsrModelOptions | undefined,
    dependencies: Qwen3AsrModelDependencies = {},
    private readonly onDispose?: () => void,
  ) {
    this.executor = createExecutor(modelId, config, backendId, loadOptions, dependencies);
  }

  async initialize(): Promise<void> {
    await this.executor?.ready?.();
  }

  async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options: Qwen3AsrTranscriptionOptions & { readonly responseFlavor?: TFlavor } = {},
  ): Promise<TranscriptResponse<Qwen3AsrNativeTranscript, TFlavor>> {
    if (!this.executor) {
      throw createExperimentalArtifactMissingError('qwen-asr', this.modelId);
    }
    const audio = normalizePcmInput(input).toMono();
    const nativeTranscript = await this.executor.transcribe(audio, options, {
      modelId: this.modelId,
      classification: this.classification,
      config: this.config,
    });
    const canonical = mapQwen3AsrNativeToCanonical(nativeTranscript, this.classification, {
      detailLevel: options.detail,
      backendId: this.backendId,
      modelId: this.modelId,
      sampleRate: audio.sampleRate,
      durationSeconds: audio.durationSeconds,
      language: nativeTranscript.language,
      metrics: nativeTranscript.metrics,
    });
    const responseFlavor = options.responseFlavor ?? 'canonical';
    if (responseFlavor === 'native') {
      return nativeTranscript as TranscriptResponse<Qwen3AsrNativeTranscript, TFlavor>;
    }
    if (responseFlavor === 'canonical+native') {
      return { canonical, native: nativeTranscript } as TranscriptResponse<
        Qwen3AsrNativeTranscript,
        TFlavor
      >;
    }
    return canonical as TranscriptResponse<Qwen3AsrNativeTranscript, TFlavor>;
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await this.executor?.dispose();
    this.onDispose?.();
  }
}

export class Qwen3AsrSpeechModel
  implements SpeechModel<Qwen3AsrModelOptions, Qwen3AsrTranscriptionOptions, Qwen3AsrNativeTranscript> {
  readonly loadOptions?: Qwen3AsrModelOptions;
  readonly info;
  private readonly sessions = new Set<Qwen3AsrSpeechSession>();
  private disposed = false;

  constructor(
    readonly backend: SpeechModel<
      Qwen3AsrModelOptions,
      Qwen3AsrTranscriptionOptions,
      Qwen3AsrNativeTranscript
    >['backend'],
    readonly family: string,
    readonly modelId: string,
    readonly classification: ModelClassification,
    readonly config: Qwen3AsrModelConfig,
    readonly resolvedPreset: string | undefined,
    loadOptions: Qwen3AsrModelOptions | undefined,
    private readonly dependencies: Qwen3AsrModelDependencies,
    describeModel: (
      modelId: string,
      classification: ModelClassification,
      config: Qwen3AsrModelConfig,
    ) => string,
  ) {
    this.loadOptions = loadOptions;
    this.info = {
      family,
      modelId,
      classification,
      preset: resolvedPreset,
      architecture: createModelArchitecture({
        processor: {
          layer: 'processor',
          module: 'audio',
          implementation: config.processorArchitecture,
          shared: true,
          notes: ['128-bin Qwen/Whisper-compatible log-mel frontend with graph-aligned padding.'],
        },
        encoder: {
          layer: 'encoder',
          module: 'qwen-asr',
          implementation: config.encoderArchitecture,
          notes: ['FP16 audio encoder graph with an audio-token mask.'],
        },
        decoder: {
          layer: 'decoder',
          module: 'qwen-asr',
          implementation: config.decoderArchitecture,
          notes: ['Merged prefill/one-token graph with explicit KV cache for the reference artifact.'],
        },
        decoding: {
          layer: 'decoding',
          module: 'inference',
          implementation: 'greedy-autoregressive',
          shared: true,
          notes: ['Beam batching is not exposed until the batch-1 graph contract is replaced.'],
        },
        tokenizer: {
          layer: 'tokenizer',
          module: 'inference',
          implementation: config.tokenizer.kind,
          shared: false,
          notes: ['Qwen GPT/ByteLevel BPE with audio/chat special tokens.'],
        },
      }),
      description: describeModel(modelId, classification, config),
      nativeOutputName: 'Qwen3AsrNativeTranscript',
      inference: {
        sampleRate: config.sampleRate,
        maxInputDurationSec: config.maxInputDurationSec,
        recommendedWindowDurationSec: config.maxInputDurationSec,
        supportsWordTimestamps: false,
        supportsTokenTimestamps: false,
        supportsSegmentTimestamps: false,
        supportsConfidence: false,
        defaultSegmentationStrategy: 'none' as const,
        defaultMergeStrategy: 'concat' as const,
      },
    };
  }

  async createSession(_options: BaseSessionOptions = {}): Promise<Qwen3AsrSpeechSession> {
    const session = new Qwen3AsrSpeechSession(
      this.modelId,
      this.classification,
      this.config,
      this.backend.id,
      this.loadOptions,
      this.dependencies,
      () => this.sessions.delete(session),
    );
    this.sessions.add(session);
    await session.initialize();
    return session;
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    const sessions = [...this.sessions];
    this.sessions.clear();
    await Promise.all(sessions.map(async (session) => session.dispose()));
  }
}

export interface CreateQwen3AsrModelFamilyOptions {
  readonly dependencies?: Qwen3AsrModelDependencies;
  readonly family?: string;
  readonly classification?: Partial<ModelClassification>;
  readonly supportsModel?: (
    modelId: string,
    classification?: Partial<ModelClassification>,
  ) => boolean;
  readonly resolveConfig?: (
    modelId: string,
    request: FamilyModelLoadRequest<Qwen3AsrModelOptions>,
  ) => Qwen3AsrModelConfig;
  readonly describeModel?: (
    modelId: string,
    classification: ModelClassification,
    config: Qwen3AsrModelConfig,
  ) => string;
}

export function createQwen3AsrModelFamily(
  options: CreateQwen3AsrModelFamilyOptions = {},
): SpeechModelFactory<
  Qwen3AsrModelOptions,
  Qwen3AsrTranscriptionOptions,
  Qwen3AsrNativeTranscript
> {
  const family = options.family ?? 'qwen-asr';
  const factoryClassification = resolveClassification(options.classification);
  return {
    family,
    classification: factoryClassification,
    supports(modelId: string): boolean {
      if (options.supportsModel) return options.supportsModel(modelId);
      return /qwen(?:3)?[-_]?asr/i.test(modelId);
    },
    matchesClassification(classification: Partial<ModelClassification>): boolean {
      if (options.supportsModel) return options.supportsModel('', classification);
      return classificationContains(factoryClassification, classification);
    },
    async createModel(request, context): Promise<Qwen3AsrSpeechModel> {
      const classification = resolveClassification(factoryClassification, request.classification);
      const config = applyOfficialQwen3AsrGraphDefaults(
        options.resolveConfig
          ? options.resolveConfig(request.modelId, request)
          : parseQwen3AsrConfig(request.modelId, request.options?.config),
        request.options?.source,
      );
      const dependencies: Qwen3AsrModelDependencies = {
        ...(options.dependencies ?? {}),
        assetProvider: options.dependencies?.assetProvider ?? context.assetProvider,
        runtimeHooks: options.dependencies?.runtimeHooks ?? context.hooks,
        signal: options.dependencies?.signal ?? context.signal,
      };
      context.hooks.logger?.info?.('Creating Qwen3-ASR model', {
        family,
        modelId: request.modelId,
        backendId: context.backend.id,
        artifactSource: request.options?.source?.kind ??
          (dependencies.executor ? 'injected' : 'required'),
      });
      return new Qwen3AsrSpeechModel(
        context.backend,
        family,
        request.modelId,
        classification,
        config,
        request.resolvedPreset,
        request.options,
        dependencies,
        options.describeModel ?? describeQwen3AsrModel,
      );
    },
  };
}

import { normalizePcmInput } from '../../audio/index.js';
import {
  CTC_GREEDY_DECODING,
  CTC_HEAD_DECODER,
  WAV2VEC2_CONFORMER_ENCODER,
} from '../../inference/index.js';
import type {
  AudioInputLike,
  BaseSessionOptions,
  FamilyModelLoadRequest,
  ModelClassification,
  ModelInferenceLimits,
  SpeechModel,
  SpeechModelFactory,
  SpeechModelFactoryContext,
  SpeechSession,
  TranscriptResponse,
  TranscriptResponseFlavor,
} from '../../types/index.js';
import { createModelArchitecture } from '../../types/index.js';
import {
  DEFAULT_WAV2VEC2_CLASSIFICATION,
  DEFAULT_WAV2VEC2_CONFIG,
  describeWav2Vec2Model,
} from './config.js';
import { OrtWav2Vec2Executor } from './executor.js';
import { mapWav2Vec2NativeToCanonical } from './mapping.js';
import type {
  Wav2Vec2Executor,
  Wav2Vec2ModelConfig,
  Wav2Vec2ModelDependencies,
  Wav2Vec2ModelOptions,
  Wav2Vec2NativeToken,
  Wav2Vec2NativeTranscript,
  Wav2Vec2NativeWord,
  Wav2Vec2TranscriptionOptions,
} from './types.js';

function classificationContains(
  candidate: Partial<ModelClassification>,
  requested: Partial<ModelClassification>,
): boolean {
  return Object.entries(requested).every(([key, value]) => {
    if (value === undefined) {
      return true;
    }
    return candidate[key as keyof ModelClassification] === value;
  });
}

function resolveClassification(
  base: Partial<ModelClassification> = {},
  request: Partial<ModelClassification> = {},
): ModelClassification {
  return {
    ...DEFAULT_WAV2VEC2_CLASSIFICATION,
    family: 'wav2vec2',
    ...base,
    ...request,
  };
}

function resolveConfig(config: Partial<Wav2Vec2ModelConfig> | undefined): Wav2Vec2ModelConfig {
  return {
    ...DEFAULT_WAV2VEC2_CONFIG,
    ...(config ?? {}),
    tokenizer: {
      ...DEFAULT_WAV2VEC2_CONFIG.tokenizer,
      ...(config?.tokenizer ?? {}),
    },
  };
}

function createWav2Vec2InferenceLimits(config: Wav2Vec2ModelConfig): ModelInferenceLimits {
  return {
    sampleRate: config.sampleRate,
    maxInputDurationSec: 60,
    recommendedWindowDurationSec: 30,
    minWindowDurationSec: 5,
    maxWindowDurationSec: 60,
    autoWindowThresholdSec: 60,
    defaultOverlapSec: 5,
    supportsWordTimestamps: true,
    supportsTokenTimestamps: true,
    supportsSegmentTimestamps: true,
    supportsConfidence: true,
    defaultSegmentationStrategy: 'ctc-frame',
    defaultMergeStrategy: 'ctc-collapse',
  };
}

function createExecutor(
  modelId: string,
  classification: ModelClassification,
  config: Wav2Vec2ModelConfig,
  backendId: string,
  loadOptions: Wav2Vec2ModelOptions | undefined,
  dependencies: Wav2Vec2ModelDependencies,
): Wav2Vec2Executor | undefined {
  if (dependencies.executor) {
    return dependencies.executor;
  }
  if (!loadOptions?.source) {
    return undefined;
  }
  return new OrtWav2Vec2Executor(modelId, classification, config, backendId, loadOptions, {
    assetProvider: dependencies.assetProvider,
    runtimeHooks: dependencies.runtimeHooks,
  });
}

function buildStubWords(durationSeconds: number): Wav2Vec2NativeWord[] {
  const lexemes = ['wav2vec2', 'ctc', 'scaffold'];
  const span = Math.max(durationSeconds, 0.3) / lexemes.length;

  return lexemes.map((text, index) => ({
    index,
    text,
    startTime: Number((index * span).toFixed(3)),
    endTime: Number(((index + 1) * span).toFixed(3)),
    confidence: 0.91,
    tokenIndices: [index],
  }));
}

function buildStubTokens(
  words: readonly Wav2Vec2NativeWord[],
  options: Wav2Vec2TranscriptionOptions,
): Wav2Vec2NativeToken[] {
  return words.map((word, index) => ({
    index,
    id: options.returnTokenIds ? index + 1 : undefined,
    text: word.text,
    startTime: word.startTime,
    endTime: word.endTime,
    confidence: options.returnConfidence ? word.confidence : undefined,
  }));
}

class Wav2Vec2SpeechSession implements SpeechSession<
  Wav2Vec2TranscriptionOptions,
  Wav2Vec2NativeTranscript
> {
  private readonly executor?: Wav2Vec2Executor;
  private disposed = false;

  constructor(
    private readonly modelId: string,
    private readonly classification: ModelClassification,
    private readonly config: Wav2Vec2ModelConfig,
    private readonly backendId: string,
    loadOptions: Wav2Vec2ModelOptions | undefined,
    dependencies: Wav2Vec2ModelDependencies = {},
    private readonly onDispose?: () => void,
  ) {
    this.executor = createExecutor(
      modelId,
      classification,
      config,
      backendId,
      loadOptions,
      dependencies,
    );
  }

  async initialize(): Promise<void> {
    await this.executor?.ready?.();
  }

  async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options: Wav2Vec2TranscriptionOptions & { readonly responseFlavor?: TFlavor } = {},
  ): Promise<TranscriptResponse<Wav2Vec2NativeTranscript, TFlavor>> {
    const audio = normalizePcmInput(input).toMono();
    const nativeTranscript = this.executor
      ? await this.executor.transcribe(audio, options)
      : this.transcribeWithStub(audio.durationSeconds, options);

    const canonical = mapWav2Vec2NativeToCanonical(nativeTranscript, this.classification, {
      detailLevel: options.detail,
      backendId: this.backendId,
      modelId: this.modelId,
      language: nativeTranscript.language ?? this.config.languages[0],
      sampleRate: audio.sampleRate,
      durationSeconds: audio.durationSeconds,
    });
    const responseFlavor = options.responseFlavor ?? 'canonical';

    if (responseFlavor === 'native') {
      return nativeTranscript as TranscriptResponse<Wav2Vec2NativeTranscript, TFlavor>;
    }
    if (responseFlavor === 'canonical+native') {
      return {
        canonical,
        native: nativeTranscript,
      } as TranscriptResponse<Wav2Vec2NativeTranscript, TFlavor>;
    }

    return canonical as TranscriptResponse<Wav2Vec2NativeTranscript, TFlavor>;
  }

  private transcribeWithStub(
    durationSeconds: number,
    options: Wav2Vec2TranscriptionOptions,
  ): Wav2Vec2NativeTranscript {
    const words = buildStubWords(durationSeconds);
    const tokens = buildStubTokens(words, options);
    const utteranceText = words.map((word) => word.text).join(' ');

    return {
      utteranceText,
      isFinal: true,
      language: this.config.languages[0],
      segments: [
        {
          index: 0,
          text: utteranceText,
          startTime: words[0]?.startTime ?? 0,
          endTime: words[words.length - 1]?.endTime ?? durationSeconds,
          confidence: 0.91,
        },
      ],
      words,
      tokens,
      warnings: [
        {
          code: 'wav2vec2.stubbed-decoder',
          message:
            'Wav2Vec2 CTC execution is scaffolded. Provide ONNX model artifacts to activate the real raw-waveform path.',
        },
      ],
    };
  }

  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    await this.executor?.dispose();
    this.onDispose?.();
  }
}

class Wav2Vec2SpeechModel implements SpeechModel<
  Wav2Vec2ModelOptions,
  Wav2Vec2TranscriptionOptions,
  Wav2Vec2NativeTranscript
> {
  readonly info;
  readonly loadOptions?: Wav2Vec2ModelOptions;
  private readonly sessions = new Set<Wav2Vec2SpeechSession>();
  private disposed = false;

  constructor(
    readonly backend: SpeechModel<
      Wav2Vec2ModelOptions,
      Wav2Vec2TranscriptionOptions,
      Wav2Vec2NativeTranscript
    >['backend'],
    readonly family: string,
    readonly modelId: string,
    readonly classification: ModelClassification,
    readonly config: Wav2Vec2ModelConfig,
    readonly resolvedPreset: string | undefined,
    loadOptions: Wav2Vec2ModelOptions | undefined,
    private readonly dependencies: Wav2Vec2ModelDependencies,
    describeModel: (
      modelId: string,
      classification: ModelClassification,
      config: Wav2Vec2ModelConfig,
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
          implementation: classification.processor ?? config.processorArchitecture,
          shared: true,
          notes: ['Raw waveform input; convolutional feature extractor is inside the ONNX graph.'],
        },
        encoder: {
          layer: 'encoder',
          module: WAV2VEC2_CONFORMER_ENCODER.sharedModule,
          implementation: config.encoderArchitecture,
          shared: true,
          notes: [`Output stride ${config.outputStride} samples.`],
        },
        decoder: {
          layer: 'decoder',
          module: CTC_HEAD_DECODER.sharedModule,
          implementation: CTC_HEAD_DECODER.kind,
          shared: true,
        },
        decoding: {
          layer: 'decoding',
          module: 'inference',
          implementation: CTC_GREEDY_DECODING.strategy,
          shared: true,
          notes: CTC_GREEDY_DECODING.notes,
        },
        tokenizer: {
          layer: 'tokenizer',
          module: 'inference',
          implementation: config.tokenizer.kind,
          shared: true,
        },
      }),
      description: describeModel(modelId, classification, config),
      nativeOutputName: 'Wav2Vec2NativeTranscript',
      inference: createWav2Vec2InferenceLimits(config),
    };
  }

  async createSession(
    _options: BaseSessionOptions = {},
  ): Promise<SpeechSession<Wav2Vec2TranscriptionOptions, Wav2Vec2NativeTranscript>> {
    const session = new Wav2Vec2SpeechSession(
      this.modelId,
      this.classification,
      this.config,
      this.backend.id,
      this.loadOptions,
      this.dependencies,
      () => {
        this.sessions.delete(session);
      },
    );
    this.sessions.add(session);
    await session.initialize();
    return session;
  }

  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }
    this.disposed = true;

    const sessions = [...this.sessions];
    this.sessions.clear();

    await Promise.all(
      sessions.map(async (session) => {
        await session.dispose();
      }),
    );
  }
}

export interface CreateWav2Vec2ModelFamilyOptions {
  readonly dependencies?: Wav2Vec2ModelDependencies;
  readonly family?: string;
  readonly classification?: Partial<ModelClassification>;
  readonly supportsModel?: (
    modelId: string,
    classification?: Partial<ModelClassification>,
  ) => boolean;
  readonly resolveConfig?: (
    modelId: string,
    request: FamilyModelLoadRequest<Wav2Vec2ModelOptions>,
  ) => Wav2Vec2ModelConfig;
  readonly describeModel?: (
    modelId: string,
    classification: ModelClassification,
    config: Wav2Vec2ModelConfig,
  ) => string;
}

export function createWav2Vec2ModelFamily(
  options: CreateWav2Vec2ModelFamilyOptions = {},
): SpeechModelFactory<Wav2Vec2ModelOptions, Wav2Vec2TranscriptionOptions, Wav2Vec2NativeTranscript> {
  const family = options.family ?? 'wav2vec2';
  const factoryClassification = resolveClassification(options.classification);

  return {
    family,
    classification: factoryClassification,
    supports(modelId: string): boolean {
      if (options.supportsModel) {
        return options.supportsModel(modelId);
      }
      const normalizedModelId = modelId.toLowerCase();
      return normalizedModelId.includes('wav2vec2') || normalizedModelId.includes('wav2vec');
    },
    matchesClassification(classification: Partial<ModelClassification>): boolean {
      if (options.supportsModel) {
        return options.supportsModel('', classification);
      }
      return classificationContains(factoryClassification, classification);
    },
    async createModel(request, context: SpeechModelFactoryContext): Promise<Wav2Vec2SpeechModel> {
      const classification = resolveClassification(factoryClassification, request.classification);
      const config = options.resolveConfig
        ? options.resolveConfig(request.modelId, request)
        : resolveConfig(request.options?.config);
      const dependencies: Wav2Vec2ModelDependencies = {
        ...(options.dependencies ?? {}),
        assetProvider: options.dependencies?.assetProvider ?? context.assetProvider,
        runtimeHooks: options.dependencies?.runtimeHooks ?? context.hooks,
      };

      context.hooks.logger?.info?.('Creating Wav2Vec2 CTC model', {
        family,
        modelId: request.modelId,
        backendId: context.backend.id,
        artifactSource: request.options?.source?.kind ?? 'stub',
      });

      return new Wav2Vec2SpeechModel(
        context.backend,
        family,
        request.modelId,
        classification,
        config,
        request.resolvedPreset,
        request.options,
        dependencies,
        options.describeModel ??
          ((_modelId: string, _classification: ModelClassification, modelConfig: Wav2Vec2ModelConfig) =>
            describeWav2Vec2Model(modelConfig)),
      );
    },
  };
}

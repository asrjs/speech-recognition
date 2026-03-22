import { normalizePcmInput } from '../../audio/index.js';
import {
  CONFORMER_ENCODER,
  CTC_GREEDY_DECODING,
  CTC_HEAD_DECODER,
} from '../../inference/index.js';
import { StubTextTokenizer } from '../../tokenizers/index.js';
import type {
  AudioInputLike,
  BaseSessionOptions,
  FamilyModelLoadRequest,
  ModelClassification,
  SpeechModel,
  SpeechModelFactory,
  SpeechModelFactoryContext,
  SpeechSession,
  TranscriptResponse,
  TranscriptResponseFlavor,
  TranscriptWord,
} from '../../types/index.js';
import { createModelArchitecture } from '../../types/index.js';
import { DEFAULT_LASR_CTC_CLASSIFICATION, describeLasrCtcModel, parseLasrCtcConfig } from './config.js';
import { OrtLasrCtcExecutor } from './executor.js';
import { mapLasrCtcNativeToCanonical } from './mapping.js';
import type {
  LasrCtcExecutor,
  LasrCtcModelConfig,
  LasrCtcModelDependencies,
  LasrCtcModelOptions,
  LasrCtcNativeToken,
  LasrCtcNativeTranscript,
  LasrCtcTranscriptionOptions,
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
    ...DEFAULT_LASR_CTC_CLASSIFICATION,
    ...base,
    ...request,
  };
}

function buildStubWords(durationSeconds: number): TranscriptWord[] {
  const lexemes = ['Medical', 'CTC', 'scaffold'];
  const span = Math.max(durationSeconds, 0.3) / lexemes.length;

  return lexemes.map((text, index) => ({
    index,
    text,
    startTime: Number((index * span).toFixed(3)),
    endTime: Number(((index + 1) * span).toFixed(3)),
    confidence: 0.91,
  }));
}

function buildStubTokens(
  words: readonly TranscriptWord[],
  options: LasrCtcTranscriptionOptions,
): LasrCtcNativeToken[] {
  return words.map((word, index) => ({
    index,
    id: options.returnTokenIds ? index + 1 : undefined,
    text: word.text,
    startTime: word.startTime,
    endTime: word.endTime,
    confidence: word.confidence,
    logitIndex: options.returnLogitIndices ? index : undefined,
  }));
}

function createExecutor(
  modelId: string,
  classification: ModelClassification,
  config: LasrCtcModelConfig,
  backendId: string,
  loadOptions: LasrCtcModelOptions | undefined,
  dependencies: LasrCtcModelDependencies,
): LasrCtcExecutor | undefined {
  if (dependencies.executor) {
    return dependencies.executor;
  }

  if (!loadOptions?.source) {
    return undefined;
  }

  return new OrtLasrCtcExecutor(modelId, classification, config, backendId, loadOptions, {
    assetProvider: dependencies.assetProvider,
    runtimeHooks: dependencies.runtimeHooks,
    preprocessor: dependencies.preprocessor,
  });
}

class LasrCtcSpeechSession implements SpeechSession<
  LasrCtcTranscriptionOptions,
  LasrCtcNativeTranscript
> {
  private readonly tokenizer;
  private readonly executor?: LasrCtcExecutor;
  private disposed = false;

  constructor(
    private readonly modelId: string,
    private readonly classification: ModelClassification,
    private readonly config: LasrCtcModelConfig,
    private readonly backendId: string,
    private readonly loadOptions: LasrCtcModelOptions | undefined,
    dependencies: LasrCtcModelDependencies = {},
    private readonly onDispose?: () => void,
  ) {
    this.tokenizer = dependencies.tokenizer ?? new StubTextTokenizer(config.tokenizer.kind, 'lasr');
    this.executor = createExecutor(
      modelId,
      classification,
      config,
      backendId,
      this.loadOptions,
      dependencies,
    );
  }

  async initialize(): Promise<void> {
    await this.executor?.ready?.();
  }

  async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options: LasrCtcTranscriptionOptions & { readonly responseFlavor?: TFlavor } = {},
  ): Promise<TranscriptResponse<LasrCtcNativeTranscript, TFlavor>> {
    const audio = normalizePcmInput(input).toMono();
    const nativeTranscript = this.executor
      ? await this.executor.transcribe(audio, options, {
          modelId: this.modelId,
          classification: this.classification,
          config: this.config,
          tokenizer: this.tokenizer,
        })
      : this.decodeWithStub(audio.durationSeconds, options);

    const canonical = mapLasrCtcNativeToCanonical(nativeTranscript, this.classification, {
      detailLevel: options.detail,
      backendId: this.backendId,
      modelId: this.modelId,
      language: this.config.languages[0],
      sampleRate: audio.sampleRate,
      durationSeconds: audio.durationSeconds,
      metrics: nativeTranscript.metrics,
    });
    const responseFlavor = options.responseFlavor ?? 'canonical';

    if (responseFlavor === 'native') {
      return nativeTranscript as TranscriptResponse<LasrCtcNativeTranscript, TFlavor>;
    }
    if (responseFlavor === 'canonical+native') {
      return {
        canonical,
        native: nativeTranscript,
      } as TranscriptResponse<LasrCtcNativeTranscript, TFlavor>;
    }

    return canonical as TranscriptResponse<LasrCtcNativeTranscript, TFlavor>;
  }

  private decodeWithStub(
    durationSeconds: number,
    options: LasrCtcTranscriptionOptions,
  ): LasrCtcNativeTranscript {
    const words = buildStubWords(durationSeconds);
    const tokens = buildStubTokens(words, options);

    return {
      utteranceText: words.map((word) => word.text).join(' '),
      isFinal: true,
      words,
      tokens,
      confidence: {
        utterance: 0.91,
        wordAverage: 0.91,
        tokenAverage: 0.91,
      },
      warnings: [
        {
          code: 'lasr-ctc.stubbed-decoder',
          message:
            'LASR CTC model execution is scaffolded. Integrate processor, encoder, and CTC logits execution to replace the stub.',
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

class LasrCtcSpeechModel implements SpeechModel<
  LasrCtcModelOptions,
  LasrCtcTranscriptionOptions,
  LasrCtcNativeTranscript
> {
  readonly info;
  readonly loadOptions?: LasrCtcModelOptions;
  private readonly sessions = new Set<LasrCtcSpeechSession>();
  private disposed = false;

  constructor(
    readonly backend: SpeechModel<
      LasrCtcModelOptions,
      LasrCtcTranscriptionOptions,
      LasrCtcNativeTranscript
    >['backend'],
    readonly family: string,
    readonly modelId: string,
    readonly classification: ModelClassification,
    readonly config: LasrCtcModelConfig,
    readonly resolvedPreset: string | undefined,
    loadOptions: LasrCtcModelOptions | undefined,
    private readonly dependencies: LasrCtcModelDependencies,
    describeModel: (
      modelId: string,
      classification: ModelClassification,
      config: LasrCtcModelConfig,
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
        },
        encoder: {
          layer: 'encoder',
          module: CONFORMER_ENCODER.sharedModule,
          implementation: config.encoderArchitecture,
          shared: true,
          notes: [`Output stride ${config.rawStride}.`],
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
      nativeOutputName: 'LasrCtcNativeTranscript',
    };
  }

  async createSession(
    _options: BaseSessionOptions = {},
  ): Promise<SpeechSession<LasrCtcTranscriptionOptions, LasrCtcNativeTranscript>> {
    const session = new LasrCtcSpeechSession(
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

export interface CreateLasrCtcModelFamilyOptions {
  readonly dependencies?: LasrCtcModelDependencies;
  readonly family?: string;
  readonly classification?: Partial<ModelClassification>;
  readonly supportsModel?: (
    modelId: string,
    classification?: Partial<ModelClassification>,
  ) => boolean;
  readonly resolveConfig?: (
    modelId: string,
    request: FamilyModelLoadRequest<LasrCtcModelOptions>,
  ) => LasrCtcModelConfig;
  readonly describeModel?: (
    modelId: string,
    classification: ModelClassification,
    config: LasrCtcModelConfig,
  ) => string;
}

export function createLasrCtcModelFamily(
  options: CreateLasrCtcModelFamilyOptions = {},
): SpeechModelFactory<LasrCtcModelOptions, LasrCtcTranscriptionOptions, LasrCtcNativeTranscript> {
  const family = options.family ?? 'lasr-ctc';
  const factoryClassification = resolveClassification(options.classification);

  return {
    family,
    classification: factoryClassification,
    supports(modelId: string): boolean {
      if (options.supportsModel) {
        return options.supportsModel(modelId);
      }

      const normalizedModelId = modelId.toLowerCase();
      return (
        normalizedModelId.includes('medasr') ||
        normalizedModelId.includes('lasr') ||
        normalizedModelId.includes('conformer') ||
        normalizedModelId.includes('ctc')
      );
    },
    matchesClassification(classification: Partial<ModelClassification>): boolean {
      if (options.supportsModel) {
        return options.supportsModel('', classification);
      }
      return classificationContains(factoryClassification, classification);
    },
    async createModel(request, context: SpeechModelFactoryContext): Promise<LasrCtcSpeechModel> {
      const classification = resolveClassification(factoryClassification, request.classification);
      const config = options.resolveConfig
        ? options.resolveConfig(request.modelId, request)
        : parseLasrCtcConfig(request.modelId, request.options?.config);
      const dependencies: LasrCtcModelDependencies = {
        ...(options.dependencies ?? {}),
        assetProvider: options.dependencies?.assetProvider ?? context.assetProvider,
        runtimeHooks: options.dependencies?.runtimeHooks ?? context.hooks,
      };

      context.hooks.logger?.info?.('Creating LASR CTC model', {
        family,
        modelId: request.modelId,
        backendId: context.backend.id,
        artifactSource: request.options?.source?.kind ?? 'stub',
      });

      return new LasrCtcSpeechModel(
        context.backend,
        family,
        request.modelId,
        classification,
        config,
        request.resolvedPreset,
        request.options,
        dependencies,
        options.describeModel ?? describeLasrCtcModel,
      );
    },
  };
}

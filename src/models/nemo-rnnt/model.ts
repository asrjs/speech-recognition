import { normalizePcmInput } from '../../audio/index.js';
import {
  DefaultStreamingTranscriber,
  FASTCONFORMER_ENCODER,
  RNNT_GREEDY_DECODING,
  RNNT_TRANSDUCER_DECODER,
} from '../../inference/index.js';
import {
  buildStubTimedWords,
  createModelClassification,
  defaultNemoConfidenceReconstructor,
  defaultNemoTimestampReconstructor,
  mapNemoNativeToCanonical,
  StubNemoFeatureExtractor,
  StubNemoTokenizer,
  type NemoConfidenceReconstructor,
  type NemoDecodeContext,
  type NemoFeatureExtractor,
  type NemoTimestampReconstructor,
  type NemoTokenizer,
} from '../nemo-common/index.js';
import type {
  AudioInputLike,
  BaseSessionOptions,
  FamilyModelLoadRequest,
  ModelClassification,
  SpeechModel,
  SpeechModelFactory,
  SpeechModelFactoryContext,
  SpeechSession,
  StreamingSessionOptions,
  StreamingTranscriber,
  TranscriptResponse,
  TranscriptResponseFlavor,
} from '../../types/index.js';
import { createModelArchitecture } from '../../types/index.js';
import {
  DEFAULT_NEMO_RNNT_CLASSIFICATION,
  describeNemoRnntModel,
  parseNemoRnntConfig,
} from './config.js';
import { OrtNemoRnntExecutor } from './executor.js';
import type {
  NemoRnntDecoder,
  NemoRnntExecutor,
  NemoRnntModelConfig,
  NemoRnntModelDependencies,
  NemoRnntModelOptions,
  NemoRnntNativeTranscript,
  NemoRnntTranscriptionOptions,
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
  defaultClassification: Partial<ModelClassification> = {},
  requestClassification: Partial<ModelClassification> = {},
): ModelClassification {
  return createModelClassification(DEFAULT_NEMO_RNNT_CLASSIFICATION, {
    ...defaultClassification,
    ...requestClassification,
  });
}

function buildStubTokens(words: ReturnType<typeof buildStubTimedWords>, frameCount: number) {
  return words.map((word, index) => ({
    index,
    id: index + 1,
    text: word.text,
    rawText: word.text.toLowerCase(),
    isWordStart: true,
    startTime: word.startTime,
    endTime: word.endTime,
    confidence: word.confidence,
    frameIndex: index * Math.max(1, Math.floor(frameCount / words.length)),
    logProb: -0.08,
  }));
}

class StubNemoRnntDecoder implements NemoRnntDecoder {
  decode(
    features: Awaited<ReturnType<NemoFeatureExtractor<NemoRnntModelConfig>['compute']>>,
    options: NemoRnntTranscriptionOptions,
    _context: NemoDecodeContext<NemoRnntModelConfig>,
  ): NemoRnntNativeTranscript {
    const words = buildStubTimedWords(['Nemo', 'RNNT', 'scaffold'], features.durationSeconds);
    const tokens = buildStubTokens(words, features.frameCount);

    return {
      utteranceText: words.map((word) => word.text).join(' '),
      rawUtteranceText: words.map((word) => word.text).join(' '),
      isFinal: true,
      words,
      tokens,
      specialTokens: [],
      control: {
        containsEou: false,
        containsEob: false,
      },
      confidence: {
        utterance: 0.92,
        wordAverage: 0.92,
        tokenAverage: 0.92,
        frameAverage: 0.92,
        averageLogProb: -0.08,
        frames: Array.from({ length: Math.max(1, features.frameCount) }, () => 0.92),
      },
      metrics: {
        preprocessMs: 0.1,
        encodeMs: 0.1,
        decodeMs: 0.1,
        tokenizeMs: 0.05,
        totalMs: 0.35,
        wallMs: 0.35,
        audioDurationSec: features.durationSeconds,
        rtf: features.durationSeconds > 0 ? 0.35 / (features.durationSeconds * 1000) : 0,
        rtfx: features.durationSeconds > 0 ? features.durationSeconds / (0.35 / 1000) : undefined,
        preprocessorBackend: 'stub',
        encoderFrameCount: features.frameCount,
        decodeIterations: Math.max(1, tokens.length),
        emittedTokenCount: tokens.length,
        emittedWordCount: words.length,
      },
      warnings: [
        {
          code: 'nemo-rnnt.stubbed-decoder',
          message:
            'NeMo RNNT model execution is scaffolded. Provide model artifacts to activate the restored ORT path.',
        },
      ],
      debug: {
        tokenIds: options.returnTokenIds ? tokens.map((token) => token.id ?? -1) : undefined,
        frameIndices: options.returnFrameIndices
          ? tokens.map((token) => token.frameIndex ?? 0)
          : undefined,
        logProbs: options.returnLogProbs ? tokens.map((token) => token.logProb ?? 0) : undefined,
      },
    };
  }
}

function createExecutor(
  modelId: string,
  classification: ModelClassification,
  config: NemoRnntModelConfig,
  backendId: string,
  loadOptions: NemoRnntModelOptions | undefined,
  dependencies: NemoRnntModelDependencies,
): NemoRnntExecutor | undefined {
  if (dependencies.executor) {
    return dependencies.executor;
  }

  if (!loadOptions?.source) {
    return undefined;
  }

  return new OrtNemoRnntExecutor(
    modelId,
    classification,
    config,
    backendId,
    loadOptions,
    dependencies,
  );
}

export class NemoRnntSpeechSession implements SpeechSession<
  NemoRnntTranscriptionOptions,
  NemoRnntNativeTranscript
> {
  private readonly featureExtractor: NemoFeatureExtractor<NemoRnntModelConfig>;
  private readonly decoder: NemoRnntDecoder;
  private readonly executor?: NemoRnntExecutor;
  private readonly timestampReconstructor: NemoTimestampReconstructor<
    NemoRnntNativeTranscript,
    NemoRnntTranscriptionOptions
  >;
  private readonly confidenceReconstructor: NemoConfidenceReconstructor<NemoRnntNativeTranscript>;
  private readonly tokenizer: NemoTokenizer;
  private disposed = false;

  constructor(
    private readonly modelId: string,
    private readonly classification: ModelClassification,
    private readonly config: NemoRnntModelConfig,
    private readonly backendId: string,
    loadOptions: NemoRnntModelOptions | undefined,
    dependencies: NemoRnntModelDependencies = {},
    private readonly onDispose?: () => void,
  ) {
    this.featureExtractor = dependencies.featureExtractor ?? new StubNemoFeatureExtractor();
    this.decoder = dependencies.decoder ?? new StubNemoRnntDecoder();
    this.executor = createExecutor(
      modelId,
      classification,
      config,
      backendId,
      loadOptions,
      dependencies,
    );
    this.timestampReconstructor =
      dependencies.timestampReconstructor ?? defaultNemoTimestampReconstructor;
    this.confidenceReconstructor =
      dependencies.confidenceReconstructor ?? defaultNemoConfidenceReconstructor;
    this.tokenizer = dependencies.tokenizer ?? new StubNemoTokenizer();
  }

  async initialize(): Promise<void> {
    await this.executor?.ready?.();
  }

  async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options: NemoRnntTranscriptionOptions & { readonly responseFlavor?: TFlavor } = {},
  ): Promise<TranscriptResponse<NemoRnntNativeTranscript, TFlavor>> {
    const audio = normalizePcmInput(input).toMono();
    const nativeTranscript = this.executor
      ? await this.executor.transcribe(audio, options, {
          modelId: this.modelId,
          classification: this.classification,
          config: this.config,
          tokenizer: this.tokenizer,
        })
      : await this.decodeWithStub(audio, options);

    const canonical = mapNemoNativeToCanonical(
      nativeTranscript,
      this.classification,
      {
        detailLevel: options.detail ?? 'segments',
        backendId: this.backendId,
        sampleRate: audio.sampleRate,
        durationSeconds: audio.durationSeconds,
        language: this.config.languages[0],
        modelId: this.modelId,
        metrics: nativeTranscript.metrics
          ? {
              preprocessMs: nativeTranscript.metrics.preprocessMs,
              encodeMs: nativeTranscript.metrics.encodeMs,
              decodeMs: nativeTranscript.metrics.decodeMs,
              tokenizeMs: nativeTranscript.metrics.tokenizeMs,
              postprocessMs: nativeTranscript.metrics.tokenizeMs,
              totalMs: nativeTranscript.metrics.totalMs,
              wallMs: nativeTranscript.metrics.wallMs,
              audioDurationSec: nativeTranscript.metrics.audioDurationSec,
              rtf: nativeTranscript.metrics.rtf,
              rtfx: nativeTranscript.metrics.rtfx,
              requestedPreprocessorBackend: nativeTranscript.metrics.requestedPreprocessorBackend,
              preprocessorBackend: nativeTranscript.metrics.preprocessorBackend,
              decodeAudioMs: nativeTranscript.metrics.decodeAudioMs,
              downmixMs: nativeTranscript.metrics.downmixMs,
              resampleMs: nativeTranscript.metrics.resampleMs,
              audioPreparationMs: nativeTranscript.metrics.audioPreparationMs,
              inputSampleRate: nativeTranscript.metrics.inputSampleRate,
              outputSampleRate: nativeTranscript.metrics.outputSampleRate,
              resampler: nativeTranscript.metrics.resampler,
              resamplerQuality: nativeTranscript.metrics.resamplerQuality,
              encoderFrameCount: nativeTranscript.metrics.encoderFrameCount,
              decodeIterations: nativeTranscript.metrics.decodeIterations,
              emittedTokenCount: nativeTranscript.metrics.emittedTokenCount,
              emittedWordCount: nativeTranscript.metrics.emittedWordCount,
            }
          : undefined,
      },
      this.timestampReconstructor,
      this.confidenceReconstructor,
    );

    const responseFlavor = options.responseFlavor ?? 'canonical';
    if (responseFlavor === 'native') {
      return nativeTranscript as TranscriptResponse<NemoRnntNativeTranscript, TFlavor>;
    }
    if (responseFlavor === 'canonical+native') {
      return {
        canonical,
        native: nativeTranscript,
      } as TranscriptResponse<NemoRnntNativeTranscript, TFlavor>;
    }

    return canonical as TranscriptResponse<NemoRnntNativeTranscript, TFlavor>;
  }

  private async decodeWithStub(
    audio: ReturnType<typeof normalizePcmInput>,
    options: NemoRnntTranscriptionOptions,
  ): Promise<NemoRnntNativeTranscript> {
    const features = await this.featureExtractor.compute(audio, this.config);
    return this.decoder.decode(features, options, {
      modelId: this.modelId,
      classification: this.classification,
      config: this.config,
      tokenizer: this.tokenizer,
    });
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.executor?.dispose();
    this.onDispose?.();
  }
}

export class NemoRnntSpeechModel implements SpeechModel<
  NemoRnntModelOptions,
  NemoRnntTranscriptionOptions,
  NemoRnntNativeTranscript
> {
  readonly info;
  readonly loadOptions?: NemoRnntModelOptions;
  private readonly sessions = new Set<NemoRnntSpeechSession>();
  private disposed = false;

  constructor(
    readonly backend: SpeechModel<
      NemoRnntModelOptions,
      NemoRnntTranscriptionOptions,
      NemoRnntNativeTranscript
    >['backend'],
    readonly family: string,
    readonly modelId: string,
    readonly classification: ModelClassification,
    readonly config: NemoRnntModelConfig,
    readonly resolvedPreset: string | undefined,
    loadOptions: NemoRnntModelOptions | undefined,
    private readonly dependencies: NemoRnntModelDependencies,
    describeModel: (
      modelId: string,
      classification: ModelClassification,
      config: NemoRnntModelConfig,
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
          implementation: classification.processor ?? 'nemo-mel',
          shared: true,
          notes: [`${config.melBins}-bin mel processor for NeMo-style models.`],
        },
        encoder: {
          layer: 'encoder',
          module: FASTCONFORMER_ENCODER.sharedModule,
          implementation: config.encoderArchitecture,
          shared: true,
          notes: [`Subsampling factor ${config.subsamplingFactor}.`],
        },
        decoder: {
          layer: 'decoder',
          module: RNNT_TRANSDUCER_DECODER.sharedModule,
          implementation: RNNT_TRANSDUCER_DECODER.kind,
          shared: true,
          notes: RNNT_TRANSDUCER_DECODER.notes,
        },
        decoding: {
          layer: 'decoding',
          module: 'inference',
          implementation: RNNT_GREEDY_DECODING.strategy,
          shared: true,
          notes: RNNT_GREEDY_DECODING.notes,
        },
        tokenizer: {
          layer: 'tokenizer',
          module: 'inference',
          implementation: config.tokenizer.kind,
          shared: true,
        },
      }),
      description: describeModel(modelId, classification, config),
      nativeOutputName: 'NemoRnntNativeTranscript',
    };
  }

  async createSession(
    _options: BaseSessionOptions = {},
  ): Promise<SpeechSession<NemoRnntTranscriptionOptions, NemoRnntNativeTranscript>> {
    const session = new NemoRnntSpeechSession(
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

  async createStreamingTranscriber(
    options: StreamingSessionOptions = {},
  ): Promise<StreamingTranscriber> {
    const session = await this.createSession();
    return new DefaultStreamingTranscriber(session, options);
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

export interface CreateNemoRnntModelFamilyOptions {
  readonly dependencies?: NemoRnntModelDependencies;
  readonly family?: string;
  readonly classification?: Partial<ModelClassification>;
  readonly supportsModel?: (
    modelId: string,
    classification?: Partial<ModelClassification>,
  ) => boolean;
  readonly resolveConfig?: (
    modelId: string,
    request: FamilyModelLoadRequest<NemoRnntModelOptions>,
  ) => NemoRnntModelConfig;
  readonly describeModel?: (
    modelId: string,
    classification: ModelClassification,
    config: NemoRnntModelConfig,
  ) => string;
}

export function createNemoRnntModelFamily(
  options: CreateNemoRnntModelFamilyOptions = {},
): SpeechModelFactory<
  NemoRnntModelOptions,
  NemoRnntTranscriptionOptions,
  NemoRnntNativeTranscript
> {
  const family = options.family ?? 'nemo-rnnt';
  const factoryClassification = resolveClassification(options.classification);

  return {
    family,
    classification: factoryClassification,
    supports(modelId: string): boolean {
      if (options.supportsModel) {
        return options.supportsModel(modelId);
      }

      const normalizedModelId = modelId.toLowerCase();
      return normalizedModelId.includes('rnnt') || normalizedModelId.includes('realtime');
    },
    matchesClassification(classification: Partial<ModelClassification>): boolean {
      if (options.supportsModel) {
        return options.supportsModel('', classification);
      }

      return classificationContains(factoryClassification, classification);
    },
    async createModel(request, context: SpeechModelFactoryContext): Promise<NemoRnntSpeechModel> {
      const classification = resolveClassification(factoryClassification, request.classification);
      const config = options.resolveConfig
        ? options.resolveConfig(request.modelId, request)
        : parseNemoRnntConfig(request.modelId, request.options?.config);
      const dependencies: NemoRnntModelDependencies = {
        ...(options.dependencies ?? {}),
        assetProvider: options.dependencies?.assetProvider ?? context.assetProvider,
        runtimeHooks: options.dependencies?.runtimeHooks ?? context.hooks,
      };

      context.hooks.logger?.info?.('Creating NeMo RNNT model', {
        family,
        modelId: request.modelId,
        backendId: context.backend.id,
        artifactSource: request.options?.source?.kind ?? 'stub',
      });

      return new NemoRnntSpeechModel(
        context.backend,
        family,
        request.modelId,
        classification,
        config,
        request.resolvedPreset,
        request.options,
        dependencies,
        options.describeModel ?? describeNemoRnntModel,
      );
    },
  };
}

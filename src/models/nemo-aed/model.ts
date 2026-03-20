import { normalizePcmInput } from '../../audio/index.js';
import {
  AED_GENERATE_DECODING,
  FASTCONFORMER_ENCODER,
  TRANSFORMER_SEQ2SEQ_DECODER,
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
  TranscriptResponse,
  TranscriptResponseFlavor,
} from '../../types/index.js';
import { createModelArchitecture } from '../../types/index.js';
import {
  DEFAULT_NEMO_AED_CLASSIFICATION,
  describeNemoAedModel,
  parseNemoAedConfig,
} from './config.js';
import { OrtNemoAedExecutor } from './executor.js';
import type {
  NemoAedDecoder,
  NemoAedExecutor,
  NemoAedModelConfig,
  NemoAedModelDependencies,
  NemoAedModelOptions,
  NemoAedNativeTranscript,
  NemoAedTranscriptionOptions,
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
  return createModelClassification(DEFAULT_NEMO_AED_CLASSIFICATION, {
    ...defaultClassification,
    ...requestClassification,
  });
}

function buildStubTokens(words: ReturnType<typeof buildStubTimedWords>) {
  return words.map((word, index) => ({
    index,
    id: index + 1,
    text: word.text,
    rawText: word.text,
    isWordStart: true,
    startTime: word.startTime,
    endTime: word.endTime,
    confidence: word.confidence,
  }));
}

const nemoAedTimestampReconstructor: NemoTimestampReconstructor<
  NemoAedNativeTranscript,
  NemoAedTranscriptionOptions
> = {
  reconstruct(nativeTranscript, detail) {
    const defaultReconstructed = defaultNemoTimestampReconstructor.reconstruct(
      nativeTranscript,
      detail,
    );
    if (
      defaultReconstructed.segments?.length ||
      defaultReconstructed.words?.length ||
      detail === 'text'
    ) {
      return defaultReconstructed;
    }

    const duration = nativeTranscript.metrics?.audioDurationSec ?? 0;
    const segments =
      nativeTranscript.utteranceText.length > 0
        ? [
            {
              index: 0,
              text: nativeTranscript.utteranceText,
              startTime: 0,
              endTime: duration,
              confidence: nativeTranscript.confidence?.utterance,
            },
          ]
        : [];

    if (detail === 'segments' || detail === 'words') {
      return {
        segments,
      };
    }

    return {
      segments,
      tokens: (nativeTranscript.tokens ?? []).map((token) => ({
        index: token.index,
        id: token.id,
        text: token.text,
        rawText: token.rawText,
        isWordStart: token.isWordStart,
        confidence: token.confidence,
        logProb: token.logProb,
      })),
    };
  },
};

class StubNemoAedDecoder implements NemoAedDecoder {
  decode(
    features: Awaited<ReturnType<NemoFeatureExtractor<NemoAedModelConfig>['compute']>>,
    options: NemoAedTranscriptionOptions,
    _context: NemoDecodeContext<NemoAedModelConfig>,
  ): NemoAedNativeTranscript {
    const words = buildStubTimedWords(['Canary', 'AED', 'scaffold'], features.durationSeconds);
    const tokens = buildStubTokens(words);
    return {
      utteranceText: words.map((word) => word.text).join(' '),
      isFinal: true,
      language: options.language ?? 'en',
      words,
      tokens,
      confidence: {
        utterance: 0.92,
        wordAverage: 0.92,
        tokenAverage: 0.92,
        averageLogProb: -0.08,
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
        decodeIterations: tokens.length,
        emittedTokenCount: tokens.length,
        emittedWordCount: words.length,
      },
      warnings: [
        {
          code: 'nemo-aed.stubbed-decoder',
          message:
            'NeMo AED model execution is scaffolded. Provide model artifacts to activate the restored ORT path.',
        },
      ],
      debug: {
        tokenIds: options.returnTokenIds ? tokens.map((token) => token.id ?? -1) : undefined,
        promptIds: undefined,
        logProbs: options.returnLogProbs
          ? Array.from({ length: tokens.length }, () => -0.08)
          : undefined,
      },
    };
  }
}

function createExecutor(
  modelId: string,
  classification: ModelClassification,
  config: NemoAedModelConfig,
  backendId: string,
  loadOptions: NemoAedModelOptions | undefined,
  dependencies: NemoAedModelDependencies,
): NemoAedExecutor | undefined {
  if (dependencies.executor) {
    return dependencies.executor;
  }
  if (!loadOptions?.source) {
    return undefined;
  }
  return new OrtNemoAedExecutor(modelId, classification, config, backendId, loadOptions, {
    assetProvider: dependencies.assetProvider,
    runtimeHooks: dependencies.runtimeHooks,
  });
}

export class NemoAedSpeechSession implements SpeechSession<
  NemoAedTranscriptionOptions,
  NemoAedNativeTranscript
> {
  private readonly featureExtractor: NemoFeatureExtractor<NemoAedModelConfig>;
  private readonly decoder: NemoAedDecoder;
  private readonly executor?: NemoAedExecutor;
  private readonly timestampReconstructor: NemoTimestampReconstructor<
    NemoAedNativeTranscript,
    NemoAedTranscriptionOptions
  >;
  private readonly confidenceReconstructor: NemoConfidenceReconstructor<NemoAedNativeTranscript>;
  private readonly tokenizer: NemoTokenizer;
  private disposed = false;

  constructor(
    private readonly modelId: string,
    private readonly classification: ModelClassification,
    private readonly config: NemoAedModelConfig,
    private readonly backendId: string,
    loadOptions: NemoAedModelOptions | undefined,
    dependencies: NemoAedModelDependencies = {},
    private readonly onDispose?: () => void,
  ) {
    this.featureExtractor = dependencies.featureExtractor ?? new StubNemoFeatureExtractor();
    this.decoder = dependencies.decoder ?? new StubNemoAedDecoder();
    this.executor = createExecutor(
      modelId,
      classification,
      config,
      backendId,
      loadOptions,
      dependencies,
    );
    this.timestampReconstructor =
      dependencies.timestampReconstructor ?? nemoAedTimestampReconstructor;
    this.confidenceReconstructor =
      dependencies.confidenceReconstructor ?? defaultNemoConfidenceReconstructor;
    this.tokenizer = dependencies.tokenizer ?? new StubNemoTokenizer();
  }

  async initialize(): Promise<void> {
    await this.executor?.ready?.();
  }

  async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options: NemoAedTranscriptionOptions & { readonly responseFlavor?: TFlavor } = {},
  ): Promise<TranscriptResponse<NemoAedNativeTranscript, TFlavor>> {
    const audio = normalizePcmInput(input).toMono();
    const startMs = performance.now();
    if (!this.executor) {
      options.onProgress?.({
        stage: 'start',
        progress: 0,
        elapsedMs: 0,
        modelId: this.modelId,
        backendId: this.backendId,
        message: `Starting transcription for ${this.modelId}.`,
      });
    }

    const nativeTranscript = this.executor
      ? await this.executor.transcribe(audio, options, {
          modelId: this.modelId,
          classification: this.classification,
          config: this.config,
          tokenizer: this.tokenizer,
        })
      : await this.decodeWithStub(audio, options);

    if (!this.executor) {
      const elapsedMs = nativeTranscript.metrics?.totalMs ?? performance.now() - startMs;
      options.onProgress?.({
        stage: 'complete',
        progress: 1,
        elapsedMs,
        modelId: this.modelId,
        backendId: this.backendId,
        message: `Finished transcription for ${this.modelId}.`,
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
      });
    }

    const canonical = mapNemoNativeToCanonical(
      nativeTranscript,
      this.classification,
      {
        detailLevel: options.detail ?? 'segments',
        backendId: this.backendId,
        sampleRate: audio.sampleRate,
        durationSeconds: audio.durationSeconds,
        language: nativeTranscript.language ?? options.language ?? this.config.languages[0],
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
      return nativeTranscript as TranscriptResponse<NemoAedNativeTranscript, TFlavor>;
    }
    if (responseFlavor === 'canonical+native') {
      return {
        canonical,
        native: nativeTranscript,
      } as TranscriptResponse<NemoAedNativeTranscript, TFlavor>;
    }

    return canonical as TranscriptResponse<NemoAedNativeTranscript, TFlavor>;
  }

  private async decodeWithStub(
    audio: ReturnType<typeof normalizePcmInput>,
    options: NemoAedTranscriptionOptions,
  ): Promise<NemoAedNativeTranscript> {
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

export class NemoAedSpeechModel implements SpeechModel<
  NemoAedModelOptions,
  NemoAedTranscriptionOptions,
  NemoAedNativeTranscript
> {
  readonly info;
  readonly loadOptions?: NemoAedModelOptions;
  private readonly sessions = new Set<NemoAedSpeechSession>();
  private disposed = false;

  constructor(
    readonly backend: SpeechModel<
      NemoAedModelOptions,
      NemoAedTranscriptionOptions,
      NemoAedNativeTranscript
    >['backend'],
    readonly family: string,
    readonly modelId: string,
    readonly classification: ModelClassification,
    readonly config: NemoAedModelConfig,
    readonly resolvedPreset: string | undefined,
    loadOptions: NemoAedModelOptions | undefined,
    private readonly dependencies: NemoAedModelDependencies,
    describeModel: (
      modelId: string,
      classification: ModelClassification,
      config: NemoAedModelConfig,
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
          module: TRANSFORMER_SEQ2SEQ_DECODER.sharedModule,
          implementation: TRANSFORMER_SEQ2SEQ_DECODER.kind,
          shared: true,
          notes: [`Prompt format ${config.promptFormat}.`],
        },
        decoding: {
          layer: 'decoding',
          module: 'inference',
          implementation: AED_GENERATE_DECODING.strategy,
          shared: true,
          notes: AED_GENERATE_DECODING.notes,
        },
        tokenizer: {
          layer: 'tokenizer',
          module: 'inference',
          implementation: config.tokenizer.kind,
          shared: true,
        },
      }),
      description: describeModel(modelId, classification, config),
      nativeOutputName: 'NemoAedNativeTranscript',
    };
  }

  async createSession(
    _options: BaseSessionOptions = {},
  ): Promise<SpeechSession<NemoAedTranscriptionOptions, NemoAedNativeTranscript>> {
    const session = new NemoAedSpeechSession(
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

export interface CreateNemoAedModelFamilyOptions {
  readonly dependencies?: NemoAedModelDependencies;
  readonly family?: string;
  readonly classification?: Partial<ModelClassification>;
  readonly supportsModel?: (
    modelId: string,
    classification?: Partial<ModelClassification>,
  ) => boolean;
  readonly resolveConfig?: (
    modelId: string,
    request: FamilyModelLoadRequest<NemoAedModelOptions>,
  ) => NemoAedModelConfig;
  readonly describeModel?: (
    modelId: string,
    classification: ModelClassification,
    config: NemoAedModelConfig,
  ) => string;
}

export function createNemoAedModelFamily(
  options: CreateNemoAedModelFamilyOptions = {},
): SpeechModelFactory<NemoAedModelOptions, NemoAedTranscriptionOptions, NemoAedNativeTranscript> {
  const family = options.family ?? 'nemo-aed';
  const factoryClassification = resolveClassification(options.classification);

  return {
    family,
    classification: factoryClassification,
    supports(modelId: string): boolean {
      if (options.supportsModel) {
        return options.supportsModel(modelId);
      }
      const normalizedModelId = modelId.toLowerCase();
      return normalizedModelId.includes('canary') || normalizedModelId.includes('nemo-aed');
    },
    matchesClassification(classification: Partial<ModelClassification>): boolean {
      if (options.supportsModel) {
        return options.supportsModel('', classification);
      }
      return classificationContains(factoryClassification, classification);
    },
    async createModel(request, context: SpeechModelFactoryContext): Promise<NemoAedSpeechModel> {
      const classification = resolveClassification(factoryClassification, request.classification);
      const config = options.resolveConfig
        ? options.resolveConfig(request.modelId, request)
        : parseNemoAedConfig(request.modelId, request.options?.config);
      const dependencies: NemoAedModelDependencies = {
        ...(options.dependencies ?? {}),
        assetProvider: options.dependencies?.assetProvider ?? context.assetProvider,
        runtimeHooks: options.dependencies?.runtimeHooks ?? context.hooks,
      };

      context.hooks.logger?.info?.('Creating NeMo AED model', {
        family,
        modelId: request.modelId,
        backendId: context.backend.id,
        artifactSource: request.options?.source?.kind ?? 'stub',
      });

      return new NemoAedSpeechModel(
        context.backend,
        family,
        request.modelId,
        classification,
        config,
        request.resolvedPreset,
        request.options,
        dependencies,
        options.describeModel ?? describeNemoAedModel,
      );
    },
  };
}

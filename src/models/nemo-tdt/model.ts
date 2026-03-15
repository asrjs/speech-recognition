import { normalizePcmInput } from '../../audio/index.js';
import {
  FASTCONFORMER_ENCODER,
  TDT_GREEDY_DECODING,
  TDT_TRANSDUCER_DECODER,
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
  DEFAULT_NEMO_TDT_CLASSIFICATION,
  describeNemoTdtModel,
  parseNemoTdtConfig,
} from './config.js';
import { OrtNemoTdtExecutor } from './executor.js';
import type {
  NemoTdtDecoder,
  NemoTdtExecutor,
  NemoTdtModelConfig,
  NemoTdtModelDependencies,
  NemoTdtModelOptions,
  NemoTdtNativeTranscript,
  NemoTdtTranscriptionOptions,
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
  return createModelClassification(DEFAULT_NEMO_TDT_CLASSIFICATION, {
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
    tdtStep: 1,
  }));
}

class StubNemoTdtDecoder implements NemoTdtDecoder {
  decode(
    features: Awaited<ReturnType<NemoFeatureExtractor<NemoTdtModelConfig>['compute']>>,
    options: NemoTdtTranscriptionOptions,
    _context: NemoDecodeContext<NemoTdtModelConfig>,
  ): NemoTdtNativeTranscript {
    const words = buildStubTimedWords(['Nemo', 'TDT', 'scaffold'], features.durationSeconds);
    const tokens = buildStubTokens(words, features.frameCount);

    return {
      utteranceText: words.map((word) => word.text).join(' '),
      isFinal: true,
      words,
      tokens,
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
          code: 'nemo-tdt.stubbed-decoder',
          message:
            'NeMo TDT model execution is scaffolded. Provide model artifacts to activate the restored ORT path.',
        },
      ],
      debug: {
        tokenIds: options.returnTokenIds ? tokens.map((token) => token.id ?? -1) : undefined,
        frameIndices: options.returnFrameIndices
          ? tokens.map((token) => token.frameIndex ?? 0)
          : undefined,
        logProbs: options.returnLogProbs ? tokens.map((token) => token.logProb ?? 0) : undefined,
        tdtSteps: options.returnTdtSteps ? tokens.map((token) => token.tdtStep ?? 0) : undefined,
      },
    };
  }
}

function createExecutor(
  modelId: string,
  classification: ModelClassification,
  config: NemoTdtModelConfig,
  backendId: string,
  loadOptions: NemoTdtModelOptions | undefined,
  dependencies: NemoTdtModelDependencies,
): NemoTdtExecutor | undefined {
  if (dependencies.executor) {
    return dependencies.executor;
  }

  if (!loadOptions?.source) {
    return undefined;
  }

  return new OrtNemoTdtExecutor(
    modelId,
    classification,
    config,
    backendId,
    loadOptions,
    dependencies,
  );
}

export class NemoTdtSpeechSession implements SpeechSession<
  NemoTdtTranscriptionOptions,
  NemoTdtNativeTranscript
> {
  private readonly featureExtractor: NemoFeatureExtractor<NemoTdtModelConfig>;
  private readonly decoder: NemoTdtDecoder;
  private readonly executor?: NemoTdtExecutor;
  private readonly timestampReconstructor: NemoTimestampReconstructor<
    NemoTdtNativeTranscript,
    NemoTdtTranscriptionOptions
  >;
  private readonly confidenceReconstructor: NemoConfidenceReconstructor<NemoTdtNativeTranscript>;
  private readonly tokenizer: NemoTokenizer;
  private disposed = false;

  constructor(
    private readonly modelId: string,
    private readonly classification: ModelClassification,
    private readonly config: NemoTdtModelConfig,
    private readonly backendId: string,
    loadOptions: NemoTdtModelOptions | undefined,
    dependencies: NemoTdtModelDependencies = {},
    private readonly onDispose?: () => void,
  ) {
    this.featureExtractor = dependencies.featureExtractor ?? new StubNemoFeatureExtractor();
    this.decoder = dependencies.decoder ?? new StubNemoTdtDecoder();
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
    options: NemoTdtTranscriptionOptions & { readonly responseFlavor?: TFlavor } = {},
  ): Promise<TranscriptResponse<NemoTdtNativeTranscript, TFlavor>> {
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
      return nativeTranscript as TranscriptResponse<NemoTdtNativeTranscript, TFlavor>;
    }
    if (responseFlavor === 'canonical+native') {
      return {
        canonical,
        native: nativeTranscript,
      } as TranscriptResponse<NemoTdtNativeTranscript, TFlavor>;
    }

    return canonical as TranscriptResponse<NemoTdtNativeTranscript, TFlavor>;
  }

  private async decodeWithStub(
    audio: ReturnType<typeof normalizePcmInput>,
    options: NemoTdtTranscriptionOptions,
  ): Promise<NemoTdtNativeTranscript> {
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

export class NemoTdtSpeechModel implements SpeechModel<
  NemoTdtModelOptions,
  NemoTdtTranscriptionOptions,
  NemoTdtNativeTranscript
> {
  readonly info;
  readonly loadOptions?: NemoTdtModelOptions;
  private readonly sessions = new Set<NemoTdtSpeechSession>();
  private disposed = false;

  constructor(
    readonly backend: SpeechModel<
      NemoTdtModelOptions,
      NemoTdtTranscriptionOptions,
      NemoTdtNativeTranscript
    >['backend'],
    readonly family: string,
    readonly modelId: string,
    readonly classification: ModelClassification,
    readonly config: NemoTdtModelConfig,
    readonly resolvedPreset: string | undefined,
    loadOptions: NemoTdtModelOptions | undefined,
    private readonly dependencies: NemoTdtModelDependencies,
    describeModel: (
      modelId: string,
      classification: ModelClassification,
      config: NemoTdtModelConfig,
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
          module: TDT_TRANSDUCER_DECODER.sharedModule,
          implementation: TDT_TRANSDUCER_DECODER.kind,
          shared: true,
          notes: TDT_TRANSDUCER_DECODER.notes,
        },
        decoding: {
          layer: 'decoding',
          module: 'inference',
          implementation: TDT_GREEDY_DECODING.strategy,
          shared: true,
          notes: TDT_GREEDY_DECODING.notes,
        },
        tokenizer: {
          layer: 'tokenizer',
          module: 'inference',
          implementation: config.tokenizer.kind,
          shared: true,
        },
      }),
      description: describeModel(modelId, classification, config),
      nativeOutputName: 'NemoTdtNativeTranscript',
    };
  }

  async createSession(
    _options: BaseSessionOptions = {},
  ): Promise<SpeechSession<NemoTdtTranscriptionOptions, NemoTdtNativeTranscript>> {
    const session = new NemoTdtSpeechSession(
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

export interface CreateNemoTdtModelFamilyOptions {
  readonly dependencies?: NemoTdtModelDependencies;
  readonly family?: string;
  readonly classification?: Partial<ModelClassification>;
  readonly supportsModel?: (
    modelId: string,
    classification?: Partial<ModelClassification>,
  ) => boolean;
  readonly resolveConfig?: (
    modelId: string,
    request: FamilyModelLoadRequest<NemoTdtModelOptions>,
  ) => NemoTdtModelConfig;
  readonly describeModel?: (
    modelId: string,
    classification: ModelClassification,
    config: NemoTdtModelConfig,
  ) => string;
}

export function createNemoTdtModelFamily(
  options: CreateNemoTdtModelFamilyOptions = {},
): SpeechModelFactory<NemoTdtModelOptions, NemoTdtTranscriptionOptions, NemoTdtNativeTranscript> {
  const family = options.family ?? 'nemo-tdt';
  const factoryClassification = resolveClassification(options.classification);

  return {
    family,
    classification: factoryClassification,
    supports(modelId: string): boolean {
      if (options.supportsModel) {
        return options.supportsModel(modelId);
      }

      const normalizedModelId = modelId.toLowerCase();
      return normalizedModelId.includes('tdt') || normalizedModelId.includes('nemo');
    },
    matchesClassification(classification: Partial<ModelClassification>): boolean {
      if (options.supportsModel) {
        return options.supportsModel('', classification);
      }

      return classificationContains(factoryClassification, classification);
    },
    async createModel(request, context: SpeechModelFactoryContext): Promise<NemoTdtSpeechModel> {
      const classification = resolveClassification(factoryClassification, request.classification);
      const config = options.resolveConfig
        ? options.resolveConfig(request.modelId, request)
        : parseNemoTdtConfig(request.modelId, request.options?.config);
      const dependencies: NemoTdtModelDependencies = {
        ...(options.dependencies ?? {}),
        assetProvider: options.dependencies?.assetProvider ?? context.assetProvider,
        runtimeHooks: options.dependencies?.runtimeHooks ?? context.hooks,
      };

      context.hooks.logger?.info?.('Creating NeMo TDT model', {
        family,
        modelId: request.modelId,
        backendId: context.backend.id,
        artifactSource: request.options?.source?.kind ?? 'stub',
      });

      return new NemoTdtSpeechModel(
        context.backend,
        family,
        request.modelId,
        classification,
        config,
        request.resolvedPreset,
        request.options,
        dependencies,
        options.describeModel ?? describeNemoTdtModel,
      );
    },
  };
}

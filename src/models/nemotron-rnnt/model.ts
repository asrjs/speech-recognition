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
  type NemoConfidenceReconstructor,
  type NemoDecodeContext,
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
  DEFAULT_NEMOTRON_RNNT_CLASSIFICATION,
  describeNemotronRnntModel,
  parseNemotronRnntConfig,
} from './config.js';
import { OrtNemotronRnntExecutor } from './executor.js';
import type {
  NemotronRnntDecoder,
  NemotronRnntExecutor,
  NemotronRnntModelConfig,
  NemotronRnntModelDependencies,
  NemotronRnntModelOptions,
  NemotronRnntNativeTranscript,
  NemotronRnntTranscriptionOptions,
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
  return createModelClassification(DEFAULT_NEMOTRON_RNNT_CLASSIFICATION, {
    ...defaultClassification,
    ...requestClassification,
  });
}

function buildStubTokens(
  words: ReturnType<typeof buildStubTimedWords>,
  frameCount: number,
) {
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

class StubNemotronRnntDecoder implements NemotronRnntDecoder {
  decode(
    features: {
      readonly features: Float32Array;
      readonly frameCount: number;
      readonly durationSeconds: number;
    },
    options: NemotronRnntTranscriptionOptions,
    _context: NemoDecodeContext<NemotronRnntModelConfig>,
  ): NemotronRnntNativeTranscript {
    const words = buildStubTimedWords(
      ['Nemotron', 'RNNT', 'scaffold'],
      features.durationSeconds,
    );
    const tokens = buildStubTokens(words, features.frameCount);
    const totalMs = 0.35;
    return {
      utteranceText: words.map((word) => word.text).join(' '),
      rawUtteranceText: words.map((word) => word.text).join(' '),
      isFinal: true,
      words,
      tokens,
      specialTokens: [],
      control: { containsLangSegment: false },
      confidence: {
        utterance: 0.92,
        tokenAverage: 0.92,
        wordAverage: 0.92,
        frameAverage: 0.92,
        averageLogProb: -0.08,
        frames: Array.from({ length: Math.max(1, features.frameCount) }, () => 0.92),
      },
      metrics: {
        preprocessMs: 0.1,
        encodeMs: 0.1,
        decodeMs: 0.1,
        totalMs,
        wallMs: totalMs,
        audioDurationSec: features.durationSeconds,
        rtf: features.durationSeconds > 0 ? totalMs / (features.durationSeconds * 1000) : 0,
        rtfx: features.durationSeconds > 0
          ? features.durationSeconds / (totalMs / 1000)
          : undefined,
        preprocessorBackend: 'js',
        encoderFrameCount: features.frameCount,
        decodeIterations: Math.max(1, tokens.length),
        emittedTokenCount: tokens.length,
        emittedWordCount: words.length,
      },
      warnings: [
        {
          code: 'nemotron-rnnt.stubbed-decoder',
          message:
            'Nemotron RNNT model execution is scaffolded. Provide model artifacts to activate the restored ORT path.',
        },
      ],
      debug: {
        tokenIds: options.returnTokenIds ? tokens.map((t) => t.id ?? -1) : undefined,
        frameIndices: options.returnFrameIndices
          ? tokens.map((t) => t.frameIndex ?? 0)
          : undefined,
        logProbs: options.returnLogProbs ? tokens.map(() => -0.08) : undefined,
      },
    };
  }
}

function createExecutor(
  modelId: string,
  classification: ModelClassification,
  config: NemotronRnntModelConfig,
  backendId: string,
  loadOptions: NemotronRnntModelOptions | undefined,
  dependencies: NemotronRnntModelDependencies,
): NemotronRnntExecutor | undefined {
  if (dependencies.executor) {
    return dependencies.executor;
  }
  if (!loadOptions?.source) {
    return undefined;
  }
  return new OrtNemotronRnntExecutor(
    modelId,
    classification,
    config,
    backendId,
    loadOptions.source,
    {
      assetProvider: dependencies.assetProvider,
      runtimeHooks: dependencies.runtimeHooks,
      signal: dependencies.signal ?? null,
    },
  );
}

export class NemotronRnntSpeechSession implements SpeechSession<
  NemotronRnntTranscriptionOptions,
  NemotronRnntNativeTranscript
> {
  private disposed = false;

  constructor(
    private readonly modelId: string,
    private readonly classification: ModelClassification,
    private readonly config: NemotronRnntModelConfig,
    private readonly backendId: string,
    loadOptions: NemotronRnntModelOptions | undefined,
    private readonly decoder: NemotronRnntDecoder,
    private readonly executor: NemotronRnntExecutor | undefined,
    private readonly timestampReconstructor: NemoTimestampReconstructor<
      NemotronRnntNativeTranscript,
      NemotronRnntTranscriptionOptions
    >,
    private readonly confidenceReconstructor: NemoConfidenceReconstructor<NemotronRnntNativeTranscript>,
    private readonly tokenizer: NemoTokenizer | undefined,
    private readonly onDispose?: () => void,
  ) {
    void loadOptions;
  }

  async initialize(): Promise<void> {
    await this.executor?.ready?.();
  }

  async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options: NemotronRnntTranscriptionOptions & { readonly responseFlavor?: TFlavor } = {},
  ): Promise<TranscriptResponse<NemotronRnntNativeTranscript, TFlavor>> {
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
      return nativeTranscript as TranscriptResponse<NemotronRnntNativeTranscript, TFlavor>;
    }
    if (responseFlavor === 'canonical+native') {
      return {
        canonical,
        native: nativeTranscript,
      } as TranscriptResponse<NemotronRnntNativeTranscript, TFlavor>;
    }
    return canonical as TranscriptResponse<NemotronRnntNativeTranscript, TFlavor>;
  }

  private async decodeWithStub(
    audio: ReturnType<typeof normalizePcmInput>,
    options: NemotronRnntTranscriptionOptions,
  ): Promise<NemotronRnntNativeTranscript> {
    const frameCount = Math.max(1, Math.floor(audio.durationSeconds * 100));
    const features = {
      features: new Float32Array(frameCount * this.config.melBins),
      frameCount,
      durationSeconds: audio.durationSeconds,
    };
    return Promise.resolve(this.decoder.decode(features, options, {
      modelId: this.modelId,
      classification: this.classification,
      config: this.config,
      tokenizer: this.tokenizer,
    }));
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await Promise.resolve(this.executor?.dispose());
    this.onDispose?.();
  }
}

export class NemotronRnntSpeechModel implements SpeechModel<
  NemotronRnntModelOptions,
  NemotronRnntTranscriptionOptions,
  NemotronRnntNativeTranscript
> {
  readonly info;
  readonly loadOptions?: NemotronRnntModelOptions;
  private readonly sessions = new Set<NemotronRnntSpeechSession>();
  private disposed = false;

  constructor(
    readonly backend: SpeechModel<
      NemotronRnntModelOptions,
      NemotronRnntTranscriptionOptions,
      NemotronRnntNativeTranscript
    >['backend'],
    readonly family: string,
    readonly modelId: string,
    readonly classification: ModelClassification,
    readonly config: NemotronRnntModelConfig,
    readonly resolvedPreset: string | undefined,
    loadOptions: NemotronRnntModelOptions | undefined,
    private readonly dependencies: NemotronRnntModelDependencies,
    describeModel: (
      modelId: string,
      classification: ModelClassification,
      config: NemotronRnntModelConfig,
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
          notes: [`${config.melBins}-bin mel processor for Nemotron RNNT models.`],
        },
        encoder: {
          layer: 'encoder',
          module: FASTCONFORMER_ENCODER.sharedModule,
          implementation: config.encoderArchitecture,
          shared: true,
          notes: [
            `Subsampling factor ${config.subsamplingFactor}. Cache-aware streaming encoder with ${config.encoderCache.channelLayers} Conformer layers.`,
          ],
        },
        decoder: {
          layer: 'decoder',
          module: RNNT_TRANSDUCER_DECODER.sharedModule,
          implementation: RNNT_TRANSDUCER_DECODER.kind,
          shared: true,
          notes: [
            ...(RNNT_TRANSDUCER_DECODER.notes ?? []),
            `Cache-aware chunked encoder (chunk=${config.chunkFrames} mel frames, ${config.encoderOutputFramesPerChunk} encoder frames per chunk).`,
            `Prompt IDs: auto=${config.promptIds.auto}, en=${config.promptIds.en}, tr=${config.promptIds.tr}.`,
          ],
        },
        decoding: {
          layer: 'decoding',
          module: 'inference',
          implementation: RNNT_GREEDY_DECODING.strategy,
          shared: true,
          notes: [
            ...(RNNT_GREEDY_DECODING.notes ?? []),
            'Nemotron-specific: per-step joint over remaining encoder frames; emit first non-blank argmax of last decoder column.',
          ],
        },
        tokenizer: {
          layer: 'tokenizer',
          module: 'inference',
          implementation: config.tokenizer.kind,
          shared: true,
          notes: [`Vocab size ${config.vocabularySize ?? '?'}, blank ${config.blankTokenId}.`],
        },
      }),
      description: describeModel(modelId, classification, config),
      nativeOutputName: 'NemotronRnntNativeTranscript',
    };
  }

  async createSession(
    _options: BaseSessionOptions = {},
  ): Promise<SpeechSession<NemotronRnntTranscriptionOptions, NemotronRnntNativeTranscript>> {
    const decoder: NemotronRnntDecoder =
      this.dependencies.decoder ?? new StubNemotronRnntDecoder();
    const executor = createExecutor(
      this.modelId,
      this.classification,
      this.config,
      this.backend.id,
      this.loadOptions,
      this.dependencies,
    );
    const session = new NemotronRnntSpeechSession(
      this.modelId,
      this.classification,
      this.config,
      this.backend.id,
      this.loadOptions,
      decoder,
      executor,
      this.dependencies.timestampReconstructor ?? defaultNemoTimestampReconstructor,
      this.dependencies.confidenceReconstructor ?? defaultNemoConfidenceReconstructor,
      this.dependencies.tokenizer,
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
    if (this.disposed) return;
    this.disposed = true;
    const sessions = [...this.sessions];
    this.sessions.clear();
    await Promise.all(sessions.map((session) => session.dispose()));
  }
}

export interface CreateNemotronRnntModelFamilyOptions {
  readonly dependencies?: NemotronRnntModelDependencies;
  readonly family?: string;
  readonly classification?: Partial<ModelClassification>;
  readonly supportsModel?: (
    modelId: string,
    classification?: Partial<ModelClassification>,
  ) => boolean;
  readonly resolveConfig?: (
    modelId: string,
    request: FamilyModelLoadRequest<NemotronRnntModelOptions>,
  ) => NemotronRnntModelConfig;
  readonly describeModel?: (
    modelId: string,
    classification: ModelClassification,
    config: NemotronRnntModelConfig,
  ) => string;
}

export function createNemotronRnntModelFamily(
  options: CreateNemotronRnntModelFamilyOptions = {},
): SpeechModelFactory<
  NemotronRnntModelOptions,
  NemotronRnntTranscriptionOptions,
  NemotronRnntNativeTranscript
> {
  const family = options.family ?? 'nemotron-rnnt';
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
        normalizedModelId.includes('nemotron') ||
        normalizedModelId.includes('nvidia/nemotron-3.5')
      );
    },
    matchesClassification(classification: Partial<ModelClassification>): boolean {
      if (options.supportsModel) {
        return options.supportsModel('', classification);
      }
      return classificationContains(factoryClassification, classification);
    },
    async createModel(
      request,
      context: SpeechModelFactoryContext,
    ): Promise<NemotronRnntSpeechModel> {
      const classification = resolveClassification(factoryClassification, request.classification);
      const config = options.resolveConfig
        ? options.resolveConfig(request.modelId, request)
        : parseNemotronRnntConfig(request.modelId, request.options?.config);
      const dependencies: NemotronRnntModelDependencies = {
        ...(options.dependencies ?? {}),
        assetProvider: options.dependencies?.assetProvider ?? context.assetProvider,
        runtimeHooks: options.dependencies?.runtimeHooks ?? context.hooks,
        signal: options.dependencies?.signal ?? context.signal,
      };

      context.hooks.logger?.info?.('Creating Nemotron RNNT model', {
        family,
        modelId: request.modelId,
        backendId: context.backend.id,
        artifactSource: request.options?.source?.kind ?? 'stub',
      });

      return new NemotronRnntSpeechModel(
        context.backend,
        family,
        request.modelId,
        classification,
        config,
        request.resolvedPreset,
        request.options,
        dependencies,
        options.describeModel ?? describeNemotronRnntModel,
      );
    },
  };
}

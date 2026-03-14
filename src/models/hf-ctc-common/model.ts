import { normalizePcmInput } from '../../processors/index.js';
import {
  CTC_GREEDY_DECODING,
  CTC_HEAD_DECODER,
  WAV2VEC2_CONFORMER_ENCODER
} from '../../inference/index.js';
import { StubTextTokenizer } from '../../tokenizers/index.js';
import type {
  AudioInputLike,
  BaseSessionOptions,
  ModelClassification,
  ModelLoadRequest,
  SpeechModel,
  SpeechModelFactory,
  SpeechModelFactoryContext,
  SpeechSession,
  TranscriptResponse,
  TranscriptResponseFlavor,
  TranscriptWord
} from '../../types/index.js';
import { createModelArchitecture } from '../../types/index.js';
import {
  DEFAULT_HF_CTC_CLASSIFICATION,
  describeHfCtcModel,
  parseHfCtcConfig
} from './config.js';
import { mapHfCtcNativeToCanonical } from './mapping.js';
import type {
  HfCtcModelConfig,
  HfCtcModelDependencies,
  HfCtcModelOptions,
  HfCtcNativeToken,
  HfCtcNativeTranscript,
  HfCtcTranscriptionOptions
} from './types.js';

function classificationContains(
  candidate: Partial<ModelClassification>,
  requested: Partial<ModelClassification>
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
  request: Partial<ModelClassification> = {}
): ModelClassification {
  return {
    ...DEFAULT_HF_CTC_CLASSIFICATION,
    ...base,
    ...request
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
    confidence: 0.91
  }));
}

function buildStubTokens(
  words: readonly TranscriptWord[],
  options: HfCtcTranscriptionOptions
): HfCtcNativeToken[] {
  return words.map((word, index) => ({
    index,
    id: options.returnTokenIds ? index + 1 : undefined,
    text: word.text,
    startTime: word.startTime,
    endTime: word.endTime,
    confidence: word.confidence,
    logitIndex: options.returnLogitIndices ? index : undefined
  }));
}

class HfCtcSpeechSession
  implements SpeechSession<HfCtcTranscriptionOptions, HfCtcNativeTranscript> {
  private readonly tokenizer;

  constructor(
    private readonly modelId: string,
    private readonly classification: ModelClassification,
    private readonly config: HfCtcModelConfig,
    private readonly backendId: string,
    dependencies: HfCtcModelDependencies = {}
  ) {
    this.tokenizer = dependencies.tokenizer ?? new StubTextTokenizer(config.tokenizer.kind, 'hf');
  }

  async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options: HfCtcTranscriptionOptions & { readonly responseFlavor?: TFlavor } = {}
  ): Promise<TranscriptResponse<HfCtcNativeTranscript, TFlavor>> {
    const audio = normalizePcmInput(input).toMono();
    const words = buildStubWords(audio.durationSeconds);
    const tokens = buildStubTokens(words, options);
    const nativeTranscript: HfCtcNativeTranscript = {
      utteranceText: words.map((word) => word.text).join(' '),
      isFinal: true,
      words,
      tokens,
      confidence: {
        utterance: 0.91,
        wordAverage: 0.91,
        tokenAverage: 0.91
      },
      warnings: [{
        code: 'hf-ctc.stubbed-decoder',
        message: 'HF CTC model execution is scaffolded. Integrate processor, encoder, and CTC logits execution to replace the stub.'
      }]
    };

    const canonical = mapHfCtcNativeToCanonical(nativeTranscript, this.classification, {
      detailLevel: options.detail,
      backendId: this.backendId,
      modelId: this.modelId,
      language: this.config.languages[0],
      sampleRate: audio.sampleRate,
      durationSeconds: audio.durationSeconds
    });
    const responseFlavor = options.responseFlavor ?? 'canonical';

    if (responseFlavor === 'native') {
      return nativeTranscript as TranscriptResponse<HfCtcNativeTranscript, TFlavor>;
    }
    if (responseFlavor === 'canonical+native') {
      return {
        canonical,
        native: nativeTranscript
      } as TranscriptResponse<HfCtcNativeTranscript, TFlavor>;
    }

    return canonical as TranscriptResponse<HfCtcNativeTranscript, TFlavor>;
  }

  dispose(): void {
    void this.tokenizer;
  }
}

class HfCtcSpeechModel
  implements SpeechModel<HfCtcModelOptions, HfCtcTranscriptionOptions, HfCtcNativeTranscript> {
  readonly info;
  readonly loadOptions?: HfCtcModelOptions;

  constructor(
    readonly backend: SpeechModel<HfCtcModelOptions, HfCtcTranscriptionOptions, HfCtcNativeTranscript>['backend'],
    readonly family: string,
    readonly modelId: string,
    readonly classification: ModelClassification,
    readonly config: HfCtcModelConfig,
    loadOptions: HfCtcModelOptions | undefined,
    private readonly dependencies: HfCtcModelDependencies,
    private readonly describeModel: (
      modelId: string,
      classification: ModelClassification,
      config: HfCtcModelConfig
    ) => string
  ) {
    this.loadOptions = loadOptions;
    this.info = {
      family,
      modelId,
      classification,
      architecture: createModelArchitecture({
        processor: {
          layer: 'processor',
          module: 'processors',
          implementation: classification.processor ?? config.processorArchitecture,
          shared: true
        },
        encoder: {
          layer: 'encoder',
          module: WAV2VEC2_CONFORMER_ENCODER.sharedModule,
          implementation: config.encoderArchitecture,
          shared: true,
          notes: [`Output stride ${config.rawStride}.`]
        },
        decoder: {
          layer: 'decoder',
          module: CTC_HEAD_DECODER.sharedModule,
          implementation: CTC_HEAD_DECODER.kind,
          shared: true
        },
        decoding: {
          layer: 'decoding',
          module: 'inference',
          implementation: CTC_GREEDY_DECODING.strategy,
          shared: true,
          notes: CTC_GREEDY_DECODING.notes
        },
        tokenizer: {
          layer: 'tokenizer',
          module: 'inference',
          implementation: config.tokenizer.kind,
          shared: true
        }
      }),
      description: describeModel(modelId, classification, config),
      nativeOutputName: 'HfCtcNativeTranscript'
    };
  }

  async createSession(_options: BaseSessionOptions = {}): Promise<SpeechSession<HfCtcTranscriptionOptions, HfCtcNativeTranscript>> {
    return new HfCtcSpeechSession(this.modelId, this.classification, this.config, this.backend.id, this.dependencies);
  }

  dispose(): void {
    return undefined;
  }
}

export interface CreateHfCtcModelFamilyOptions {
  readonly dependencies?: HfCtcModelDependencies;
  readonly family?: string;
  readonly classification?: Partial<ModelClassification>;
  readonly supportsModel?: (modelId: string, classification?: Partial<ModelClassification>) => boolean;
  readonly resolveConfig?: (
    modelId: string,
    request: ModelLoadRequest<HfCtcModelOptions>
  ) => HfCtcModelConfig;
  readonly describeModel?: (
    modelId: string,
    classification: ModelClassification,
    config: HfCtcModelConfig
  ) => string;
}

export function createHfCtcModelFamily(
  options: CreateHfCtcModelFamilyOptions = {}
): SpeechModelFactory<HfCtcModelOptions, HfCtcTranscriptionOptions, HfCtcNativeTranscript> {
  const family = options.family ?? 'hf-ctc';
  const factoryClassification = resolveClassification(options.classification);

  return {
    family,
    classification: factoryClassification,
    supports(modelId: string): boolean {
      if (options.supportsModel) {
        return options.supportsModel(modelId);
      }

      const normalizedModelId = modelId.toLowerCase();
      return normalizedModelId.includes('medasr')
        || normalizedModelId.includes('wav2vec')
        || normalizedModelId.includes('ctc');
    },
    matchesClassification(classification: Partial<ModelClassification>): boolean {
      if (options.supportsModel) {
        return options.supportsModel('', classification);
      }
      return classificationContains(factoryClassification, classification);
    },
    async createModel(
      request,
      context: SpeechModelFactoryContext
    ): Promise<HfCtcSpeechModel> {
      const classification = resolveClassification(factoryClassification, request.classification);
      const config = options.resolveConfig
        ? options.resolveConfig(request.modelId, request)
        : parseHfCtcConfig(request.modelId, request.options?.config);

      context.hooks.logger?.info?.('Creating HF CTC scaffold model', {
        family,
        modelId: request.modelId,
        backendId: context.backend.id
      });

      return new HfCtcSpeechModel(
        context.backend,
        family,
        request.modelId,
        classification,
        config,
        request.options,
        options.dependencies ?? {},
        options.describeModel ?? describeHfCtcModel
      );
    }
  };
}

import { normalizePcmInput } from '../../processors/index.js';
import {
  TRANSFORMER_SEQ2SEQ_DECODER,
  WHISPER_GENERATE_DECODING,
  WHISPER_TRANSFORMER_ENCODER
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
  TranscriptResponseFlavor
} from '../../types/index.js';
import { createModelArchitecture } from '../../types/index.js';
import {
  DEFAULT_WHISPER_CLASSIFICATION,
  describeWhisperSeq2SeqModel,
  parseWhisperSeq2SeqConfig
} from './config.js';
import { mapWhisperNativeToCanonical } from './mapping.js';
import type {
  WhisperNativeSegment,
  WhisperNativeTranscript,
  WhisperSeq2SeqModelConfig,
  WhisperSeq2SeqModelDependencies,
  WhisperSeq2SeqModelOptions,
  WhisperSeq2SeqTranscriptionOptions
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
    ...DEFAULT_WHISPER_CLASSIFICATION,
    ...base,
    ...request
  };
}

function buildStubSegments(
  durationSeconds: number,
  task: WhisperSeq2SeqTranscriptionOptions['task']
): WhisperNativeSegment[] {
  const texts = task === 'translate'
    ? ['Translated', 'Whisper', 'scaffold']
    : ['Whisper', 'seq2seq', 'scaffold'];
  const span = Math.max(durationSeconds, 0.3) / texts.length;

  return texts.map((text, index) => ({
    index,
    text,
    startTime: Number((index * span).toFixed(3)),
    endTime: Number(((index + 1) * span).toFixed(3)),
    confidence: 0.9
  }));
}

class WhisperSeq2SeqSpeechSession
  implements SpeechSession<WhisperSeq2SeqTranscriptionOptions, WhisperNativeTranscript> {
  private readonly tokenizer;

  constructor(
    private readonly modelId: string,
    private readonly classification: ModelClassification,
    private readonly config: WhisperSeq2SeqModelConfig,
    private readonly backendId: string,
    dependencies: WhisperSeq2SeqModelDependencies = {}
  ) {
    this.tokenizer = dependencies.tokenizer ?? new StubTextTokenizer('tiktoken', 'wh');
  }

  async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options: WhisperSeq2SeqTranscriptionOptions & { readonly responseFlavor?: TFlavor } = {}
  ): Promise<TranscriptResponse<WhisperNativeTranscript, TFlavor>> {
    const audio = normalizePcmInput(input).toMono();
    const segments = buildStubSegments(audio.durationSeconds, options.task);
    const tokens = segments.map((segment, index) => ({
      index,
      id: index + 100,
      text: segment.text,
      startTime: segment.startTime,
      endTime: segment.endTime,
      confidence: segment.confidence,
      special: false
    }));
    const nativeTranscript: WhisperNativeTranscript = {
      utteranceText: segments.map((segment) => segment.text).join(' '),
      isFinal: true,
      language: options.language ?? this.config.languages[0],
      segments,
      tokens: options.returnSpecialTokens ? tokens : tokens.filter((token) => !token.special),
      warnings: [{
        code: 'whisper-seq2seq.stubbed-decoder',
        message: 'Whisper seq2seq execution is scaffolded. Integrate encoder/decoder generation to replace the stub.'
      }]
    };

    const canonical = mapWhisperNativeToCanonical(nativeTranscript, this.classification, {
      detailLevel: options.detail,
      backendId: this.backendId,
      modelId: this.modelId,
      sampleRate: audio.sampleRate,
      durationSeconds: audio.durationSeconds,
      language: nativeTranscript.language
    });
    const responseFlavor = options.responseFlavor ?? 'canonical';

    if (responseFlavor === 'native') {
      return nativeTranscript as TranscriptResponse<WhisperNativeTranscript, TFlavor>;
    }
    if (responseFlavor === 'canonical+native') {
      return {
        canonical,
        native: nativeTranscript
      } as TranscriptResponse<WhisperNativeTranscript, TFlavor>;
    }

    return canonical as TranscriptResponse<WhisperNativeTranscript, TFlavor>;
  }

  dispose(): void {
    void this.tokenizer;
  }
}

class WhisperSeq2SeqSpeechModel
  implements SpeechModel<WhisperSeq2SeqModelOptions, WhisperSeq2SeqTranscriptionOptions, WhisperNativeTranscript> {
  readonly info;
  readonly loadOptions?: WhisperSeq2SeqModelOptions;

  constructor(
    readonly backend: SpeechModel<WhisperSeq2SeqModelOptions, WhisperSeq2SeqTranscriptionOptions, WhisperNativeTranscript>['backend'],
    readonly family: string,
    readonly modelId: string,
    readonly classification: ModelClassification,
    readonly config: WhisperSeq2SeqModelConfig,
    loadOptions: WhisperSeq2SeqModelOptions | undefined,
    private readonly dependencies: WhisperSeq2SeqModelDependencies,
    private readonly describeModel: (
      modelId: string,
      classification: ModelClassification,
      config: WhisperSeq2SeqModelConfig
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
          module: WHISPER_TRANSFORMER_ENCODER.sharedModule,
          implementation: config.encoderArchitecture,
          shared: true,
          notes: [`Max source positions ${config.maxSourcePositions}.`]
        },
        decoder: {
          layer: 'decoder',
          module: TRANSFORMER_SEQ2SEQ_DECODER.sharedModule,
          implementation: TRANSFORMER_SEQ2SEQ_DECODER.kind,
          shared: true
        },
        decoding: {
          layer: 'decoding',
          module: 'inference',
          implementation: WHISPER_GENERATE_DECODING.strategy,
          shared: true,
          notes: WHISPER_GENERATE_DECODING.notes
        },
        tokenizer: {
          layer: 'tokenizer',
          module: 'inference',
          implementation: config.tokenizer.kind,
          shared: true
        }
      }),
      description: describeModel(modelId, classification, config),
      nativeOutputName: 'WhisperNativeTranscript'
    };
  }

  async createSession(_options: BaseSessionOptions = {}): Promise<SpeechSession<WhisperSeq2SeqTranscriptionOptions, WhisperNativeTranscript>> {
    return new WhisperSeq2SeqSpeechSession(this.modelId, this.classification, this.config, this.backend.id, this.dependencies);
  }

  dispose(): void {
    return undefined;
  }
}

export interface CreateWhisperSeq2SeqModelFamilyOptions {
  readonly dependencies?: WhisperSeq2SeqModelDependencies;
  readonly family?: string;
  readonly classification?: Partial<ModelClassification>;
  readonly supportsModel?: (modelId: string, classification?: Partial<ModelClassification>) => boolean;
  readonly resolveConfig?: (
    modelId: string,
    request: ModelLoadRequest<WhisperSeq2SeqModelOptions>
  ) => WhisperSeq2SeqModelConfig;
  readonly describeModel?: (
    modelId: string,
    classification: ModelClassification,
    config: WhisperSeq2SeqModelConfig
  ) => string;
}

export function createWhisperSeq2SeqModelFamily(
  options: CreateWhisperSeq2SeqModelFamilyOptions = {}
): SpeechModelFactory<WhisperSeq2SeqModelOptions, WhisperSeq2SeqTranscriptionOptions, WhisperNativeTranscript> {
  const family = options.family ?? 'whisper-seq2seq';
  const factoryClassification = resolveClassification(options.classification);

  return {
    family,
    classification: factoryClassification,
    supports(modelId: string): boolean {
      if (options.supportsModel) {
        return options.supportsModel(modelId);
      }

      return modelId.toLowerCase().includes('whisper');
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
    ): Promise<WhisperSeq2SeqSpeechModel> {
      const classification = resolveClassification(factoryClassification, request.classification);
      const config = options.resolveConfig
        ? options.resolveConfig(request.modelId, request)
        : parseWhisperSeq2SeqConfig(request.modelId, request.options?.config);

      context.hooks.logger?.info?.('Creating Whisper seq2seq scaffold model', {
        family,
        modelId: request.modelId,
        backendId: context.backend.id
      });

      return new WhisperSeq2SeqSpeechModel(
        context.backend,
        family,
        request.modelId,
        classification,
        config,
        request.options,
        options.dependencies ?? {},
        options.describeModel ?? describeWhisperSeq2SeqModel
      );
    }
  };
}

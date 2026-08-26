import { normalizePcmInput } from '../../audio/index.js';
import { CONFORMER_ENCODER, CTC_GREEDY_DECODING, CTC_HEAD_DECODER } from '../../inference/index.js';
import type {
  AudioInputLike,
  BaseSessionOptions,
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
import { OrtSenseVoiceExecutor } from './executor.js';
import { mapSenseVoiceNativeToCanonical } from './mapping.js';
import type {
  SenseVoiceExecutor,
  SenseVoiceModelConfig,
  SenseVoiceModelDependencies,
  SenseVoiceModelOptions,
  SenseVoiceNativeTranscript,
  SenseVoiceTranscriptionOptions,
} from './types.js';

const DEFAULT_CONFIG: SenseVoiceModelConfig = {
  ecosystem: 'funasr',
  architecture: 'sensevoice',
  processorArchitecture: 'kaldi-fbank',
  encoderArchitecture: 'sensevoice-conformer',
  decoderArchitecture: 'ctc',
  sampleRate: 16000,
  featureHopSeconds: 0.01,
  nMels: 80,
  vocabularySize: 25055,
  ctcBlankId: 0,
  languages: ['auto', 'zh', 'en', 'yue', 'ja', 'ko'],
};

const DEFAULT_CLASSIFICATION: ModelClassification = {
  family: 'sensevoice',
  ecosystem: 'funasr',
  processor: 'kaldi-mel',
  encoder: 'conformer',
  decoder: 'ctc',
  topology: 'ctc',
  task: 'asr',
};

function inferenceLimits(): ModelInferenceLimits {
  return {
    sampleRate: 16000,
    maxInputDurationSec: 30,
    recommendedWindowDurationSec: 15,
    minWindowDurationSec: 1,
    maxWindowDurationSec: 30,
    autoWindowThresholdSec: 30,
    defaultOverlapSec: 0,
    supportsWordTimestamps: false,
    supportsTokenTimestamps: true,
    supportsSegmentTimestamps: true,
    supportsConfidence: true,
    defaultSegmentationStrategy: 'ctc-frame',
    defaultMergeStrategy: 'ctc-collapse',
  };
}

class SenseVoiceSession implements SpeechSession<SenseVoiceTranscriptionOptions, SenseVoiceNativeTranscript> {
  private disposed = false;
  constructor(
    private readonly modelId: string,
    private readonly classification: ModelClassification,
    _config: SenseVoiceModelConfig,
    private readonly backendId: string,
    private readonly executor: SenseVoiceExecutor,
    private readonly onDispose?: () => void,
  ) {}

  async initialize(): Promise<void> {
    await this.executor.ready?.();
  }

  async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options: SenseVoiceTranscriptionOptions & { readonly responseFlavor?: TFlavor } = {},
  ): Promise<TranscriptResponse<SenseVoiceNativeTranscript, TFlavor>> {
    const audio = normalizePcmInput(input).toMono();
    const native = await this.executor.transcribe(audio, options);
    const canonical = mapSenseVoiceNativeToCanonical(native, this.classification, {
      detailLevel: options.detail,
      backendId: this.backendId,
      modelId: this.modelId,
      language: native.language,
      sampleRate: audio.sampleRate,
      durationSeconds: audio.durationSeconds,
      metrics: native.metrics,
    });
    const flavor = options.responseFlavor ?? 'canonical';
    if (flavor === 'native') return native as TranscriptResponse<SenseVoiceNativeTranscript, TFlavor>;
    if (flavor === 'canonical+native') return { canonical, native } as TranscriptResponse<SenseVoiceNativeTranscript, TFlavor>;
    return canonical as TranscriptResponse<SenseVoiceNativeTranscript, TFlavor>;
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await this.executor.dispose();
    this.onDispose?.();
  }
}

class SenseVoiceModel implements SpeechModel<SenseVoiceModelOptions, SenseVoiceTranscriptionOptions, SenseVoiceNativeTranscript> {
  readonly loadOptions?: SenseVoiceModelOptions;
  readonly info;
  private readonly sessions = new Set<SenseVoiceSession>();
  private disposed = false;

  constructor(
    readonly backend: SpeechModel<SenseVoiceModelOptions, SenseVoiceTranscriptionOptions, SenseVoiceNativeTranscript>['backend'],
    readonly family: string,
    readonly modelId: string,
    readonly classification: ModelClassification,
    readonly config: SenseVoiceModelConfig,
    readonly resolvedPreset: string | undefined,
    loadOptions: SenseVoiceModelOptions | undefined,
    private readonly dependencies: SenseVoiceModelDependencies,
  ) {
    this.loadOptions = loadOptions;
    this.info = {
      family,
      modelId,
      classification,
      preset: resolvedPreset,
      architecture: createModelArchitecture({
        processor: { layer: 'processor', module: 'audio', implementation: 'kaldi-fbank', shared: false },
        encoder: { layer: 'encoder', module: CONFORMER_ENCODER.sharedModule, implementation: config.encoderArchitecture, shared: false },
        decoder: { layer: 'decoder', module: CTC_HEAD_DECODER.sharedModule, implementation: CTC_HEAD_DECODER.kind, shared: true },
        decoding: { layer: 'decoding', module: 'inference', implementation: CTC_GREEDY_DECODING.strategy, shared: true },
        tokenizer: { layer: 'tokenizer', module: 'inference', implementation: 'sentencepiece', shared: false },
      }),
      description: `SenseVoiceSmall non-autoregressive CTC family for ${modelId}.`,
      nativeOutputName: 'SenseVoiceNativeTranscript',
      inference: inferenceLimits(),
    };
  }

  async createSession(_options: BaseSessionOptions = {}): Promise<SpeechSession<SenseVoiceTranscriptionOptions, SenseVoiceNativeTranscript>> {
    const executor = this.dependencies.executor ?? new OrtSenseVoiceExecutor(this.modelId, this.backend.id, this.loadOptions);
    const session = new SenseVoiceSession(this.modelId, this.classification, this.config, this.backend.id, executor, () => this.sessions.delete(session));
    this.sessions.add(session);
    await session.initialize();
    return session;
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await Promise.all([...this.sessions].map((session) => session.dispose()));
    this.sessions.clear();
  }
}

export function createSenseVoiceModelFamily(): SpeechModelFactory<SenseVoiceModelOptions, SenseVoiceTranscriptionOptions, SenseVoiceNativeTranscript> {
  return {
    family: 'sensevoice',
    classification: DEFAULT_CLASSIFICATION,
    supports(modelId: string): boolean {
      const normalized = modelId.toLowerCase();
      return normalized.includes('sensevoice');
    },
    matchesClassification(classification: Partial<ModelClassification>): boolean {
      return Object.entries(classification).every(([key, value]) => DEFAULT_CLASSIFICATION[key as keyof ModelClassification] === value);
    },
    async createModel(request, context: SpeechModelFactoryContext): Promise<SpeechModel<SenseVoiceModelOptions, SenseVoiceTranscriptionOptions, SenseVoiceNativeTranscript>> {
      const classification = { ...DEFAULT_CLASSIFICATION, ...(request.classification ?? {}) };
      const config = { ...DEFAULT_CONFIG, ...(request.options?.config ?? {}) };
      const dependencies: SenseVoiceModelDependencies = {
        assetProvider: context.assetProvider,
        runtimeHooks: context.hooks,
      };
      context.hooks.logger?.info?.('Creating SenseVoice CTC model', {
        family: 'sensevoice', modelId: request.modelId, backendId: context.backend.id,
        artifactSource: request.options?.source?.kind ?? 'none',
      });
      return new SenseVoiceModel(context.backend, 'sensevoice', request.modelId, classification, config, request.resolvedPreset, request.options, dependencies);
    },
  };
}

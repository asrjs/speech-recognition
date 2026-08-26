import { normalizePcmInput } from '../../audio/index.js';
import { CTC_GREEDY_DECODING, CTC_HEAD_DECODER, CONFORMER_ENCODER } from '../../inference/index.js';
import { mapLasrCtcNativeToCanonical } from '../lasr-ctc/mapping.js';
import type {
  AudioInputLike,
  BaseSessionOptions,
  ModelClassification,
  SpeechModel,
  SpeechModelFactory,
  SpeechSession,
  TranscriptResponse,
  TranscriptResponseFlavor,
} from '../../types/index.js';
import { createModelArchitecture } from '../../types/index.js';
import { OrtGigaAmCtcExecutor } from './executor.js';
import type { GigaAmModelConfig, GigaAmModelFamilyOptions, GigaAmModelOptions } from './types.js';
import type { LasrCtcNativeTranscript, LasrCtcTranscriptionOptions } from '../lasr-ctc/types.js';

const DEFAULT_CONFIG: GigaAmModelConfig = {
  ecosystem: 'gigaam', architecture: 'gigaam-ctc', processorArchitecture: 'gigaam-fbank',
  encoderArchitecture: 'gigaam-conformer', decoderArchitecture: 'ctc', sampleRate: 16000,
  rawStride: 4, nMels: 64, featureHopSeconds: 0.01, vocabularySize: 71,
  languages: ['ru', 'en', 'kk', 'ky', 'uz'],
  tokenizer: { kind: 'sentencepiece', blankTokenId: 70 },
  nFft: 320, winLength: 320, hopLength: 160, featureLayout: 'mel-major',
};

const CLASSIFICATION: ModelClassification = {
  family: 'gigaam-ctc', ecosystem: 'gigaam', processor: 'gigaam-fbank', encoder: 'conformer', decoder: 'ctc', topology: 'ctc', task: 'asr',
};

class GigaAmSession implements SpeechSession<LasrCtcTranscriptionOptions, LasrCtcNativeTranscript> {
  private disposed = false;
  constructor(private readonly modelId: string, private readonly backendId: string, private readonly executor: OrtGigaAmCtcExecutor, private readonly onDispose: () => void) {}
  async initialize(): Promise<void> { await this.executor.ready(); }
  async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(input: AudioInputLike, options: LasrCtcTranscriptionOptions & { readonly responseFlavor?: TFlavor } = {}): Promise<TranscriptResponse<LasrCtcNativeTranscript, TFlavor>> {
    const audio = normalizePcmInput(input).toMono();
    const native = await this.executor.transcribe(audio, options);
    const canonical = mapLasrCtcNativeToCanonical(native, CLASSIFICATION, { detailLevel: options.detail, backendId: this.backendId, modelId: this.modelId, language: undefined, sampleRate: audio.sampleRate, durationSeconds: audio.durationSeconds, metrics: native.metrics });
    const flavor = options.responseFlavor ?? 'canonical';
    if (flavor === 'native') return native as TranscriptResponse<LasrCtcNativeTranscript, TFlavor>;
    if (flavor === 'canonical+native') return { canonical, native } as TranscriptResponse<LasrCtcNativeTranscript, TFlavor>;
    return canonical as TranscriptResponse<LasrCtcNativeTranscript, TFlavor>;
  }
  async dispose(): Promise<void> { if (this.disposed) return; this.disposed = true; this.executor.dispose(); this.onDispose(); }
}

class GigaAmModel implements SpeechModel<GigaAmModelOptions, LasrCtcTranscriptionOptions, LasrCtcNativeTranscript> {
  readonly loadOptions?: GigaAmModelOptions;
  readonly info;
  private readonly sessions = new Set<GigaAmSession>();
  private disposed = false;
  constructor(readonly backend: SpeechModel<GigaAmModelOptions, LasrCtcTranscriptionOptions, LasrCtcNativeTranscript>['backend'], readonly family: string, readonly modelId: string, readonly config: GigaAmModelConfig, loadOptions: GigaAmModelOptions | undefined, private readonly dependencies: NonNullable<GigaAmModelFamilyOptions['dependencies']>) {
    this.loadOptions = loadOptions;
    this.info = { family, modelId, classification: CLASSIFICATION, architecture: createModelArchitecture({ processor: { layer: 'processor', module: 'audio', implementation: config.processorArchitecture, shared: false }, encoder: { layer: 'encoder', module: CONFORMER_ENCODER.sharedModule, implementation: config.encoderArchitecture, shared: false }, decoder: { layer: 'decoder', module: CTC_HEAD_DECODER.sharedModule, implementation: 'ctc', shared: true }, decoding: { layer: 'decoding', module: 'inference', implementation: CTC_GREEDY_DECODING.strategy, shared: true }, tokenizer: { layer: 'tokenizer', module: 'inference', implementation: 'character', shared: false } }), description: `GigaAM Multilingual character CTC model for ${modelId}.`, nativeOutputName: 'LasrCtcNativeTranscript' };
  }
  async createSession(_options: BaseSessionOptions = {}): Promise<SpeechSession<LasrCtcTranscriptionOptions, LasrCtcNativeTranscript>> {
    const executor = this.dependencies.executor ?? new OrtGigaAmCtcExecutor(this.modelId, this.backend.id, this.config, this.loadOptions, { assetProvider: this.dependencies.assetProvider, runtimeHooks: this.dependencies.runtimeHooks });
    const session = new GigaAmSession(this.modelId, this.backend.id, executor, () => this.sessions.delete(session));
    this.sessions.add(session); await session.initialize(); return session;
  }
  async dispose(): Promise<void> { if (this.disposed) return; this.disposed = true; await Promise.all([...this.sessions].map((session) => session.dispose())); this.sessions.clear(); }
}

export function createGigaAmCtcModelFamily(options: GigaAmModelFamilyOptions = {}): SpeechModelFactory<GigaAmModelOptions, LasrCtcTranscriptionOptions, LasrCtcNativeTranscript> {
  return {
    family: 'gigaam-ctc', classification: CLASSIFICATION,
    supports(modelId: string): boolean { const value = modelId.toLowerCase(); return value.includes('gigaam') && value.includes('ctc'); },
    matchesClassification(classification): boolean { return Object.entries(classification).every(([key, value]) => CLASSIFICATION[key as keyof ModelClassification] === value); },
    async createModel(request, context) {
      const dependencies = { ...(options.dependencies ?? {}), assetProvider: options.dependencies?.assetProvider ?? context.assetProvider, runtimeHooks: options.dependencies?.runtimeHooks ?? context.hooks };
      return new GigaAmModel(context.backend, 'gigaam-ctc', request.modelId, { ...DEFAULT_CONFIG, ...(request.options?.config ?? {}) }, request.options, dependencies);
    },
  };
}

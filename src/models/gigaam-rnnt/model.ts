import { normalizePcmInput } from '../../audio/index.js';
import { CONFORMER_ENCODER, RNNT_GREEDY_DECODING, RNNT_TRANSDUCER_DECODER } from '../../inference/index.js';
import type { AudioInputLike, BaseSessionOptions, ModelClassification, SpeechModel, SpeechModelFactory, SpeechSession, TranscriptResponse, TranscriptResponseFlavor } from '../../types/index.js';
import { createModelArchitecture } from '../../types/index.js';
import { mapLasrCtcNativeToCanonical } from '../lasr-ctc/mapping.js';
import { OrtGigaAmRnntExecutor } from './executor.js';
import type { GigaAmRnntModelConfig, GigaAmRnntModelFamilyOptions, GigaAmRnntModelOptions, GigaAmRnntNativeTranscript, GigaAmRnntTranscriptionOptions } from './types.js';

const DEFAULT_CONFIG: GigaAmRnntModelConfig = {
  ecosystem: 'gigaam', architecture: 'gigaam-rnnt', processorArchitecture: 'gigaam-fbank', encoderArchitecture: 'gigaam-conformer', decoderArchitecture: 'rnnt', sampleRate: 16000, rawStride: 4, nMels: 64, featureHopSeconds: 0.01, vocabularySize: 1025, languages: ['ru'], tokenizer: { kind: 'sentencepiece', blankTokenId: 1024 }, nFft: 320, winLength: 320, hopLength: 160, featureLayout: 'mel-major', predictionHiddenSize: 320, predictionRnnLayers: 1, maxTokensPerFrame: 10,
};

const CLASSIFICATION: ModelClassification = { family: 'gigaam-rnnt', ecosystem: 'gigaam', processor: 'gigaam-fbank', encoder: 'conformer', decoder: 'rnnt', topology: 'rnnt', task: 'asr' };

class GigaAmRnntSession implements SpeechSession<GigaAmRnntTranscriptionOptions, GigaAmRnntNativeTranscript> {
  private disposed = false;
  constructor(private readonly modelId: string, private readonly backendId: string, private readonly executor: OrtGigaAmRnntExecutor, private readonly onDispose: () => void) {}
  async initialize(): Promise<void> { await this.executor.ready(); }
  async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(input: AudioInputLike, options: GigaAmRnntTranscriptionOptions & { readonly responseFlavor?: TFlavor } = {}): Promise<TranscriptResponse<GigaAmRnntNativeTranscript, TFlavor>> {
    const audio = normalizePcmInput(input).toMono(); const native = await this.executor.transcribe(audio, options); const canonical = mapLasrCtcNativeToCanonical(native, CLASSIFICATION, { detailLevel: options.detail, backendId: this.backendId, modelId: this.modelId, sampleRate: audio.sampleRate, durationSeconds: audio.durationSeconds, metrics: native.metrics }); const flavor = options.responseFlavor ?? 'canonical';
    if (flavor === 'native') return native as TranscriptResponse<GigaAmRnntNativeTranscript, TFlavor>;
    if (flavor === 'canonical+native') return { canonical, native } as TranscriptResponse<GigaAmRnntNativeTranscript, TFlavor>;
    return canonical as TranscriptResponse<GigaAmRnntNativeTranscript, TFlavor>;
  }
  async dispose(): Promise<void> { if (this.disposed) return; this.disposed = true; await Promise.resolve(this.executor.dispose()); this.onDispose(); }
}

class GigaAmRnntModel implements SpeechModel<GigaAmRnntModelOptions, GigaAmRnntTranscriptionOptions, GigaAmRnntNativeTranscript> {
  readonly loadOptions?: GigaAmRnntModelOptions;
  readonly info;
  private readonly sessions = new Set<GigaAmRnntSession>();
  private disposed = false;
  constructor(readonly backend: SpeechModel<GigaAmRnntModelOptions, GigaAmRnntTranscriptionOptions, GigaAmRnntNativeTranscript>['backend'], readonly family: string, readonly modelId: string, readonly config: GigaAmRnntModelConfig, loadOptions: GigaAmRnntModelOptions | undefined, private readonly dependencies: NonNullable<GigaAmRnntModelFamilyOptions['dependencies']>) {
    this.loadOptions = loadOptions;
    this.info = { family, modelId, classification: CLASSIFICATION, architecture: createModelArchitecture({ processor: { layer: 'processor', module: 'audio', implementation: config.processorArchitecture, shared: false }, encoder: { layer: 'encoder', module: CONFORMER_ENCODER.sharedModule, implementation: config.encoderArchitecture, shared: false }, decoder: { layer: 'decoder', module: RNNT_TRANSDUCER_DECODER.sharedModule, implementation: 'rnnt', shared: false }, decoding: { layer: 'decoding', module: 'inference', implementation: RNNT_GREEDY_DECODING.strategy, shared: true }, tokenizer: { layer: 'tokenizer', module: 'inference', implementation: 'character', shared: false } }), description: `GigaAM v3 end-to-end RNN-T model for ${modelId}.`, nativeOutputName: 'GigaAmRnntNativeTranscript' };
  }
  async createSession(_options: BaseSessionOptions = {}): Promise<SpeechSession<GigaAmRnntTranscriptionOptions, GigaAmRnntNativeTranscript>> { const executor = this.dependencies.executor ?? new OrtGigaAmRnntExecutor(this.modelId, this.backend.id, this.config, this.loadOptions, { assetProvider: this.dependencies.assetProvider, runtimeHooks: this.dependencies.runtimeHooks, signal: this.dependencies.signal }); const session = new GigaAmRnntSession(this.modelId, this.backend.id, executor, () => this.sessions.delete(session)); this.sessions.add(session); await session.initialize(); return session; }
  async dispose(): Promise<void> { if (this.disposed) return; this.disposed = true; await Promise.all([...this.sessions].map((session) => session.dispose())); this.sessions.clear(); }
}

export function createGigaAmRnntModelFamily(options: GigaAmRnntModelFamilyOptions = {}): SpeechModelFactory<GigaAmRnntModelOptions, GigaAmRnntTranscriptionOptions, GigaAmRnntNativeTranscript> {
  return { family: 'gigaam-rnnt', classification: CLASSIFICATION, supports(modelId: string): boolean { const value = modelId.toLowerCase(); return value.includes('gigaam') && (value.includes('rnnt') || value.includes('rnn-t') || value.includes('e2e')); }, matchesClassification(classification): boolean { return Object.entries(classification).every(([key, value]) => CLASSIFICATION[key as keyof ModelClassification] === value); }, async createModel(request, context) { const dependencies = { ...(options.dependencies ?? {}), assetProvider: options.dependencies?.assetProvider ?? context.assetProvider, runtimeHooks: options.dependencies?.runtimeHooks ?? context.hooks, signal: options.dependencies?.signal ?? context.signal }; return new GigaAmRnntModel(context.backend, 'gigaam-rnnt', request.modelId, { ...DEFAULT_CONFIG, ...(request.options?.config ?? {}) }, request.options, dependencies); } };
}

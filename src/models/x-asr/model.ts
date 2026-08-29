import { normalizePcmInput } from '../../audio/index.js';
import { RNNT_GREEDY_DECODING, RNNT_TRANSDUCER_DECODER } from '../../inference/index.js';
import { TranscriptAccumulator } from '../../inference/streaming/accumulator.js';
import { joinTranscriptFragments } from '../../inference/streaming/merge.js';
import { mapXAsrNativeToCanonical } from './mapping.js';
import type {
  AudioInputLike,
  BaseSessionOptions,
  ModelClassification,
  PartialTranscript,
  SpeechModel,
  SpeechModelFactory,
  SpeechSession,
  StreamingSessionOptions,
  StreamingTranscriber,
  StreamingTranscriberState,
  TranscriptResponse,
  TranscriptResponseFlavor,
} from '../../types/index.js';
import { createModelArchitecture } from '../../types/index.js';
import { OrtXAsrExecutor, type XAsrExecutor, type XAsrStreamState } from './executor.js';
import type {
  XAsrModelConfig,
  XAsrModelDependencies,
  XAsrModelOptions,
  XAsrNativeTranscript,
  XAsrTranscriptionOptions,
} from './types.js';

const CLASSIFICATION: ModelClassification = {
  family: 'x-asr',
  ecosystem: 'x-asr',
  processor: 'kaldi-fbank',
  encoder: 'zipformer2',
  decoder: 'stateless-rnnt',
  topology: 'stateless-rnnt',
  task: 'asr',
};

/** State outputs for the official 19-layer X-ASR Zipformer2 graph family. */
export const DEFAULT_XASR_ENCODER_STATE_OUTPUTS = [
  ...Array.from({ length: 19 }, (_, layer) =>
    ['key', 'nonlin_attn', 'val1', 'val2', 'conv1', 'conv2'].map(
      (kind) => `new_cached_${kind}_${layer}`,
    ),
  ).flat(),
  'new_embed_states',
  'new_processed_lens',
] as const;

const DEFAULT_GRAPH = {
  encoderStateInputs: [],
  encoderStateOutputs: DEFAULT_XASR_ENCODER_STATE_OUTPUTS,
  encoderFrameSize: 29,
  encoderFrameShift: 16,
  decoderContextSize: 2,
  featureInputName: 'x',
  encoderOutputName: 'encoder_out',
  decoderInputName: 'y',
  decoderOutputName: 'decoder_out',
  joinerEncoderInputName: 'encoder_out',
  joinerDecoderInputName: 'decoder_out',
  joinerOutputName: 'logit',
} as const;
const DEFAULT_CONFIG: XAsrModelConfig = {
  ecosystem: 'x-asr',
  architecture: 'zipformer2-streaming-rnnt',
  processorArchitecture: 'kaldi-fbank',
  encoderArchitecture: 'zipformer2',
  decoderArchitecture: 'stateless-rnnt',
  sampleRate: 16000,
  featureDim: 80,
  featureHopSeconds: 0.01,
  rawStride: 1,
  languages: ['zh', 'en'],
  chunkMs: 160,
  graph: DEFAULT_GRAPH,
};

function canonical(
  native: XAsrNativeTranscript,
  modelId: string,
  backendId: string,
  sampleRate: number,
  durationSeconds: number,
  detail: XAsrTranscriptionOptions['detail'],
) {
  return mapXAsrNativeToCanonical(native, CLASSIFICATION, {
    detailLevel: detail,
    backendId,
    modelId,
    sampleRate,
    durationSeconds,
    metrics: native.metrics,
  });
}

class XAsrSpeechSession implements SpeechSession<XAsrTranscriptionOptions, XAsrNativeTranscript> {
  private disposed = false;
  constructor(
    private readonly modelId: string,
    private readonly backendId: string,
    private readonly config: XAsrModelConfig,
    private readonly executor: XAsrExecutor,
    private readonly onDispose: () => void,
  ) {}
  async initialize(): Promise<void> {
    await this.executor.ready();
  }
  async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options: XAsrTranscriptionOptions & { readonly responseFlavor?: TFlavor } = {},
  ): Promise<TranscriptResponse<XAsrNativeTranscript, TFlavor>> {
    const audio = normalizePcmInput(input).toMono();
    const native = await this.executor.transcribe(audio, options);
    const value =
      options.responseFlavor === 'native'
        ? native
        : canonical(
            native,
            this.modelId,
            this.backendId,
            audio.sampleRate,
            audio.durationSeconds,
            options.detail,
          );
    if (options.responseFlavor === 'canonical+native')
      return { canonical: value as never, native } as unknown as TranscriptResponse<
        XAsrNativeTranscript,
        TFlavor
      >;
    return value as TranscriptResponse<XAsrNativeTranscript, TFlavor>;
  }
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await Promise.resolve(this.executor.dispose());
    this.onDispose();
  }
  createStreamingTranscriber(options: StreamingSessionOptions = {}): StreamingTranscriber {
    return new XAsrStreamingTranscriber(
      this.modelId,
      this.backendId,
      this.config,
      this.executor,
      options,
      () => this.dispose(),
    );
  }
}

class XAsrStreamingTranscriber implements StreamingTranscriber {
  private stream: XAsrStreamState;
  private readonly accumulator = new TranscriptAccumulator();
  private readonly detail: NonNullable<StreamingSessionOptions['detail']>;
  private readonly emitPartials: boolean;
  private finalized = false;
  private durationSeconds = 0;
  private revision = 0;
  private disposed = false;
  private generation = 0;
  private operationAbortController = new AbortController();
  private operationTail: Promise<void> = Promise.resolve();
  private readonly retiredStreams: XAsrStreamState[] = [];
  constructor(
    private readonly modelId: string,
    private readonly backendId: string,
    private readonly config: XAsrModelConfig,
    private readonly executor: XAsrExecutor,
    options: StreamingSessionOptions,
    private readonly onDispose?: () => Promise<void> | void,
  ) {
    this.detail = options.detail ?? 'segments';
    this.emitPartials = options.emitPartials ?? true;
    this.stream = executor.createStream();
  }
  async pushAudio(input: AudioInputLike): Promise<PartialTranscript> {
    const generation = this.generation;
    return this.enqueue(() => {
      this.assertOpen();
      if (generation !== this.generation) return this.staleUpdate();
      const audio = normalizePcmInput(input).toMono();
      this.durationSeconds += audio.durationSeconds;
      const pcm = audio.channels?.[0] ?? new Float32Array(0);
      if (!this.emitPartials) return this.blank('partial');
      return this.transcribeStream(pcm, false, generation);
    });
  }
  async flush(): Promise<PartialTranscript> {
    const generation = this.generation;
    return this.enqueue(() => {
      this.assertOpen();
      return this.transcribeStream(new Float32Array(0), false, generation);
    });
  }
  async finalize(): Promise<PartialTranscript> {
    const generation = this.generation;
    return this.enqueue(async () => {
      this.assertOpen();
      const update = await this.transcribeStream(new Float32Array(0), true, generation);
      if (generation !== this.generation) return update;
      this.finalized = true;
      return update;
    });
  }
  async reset(): Promise<void> {
    this.assertOpen();
    this.generation += 1;
    this.operationAbortController.abort();
    this.operationAbortController = new AbortController();
    this.retiredStreams.push(this.stream);
    this.stream = this.executor.createStream();
    this.accumulator.reset();
    this.durationSeconds = 0;
    this.revision = 0;
    this.finalized = false;
    void this.operationTail.then(() => this.disposeRetiredStreams());
  }
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    this.generation += 1;
    this.operationAbortController.abort();
    await this.operationTail;
    this.disposeRetiredStreams();
    this.executor.disposeStream(this.stream);
    await this.onDispose?.();
  }
  getState(): StreamingTranscriberState {
    const state = this.accumulator.getState();
    return {
      revision: state.revision,
      bufferedDurationSeconds: this.durationSeconds,
      committedText: state.committedText,
      previewText: state.previewText,
      isFinalized: this.finalized,
    };
  }
  private async transcribeStream(
    audio: Float32Array,
    final: boolean,
    generation: number,
  ): Promise<PartialTranscript> {
    if (generation !== this.generation) return this.staleUpdate();
    const signal = this.operationAbortController.signal;
    let result: Awaited<ReturnType<XAsrExecutor['pushStream']>>;
    try {
      result = await this.executor.pushStream(this.stream, audio, final, {
        detail: this.detail,
        signal,
      });
    } catch (error) {
      if (generation !== this.generation || signal.aborted) return this.staleUpdate();
      throw error;
    }
    if (generation !== this.generation) return this.staleUpdate();
    this.stream = result.state;
    const update = this.update(result);
    if (final) this.executor.disposeStream(result.state);
    return final ? { ...update, kind: 'final' } : update;
  }
  private staleUpdate(): PartialTranscript {
    const state = this.accumulator.getState();
    return {
      kind: 'partial',
      revision: state.revision,
      text: joinTranscriptFragments(state.committedText, state.previewText),
      committedText: state.committedText,
      previewText: state.previewText,
      warnings: [],
      meta: {
        detailLevel: this.detail,
        isFinal: false,
        modelFamily: 'x-asr',
        modelId: this.modelId,
        durationSeconds: this.durationSeconds,
      },
    };
  }
  private disposeRetiredStreams(): void {
    while (this.retiredStreams.length > 0) this.executor.disposeStream(this.retiredStreams.pop()!);
  }
  private enqueue<T>(operation: () => Promise<T> | T): Promise<T> {
    const run = this.operationTail.then(operation, operation);
    this.operationTail = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }
  private update(
    result: Awaited<ReturnType<XAsrExecutor['pushStream']>>,
    kind: 'partial' | 'final' = 'partial',
  ): PartialTranscript {
    this.stream = result.state;
    const audioSeconds = this.durationSeconds;
    const value = canonical(
      result.transcript,
      this.modelId,
      this.backendId,
      this.config.sampleRate,
      audioSeconds,
      this.detail,
    );
    return this.accumulator.update(value, kind);
  }
  private blank(kind: 'partial' | 'final'): PartialTranscript {
    this.revision += 1;
    return {
      kind,
      revision: this.revision,
      text: '',
      committedText: '',
      previewText: '',
      warnings: [],
      meta: {
        detailLevel: this.detail,
        isFinal: kind === 'final',
        modelFamily: 'x-asr',
        modelId: this.modelId,
        durationSeconds: this.durationSeconds,
      },
    };
  }
  private assertOpen(): void {
    if (this.disposed) throw new Error('Streaming transcriber is disposed.');
    if (this.finalized)
      throw new Error('Streaming transcriber is finalized. Call reset() before pushing new audio.');
  }
}

class XAsrSpeechModel implements SpeechModel<
  XAsrModelOptions,
  XAsrTranscriptionOptions,
  XAsrNativeTranscript
> {
  readonly loadOptions?: XAsrModelOptions;
  readonly info;
  private readonly sessions = new Set<XAsrSpeechSession>();
  private disposed = false;
  constructor(
    readonly backend: SpeechModel<
      XAsrModelOptions,
      XAsrTranscriptionOptions,
      XAsrNativeTranscript
    >['backend'],
    readonly family: string,
    readonly modelId: string,
    readonly config: XAsrModelConfig,
    loadOptions: XAsrModelOptions | undefined,
    private readonly dependencies: XAsrModelDependencies,
  ) {
    this.loadOptions = loadOptions;
    this.info = {
      family,
      modelId,
      classification: CLASSIFICATION,
      architecture: createModelArchitecture({
        processor: {
          layer: 'processor',
          module: 'audio',
          implementation: config.processorArchitecture,
          shared: false,
        },
        encoder: {
          layer: 'encoder',
          module: 'inference',
          implementation: config.encoderArchitecture,
          shared: false,
          notes: ['Cache-aware streaming Zipformer2 encoder.'],
        },
        decoder: {
          layer: 'decoder',
          module: RNNT_TRANSDUCER_DECODER.sharedModule,
          implementation: 'stateless-rnnt',
          shared: false,
        },
        decoding: {
          layer: 'decoding',
          module: 'inference',
          implementation: RNNT_GREEDY_DECODING.strategy,
          shared: true,
          notes: ['Stateful encoder cache with greedy transducer decoding.'],
        },
        tokenizer: {
          layer: 'tokenizer',
          module: 'inference',
          implementation: 'bpe',
          shared: false,
        },
      }),
      description: `X-ASR ${config.chunkMs} ms streaming Zipformer2 transducer for ${modelId}.`,
      nativeOutputName: 'XAsrNativeTranscript',
    };
  }
  async createSession(
    _options: BaseSessionOptions = {},
  ): Promise<SpeechSession<XAsrTranscriptionOptions, XAsrNativeTranscript>> {
    const executor =
      this.dependencies.executor ??
      new OrtXAsrExecutor(this.modelId, this.backend.id, this.config, this.loadOptions, {
        assetProvider: this.dependencies.assetProvider,
        runtimeHooks: this.dependencies.runtimeHooks,
        signal: this.dependencies.signal,
      });
    const session = new XAsrSpeechSession(
      this.modelId,
      this.backend.id,
      this.config,
      executor,
      () => this.sessions.delete(session),
    );
    this.sessions.add(session);
    await session.initialize();
    return session;
  }
  async createStreamingTranscriber(
    options: StreamingSessionOptions = {},
  ): Promise<StreamingTranscriber> {
    const session = await this.createSession();
    return (session as XAsrSpeechSession).createStreamingTranscriber(options);
  }
  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    await Promise.all([...this.sessions].map((session) => session.dispose()));
    this.sessions.clear();
  }
}

export interface CreateXAsrModelFamilyOptions {
  readonly dependencies?: XAsrModelDependencies;
}

export function createXAsrModelFamily(
  options: CreateXAsrModelFamilyOptions = {},
): SpeechModelFactory<XAsrModelOptions, XAsrTranscriptionOptions, XAsrNativeTranscript> {
  return {
    family: 'x-asr',
    classification: CLASSIFICATION,
    supports(modelId) {
      const value = modelId.toLowerCase();
      return value.includes('x-asr') || value.includes('xasr');
    },
    matchesClassification(classification) {
      return Object.entries(classification).every(
        ([key, value]) => CLASSIFICATION[key as keyof ModelClassification] === value,
      );
    },
    async createModel(request, context) {
      const graph = {
        ...DEFAULT_GRAPH,
        ...(request.options?.config?.graph ?? {}),
      } as XAsrModelConfig['graph'];
      const config: XAsrModelConfig = {
        ...DEFAULT_CONFIG,
        ...(request.options?.config ?? {}),
        graph,
      };
      return new XAsrSpeechModel(
        context.backend,
        'x-asr',
        request.modelId,
        config,
        request.options,
        {
          ...(options.dependencies ?? {}),
          assetProvider: options.dependencies?.assetProvider ?? context.assetProvider,
          runtimeHooks: options.dependencies?.runtimeHooks ?? context.hooks,
          signal: options.dependencies?.signal ?? context.signal,
        },
      );
    },
  };
}

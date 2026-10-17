import { normalizePcmInput } from '../../audio/index.js';
import type { AudioInputLike } from '../../types/audio.js';
import type {
  BaseSessionOptions,
  BaseTranscriptionOptions,
  FamilyModelLoadRequest,
  ModelClassification,
  SpeechModel,
  SpeechModelFactory,
  SpeechModelFactoryContext,
  SpeechSession,
  TranscriptDetailLevel,
  TranscriptMetrics,
  TranscriptResponse,
  TranscriptResponseFlavor,
  TranscriptResult,
  TranscriptSegment,
  TranscriptWord,
} from '../../types/index.js';
import { createModelArchitecture } from '../../types/index.js';
import type { StreamingSessionOptions, StreamingTranscriber } from '../../types/streaming.js';
import { createExperimentalArtifactMissingError } from '../../runtime/experimental-families.js';
import {
  assertSessionOwnerActive,
  disposeOwnedResources,
  initializeOwnedSession,
} from '../../runtime/owned-session.js';
import * as ownedSessionModule from '../../runtime/owned-session.js';
import { initOrt, type OrtSessionLike } from '../lasr-ctc/ort.js';
import {
  MoonshineSession,
  MoonshineTokenizer,
  type MoonshineDecodeResult,
  type MoonshineSessionOptions,
} from './session.js';
import { loadMoonshineStreaming, type LoadedMoonshineStreaming } from './streaming-loader.js';
import type { MoonshineLoadOptions, MoonshineStreamingAssets } from './types.js';

const registerOwnedSpeechSession =
  (ownedSessionModule as { registerOwnedSpeechSession?: typeof initializeOwnedSession })
    .registerOwnedSpeechSession ?? initializeOwnedSession;

export const DEFAULT_MOONSHINE_CLASSIFICATION: ModelClassification = {
  ecosystem: 'moonshine',
  family: 'moonshine',
  encoder: 'conformer',
  decoder: 'transformer',
  topology: 'seq2seq',
  processor: 'raw-pcm',
  task: 'transcribe',
};

function classificationContains(
  candidate: Partial<ModelClassification>,
  requested: Partial<ModelClassification>,
): boolean {
  return Object.entries(requested).every(
    ([key, value]) => value === undefined || candidate[key as keyof ModelClassification] === value,
  );
}

function resolveClassification(
  base: Partial<ModelClassification> = {},
  request: Partial<ModelClassification> = {},
): ModelClassification {
  return { ...DEFAULT_MOONSHINE_CLASSIFICATION, ...base, ...request };
}

export function mapMoonshineResultToCanonical(
  decodeResult: MoonshineDecodeResult,
  options: {
    readonly modelId: string;
    readonly backendId: string;
    readonly durationSeconds: number;
    readonly sampleRate: number;
    readonly detailLevel: TranscriptDetailLevel;
    readonly metrics?: TranscriptMetrics;
  },
): TranscriptResult {
  const words: TranscriptWord[] =
    decodeResult.text.trim().length === 0
      ? []
      : decodeResult.text.trim().split(/\s+/).map((text, index, all) => {
          const span = options.durationSeconds / Math.max(all.length, 1);
          return {
            index,
            text,
            startTime: Number((index * span).toFixed(3)),
            endTime: Number(((index + 1) * span).toFixed(3)),
          };
        });

  const segments: TranscriptSegment[] =
    decodeResult.text.trim().length === 0
      ? []
      : [
          {
            index: 0,
            text: decodeResult.text,
            startTime: 0,
            endTime: Number(options.durationSeconds.toFixed(3)),
            wordIndices: words.map((_, i) => i),
          },
        ];

  const includeWords =
    options.detailLevel === 'words' ||
    options.detailLevel === 'sentences+words' ||
    options.detailLevel === 'detailed';

  const includeSegments =
    options.detailLevel === 'segments' ||
    options.detailLevel === 'detailed';

  return {
    text: decodeResult.text,
    warnings: [],
    meta: {
      modelFamily: 'moonshine',
      modelId: options.modelId,
      backendId: options.backendId,
      sampleRate: options.sampleRate,
      durationSeconds: options.durationSeconds,
      detailLevel: options.detailLevel,
      isFinal: true,
      tokenCount: decodeResult.tokenIds.length,
      wordCount: words.length,
      segmentCount: segments.length,
      ...(options.metrics ? { metrics: options.metrics } : {}),
    },
    ...(includeSegments ? { segments } : {}),
    ...(includeWords ? { words } : {}),
  };
}

export class MoonshineSpeechSession
  implements SpeechSession<BaseTranscriptionOptions, MoonshineDecodeResult> {
  private disposed = false;
  private offlineSession?: MoonshineSession;
  private streamingLoaded?: LoadedMoonshineStreaming;
  private readonly offlineOrtSessions: OrtSessionLike[] = [];

  constructor(
    private readonly modelId: string,
    private readonly backendId: string,
    private readonly loadOptions: MoonshineLoadOptions | undefined,
    private readonly signal: { readonly aborted: boolean } | null | undefined,
    private readonly onDispose?: () => void,
  ) {}

  async initialize(): Promise<void> {
    if (
      this.loadOptions?.crossKvUrl &&
      this.loadOptions?.decoderUrl &&
      this.loadOptions?.encoderUrl &&
      this.loadOptions?.tokenizerUrl
    ) {
      const rawOrt = await initOrt(this.backendId, {
        wasmPaths: this.loadOptions.wasmPaths,
        cpuThreads: this.loadOptions.cpuThreads,
        signal: this.signal,
      });
      const ort = rawOrt as unknown as MoonshineSessionOptions['ort'];

      // The 3-graph torch.onnx.export(dynamo=True) offline Moonshine graph uses
      // dynamic 0-length past tensors and attention masks that require WASM for
      // exact numerical parity, whereas the 4-graph streaming Moonshine export
      // runs natively on WebGPU.
      const sessionOptions: Record<string, unknown> = {
        executionProviders: ['wasm'],
        ...(this.loadOptions.cpuThreads !== undefined
          ? { intraOpNumThreads: this.loadOptions.cpuThreads }
          : {}),
        ...(this.signal ? { signal: this.signal } : {}),
      };

      const withExternalData = (
        base: Record<string, unknown>,
        sidecarPath: string,
        dataUrl?: string,
      ): Record<string, unknown> =>
        dataUrl ? { ...base, externalData: [{ path: sidecarPath, data: dataUrl }] } : base;

      const [encoderSession, crossKvSession, decoderSession] = await Promise.all([
        rawOrt.InferenceSession.create(
          this.loadOptions.encoderUrl,
          withExternalData(sessionOptions, 'encoder_model.onnx.data', this.loadOptions.encoderDataUrl),
        ),
        rawOrt.InferenceSession.create(
          this.loadOptions.crossKvUrl,
          withExternalData(sessionOptions, 'decoder_cross_kv_model.onnx.data', this.loadOptions.crossKvDataUrl),
        ),
        rawOrt.InferenceSession.create(
          this.loadOptions.decoderUrl,
          withExternalData(sessionOptions, 'decoder_with_past_model.onnx.data', this.loadOptions.decoderDataUrl),
        ),
      ]);
      this.offlineOrtSessions.push(encoderSession, crossKvSession, decoderSession);

      let tokenizerJson: unknown;
      if (this.loadOptions.tokenizerUrl.startsWith('file://')) {
        try {
          const fs = await import('node:fs/promises');
          const { fileURLToPath } = await import('node:url');
          const content = await fs.readFile(fileURLToPath(this.loadOptions.tokenizerUrl), 'utf-8');
          tokenizerJson = JSON.parse(content);
        } catch {
          const res = await fetch(this.loadOptions.tokenizerUrl);
          tokenizerJson = await res.json();
        }
      } else {
        const res = await fetch(this.loadOptions.tokenizerUrl);
        tokenizerJson = await res.json();
      }

      let slots = 8;
      let heads = 8;
      let headDim = 64;
      if (this.loadOptions.receiptUrl) {
        try {
          let receiptJson: {
            graphs?: {
              decoder?: { interface?: { slots?: number; heads?: number; headDim?: number } };
            };
          } | null = null;
          if (this.loadOptions.receiptUrl.startsWith('file://')) {
            const fs = await import('node:fs/promises');
            const { fileURLToPath } = await import('node:url');
            receiptJson = JSON.parse(
              await fs.readFile(fileURLToPath(this.loadOptions.receiptUrl), 'utf-8'),
            );
          } else {
            receiptJson = (await (await fetch(this.loadOptions.receiptUrl)).json()) as typeof receiptJson;
          }
          if (receiptJson?.graphs?.decoder?.interface) {
            slots = receiptJson.graphs.decoder.interface.slots ?? slots;
            heads = receiptJson.graphs.decoder.interface.heads ?? heads;
            headDim = receiptJson.graphs.decoder.interface.headDim ?? headDim;
          }
        } catch {
          // Fallback to defaults
        }
      }

      let bosTokenId = 1;
      let eosTokenId = 2;
      if (this.loadOptions.generationUrl) {
        try {
          let genJson: { bos_token_id?: number; eos_token_id?: number } | null = null;
          if (this.loadOptions.generationUrl.startsWith('file://')) {
            const fs = await import('node:fs/promises');
            const { fileURLToPath } = await import('node:url');
            genJson = JSON.parse(
              await fs.readFile(fileURLToPath(this.loadOptions.generationUrl), 'utf-8'),
            );
          } else {
            genJson = (await (await fetch(this.loadOptions.generationUrl)).json()) as typeof genJson;
          }
          bosTokenId = genJson?.bos_token_id ?? bosTokenId;
          eosTokenId = genJson?.eos_token_id ?? eosTokenId;
        } catch {
          // Fallback to defaults
        }
      }

      const tokenizer = MoonshineTokenizer.fromTokenizerJson(
        tokenizerJson as Parameters<typeof MoonshineTokenizer.fromTokenizerJson>[0],
        eosTokenId,
      );

      this.offlineSession = new MoonshineSession({
        ort,
        encoder: encoderSession as unknown as MoonshineSessionOptions['encoder'],
        crossKv: crossKvSession as unknown as MoonshineSessionOptions['crossKv'],
        decoder: decoderSession as unknown as MoonshineSessionOptions['decoder'],
        tokenizer,
        bosTokenId,
        eosTokenId,
        slots,
        heads,
        headDim,
        ...(this.loadOptions.maxTokens !== undefined
          ? { maxTokens: this.loadOptions.maxTokens }
          : {}),
      });
      return;
    }

    const streamingAssets =
      this.loadOptions?.streaming ??
      (this.loadOptions && 'chunkedEncoderUrl' in this.loadOptions
        ? (this.loadOptions as unknown as MoonshineStreamingAssets)
        : undefined);

    if (streamingAssets) {
      this.streamingLoaded = await loadMoonshineStreaming({
        ...streamingAssets,
        modelId: this.modelId,
        encoderBackend: this.loadOptions?.encoderBackend,
        wasmPaths: this.loadOptions?.wasmPaths,
        cpuThreads: this.loadOptions?.cpuThreads,
        signal: this.signal as AbortSignal | undefined,
      });
      return;
    }

    throw createExperimentalArtifactMissingError('moonshine', this.modelId);
  }

  async transcribe<TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options?: BaseTranscriptionOptions & { readonly responseFlavor?: TFlavor },
  ): Promise<TranscriptResponse<MoonshineDecodeResult, TFlavor>> {
    if (this.disposed) {
      throw new Error('Moonshine session is disposed.');
    }

    const pcm = normalizePcmInput(input, { sampleRate: 16000 });
    const mono = pcm.channels[0];
    if (!mono) {
      throw new Error('Moonshine transcription requires mono audio.');
    }
    const durationSeconds = mono.length / 16000;

    let decodeResult: MoonshineDecodeResult;
    if (this.offlineSession) {
      decodeResult = await this.offlineSession.transcribe(mono);
    } else if (this.streamingLoaded) {
      this.streamingLoaded.transcriber.reset();
      await this.streamingLoaded.transcriber.pushAudio(mono);
      const finalized = await this.streamingLoaded.transcriber.finalize();
      decodeResult = {
        text: finalized.text,
        tokenIds: [],
        encoderFrames: Math.round(durationSeconds * 25),
        steps: 0,
      };
    } else {
      throw createExperimentalArtifactMissingError('moonshine', this.modelId);
    }

    const detailLevel: TranscriptDetailLevel = options?.detail ?? 'segments';
    const canonical = mapMoonshineResultToCanonical(decodeResult, {
      modelId: this.modelId,
      backendId: this.backendId,
      durationSeconds,
      sampleRate: 16000,
      detailLevel,
    });

    if (options?.responseFlavor === 'native') {
      return decodeResult as TranscriptResponse<MoonshineDecodeResult, TFlavor>;
    }
    if (options?.responseFlavor === 'canonical+native') {
      return { canonical, native: decodeResult } as TranscriptResponse<
        MoonshineDecodeResult,
        TFlavor
      >;
    }
    return canonical as TranscriptResponse<MoonshineDecodeResult, TFlavor>;
  }

  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    if (this.streamingLoaded) {
      await this.streamingLoaded.dispose();
      this.streamingLoaded = undefined;
    }
    await Promise.all(
      this.offlineOrtSessions.map(async (s) => {
        await s.release?.();
      }),
    );
    this.offlineOrtSessions.length = 0;
    this.onDispose?.();
  }
}

export interface CreateMoonshineModelFamilyOptions {
  readonly family?: string;
  readonly classification?: Partial<ModelClassification>;
  readonly supportsModel?: (
    modelId: string,
    classification?: Partial<ModelClassification>,
  ) => boolean;
}

export function createMoonshineModelFamily(
  options: CreateMoonshineModelFamilyOptions = {},
): SpeechModelFactory<MoonshineLoadOptions, BaseTranscriptionOptions, MoonshineDecodeResult> {
  const family = options.family ?? 'moonshine';
  const factoryClassification = resolveClassification(options.classification);

  return {
    family,
    classification: factoryClassification,
    supportedBackendIds: ['webgpu', 'wasm', 'native-cpu'],
    supports(modelId: string): boolean {
      if (options.supportsModel) {
        return options.supportsModel(modelId);
      }
      if (!modelId) return false;
      return modelId.toLowerCase().includes('moonshine');
    },
    matchesClassification(classification: Partial<ModelClassification>): boolean {
      if (options.supportsModel) {
        return options.supportsModel('', classification);
      }
      return classificationContains(factoryClassification, classification);
    },
    async createModel(
      request: FamilyModelLoadRequest<MoonshineLoadOptions>,
      context: SpeechModelFactoryContext,
    ): Promise<SpeechModel<MoonshineLoadOptions, BaseTranscriptionOptions, MoonshineDecodeResult>> {
      const sessions = new Set<MoonshineSpeechSession>();
      const transcribers = new Set<StreamingTranscriber>();
      let disposed = false;

      const classification = resolveClassification(factoryClassification, request.classification);

      return {
        info: {
          family,
          modelId: request.modelId,
          classification,
          preset: request.resolvedPreset ?? 'moonshine',
          architecture: createModelArchitecture({
            processor: {
              layer: 'processor',
              module: 'audio',
              implementation: 'raw-pcm',
              shared: true,
            },
            encoder: {
              layer: 'encoder',
              module: 'moonshine-encoder',
              implementation: 'conformer',
              shared: true,
            },
            decoder: {
              layer: 'decoder',
              module: 'moonshine-decoder',
              implementation: 'transformer',
              shared: true,
            },
            decoding: {
              layer: 'decoding',
              module: 'inference',
              implementation: 'greedy',
              shared: true,
            },
            tokenizer: {
              layer: 'tokenizer',
              module: 'tokenizers',
              implementation: 'sentencepiece',
              notes: ['Vocab size: 32768'],
              shared: true,
            },
          }),
        },
        backend: context.backend,
        loadOptions: request.options,
        async createSession(
          _options?: BaseSessionOptions,
        ): Promise<SpeechSession<BaseTranscriptionOptions, MoonshineDecodeResult>> {
          assertSessionOwnerActive(disposed);
          const session = new MoonshineSpeechSession(
            request.modelId,
            context.backend.id,
            request.options,
            request.signal,
            () => {
              sessions.delete(session);
            },
          );
          return await registerOwnedSpeechSession(session, sessions, () => disposed);
        },
        async createStreamingTranscriber(
          streamingOptions?: StreamingSessionOptions,
        ): Promise<StreamingTranscriber> {
          assertSessionOwnerActive(disposed);
          const streamingAssets =
            request.options?.streaming ??
            (request.options && 'chunkedEncoderUrl' in request.options
              ? (request.options as unknown as MoonshineStreamingAssets)
              : undefined);

          if (!streamingAssets) {
            throw new Error('Moonshine model does not have streaming assets configured.');
          }

          const loaded = await loadMoonshineStreaming({
            ...streamingAssets,
            modelId: request.modelId,
            encoderBackend: request.options?.encoderBackend,
            wasmPaths: request.options?.wasmPaths,
            cpuThreads: request.options?.cpuThreads,
            emitPartials: streamingOptions?.emitPartials,
            signal: request.signal as AbortSignal | undefined,
          });

          const transcriber = loaded.transcriber;
          let transcriberDisposed = false;
          const wrapper: StreamingTranscriber = {
            pushAudio(input: AudioInputLike) {
              return transcriber.pushAudio(input);
            },
            flush() {
              return transcriber.flush();
            },
            finalize() {
              return transcriber.finalize();
            },
            reset() {
              return transcriber.reset();
            },
            async dispose() {
              if (!transcriberDisposed) {
                transcriberDisposed = true;
                transcribers.delete(wrapper);
                await loaded.dispose();
              }
            },
            getState() {
              return transcriber.getState();
            },
          };
          transcribers.add(wrapper);
          return wrapper;
        },
        async dispose(): Promise<void> {
          if (disposed) {
            return;
          }
          disposed = true;
          const toDispose = [
            ...sessions,
            ...[...transcribers].map((t) => ({ dispose: () => t.dispose?.() })),
          ];
          sessions.clear();
          transcribers.clear();
          await disposeOwnedResources(toDispose);
        },
      };
    },
  };
}

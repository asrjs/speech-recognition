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
import { releaseMoonshineSessions, settleMoonshineSessionCreates } from './session-resources.js';
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
  const tokenCount = decodeResult.tokenCount ?? decodeResult.tokenIds?.length;
  const words: TranscriptWord[] =
    decodeResult.text.trim().length === 0
      ? []
      : decodeResult.text
          .trim()
          .split(/\s+/)
          .map((text, index, all) => {
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

  const includeSegments = options.detailLevel === 'segments' || options.detailLevel === 'detailed';

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
      ...(tokenCount === undefined ? {} : { tokenCount }),
      wordCount: words.length,
      segmentCount: segments.length,
      ...(options.metrics ? { metrics: options.metrics } : {}),
    },
    ...(includeSegments ? { segments } : {}),
    ...(includeWords ? { words } : {}),
  };
}

export class MoonshineSpeechSession implements SpeechSession<
  BaseTranscriptionOptions,
  MoonshineDecodeResult
> {
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
      const rawOrt = await initOrt('wasm', {
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

      const [encoderSession, crossKvSession, decoderSession] = await settleMoonshineSessionCreates([
        rawOrt.InferenceSession.create(
          this.loadOptions.encoderUrl,
          withExternalData(
            sessionOptions,
            'encoder_model.onnx.data',
            this.loadOptions.encoderDataUrl,
          ),
        ),
        rawOrt.InferenceSession.create(
          this.loadOptions.crossKvUrl,
          withExternalData(
            sessionOptions,
            'decoder_cross_kv_model.onnx.data',
            this.loadOptions.crossKvDataUrl,
          ),
        ),
        rawOrt.InferenceSession.create(
          this.loadOptions.decoderUrl,
          withExternalData(
            sessionOptions,
            'decoder_with_past_model.onnx.data',
            this.loadOptions.decoderDataUrl,
          ),
        ),
      ]);
      const [encoderGraph, crossKvGraph, decoderGraph] = [
        encoderSession,
        crossKvSession,
        decoderSession,
      ] as const;
      if (!encoderGraph || !crossKvGraph || !decoderGraph) {
        throw new Error('Moonshine graph creation returned an incomplete session set.');
      }
      if (this.disposed) {
        await releaseMoonshineSessions([encoderGraph, crossKvGraph, decoderGraph]);
        throw new Error('Moonshine session was disposed during initialization.');
      }
      this.offlineOrtSessions.push(encoderGraph, crossKvGraph, decoderGraph);

      const decoderInputs = 'inputNames' in decoderGraph ? decoderGraph.inputNames : undefined;
      const decoderOutputs = 'outputNames' in decoderGraph ? decoderGraph.outputNames : undefined;
      const requiredDecoderInputs = ['input_ids', 'self_past', 'cross_past', 'encoder_states'];
      if (
        !Array.isArray(decoderInputs) ||
        !requiredDecoderInputs.every((name) => decoderInputs.includes(name)) ||
        !Array.isArray(decoderOutputs) ||
        decoderOutputs[0] !== 'logits' ||
        decoderOutputs[1] !== 'self_present'
      ) {
        throw new Error(
          'Unsupported Moonshine offline decoder graph: expected the stacked self_past/cross_past ' +
            'cache interface with logits/self_present outputs. Use a matching stacked-cache export; ' +
            'per-layer past_key_values graphs require a different adapter.',
        );
      }

      const readJsonFromUrl = async <T = unknown>(targetUrl: string): Promise<T> => {
        if (targetUrl.startsWith('file://')) {
          try {
            const fsModuleId = 'node:fs/promises';
            const urlModuleId = 'node:url';
            const fs = await import(/* @vite-ignore */ fsModuleId);
            const { fileURLToPath } = await import(/* @vite-ignore */ urlModuleId);
            const content = await fs.readFile(fileURLToPath(targetUrl), 'utf-8');
            return JSON.parse(content) as T;
          } catch {
            const res = await fetch(targetUrl);
            return (await res.json()) as T;
          }
        }
        const res = await fetch(targetUrl);
        return (await res.json()) as T;
      };

      const tokenizerJson = await readJsonFromUrl<unknown>(this.loadOptions.tokenizerUrl);

      let slots = 8;
      let heads = 8;
      let headDim = 64;
      if (this.loadOptions.receiptUrl) {
        try {
          const receiptJson = await readJsonFromUrl<{
            graphs?: {
              decoder?: { interface?: { slots?: number; heads?: number; headDim?: number } };
            };
          }>(this.loadOptions.receiptUrl);
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
          const genJson = await readJsonFromUrl<{ bos_token_id?: number; eos_token_id?: number }>(
            this.loadOptions.generationUrl,
          );
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
        encoder: encoderGraph as unknown as MoonshineSessionOptions['encoder'],
        crossKv: crossKvGraph as unknown as MoonshineSessionOptions['crossKv'],
        decoder: decoderGraph as unknown as MoonshineSessionOptions['decoder'],
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
      const loaded = await loadMoonshineStreaming({
        ...streamingAssets,
        modelId: this.modelId,
        encoderBackend:
          this.loadOptions?.encoderBackend ??
          (this.backendId.startsWith('webgpu') ? 'webgpu' : 'wasm'),
        wasmPaths: this.loadOptions?.wasmPaths,
        cpuThreads: this.loadOptions?.cpuThreads,
        signal: this.signal as AbortSignal | undefined,
      });
      if (this.disposed) {
        await loaded.dispose();
        throw new Error('Moonshine session was disposed during initialization.');
      }
      this.streamingLoaded = loaded;
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
        ...(finalized.meta.tokenCount === undefined
          ? {}
          : { tokenCount: finalized.meta.tokenCount }),
        encoderFrames: Math.round(durationSeconds * 25),
        steps: 0,
      };
    } else {
      throw createExperimentalArtifactMissingError('moonshine', this.modelId);
    }

    const detailLevel: TranscriptDetailLevel = options?.detail ?? 'segments';
    const canonical = mapMoonshineResultToCanonical(decodeResult, {
      modelId: this.modelId,
      backendId: this.offlineSession ? 'wasm' : (this.streamingLoaded?.resolvedBackend ?? 'wasm'),
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
    const failures: unknown[] = [];
    if (this.streamingLoaded) {
      try {
        await this.streamingLoaded.dispose();
      } catch (error) {
        failures.push(error);
      }
      this.streamingLoaded = undefined;
    }
    try {
      await releaseMoonshineSessions(this.offlineOrtSessions);
    } catch (error) {
      failures.push(error);
    }
    this.offlineOrtSessions.length = 0;
    this.onDispose?.();
    if (failures.length === 1) throw failures[0];
    if (failures.length > 1) {
      throw new AggregateError(failures, 'Moonshine model cleanup failed.', { cause: failures[0] });
    }
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
      const streamingAssets = request.options?.streaming ??
        (request.options && 'chunkedEncoderUrl' in request.options
          ? (request.options as unknown as MoonshineStreamingAssets)
          : undefined);

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
        ...(streamingAssets ? { async createStreamingTranscriber(
          streamingOptions?: StreamingSessionOptions,
        ): Promise<StreamingTranscriber> {
          assertSessionOwnerActive(disposed);
          const loaded = await loadMoonshineStreaming({
            ...streamingAssets,
            modelId: request.modelId,
            encoderBackend:
              request.options?.encoderBackend ??
              (context.backend.id.startsWith('webgpu') ? 'webgpu' : 'wasm'),
            wasmPaths: request.options?.wasmPaths,
            cpuThreads: request.options?.cpuThreads,
            emitPartials: streamingOptions?.emitPartials,
            signal: request.signal as AbortSignal | undefined,
          });

          const transcriber = loaded.transcriber;
          if (disposed) {
            await loaded.dispose();
            assertSessionOwnerActive(disposed);
          }
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
        } } : {}),
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

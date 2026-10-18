/**
 * Public loader for the streaming Moonshine graph set.
 *
 * Wires the four exported graphs (chunked encoder, whole-utterance encoder,
 * decoder seed, decoder step) and the tokenizer into a
 * `MoonshineStreamingTranscriber` that speaks the canonical
 * `StreamingTranscriber` contract. This is the only place URL routing and ORT
 * session creation for the streaming family live; the session and transcriber
 * stay transport-free so they can be unit-tested with fakes.
 *
 * The `ort` module is injectable: production callers omit it and the loader
 * imports `onnxruntime-web` (setting `wasmPaths` when provided), tests pass a
 * fake. Session release happens in `dispose()`.
 */
import type { MoonshineLoadOptions, MoonshineStreamingAssets } from './types.js';
import { initOrt } from '../lasr-ctc/ort.js';
import { throwIfAssetAborted } from '../../io/abort.js';
import { MoonshineTokenizer } from './session.js';
import { MoonshineStreamingSession } from './streaming-session.js';
import { MoonshineStreamingTranscriber } from './streaming-transcriber.js';
import {
  MoonshineSessionBatchError,
  releaseMoonshineSessions,
  settleMoonshineSessionCreates,
} from './session-resources.js';

export interface MoonshineOrtLike {
  readonly Tensor: new (type: string, data: ArrayBufferView, dims: readonly number[]) => unknown;
  readonly InferenceSession: {
    create(url: string, options?: Record<string, unknown>): Promise<OrtSessionLike>;
  };
  readonly env?: { wasm?: { wasmPaths?: string } };
}

interface OrtSessionLike {
  run(feeds: Record<string, unknown>): Promise<Record<string, unknown>>;
  release?(): void | Promise<void>;
}

export interface MoonshineStreamingLoadOptions extends MoonshineStreamingAssets {
  readonly modelId?: string;
  readonly encoderBackend?: 'wasm' | 'webgpu';
  readonly wasmPaths?: string;
  readonly cpuThreads?: number;
  readonly emitFrames?: number;
  readonly dtype?: 'float32' | 'float16';
  readonly emitPartials?: boolean;
  /** Test/embedding seam; production omits it. */
  readonly ort?: MoonshineOrtLike;
  /** Test seam for the tokenizer fetch; defaults to global fetch + JSON. */
  readonly fetchJson?: (url: string) => Promise<unknown>;
  readonly signal?: AbortSignal;
}

export interface LoadedMoonshineStreaming {
  readonly session: MoonshineStreamingSession;
  readonly transcriber: MoonshineStreamingTranscriber;
  readonly resolvedBackend: 'webgpu' | 'wasm';
  /** @deprecated Use resolvedBackend; all four graphs use this provider. */
  readonly resolvedEncoderBackend: 'webgpu' | 'wasm';
  readonly backendFallbackReason?: string;
  dispose(): Promise<void>;
}

async function loadDefaultOrt(options: MoonshineStreamingLoadOptions): Promise<MoonshineOrtLike> {
  return await initOrt(options.encoderBackend ?? 'wasm', {
    wasmPaths: options.wasmPaths,
    cpuThreads: options.cpuThreads,
    signal: options.signal,
  }) as unknown as MoonshineOrtLike;
}

function sessionOptionsFor(
  options: MoonshineStreamingLoadOptions,
  backend: 'webgpu' | 'wasm',
): Record<string, unknown> {
  const executionProviders = [backend];
  const intraOpNumThreads = options.cpuThreads;
  return {
    executionProviders,
    ...(backend === 'webgpu' ? { graphOptimizationLevel: 'basic' } : {}),
    ...(intraOpNumThreads === undefined ? {} : { intraOpNumThreads }),
    ...(options.signal ? { signal: options.signal } : {}),
  };
}

export async function loadMoonshineStreaming(
  options: MoonshineStreamingLoadOptions,
): Promise<LoadedMoonshineStreaming> {
  throwIfAssetAborted(options.signal, 'session-create');
  const ort = options.ort ?? (await loadDefaultOrt(options));

  const createGraphSessions = async (
    backend: 'webgpu' | 'wasm',
  ): Promise<{
    sessions: OrtSessionLike[];
    chunkedEncoder: OrtSessionLike;
    wholeEncoder: OrtSessionLike;
    prefill: OrtSessionLike;
    step: OrtSessionLike;
  }> => {
    const sessionOptions = sessionOptionsFor(options, backend);
    const withData = (sidecarPath: string, dataUrl?: string): Record<string, unknown> =>
      dataUrl
        ? { ...sessionOptions, externalData: [{ path: sidecarPath, data: dataUrl }] }
        : sessionOptions;
    const sessions = await settleMoonshineSessionCreates([
      ort.InferenceSession.create(
        options.chunkedEncoderUrl,
        withData('encoder_chunked_model.onnx.data', options.chunkedEncoderDataUrl),
      ),
      ort.InferenceSession.create(
        options.encoderUrl,
        withData('encoder_model.onnx.data', options.encoderDataUrl),
      ),
      ort.InferenceSession.create(
        options.prefillUrl,
        withData('decoder_prefill_model.onnx.data', options.prefillDataUrl),
      ),
      ort.InferenceSession.create(
        options.stepUrl,
        withData('decoder_step_model.onnx.data', options.stepDataUrl),
      ),
    ]);
    const [chunkedEncoder, wholeEncoder, prefill, step] = sessions;
    if (!chunkedEncoder || !wholeEncoder || !prefill || !step) {
      await releaseMoonshineSessions(sessions);
      throw new Error('Moonshine streaming graph creation returned an incomplete session set.');
    }
    return {
      sessions,
      chunkedEncoder,
      wholeEncoder,
      prefill,
      step,
    };
  };

  let resolvedBackend: 'webgpu' | 'wasm' = options.encoderBackend === 'webgpu' ? 'webgpu' : 'wasm';
  let backendFallbackReason: string | undefined;
  let graphs: {
    sessions: OrtSessionLike[];
    chunkedEncoder: OrtSessionLike;
    wholeEncoder: OrtSessionLike;
    prefill: OrtSessionLike;
    step: OrtSessionLike;
  };

  if (options.encoderBackend === 'webgpu') {
    try {
      graphs = await createGraphSessions('webgpu');
      resolvedBackend = 'webgpu';
    } catch (error) {
      if (error instanceof MoonshineSessionBatchError && error.cleanupFailures.length > 0) {
        throw error;
      }
      throwIfAssetAborted(options.signal, 'session-create');
      backendFallbackReason = String(error instanceof Error ? error.message : error);
      graphs = await createGraphSessions('wasm');
      resolvedBackend = 'wasm';
    }
  } else {
    graphs = await createGraphSessions('wasm');
    resolvedBackend = 'wasm';
  }

  const sessions = graphs.sessions;

  try {
    throwIfAssetAborted(options.signal, 'session-create');
    const tokenizerJson = options.fetchJson
      ? await options.fetchJson(options.tokenizerUrl)
      : await (await fetch(options.tokenizerUrl)).json();
    const tokenizer = MoonshineTokenizer.fromTokenizerJson(
      tokenizerJson as Parameters<typeof MoonshineTokenizer.fromTokenizerJson>[0],
      0,
    );
    throwIfAssetAborted(options.signal, 'session-create');

    const session = new MoonshineStreamingSession({
      ort,
      chunkedEncoder: graphs.chunkedEncoder,
      wholeEncoder: graphs.wholeEncoder,
      prefill: graphs.prefill,
      step: graphs.step,
      tokenizer,
      bosTokenId: 1,
      eosTokenId: 2,
      ...(options.emitFrames === undefined ? {} : { emitFrames: options.emitFrames }),
      ...(options.dtype === undefined ? {} : { dtype: options.dtype }),
    });
    const transcriber = new MoonshineStreamingTranscriber(session, {
      ...(options.modelId === undefined ? {} : { modelId: options.modelId }),
      backendId: resolvedBackend,
      ...(options.emitPartials === undefined ? {} : { emitPartials: options.emitPartials }),
    });

    return {
      session,
      transcriber,
      resolvedBackend,
      resolvedEncoderBackend: resolvedBackend,
      ...(backendFallbackReason === undefined ? {} : { backendFallbackReason }),
      async dispose() {
        const failures: unknown[] = [];
        try {
          transcriber.dispose();
        } catch (error) {
          failures.push(error);
        }
        try {
          await releaseMoonshineSessions(sessions);
        } catch (error) {
          failures.push(error);
        }
        if (failures.length === 1) throw failures[0];
        if (failures.length > 1) {
          throw new AggregateError(failures, 'Moonshine streaming cleanup failed.', {
            cause: failures[0],
          });
        }
      },
    };
  } catch (error) {
    try {
      await releaseMoonshineSessions(sessions);
    } catch (cleanupError) {
      throw new AggregateError(
        [error, cleanupError],
        'Moonshine streaming initialization and cleanup failed.',
        { cause: error },
      );
    }
    throw error;
  }
}

/**
 * Streaming slice of the local-folder load options, when the folder carries the
 * four streaming graphs alongside (or instead of) the offline pair.
 */
export function streamingAssetsFromLoadOptions(
  options: Pick<MoonshineLoadOptions, 'encoderUrl' | 'tokenizerUrl'> & {
    readonly streaming?: MoonshineStreamingAssets;
  },
): MoonshineStreamingAssets | undefined {
  if (!options.streaming) {
    return undefined;
  }
  return { ...options.streaming };
}

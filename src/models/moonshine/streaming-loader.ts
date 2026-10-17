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
import { MoonshineTokenizer } from './session.js';
import { MoonshineStreamingSession } from './streaming-session.js';
import { MoonshineStreamingTranscriber } from './streaming-transcriber.js';

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
  readonly resolvedEncoderBackend?: 'webgpu' | 'wasm';
  dispose(): Promise<void>;
}

async function loadDefaultOrt(options: MoonshineStreamingLoadOptions): Promise<MoonshineOrtLike> {
  const imported = (await import('onnxruntime-web')) as unknown as MoonshineOrtLike & {
    readonly default?: MoonshineOrtLike;
  };
  const ort = imported.default ?? imported;
  if (options.wasmPaths && ort.env?.wasm) {
    ort.env.wasm.wasmPaths = options.wasmPaths;
  }
  return ort;
}

function sessionOptionsFor(
  options: MoonshineStreamingLoadOptions,
  backend: 'webgpu' | 'wasm',
): Record<string, unknown> {
  const executionProviders =
    backend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
  const intraOpNumThreads = options.cpuThreads;
  return {
    executionProviders,
    ...(intraOpNumThreads === undefined ? {} : { intraOpNumThreads }),
    ...(options.signal ? { signal: options.signal } : {}),
  };
}

export async function loadMoonshineStreaming(
  options: MoonshineStreamingLoadOptions,
): Promise<LoadedMoonshineStreaming> {
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
    const created: OrtSessionLike[] = [];
    try {
      const [chunkedEncoder, wholeEncoder, prefill, step] = await Promise.all([
        ort.InferenceSession.create(options.chunkedEncoderUrl, sessionOptions),
        ort.InferenceSession.create(options.encoderUrl, sessionOptions),
        ort.InferenceSession.create(options.prefillUrl, sessionOptions),
        ort.InferenceSession.create(options.stepUrl, sessionOptions),
      ]);
      created.push(chunkedEncoder, wholeEncoder, prefill, step);
      return { sessions: created, chunkedEncoder, wholeEncoder, prefill, step };
    } catch (err) {
      await Promise.all(created.map(async (owned) => {
        await owned.release?.();
      }));
      throw err;
    }
  };

  let resolvedEncoderBackend: 'webgpu' | 'wasm' =
    options.encoderBackend === 'webgpu' ? 'webgpu' : 'wasm';
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
      resolvedEncoderBackend = 'webgpu';
    } catch {
      graphs = await createGraphSessions('wasm');
      resolvedEncoderBackend = 'wasm';
    }
  } else {
    graphs = await createGraphSessions('wasm');
    resolvedEncoderBackend = 'wasm';
  }

  const sessions = graphs.sessions;

  try {
    const tokenizerJson = options.fetchJson
      ? await options.fetchJson(options.tokenizerUrl)
      : await (await fetch(options.tokenizerUrl)).json();
    const tokenizer = MoonshineTokenizer.fromTokenizerJson(
      tokenizerJson as Parameters<typeof MoonshineTokenizer.fromTokenizerJson>[0],
      0,
    );

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
      ...(options.emitPartials === undefined ? {} : { emitPartials: options.emitPartials }),
    });

    return {
      session,
      transcriber,
      resolvedEncoderBackend,
      async dispose() {
        transcriber.dispose();
        await Promise.all(sessions.map(async (owned) => {
          await owned.release?.();
        }));
      },
    };
  } catch (error) {
    await Promise.all(sessions.map(async (owned) => {
      await owned.release?.();
    }));
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

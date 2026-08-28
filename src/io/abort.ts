/**
 * Load-abort helpers for asset IO and ORT session creation.
 *
 * Kept in `src/io` so download/session-create code does not import pipeline or
 * runtime types. Callers that own the public load surface map
 * {@link AssetLoadAbortedError} to `PipelineAbortedError`.
 */

export interface AssetAbortSignalLike {
  readonly aborted: boolean;
}

export const ASSET_LOAD_ABORTED_CODE = 'asset-load-aborted' as const;

export class AssetLoadAbortedError extends Error {
  readonly code = ASSET_LOAD_ABORTED_CODE;
  readonly stageId?: string;

  constructor(stageId?: string) {
    super(stageId ? `Asset load aborted during "${stageId}".` : 'Asset load aborted.');
    this.name = 'AssetLoadAbortedError';
    this.stageId = stageId;
  }
}

export function isAssetLoadAbortedError(error: unknown): error is AssetLoadAbortedError {
  if (error instanceof AssetLoadAbortedError) {
    return true;
  }
  if (!error || typeof error !== 'object') {
    return false;
  }
  const candidate = error as { name?: unknown; code?: unknown };
  return candidate.name === 'AssetLoadAbortedError' && candidate.code === ASSET_LOAD_ABORTED_CODE;
}

export function throwIfAssetAborted(
  signal: AssetAbortSignalLike | null | undefined,
  stage = 'download',
): void {
  if (signal?.aborted) {
    throw new AssetLoadAbortedError(stage);
  }
}

export function toFetchAbortSignal(signal?: AssetAbortSignalLike | null): AbortSignal | undefined {
  if (!signal) {
    return undefined;
  }
  if (typeof AbortSignal !== 'undefined' && signal instanceof AbortSignal) {
    return signal;
  }
  return undefined;
}

type ObservableAbortSignalLike = AssetAbortSignalLike & {
  readonly addEventListener?: (
    type: 'abort',
    listener: () => void,
    options?: { readonly once?: boolean },
  ) => void;
  readonly removeEventListener?: (type: 'abort', listener: () => void) => void;
};

const ABORT_SIGNAL_POLL_INTERVAL_MS = 25;

/**
 * Observe native, cross-realm, and minimal `{ aborted }` signals.
 *
 * Public browser APIs accept a small signal shape so callers can use
 * worker-safe and cross-realm cancellation objects. Native abort events are
 * preferred, while a short polling fallback covers plain objects whose
 * `aborted` property changes without an event source.
 */
export function subscribeToAbortSignal(
  signal: AssetAbortSignalLike | null | undefined,
  onAbort: () => void,
): () => void {
  if (!signal) {
    return () => undefined;
  }

  const observable = signal as ObservableAbortSignalLike;
  if (typeof observable.addEventListener === 'function') {
    const listener = () => onAbort();
    observable.addEventListener('abort', listener, { once: true });
    return () => observable.removeEventListener?.('abort', listener);
  }

  const interval = setInterval(() => {
    if (!observable.aborted) {
      return;
    }
    clearInterval(interval);
    onAbort();
  }, ABORT_SIGNAL_POLL_INTERVAL_MS);
  const maybeNodeTimer = interval as unknown as { readonly unref?: () => void };
  maybeNodeTimer.unref?.();
  return () => clearInterval(interval);
}

export function isDomAbortError(error: unknown): boolean {
  if (typeof DOMException !== 'undefined' && error instanceof DOMException) {
    return error.name === 'AbortError';
  }
  return error instanceof Error && error.name === 'AbortError';
}

export function toAssetLoadAbortedError(error: unknown, stage = 'download'): unknown {
  if (isAssetLoadAbortedError(error)) {
    return error;
  }
  if (isDomAbortError(error)) {
    return new AssetLoadAbortedError(stage);
  }
  return error;
}

export function throwAssetLoadAborted(error: unknown, stage = 'download'): never {
  throw toAssetLoadAbortedError(error, stage);
}

export function rethrowIfAssetAborted(error: unknown, stage = 'download'): void {
  if (isAssetLoadAbortedError(error) || isDomAbortError(error)) {
    throwAssetLoadAborted(error, stage);
  }
}

export async function fetchTextHonoringAbort(
  url: string,
  signal?: AssetAbortSignalLike | null,
  options: {
    readonly stage?: string;
    readonly errorMessage?: string;
  } = {},
): Promise<string> {
  const response = await fetchResponseHonoringAbort(url, signal, options.stage);
  if (!response.ok) {
    throw new Error(
      options.errorMessage ?? `Failed to fetch "${url}": ${response.status} ${response.statusText}`,
    );
  }
  try {
    const text = await response.text();
    throwIfAssetAborted(signal, options.stage ?? 'download');
    return text;
  } catch (error) {
    throwAssetLoadAborted(error, options.stage ?? 'download');
  }
}

export async function fetchBytesHonoringAbort(
  url: string,
  signal?: AssetAbortSignalLike | null,
  options: {
    readonly stage?: string;
    readonly errorMessage?: string;
  } = {},
): Promise<Uint8Array> {
  const response = await fetchResponseHonoringAbort(url, signal, options.stage);
  if (!response.ok) {
    throw new Error(
      options.errorMessage ?? `Failed to fetch "${url}": ${response.status} ${response.statusText}`,
    );
  }
  try {
    const bytes = new Uint8Array(await response.arrayBuffer());
    throwIfAssetAborted(signal, options.stage ?? 'download');
    return bytes;
  } catch (error) {
    throwAssetLoadAborted(error, options.stage ?? 'download');
  }
}

async function fetchResponseHonoringAbort(
  url: string,
  signal?: AssetAbortSignalLike | null,
  stage = 'download',
): Promise<Response> {
  throwIfAssetAborted(signal, stage);
  const fetchSignal = toFetchAbortSignal(signal);
  try {
    return await fetch(url, fetchSignal ? { signal: fetchSignal } : undefined);
  } catch (error) {
    throwAssetLoadAborted(error, stage);
  }
}

type ReleasableSession = {
  release?: () => void | Promise<void>;
};

export function withNativeAbortSignalOption(
  options: Record<string, unknown> | undefined,
  signal?: AssetAbortSignalLike | null,
): Record<string, unknown> | undefined {
  const native = toFetchAbortSignal(signal);
  if (!native) {
    return options;
  }
  return { ...(options ?? {}), abortSignal: native };
}

/**
 * Honor abort around `InferenceSession.create`.
 *
 * onnxruntime-web does not document a supported abort API on session create.
 * When a native `AbortSignal` is available it is forwarded as `abortSignal`
 * in case a given ORT build observes it; regardless, this checkpoints
 * immediately after create and releases the session if abort already happened.
 */
export async function honorAbortAfterCreate<T extends ReleasableSession>(
  create: () => Promise<T>,
  signal: AssetAbortSignalLike | null | undefined,
  release: (value: T) => void | Promise<void>,
  stage = 'session-create',
): Promise<T> {
  throwIfAssetAborted(signal, stage);
  const created = await create();
  if (signal?.aborted) {
    try {
      await release(created);
    } catch {
      // best-effort teardown of a session that must not outlive abort
    }
    throw new AssetLoadAbortedError(stage);
  }
  return created;
}

export function withOrtCreateAbort<
  T extends {
    InferenceSession: {
      create: (url: string, options?: Record<string, unknown>) => Promise<ReleasableSession>;
    };
  },
>(ort: T, signal?: AssetAbortSignalLike | null): T {
  if (!signal) {
    return ort;
  }

  const create = ort.InferenceSession.create.bind(ort.InferenceSession);
  const InferenceSession = {
    create: (url: string, options?: Record<string, unknown>) =>
      honorAbortAfterCreate(
        () => create(url, withNativeAbortSignalOption(options, signal)),
        signal,
        (session) => {
          void session?.release?.();
        },
      ),
  };

  return new Proxy(ort, {
    get(target, prop, receiver) {
      if (prop === 'InferenceSession') {
        return InferenceSession;
      }
      return Reflect.get(target, prop, receiver);
    },
  });
}

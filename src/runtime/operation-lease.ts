import type {
  AudioInputLike,
  BaseTranscriptionOptions,
  SpeechSession,
  TranscriptResponse,
  TranscriptResponseFlavor,
} from '../types/index.js';

/**
 * Tracks asynchronous work which must finish before an owning resource can be
 * disposed. The owner closes the lease synchronously, so no new work can
 * enter after disposal starts, then waits for the work that was already in
 * flight.
 */
export interface ActiveOperationLease {
  enter(): (() => void) | undefined;
  closeAndWait(): Promise<void>;
}

export function createActiveOperationLease(): ActiveOperationLease {
  const active = new Set<Promise<void>>();
  let closed = false;
  let closePromise: Promise<void> | null = null;

  return {
    enter(): (() => void) | undefined {
      if (closed) {
        return undefined;
      }

      let resolveCompletion!: () => void;
      const completion = new Promise<void>((resolve) => {
        resolveCompletion = resolve;
      });
      active.add(completion);
      let released = false;
      return () => {
        if (released) {
          return;
        }
        released = true;
        resolveCompletion();
        active.delete(completion);
      };
    },
    closeAndWait(): Promise<void> {
      if (!closePromise) {
        closed = true;
        closePromise = Promise.allSettled([...active]).then(() => undefined);
      }
      return closePromise;
    },
  };
}

export function createGuardedSpeechSession<
  TTranscriptionOptions extends BaseTranscriptionOptions = BaseTranscriptionOptions,
  TNative = unknown,
>(
  session: SpeechSession<TTranscriptionOptions, TNative>,
  operationLease: ActiveOperationLease,
): SpeechSession<TTranscriptionOptions, TNative> {
  let disposePromise: Promise<void> | null = null;

  const transcribe = async <TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: AudioInputLike,
    options?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor },
  ): Promise<TranscriptResponse<TNative, TFlavor>> => {
    const releaseOperation = operationLease.enter();
    if (!releaseOperation) {
      throw new Error('Speech model handle is disposed.');
    }
    try {
      return await session.transcribe<TFlavor>(input, options);
    } finally {
      releaseOperation();
    }
  };

  const transcribeBatch = async <TFlavor extends TranscriptResponseFlavor = 'canonical'>(
    input: readonly AudioInputLike[],
    options?: TTranscriptionOptions & { readonly responseFlavor?: TFlavor },
  ): Promise<readonly TranscriptResponse<TNative, TFlavor>[]> => {
    const releaseOperation = operationLease.enter();
    if (!releaseOperation) {
      throw new Error('Speech model handle is disposed.');
    }
    try {
      if (typeof session.transcribeBatch !== 'function') {
        throw new Error('Speech session does not expose batch transcription.');
      }
      return await session.transcribeBatch<TFlavor>(input, options);
    } finally {
      releaseOperation();
    }
  };

  return {
    transcribe,
    ...(typeof session.transcribeBatch === 'function' ? { transcribeBatch } : {}),
    async dispose(): Promise<void> {
      if (!disposePromise) {
        disposePromise = (async () => {
          await operationLease.closeAndWait();
          await session.dispose();
        })();
      }
      await disposePromise;
    },
  } as SpeechSession<TTranscriptionOptions, TNative>;
}

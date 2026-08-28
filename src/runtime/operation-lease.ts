import type {
  AudioInputLike,
  BaseTranscriptionOptions,
  SpeechSession,
  StreamingSessionOptions,
  StreamingTranscriber,
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

export interface TrackedStreamingTranscriberFactory {
  create(options?: StreamingSessionOptions): Promise<StreamingTranscriber>;
  disposeAll(): Promise<void>;
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

/**
 * Tracks streaming transcribers created through a high-level model handle.
 *
 * Streaming work is owned by the transcriber rather than by the short-lived
 * creation call, so the owner closes its operation lease first and then calls
 * {@link disposeAll} before releasing the model/session resources underneath.
 */
export function createTrackedStreamingTranscriberFactory(
  operationLease: ActiveOperationLease,
  createTranscriber: (options: StreamingSessionOptions) => Promise<StreamingTranscriber>,
): TrackedStreamingTranscriberFactory {
  const owned = new Set<StreamingTranscriber>();

  const create = async (options: StreamingSessionOptions = {}): Promise<StreamingTranscriber> => {
    const releaseOperation = operationLease.enter();
    if (!releaseOperation) {
      throw new Error('Speech model handle is disposed.');
    }

    try {
      const transcriber = await createTranscriber(options);
      const transcriberLease = createActiveOperationLease();
      const runTranscriberOperation = async <T>(operation: () => Promise<T>): Promise<T> => {
        const releaseOperation = transcriberLease.enter();
        if (!releaseOperation) {
          throw new Error('Streaming transcriber is disposed.');
        }
        try {
          return await operation();
        } finally {
          releaseOperation();
        }
      };
      let disposePromise: Promise<void> | null = null;
      const ownedTranscriber: StreamingTranscriber = {
        pushAudio: (input) => runTranscriberOperation(() => transcriber.pushAudio(input)),
        flush: () => runTranscriberOperation(() => transcriber.flush()),
        finalize: () => runTranscriberOperation(() => transcriber.finalize()),
        reset: () => runTranscriberOperation(async () => await transcriber.reset()),
        getState: () => transcriber.getState(),
        dispose: async () => {
          if (!disposePromise) {
            disposePromise = (async () => {
              await transcriberLease.closeAndWait();
              await transcriber.dispose?.();
              owned.delete(ownedTranscriber);
            })();
          }
          await disposePromise;
        },
      };
      owned.add(ownedTranscriber);
      return ownedTranscriber;
    } finally {
      releaseOperation();
    }
  };

  return {
    create,
    async disposeAll(): Promise<void> {
      await Promise.all([...owned].map((transcriber) => transcriber.dispose?.()));
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
      if (input.length === 0) {
        return [] as readonly TranscriptResponse<TNative, TFlavor>[];
      }
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

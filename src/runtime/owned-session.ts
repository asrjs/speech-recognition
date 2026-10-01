import { SpeechRuntimeError } from './errors.js';

interface InitializableSession {
  initialize(): Promise<void>;
  dispose(): void | Promise<void>;
}

/** Settle every owned resource before reporting cleanup failures. */
export async function disposeOwnedResources(
  resources: Iterable<Pick<InitializableSession, 'dispose'>>,
): Promise<void> {
  const results = await Promise.allSettled(
    Array.from(resources, (resource) => Promise.resolve().then(() => resource.dispose())),
  );
  const failures = results.flatMap((result) =>
    result.status === 'rejected' ? [result.reason] : [],
  );
  if (failures.length === 1) throw failures[0];
  if (failures.length > 1) {
    throw new AggregateError(failures, 'Owned resource cleanup failed.', { cause: failures[0] });
  }
}

export function assertSessionOwnerActive(disposed: boolean): void {
  if (disposed) {
    throw new SpeechRuntimeError(
      'Cannot create a session from a disposed model.',
      'model-disposed',
    );
  }
}

/**
 * Model-neutral ownership for a session that has not yet reached its caller.
 * Families still construct, initialize and dispose their own execution resources.
 */
export async function initializeOwnedSession<TSession extends InitializableSession>(
  session: TSession,
  owner: Set<TSession>,
  isOwnerDisposed: () => boolean = () => false,
): Promise<TSession> {
  owner.add(session);
  try {
    assertSessionOwnerActive(isOwnerDisposed());
    await session.initialize();
    assertSessionOwnerActive(isOwnerDisposed());
    return session;
  } catch (initializationError) {
    // No caller can dispose this failed creation; remove hidden ownership even
    // when cleanup fails, and keep both failures observable.
    owner.delete(session);
    try {
      await session.dispose();
    } catch (cleanupError) {
      throw new AggregateError(
        [initializationError, cleanupError],
        'Session initialization and cleanup failed.',
        { cause: initializationError },
      );
    }
    throw initializationError;
  }
}

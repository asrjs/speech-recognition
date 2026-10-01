/** Test-only ownership for partial model-session setup, independent of ORT/family. */
export interface ReleasableFixtureSession {
  release(): void | Promise<void>;
}

export type FixtureSessionOwner = <T extends ReleasableFixtureSession>(session: T) => T;

export interface FixtureSessionGroup<T> {
  readonly value: T;
  release(): Promise<void>;
}

export async function createFixtureSessionGroup<T>(
  setup: (own: FixtureSessionOwner) => Promise<T>,
): Promise<FixtureSessionGroup<T>> {
  const sessions = new Set<ReleasableFixtureSession>();
  let cleanup: Promise<void> | undefined;
  const release = () =>
    (cleanup ??= (async () => {
      const results = await Promise.allSettled(
        [...sessions].map((session) => Promise.resolve().then(() => session.release())),
      );
      const errors = results.flatMap((result) =>
        result.status === 'rejected' ? [result.reason] : [],
      );
      if (errors.length) throw new AggregateError(errors, 'Fixture session cleanup failed');
    })());
  try {
    const value = await setup((session) => {
      sessions.add(session);
      return session;
    });
    return { value, release };
  } catch (error) {
    try {
      await release();
    } catch (cleanupError) {
      throw new AggregateError([error, cleanupError], 'Fixture session setup and cleanup failed', {
        cause: error,
      });
    }
    throw error;
  }
}

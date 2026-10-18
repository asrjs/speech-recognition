interface ReleasableSession {
  release?(): void | Promise<void>;
}

export class MoonshineSessionBatchError extends AggregateError {
  constructor(
    readonly creationFailures: readonly unknown[],
    readonly cleanupFailures: readonly unknown[],
  ) {
    const failures = [...creationFailures, ...cleanupFailures];
    const message =
      cleanupFailures.length > 0
        ? 'Moonshine session creation and cleanup failed.'
        : `Moonshine session creation failed: ${String(creationFailures[0] ?? 'unknown error')}`;
    super(failures, message, { cause: creationFailures[0] });
    this.name = 'MoonshineSessionBatchError';
  }
}

function aggregateFailures(failures: readonly unknown[], message: string): unknown {
  if (failures.length === 1) return failures[0];
  return new AggregateError(failures, message, { cause: failures[0] });
}

export async function releaseMoonshineSessions(
  sessions: readonly ReleasableSession[],
): Promise<void> {
  const settled = await Promise.allSettled(
    sessions.map((session) => Promise.resolve().then(() => session.release?.())),
  );
  const failures = settled.flatMap((result) =>
    result.status === 'rejected' ? [result.reason] : [],
  );
  if (failures.length > 0) {
    throw aggregateFailures(failures, 'Moonshine session release failed.');
  }
}

/** Wait for every create attempt before cleaning up successes from a failed batch. */
export async function settleMoonshineSessionCreates<T extends ReleasableSession>(
  pending: readonly Promise<T>[],
): Promise<T[]> {
  const settled = await Promise.allSettled(pending);
  const sessions = settled.flatMap((result) =>
    result.status === 'fulfilled' ? [result.value] : [],
  );
  const creationFailures = settled.flatMap((result) =>
    result.status === 'rejected' ? [result.reason] : [],
  );
  if (creationFailures.length === 0) return sessions;

  let cleanupFailures: readonly unknown[] = [];
  try {
    await releaseMoonshineSessions(sessions);
  } catch (error) {
    cleanupFailures = error instanceof AggregateError ? error.errors : [error];
  }
  throw new MoonshineSessionBatchError(creationFailures, cleanupFailures);
}

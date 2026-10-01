import { describe, expect, it, vi } from 'vitest';
import { createFixtureSessionGroup } from './support/fixture-sessions.js';

// Simulated ownership/error paths, not real model inference or native acceptance.
describe('fixture session group ownership', () => {
  it('transfers successful resources and awaits idempotent cleanup', async () => {
    const calls: string[] = [];
    const session = {
      release: vi.fn(async () => {
        await Promise.resolve();
        calls.push('released');
      }),
    };
    const group = await createFixtureSessionGroup(async (own) => ({
      session: own(session),
      value: 42,
    }));
    expect(group.value.session).toBe(session);
    expect(group.value.value).toBe(42);
    expect(session.release).not.toHaveBeenCalled();
    const cleanup = group.release();
    expect(group.release()).toBe(cleanup);
    await cleanup;
    expect(calls).toEqual(['released']);
    expect(session.release).toHaveBeenCalledTimes(1);
  });

  it.each([0, 1, 2, 3])(
    'releases every acquired session when creation fails after %i successes',
    async (count) => {
      const sessions = Array.from({ length: count }, () => ({ release: vi.fn(async () => {}) }));
      const failure = new Error('session creation failed');
      await expect(
        createFixtureSessionGroup(async (own) => {
          sessions.forEach((session) => own(session));
          throw failure;
        }),
      ).rejects.toBe(failure);
      for (const session of sessions) expect(session.release).toHaveBeenCalledTimes(1);
    },
  );

  it('also rolls back acquired sessions after later metadata or tokenizer validation fails', async () => {
    const session = { release: vi.fn(async () => {}) };
    const failure = new Error('post-acquisition validation failed');
    await expect(
      createFixtureSessionGroup(async (own) => {
        own(session);
        await Promise.resolve();
        throw failure;
      }),
    ).rejects.toBe(failure);
    expect(session.release).toHaveBeenCalledTimes(1);
  });

  it('keeps the setup error and all cleanup errors while attempting every release', async () => {
    const failure = new Error('setup failed');
    const releaseFailure = new Error('release failed');
    const bad = {
      release: vi.fn(async () => {
        throw releaseFailure;
      }),
    };
    const good = { release: vi.fn(async () => {}) };
    const error = await createFixtureSessionGroup(async (own) => {
      own(bad);
      own(good);
      throw failure;
    }).catch((reason: unknown) => reason);
    expect(error).toBeInstanceOf(AggregateError);
    const aggregate = error as AggregateError;
    expect(aggregate.cause).toBe(failure);
    expect(aggregate.errors[0]).toBe(failure);
    expect(aggregate.errors[1]).toBeInstanceOf(AggregateError);
    expect((aggregate.errors[1] as AggregateError).errors).toEqual([releaseFailure]);
    expect(bad.release).toHaveBeenCalledTimes(1);
    expect(good.release).toHaveBeenCalledTimes(1);
  });

  it('does not hide successful-group cleanup errors or repeat failed releases', async () => {
    const failure = new Error('release failed');
    const bad = {
      release: vi.fn(() => {
        throw failure;
      }),
    };
    const good = { release: vi.fn(async () => {}) };
    const group = await createFixtureSessionGroup(async (own) => [own(bad), own(good)]);
    const cleanup = group.release();
    await expect(cleanup).rejects.toBeInstanceOf(AggregateError);
    expect(group.release()).toBe(cleanup);
    expect(bad.release).toHaveBeenCalledTimes(1);
    expect(good.release).toHaveBeenCalledTimes(1);
  });

  it('releases the same resource once even when registered twice', async () => {
    const session = { release: vi.fn(async () => {}) };
    const group = await createFixtureSessionGroup(async (own) => [own(session), own(session)]);
    await group.release();
    expect(session.release).toHaveBeenCalledTimes(1);
  });
});

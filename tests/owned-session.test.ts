import { describe, expect, it, vi } from 'vitest';
import { disposeOwnedResources, initializeOwnedSession } from '../src/runtime/owned-session.js';

describe('model-neutral owned resource cleanup', () => {
  it('attempts sync and async failures, awaits siblings and retains ordered errors', async () => {
    const first = new Error('sync cleanup failure');
    const second = new Error('async cleanup failure');
    let finishCleanup = () => {};
    const gate = new Promise<void>((resolve) => {
      finishCleanup = resolve;
    });
    const resources = [
      {
        dispose: vi.fn(() => {
          throw first;
        }),
      },
      {
        dispose: vi.fn(async () => {
          throw second;
        }),
      },
      { dispose: vi.fn(() => gate) },
    ];
    let settled = false;
    const result = disposeOwnedResources(resources).catch((error: unknown) => {
      settled = true;
      return error;
    });
    try {
      await vi.waitFor(() => expect(resources[2]!.dispose).toHaveBeenCalledTimes(1));
      await new Promise<void>((resolve) => setImmediate(resolve));
      expect(settled).toBe(false);
    } finally {
      finishCleanup();
      await result;
    }
    const outcome = await result;
    expect(outcome).toBeInstanceOf(AggregateError);
    expect((outcome as AggregateError).errors).toEqual([first, second]);
    expect((outcome as AggregateError).cause).toBe(first);
    for (const resource of resources) expect(resource.dispose).toHaveBeenCalledTimes(1);
  });

  it('preserves one failure identity and accepts an empty ownership set', async () => {
    const failure = new Error('single cleanup failure');
    await expect(
      disposeOwnedResources([
        {
          dispose: () => {
            throw failure;
          },
        },
      ]),
    ).rejects.toBe(failure);
    await expect(disposeOwnedResources([])).resolves.toBeUndefined();
  });
});

describe('model-neutral initialized session ownership', () => {
  it('registers before initialization and returns the same initialized object', async () => {
    const initialize = vi.fn<() => Promise<void>>();
    const session = { initialize, dispose: vi.fn(async () => {}), marker: 'retained type' };
    const owner = new Set<typeof session>();
    initialize.mockImplementation(async () => {
      expect(owner.has(session)).toBe(true);
    });
    const result = await initializeOwnedSession(session, owner);
    expect(result).toBe(session);
    expect(result.marker).toBe('retained type');
    expect(owner.has(session)).toBe(true);
    expect(session.dispose).not.toHaveBeenCalled();
  });

  it('removes failed ownership and awaits cleanup before rejecting', async () => {
    const failure = new Error('initialize failed');
    let finishCleanup = () => {};
    const gate = new Promise<void>((resolve) => {
      finishCleanup = resolve;
    });
    const session = {
      initialize: vi.fn(async () => {
        throw failure;
      }),
      dispose: vi.fn(() => gate),
    };
    const owner = new Set<typeof session>();
    let settled = false;
    const result = initializeOwnedSession(session, owner).catch((error: unknown) => {
      settled = true;
      return error;
    });
    try {
      await vi.waitFor(() => expect(session.dispose).toHaveBeenCalledTimes(1));
      expect(owner.has(session)).toBe(false);
      expect(settled).toBe(false);
    } finally {
      finishCleanup();
      expect(await result).toBe(failure);
    }
  });

  it('keeps original initialization and asynchronous cleanup errors', async () => {
    const failure = new Error('initialize failed');
    const cleanupFailure = new Error('cleanup failed');
    const session = {
      initialize: vi.fn(async () => {
        throw failure;
      }),
      dispose: vi.fn(async () => {
        throw cleanupFailure;
      }),
    };
    const owner = new Set<typeof session>();
    const result = await initializeOwnedSession(session, owner).catch((error: unknown) => error);
    expect(result).toBeInstanceOf(AggregateError);
    expect((result as AggregateError).errors).toEqual([failure, cleanupFailure]);
    expect((result as AggregateError).cause).toBe(failure);
    expect(owner.size).toBe(0);
  });

  it('handles a synchronous initialization throw and only removes the failed sibling', async () => {
    const failure = new Error('sync initialize failed');
    const sibling = { initialize: vi.fn(async () => {}), dispose: vi.fn(async () => {}) };
    const failed = {
      initialize() {
        throw failure;
      },
      dispose: vi.fn(async () => {}),
    };
    const owner = new Set<{ initialize(): Promise<void>; dispose(): Promise<void> }>([sibling]);
    await expect(initializeOwnedSession(failed, owner)).rejects.toBe(failure);
    expect([...owner]).toEqual([sibling]);
    expect(failed.dispose).toHaveBeenCalledTimes(1);
    expect(sibling.dispose).not.toHaveBeenCalled();
  });
});

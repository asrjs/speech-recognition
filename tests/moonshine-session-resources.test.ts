import { describe, expect, it } from 'vitest';
import {
  releaseMoonshineSessions,
  settleMoonshineSessionCreates,
} from '../src/models/moonshine/session-resources.js';

describe('Moonshine session resource ownership', () => {
  it('waits for late create successes and releases every fulfilled session after a sibling fails', async () => {
    const released: string[] = [];
    let lateCreateCompleted = false;
    const session = (id: string) => ({
      release: () => {
        released.push(id);
      },
    });

    const pending = [
      Promise.resolve(session('first')),
      new Promise<ReturnType<typeof session>>((_, reject) => {
        setTimeout(() => reject(new Error('graph creation failed')), 2);
      }),
      new Promise<ReturnType<typeof session>>((resolve) => {
        setTimeout(() => {
          lateCreateCompleted = true;
          resolve(session('late'));
        }, 10);
      }),
    ];

    await expect(settleMoonshineSessionCreates(pending)).rejects.toThrow('graph creation failed');
    expect(lateCreateCompleted).toBe(true);
    expect(released.sort()).toEqual(['first', 'late']);
  });

  it('surfaces release failures after attempting every session', async () => {
    const attempted: string[] = [];
    await expect(
      releaseMoonshineSessions([
        {
          release: () => {
            attempted.push('first');
            throw new Error('release failed');
          },
        },
        {
          release: () => {
            attempted.push('second');
          },
        },
      ]),
    ).rejects.toThrow('release failed');
    expect(attempted.sort()).toEqual(['first', 'second']);
  });

  it('preserves both graph creation and cleanup failures', async () => {
    const pending = [
      Promise.resolve({
        release: () => {
          throw new Error('release failed');
        },
      }),
      Promise.reject(new Error('create failed')),
    ];
    await expect(settleMoonshineSessionCreates(pending)).rejects.toThrow(
      'Moonshine session creation and cleanup failed.',
    );
  });
});

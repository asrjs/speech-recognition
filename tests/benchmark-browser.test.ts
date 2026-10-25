import { measureBrowserMemory } from '@asrjs/speech-recognition/bench/browser';
import { describe, expect, it, vi } from 'vitest';

describe('browser benchmark measurements', () => {
  it('bounds a pending optional measurement without inventing memory bytes', async () => {
    vi.useFakeTimers();
    try {
      const pending = measureBrowserMemory({
        performance: { measureUserAgentSpecificMemory: () => new Promise(() => {}) },
        timeoutMs: 1000,
      });
      await vi.advanceTimersByTimeAsync(1000);
      await expect(pending).resolves.toMatchObject({
        source: 'unavailable', scope: 'unavailable', bytes: null, reason: 'timeout',
      });
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it('clears the deadline after a successful measurement', async () => {
    vi.useFakeTimers();
    try {
      await expect(measureBrowserMemory({
        performance: { measureUserAgentSpecificMemory: async () => ({ bytes: 4096 }) },
        timeoutMs: 1000,
      })).resolves.toMatchObject({ bytes: 4096 });
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it('records User-Agent Specific Memory results with explicit scope', async () => {
    const snapshot = await measureBrowserMemory({
      performance: {
        measureUserAgentSpecificMemory: async () => ({ bytes: 4096 }),
      },
      now: () => '2026-08-26T00:00:00.000Z',
    });

    expect(snapshot).toEqual({
      capturedAt: '2026-08-26T00:00:00.000Z',
      source: 'measure-user-agent-specific-memory',
      scope: 'process',
      bytes: 4096,
    });
  });

  it('reports unsupported and invalid measurements without inventing bytes', async () => {
    await expect(
      measureBrowserMemory({ performance: null, now: () => 'unsupported' }),
    ).resolves.toEqual({
      capturedAt: 'unsupported',
      source: 'unavailable',
      scope: 'unavailable',
      bytes: null,
      reason: 'unsupported',
    });

    await expect(
      measureBrowserMemory({
        performance: { measureUserAgentSpecificMemory: async () => ({ bytes: Number.NaN }) },
        now: () => 'invalid',
      }),
    ).resolves.toMatchObject({ bytes: null, reason: 'invalid-result' });
  });

  it('turns measurement failures into structured unavailable snapshots', async () => {
    const snapshot = await measureBrowserMemory({
      performance: {
        measureUserAgentSpecificMemory: async () => {
          throw new Error('not cross-origin isolated');
        },
      },
      now: () => 'failed',
    });

    expect(snapshot).toMatchObject({
      capturedAt: 'failed',
      bytes: null,
      reason: 'measurement-failed',
    });
  });
});

import { measureBrowserMemory } from '@asrjs/speech-recognition/bench/browser';
import { describe, expect, it } from 'vitest';

describe('browser benchmark measurements', () => {
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

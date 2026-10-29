import {
  MemoryAssetCache,
  UrlAssetHandle,
  reportSpeechWarning,
  setSpeechDiagnosticsLogger,
} from '@asrjs/speech-recognition/io';
import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(() => {
  setSpeechDiagnosticsLogger();
  vi.restoreAllMocks();
});

describe('speech diagnostics logger', () => {
  it('keeps the previous console.warn output by default', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const error = new Error('cache unavailable');

    reportSpeechWarning('[assets] Cache write failed.', { error });
    reportSpeechWarning('plain message');

    expect(warn).toHaveBeenNthCalledWith(1, '[assets] Cache write failed.', error);
    expect(warn).toHaveBeenNthCalledWith(2, 'plain message');
  });

  it('redirects warnings to an application logger and can silence them', () => {
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    const logger = { warn: vi.fn() };

    setSpeechDiagnosticsLogger(logger);
    reportSpeechWarning('redirected', { error: 'x' });
    setSpeechDiagnosticsLogger(null);
    reportSpeechWarning('silenced');

    expect(logger.warn).toHaveBeenCalledOnce();
    expect(logger.warn).toHaveBeenCalledWith('redirected', { error: 'x' });
    expect(consoleWarn).not.toHaveBeenCalled();
  });

  it('never lets a failing logger break the reported operation', () => {
    setSpeechDiagnosticsLogger({
      warn() {
        throw new Error('logger failed');
      },
    });

    expect(() => reportSpeechWarning('still fine')).not.toThrow();
  });

  it('receives asset cache fallbacks through the configured logger', async () => {
    const logger = { warn: vi.fn() };
    setSpeechDiagnosticsLogger(logger);
    const cache = new MemoryAssetCache();
    vi.spyOn(cache, 'set').mockRejectedValue(new Error('quota exceeded'));
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(new Uint8Array([1, 2, 3]), { status: 200 })),
    );
    const handle = new UrlAssetHandle(
      {
        id: 'diagnostics-cache',
        provider: 'url',
        url: 'https://example.com/model.bin',
        cacheKey: 'diagnostics-cache',
      },
      'https://example.com/model.bin',
      cache,
    );

    await expect(handle.readBytes()).resolves.toEqual(new Uint8Array([1, 2, 3]));

    expect(logger.warn).toHaveBeenCalledWith(
      '[assets] Cache write failed for "diagnostics-cache". Continuing without cache.',
      { error: expect.objectContaining({ message: 'quota exceeded' }) },
    );
    vi.unstubAllGlobals();
  });
});

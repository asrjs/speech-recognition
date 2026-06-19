import { describe, it, expect, vi } from 'vitest';
import { EnhancedWhisperExecutor } from '../src/models/whisper-seq2seq/enhanced-executor.js';
import type { VadSegmenterConfig } from '../src/models/whisper-seq2seq/enhanced-types.js';
import type { WhisperVadBackend } from '../src/chunking/types.js';

function mt(text: string): any {
  return { utteranceText: text, isFinal: true, language: 'en', segments: [], words: [] };
}
function mv() { return { transcribe: vi.fn(), dispose: vi.fn() }; }
function ma(s: number): any { const a = new Float32Array(16000*s); (a as any).sampleRate = 16000; return a; }

describe('EnhancedWhisperExecutor', () => {
  it('delegates single-chunk to vanilla', async () => {
    const v = mv(); v.transcribe.mockResolvedValue(mt('hello world'));
    const e = new EnhancedWhisperExecutor(v as any);
    const r: any = await e.transcribe(ma(1), { language: 'en', task: 'transcribe' as const, temperatureFallback: false } as any, undefined as any);
    expect(r.utteranceText).toBe('hello world');
    expect(v.transcribe).toHaveBeenCalledTimes(1);
  });

  it('compression gate catches repetitive text', async () => {
    const v = mv();
    v.transcribe.mockResolvedValueOnce(mt('the the the the the the the the the the'));
    v.transcribe.mockResolvedValueOnce(mt('good'));
    const e = new EnhancedWhisperExecutor(v as any);
    const r: any = await e.transcribe(ma(1), { language: 'en', task: 'transcribe' as const, temperatureFallback: true, temperatures: [0.0, 0.2] } as any, undefined as any);
    expect(r.utteranceText).toBe('good');
    expect(v.transcribe).toHaveBeenCalledTimes(2);
    expect(v.transcribe.mock.calls.map((call) => call[1].temperature)).toEqual([0.0, 0.2]);
  });

  it('preserves caller token-logit callback during fallback collection', async () => {
    const callerOnTokenLogits = vi.fn();
    const v = mv();
    v.transcribe.mockImplementation(async (_audio: any, options: any) => {
      options.onTokenLogits(42, new Float32Array([0, 1]), { tokens: [42], beginIndex: 0 });
      return mt('good');
    });
    const e = new EnhancedWhisperExecutor(v as any);
    await e.transcribe(
      ma(1),
      {
        language: 'en',
        task: 'transcribe' as const,
        temperatureFallback: true,
        temperatures: [0.0],
        onTokenLogits: callerOnTokenLogits,
      } as any,
      undefined as any,
    );
    expect(callerOnTokenLogits).toHaveBeenCalledWith(42, expect.any(Float32Array), { tokens: [42], beginIndex: 0 });
  });

  it('VAD multi-chunk pipeline', async () => {
    const v = mv();
    v.transcribe.mockResolvedValueOnce(mt('chunk1'));
    v.transcribe.mockResolvedValueOnce(mt('chunk2'));
    const vad: WhisperVadBackend = { async segment() { return [{ startSeconds:0, endSeconds:2, durationSeconds:2 }, { startSeconds:2.5, endSeconds:4.5, durationSeconds:2 }]; } };
    const cfg: VadSegmenterConfig = { backend: 'ten-vad' };
    const e = new EnhancedWhisperExecutor(v as any, cfg, vad);
    const r: any = await e.transcribe(ma(5), { language: 'en', task: 'transcribe' as const, temperatureFallback: false } as any, undefined as any);
    expect(r.utteranceText).toBe('chunk1 chunk2');
    expect(v.transcribe).toHaveBeenCalledTimes(2);
  });

  it('passes fallback temperatures through VAD chunk retries', async () => {
    const v = mv();
    v.transcribe.mockResolvedValueOnce(mt('the the the the the the the the the the'));
    v.transcribe.mockResolvedValueOnce(mt('chunk ok'));
    const vad: WhisperVadBackend = {
      async segment() {
        return [{ startSeconds: 0, endSeconds: 2, durationSeconds: 2 }];
      },
    };
    const cfg: VadSegmenterConfig = { backend: 'ten-vad' };
    const e = new EnhancedWhisperExecutor(v as any, cfg, vad);
    const r: any = await e.transcribe(
      ma(3),
      { language: 'en', task: 'transcribe' as const, temperatureFallback: true, temperatures: [0.0, 0.2] } as any,
      undefined as any,
    );
    expect(r.utteranceText).toBe('chunk ok');
    expect(v.transcribe.mock.calls.map((call) => call[1].temperature)).toEqual([0.0, 0.2]);
  });

  it('disposes vanilla', async () => {
    const v = mv(); const e = new EnhancedWhisperExecutor(v as any);
    await e.dispose(); expect(v.dispose).toHaveBeenCalledTimes(1);
  });

  it('plain vanilla fallback', async () => {
    const v = mv(); v.transcribe.mockResolvedValue(mt('direct'));
    const e = new EnhancedWhisperExecutor(v as any);
    const r: any = await e.transcribe(ma(1), { language: 'en', task: 'transcribe' as const, temperatureFallback: false } as any, undefined as any);
    expect(r.utteranceText).toBe('direct');
  });
});

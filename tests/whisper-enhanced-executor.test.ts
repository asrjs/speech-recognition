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
    expect(v.transcribe.mock.calls.map((call: any[]) => call[0].numberOfFrames)).toEqual([38400, 44800]);
  });

  it('slices AudioBufferLike VAD chunks and preserves native timing details', async () => {
    const v = mv();
    const nativeChunk = (text: string, warning?: { code: string; message: string }) => ({
      ...mt(text),
      segments: [{ index: 0, text, startTime: 0, endTime: 0.5, confidence: 0.9 }],
      words: [{ index: 0, text, startTime: 0, endTime: 0.5, confidence: 0.8 }],
      ...(warning ? { warnings: [warning] } : {}),
    });
    v.transcribe
      .mockResolvedValueOnce(nativeChunk('chunk1'))
      .mockResolvedValueOnce(nativeChunk('chunk2', { code: 'test.warning', message: 'kept' }));
    const frames = new Float32Array(5 * 16000);
    const audio = {
      sampleRate: 16000,
      numberOfChannels: 1,
      numberOfFrames: frames.length,
      durationSeconds: 5,
      channels: [frames],
      format: 'f32-planar' as const,
    };
    const vad: WhisperVadBackend = {
      async segment() {
        return [
          { startSeconds: 0, endSeconds: 2, durationSeconds: 2 },
          { startSeconds: 2.5, endSeconds: 4.5, durationSeconds: 2 },
        ];
      },
    };
    const e = new EnhancedWhisperExecutor(v as any, { backend: 'ten-vad' }, vad);
    const result: any = await e.transcribe(
      audio,
      { language: 'en', task: 'transcribe' as const, temperatureFallback: false } as any,
      undefined as any,
    );

    expect(v.transcribe.mock.calls.map((call: any[]) => call[0].numberOfFrames)).toEqual([38400, 44800]);
    expect(result.utteranceText).toBe('chunk1 chunk2');
    expect(result.segments.map((segment: any) => segment.startTime)).toEqual([0, 2.1]);
    expect(result.words.map((word: any) => word.startTime)).toEqual([0, 2.1]);
    expect(result.warnings).toEqual([{ code: 'test.warning', message: 'kept' }]);
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

  it('rejects low selected-sequence logprob and recovers with temperature', async () => {
    const v = mv();
    v.transcribe.mockImplementation(async (_audio: any, options: any) => {
      if ((options.temperature ?? 0) === 0) {
        return {
          ...mt('uncertain words here'),
          tokenTraces: [
            { tokenId: 10, logProb: -3.1, entropy: 0.4 },
            { tokenId: 11, logProb: -2.7, entropy: 0.5 },
          ],
        };
      }
      return {
        ...mt('recovered text'),
        tokenTraces: [
          { tokenId: 20, logProb: -0.12, entropy: 0.2 },
        ],
      };
    });
    const e = new EnhancedWhisperExecutor(v as any);
    const r: any = await e.transcribe(
      ma(1),
      {
        language: 'en',
        task: 'transcribe' as const,
        temperatureFallback: true,
        temperatures: [0.0, 0.2],
        numBeams: 2,
      } as any,
      undefined as any,
    );
    expect(r.utteranceText).toBe('recovered text');
    expect(v.transcribe).toHaveBeenCalledTimes(2);
    expect(v.transcribe.mock.calls.map((call: any) => call[1].temperature)).toEqual([0.0, 0.2]);
    expect(v.transcribe.mock.calls[0][1].trackQuality).toBe(true);
    expect(v.transcribe.mock.calls[0][1].numBeams).toBe(2);
  });

  it('rejects repetitive beam text via compression and recovers', async () => {
    const v = mv();
    v.transcribe.mockImplementation(async (_audio: any, options: any) => {
      if ((options.temperature ?? 0) === 0) {
        return {
          ...mt('the the the the the the the the the the'),
          tokenTraces: [
            { tokenId: 1, logProb: -0.05, entropy: 0.1 },
            { tokenId: 1, logProb: -0.04, entropy: 0.1 },
          ],
        };
      }
      return {
        ...mt('natural sentence after retry'),
        tokenTraces: [{ tokenId: 7, logProb: -0.08, entropy: 0.15 }],
      };
    });
    const e = new EnhancedWhisperExecutor(v as any);
    const r: any = await e.transcribe(
      ma(1),
      {
        language: 'en',
        task: 'transcribe' as const,
        temperatureFallback: true,
        temperatures: [0.0, 0.4],
        numBeams: 3,
      } as any,
      undefined as any,
    );
    expect(r.utteranceText).toBe('natural sentence after retry');
    expect(v.transcribe.mock.calls.map((call: any) => call[1].temperature)).toEqual([0.0, 0.4]);
  });

  it('does not retain full-vocabulary logits when selected traces exist', async () => {
    const logits = new Float32Array(64);
    logits[3] = 8;
    const v = mv();
    v.transcribe.mockImplementation(async (_audio: any, options: any) => {
      options.onTokenLogits?.(3, logits, { tokens: [3], beginIndex: 0 });
      logits.fill(99);
      return {
        ...mt('trace native'),
        tokenTraces: [{ tokenId: 3, logProb: -0.04, entropy: 0.12 }],
      };
    });
    const e = new EnhancedWhisperExecutor(v as any);
    const r: any = await e.transcribe(
      ma(1),
      { language: 'en', task: 'transcribe' as const, temperatureFallback: true, temperatures: [0.0] } as any,
      undefined as any,
    );
    expect(r.utteranceText).toBe('trace native');
    expect(v.transcribe).toHaveBeenCalledTimes(1);
  });
});

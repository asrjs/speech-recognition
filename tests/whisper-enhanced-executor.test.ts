/**
 * Tests for EnhancedWhisperExecutor — composition wrapper.
 * Phase 8: integration test with mock vanilla executor.
 */

import { describe, it, expect, vi } from 'vitest';
import { EnhancedWhisperExecutor } from '../src/models/whisper-seq2seq/enhanced-executor.js';
import type {
  EnhancedDecodeOptions,
  VadSegmenterConfig,
} from '../src/models/whisper-seq2seq/enhanced-types.js';
import type { WhisperVadBackend, VadSpeechSegment } from '../src/models/whisper-seq2seq/vad-segmenter.js';
import type { WhisperNativeTranscript } from '../src/models/whisper-seq2seq/types.js';

// ---------------------------------------------------------------------------
// Mock vanilla executor
// ---------------------------------------------------------------------------

function createMockVanilla(): { transcribe: ReturnType<typeof vi.fn>; dispose: ReturnType<typeof vi.fn> } {
  return {
    transcribe: vi.fn(),
    dispose: vi.fn(),
  };
}

function makeTranscript(text: string, segments: Array<{ start: number; end: number; words: Array<{ start: number; end: number; word: string; probability: number }> }> = []): WhisperNativeTranscript {
  return {
    text,
    segments: segments.map((s, i) => ({
      id: i,
      start: s.start,
      end: s.end,
      text: s.words.map(w => w.word).join(' '),
      words: s.words.map(w => ({ ...w, start: w.start, end: w.end })),
    })),
    language: 'en',
    duration: segments.reduce((sum, s) => sum + (s.end - s.start), 0),
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('EnhancedWhisperExecutor', () => {
  it('works with same API as vanilla executor', async () => {
    const vanilla = createMockVanilla();
    vanilla.transcribe.mockResolvedValue(makeTranscript('hello world'));

    const executor = new EnhancedWhisperExecutor(vanilla as any);
    const result = await executor.transcribe(
      new Float32Array(16000),
      { language: 'en', task: 'transcribe' as const, noTimestamps: true },
      undefined as any,
    );

    expect(result.text).toBe('hello world');
    expect(vanilla.transcribe).toHaveBeenCalledTimes(1);
  });

  it('delegates non-enhanced options to vanilla', async () => {
    const vanilla = createMockVanilla();
    vanilla.transcribe.mockResolvedValue(makeTranscript('test'));

    const executor = new EnhancedWhisperExecutor(vanilla as any);
    await executor.transcribe(
      new Float32Array(16000),
      { language: 'en', task: 'transcribe' as const, noTimestamps: true },
      undefined as any,
    );

    // Verify vanilla was called with correct options
    const callArgs = vanilla.transcribe.mock.calls[0]![1];
    expect(callArgs.language).toBe('en');
  });

  it('disposes underlying vanilla executor', async () => {
    const vanilla = createMockVanilla();
    const executor = new EnhancedWhisperExecutor(vanilla as any);
    await executor.dispose();
    expect(vanilla.dispose).toHaveBeenCalledTimes(1);
  });

  it('passes VAD config to constructor', () => {
    const vanilla = createMockVanilla();
    const vadConfig: VadSegmenterConfig = {
      backend: 'ten-vad',
      speechThreshold: 0.7,
    };
    const executor = new EnhancedWhisperExecutor(vanilla as any, vadConfig);
    expect(executor).toBeDefined();
  });

  it('handles quality gate enhanced options', async () => {
    const vanilla = createMockVanilla();
    vanilla.transcribe.mockResolvedValue(makeTranscript('good transcription'));

    const executor = new EnhancedWhisperExecutor(vanilla as any);
    const options: EnhancedDecodeOptions & any = {
      language: 'en',
      task: 'transcribe',
      noTimestamps: true,
      // Enhanced options
      compressionRatioThreshold: 3.0,
      temperatureFallback: false,
      conditionOnPreviousText: false,
    };

    const result = await executor.transcribe(
      new Float32Array(16000),
      options,
      undefined as any,
    );
    expect(result.text).toBe('good transcription');
  });
});

describe('EnhancedWhisperExecutor with VAD', () => {
  it('uses VAD backend when configured', async () => {
    const vanilla = createMockVanilla();
    vanilla.transcribe.mockResolvedValue(makeTranscript('chunk text'));

    const mockVad: WhisperVadBackend = {
      async segment(_audio, _sampleRate, _threshold) {
        return [
          { startSeconds: 0, endSeconds: 2.0, durationSeconds: 2.0 },
          { startSeconds: 2.5, endSeconds: 4.5, durationSeconds: 2.0 },
        ];
      },
    };

    const vadConfig: VadSegmenterConfig = {
      backend: 'ten-vad',
      speechThreshold: 0.5,
    };

    const executor = new EnhancedWhisperExecutor(vanilla as any, vadConfig, mockVad);
    await executor.transcribe(
      new Float32Array(16000 * 10),
      { language: 'en', task: 'transcribe' as const, conditionOnPreviousText: false },
      undefined as any,
    );

    // Should call vanilla.transcribe twice (once per VAD segment)
    expect(vanilla.transcribe).toHaveBeenCalledTimes(2);
  });
});

import {
  ProductionWhisperPipeline,
  createWhisperProductionPipeline,
  type ProductionWhisperPipelineOptions,
  type ProductionTranscript,
} from '@asrjs/speech-recognition/pipeline';
import { describe, expect, it } from 'vitest';

// ---------------------------------------------------------------------------
// Mock EnhancedWhisperExecutor
// ---------------------------------------------------------------------------

function makeMockEnhancedExecutor(mockText = 'hello world.', mockWords?: any[]) {
  const defaultWords = [
    { word: 'hello', start: 0.0, end: 0.5, probability: 0.95 },
    { word: 'world', start: 0.6, end: 1.0, probability: 0.9 },
  ];

  return {
    transcribe: async () => ({
      utteranceText: mockText,
      isFinal: true,
      language: 'en',
      segments: [{
        text: mockText,
        start: 0,
        end: 1.0,
        words: mockWords ?? defaultWords,
      }],
      words: mockWords ?? defaultWords,
    }),
    dispose: async () => {},
  } as any;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('ProductionWhisperPipeline', () => {
  it('transcribes audio through pipeline stages', async () => {
    const mockExecutor = makeMockEnhancedExecutor();
    const pipeline = createWhisperProductionPipeline({
      enhancedExecutor: mockExecutor,
      outputFormats: ['sentences', 'srt'],
    });

    const audio = new Float32Array(16000); // 1 second @ 16kHz
    const result = await pipeline.transcribe(audio, { sampleRate: 16000 });

    expect(result).toBeDefined();
    // Raw text from executor (mock returns 'hello world.')
    expect(result.text).toBe('hello world.');
    expect(result.raw).toBe('hello world.');
    expect(result.metrics.duration).toBeGreaterThan(0);
    expect(result.metrics.wordCount).toBe(2);
  });

  it('includes sentence segmentation', async () => {
    const mockExecutor = makeMockEnhancedExecutor('first sentence. second sentence.');
    const pipeline = createWhisperProductionPipeline({
      enhancedExecutor: mockExecutor,
      outputFormats: ['sentences'],
    });

    const audio = new Float32Array(16000);
    const result = await pipeline.transcribe(audio, { sampleRate: 16000 });

    expect(result.sentences.length).toBeGreaterThanOrEqual(1);
    expect(result.text).toContain('first sentence');
  });

  it('generates SRT subtitles when requested', async () => {
    const mockExecutor = makeMockEnhancedExecutor('hello world.');
    const pipeline = createWhisperProductionPipeline({
      enhancedExecutor: mockExecutor,
      outputFormats: ['sentences', 'srt', 'vtt'],
    });

    const audio = new Float32Array(16000);
    const result = await pipeline.transcribe(audio, { sampleRate: 16000 });

    expect(result.subtitles.srt).toBeTruthy();
    expect(result.subtitles.srt).toContain('00:00:00');
    expect(result.subtitles.vtt).toBeTruthy();
  });

  it('returns metrics in sidecars', async () => {
    const mockExecutor = makeMockEnhancedExecutor('test.');
    const pipeline = createWhisperProductionPipeline({
      enhancedExecutor: mockExecutor,
      outputFormats: ['sentences'],
    });

    const audio = new Float32Array(16000);
    const result = await pipeline.transcribe(audio, { sampleRate: 16000 });

    expect(result.metrics).toBeDefined();
    expect(result.metrics.duration).toBeGreaterThan(0);
    expect(result.metrics.wordCount).toBeGreaterThanOrEqual(1);
  });

  it('passes through VAD options', async () => {
    const mockExecutor = makeMockEnhancedExecutor();
    const pipeline = createWhisperProductionPipeline({
      enhancedExecutor: mockExecutor,
      outputFormats: ['sentences'],
      vadConfig: {
        minSilenceDurationMs: 200,
        maxSegmentDurationMs: 15000,
      },
    });

    const audio = new Float32Array(16000);
    const result = await pipeline.transcribe(audio, {
      sampleRate: 16000,
      language: 'en',
    });

    expect(result.text).toBeDefined();
  });

  it('handles empty audio gracefully', async () => {
    const mockExecutor = {
      transcribe: async () => ({
        utteranceText: '[no speech detected]',
        isFinal: true,
        language: 'en',
        segments: [],
        words: [],
      }),
      dispose: async () => {},
    } as any;

    const pipeline = createWhisperProductionPipeline({
      enhancedExecutor: mockExecutor,
      outputFormats: ['sentences'],
    });

    const audio = new Float32Array(0);
    const result = await pipeline.transcribe(audio, { sampleRate: 16000 });

    expect(result.text).toBe('[no speech detected]');
    expect(result.metrics.wordCount).toBe(0);
  });

  it('disposes underlying executor', async () => {
    let disposed = false;
    const mockExecutor = {
      transcribe: async () => ({ utteranceText: '', isFinal: true, language: 'en', segments: [], words: [] }),
      dispose: async () => { disposed = true; },
    } as any;

    const pipeline = createWhisperProductionPipeline({
      enhancedExecutor: mockExecutor,
      outputFormats: ['sentences'],
    });

    await pipeline.dispose();
    expect(disposed).toBe(true);
  });
});

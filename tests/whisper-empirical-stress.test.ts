import { describe, expect, it, vi } from 'vitest';
import {
  shouldChunkWhisperAudio,
  transcribeWhisperLongAudio,
  mergeWhisperChunkTranscripts,
} from '../src/models/whisper-seq2seq/chunking.js';
import {
  hasVerifiedWhisperDecoderAlignment,
  executeAlignmentReferenceEncoder,
  finalizeMergedWhisperTranscript,
} from '../src/models/whisper-seq2seq/executor-alignment.js';
import { WhisperOnnxExecutor } from '../src/models/whisper-seq2seq/executor.js';
import type { OrtSessionLike, OrtTensorLike } from '../src/models/whisper-seq2seq/ort.js';
import type { AudioBufferLike } from '../src/types/index.js';
import type {
  WhisperNativeToken,
  WhisperNativeTranscript,
  WhisperSeq2SeqModelConfig,
  WhisperSeq2SeqTranscriptionOptions,
} from '../src/models/whisper-seq2seq/types.js';

const mockModelConfig: WhisperSeq2SeqModelConfig = {
  ecosystem: 'openai',
  architecture: 'whisper-seq2seq',
  processorArchitecture: 'whisper-mel',
  encoderArchitecture: 'whisper-transformer',
  decoderArchitecture: 'transformer-decoder',
  sampleRate: 16000,
  melBins: 80,
  maxSourcePositions: 1500,
  maxTargetPositions: 448,
  languages: ['en'],
  tokenizer: { kind: 'bpe', vocabSize: 51865 },
};

function createMockSession(_id: string): OrtSessionLike & { released: boolean } {
  return {
    inputNames: ['input'],
    outputNames: ['output'],
    run: vi.fn().mockResolvedValue({}),
    release: vi.fn(),
    released: false,
  };
}

describe('Challenger Stress: WhisperOnnxExecutor.lazyLoadSession Concurrency & Disposal', () => {
  it('Scenario 1: caches session on sequential calls', async () => {
    const executor = new WhisperOnnxExecutor('test-model', mockModelConfig, 'wasm');
    const lazyLoad = (executor as any).lazyLoadSession.bind(executor);

    let cached: OrtSessionLike | undefined;
    let promise: Promise<OrtSessionLike | undefined> | undefined;
    let creationCalls = 0;

    const createFn = async () => {
      creationCalls++;
      return createMockSession('session-1');
    };

    const s1 = await lazyLoad(
      () => cached,
      (s: OrtSessionLike) => { cached = s; },
      () => promise,
      (p: any) => { promise = p; },
      createFn,
    );

    const s2 = await lazyLoad(
      () => cached,
      (s: OrtSessionLike) => { cached = s; },
      () => promise,
      (p: any) => { promise = p; },
      createFn,
    );

    expect(creationCalls).toBe(1);
    expect(s1).toBe(s2);
    expect(cached).toBe(s1);
  });

  it('Scenario 2: stampede concurrency deduplicates in-flight creation', async () => {
    const executor = new WhisperOnnxExecutor('test-model', mockModelConfig, 'wasm');
    const lazyLoad = (executor as any).lazyLoadSession.bind(executor);

    let cached: OrtSessionLike | undefined;
    let promise: Promise<OrtSessionLike | undefined> | undefined;
    let creationCalls = 0;

    const createFn = async () => {
      creationCalls++;
      await new Promise((resolve) => setTimeout(resolve, 30));
      return createMockSession('session-stampede');
    };

    const callers = Array.from({ length: 10 }, () =>
      lazyLoad(
        () => cached,
        (s: OrtSessionLike) => { cached = s; },
        () => promise,
        (p: any) => { promise = p; },
        createFn,
      ),
    );

    const results = await Promise.all(callers);

    expect(creationCalls).toBe(1);
    expect(results.length).toBe(10);
    const first = results[0];
    for (const res of results) {
      expect(res).toBe(first);
    }
    expect(cached).toBe(first);
  });

  it('Scenario 3: disposal during in-flight creation returns undefined and releases created session', async () => {
    const executor = new WhisperOnnxExecutor('test-model', mockModelConfig, 'wasm');
    const lazyLoad = (executor as any).lazyLoadSession.bind(executor);

    let cached: OrtSessionLike | undefined;
    let promise: Promise<OrtSessionLike | undefined> | undefined;
    let sessionReleased = false;

    const mockSession = {
      ...createMockSession('session-race'),
      release: vi.fn(() => {
        sessionReleased = true;
      }),
    };

    let finishCreation: () => void;
    const creationStarted = new Promise<void>((r) => { finishCreation = r; });

    const createFn = async () => {
      await creationStarted;
      return mockSession;
    };

    // Start lazy load
    const loadPromise = lazyLoad(
      () => cached,
      (s: OrtSessionLike) => { cached = s; },
      () => promise,
      (p: any) => { promise = p; },
      createFn,
    );

    // Call dispose while in flight
    (executor as any).disposed = true;

    // Now let the creation finish
    finishCreation!();

    const result = await loadPromise;

    // Disposed session loader must return undefined
    expect(result).toBeUndefined();
    // Dead session must never be cached
    expect(cached).toBeUndefined();
    // Session created during disposal must be released
    expect(mockSession.release).toHaveBeenCalled();
    expect(sessionReleased).toBe(true);
  });

  it('Scenario 4: immediate undefined when called after executor is already disposed', async () => {
    const executor = new WhisperOnnxExecutor('test-model', mockModelConfig, 'wasm');
    (executor as any).disposed = true;

    const lazyLoad = (executor as any).lazyLoadSession.bind(executor);
    let creationCalls = 0;

    const result = await lazyLoad(
      () => undefined,
      () => {},
      () => undefined,
      () => {},
      async () => {
        creationCalls++;
        return createMockSession('session');
      },
    );

    expect(result).toBeUndefined();
    expect(creationCalls).toBe(0);
  });

  it('Scenario 5: creation error resets in-flight promise and permits retry', async () => {
    const executor = new WhisperOnnxExecutor('test-model', mockModelConfig, 'wasm');
    const lazyLoad = (executor as any).lazyLoadSession.bind(executor);

    let cached: OrtSessionLike | undefined;
    let promise: Promise<OrtSessionLike | undefined> | undefined;
    let attempt = 0;

    const createFn = async () => {
      attempt++;
      if (attempt === 1) {
        throw new Error('Simulated network load failure');
      }
      return createMockSession('session-retry');
    };

    // First attempt fails
    await expect(
      lazyLoad(
        () => cached,
        (s: OrtSessionLike) => { cached = s; },
        () => promise,
        (p: any) => { promise = p; },
        createFn,
      ),
    ).rejects.toThrow('Simulated network load failure');

    // Promise must be cleared
    expect(promise).toBeUndefined();
    expect(cached).toBeUndefined();

    // Second attempt succeeds
    const s2 = await lazyLoad(
      () => cached,
      (s: OrtSessionLike) => { cached = s; },
      () => promise,
      (p: any) => { promise = p; },
      createFn,
    );

    expect(s2).toBeDefined();
    expect(cached).toBe(s2);
    expect(attempt).toBe(2);
  });

  it('Scenario 6: concurrent stampede error propagates to all callers and resets promise', async () => {
    const executor = new WhisperOnnxExecutor('test-model', mockModelConfig, 'wasm');
    const lazyLoad = (executor as any).lazyLoadSession.bind(executor);

    let cached: OrtSessionLike | undefined;
    let promise: Promise<OrtSessionLike | undefined> | undefined;

    const createFn = async () => {
      await new Promise((r) => setTimeout(r, 20));
      throw new Error('Stampede crash');
    };

    const callers = Array.from({ length: 5 }, () =>
      lazyLoad(
        () => cached,
        (s: OrtSessionLike) => { cached = s; },
        () => promise,
        (p: any) => { promise = p; },
        createFn,
      ),
    );

    const settled = await Promise.allSettled(callers);
    expect(settled.every((s) => s.status === 'rejected')).toBe(true);
    expect(promise).toBeUndefined();
  });

  it('Scenario 7: executor.dispose flushes in-flight lazy load promises and frees sessions', async () => {
    const executor = new WhisperOnnxExecutor('test-model', mockModelConfig, 'wasm');

    const s1 = createMockSession('decoder-align');
    const s2 = createMockSession('align-ref-encoder');
    const s3 = createMockSession('align-ref-decoder');

    (executor as any).decoderAlignLoadPromise = Promise.resolve(s1);
    (executor as any).alignmentReferenceEncoderLoadPromise = Promise.resolve(s2);
    (executor as any).alignmentReferenceDecoderAlignLoadPromise = Promise.resolve(s3);

    await executor.dispose();

    expect(s1.release).toHaveBeenCalled();
    expect(s2.release).toHaveBeenCalled();
    expect(s3.release).toHaveBeenCalled();

    expect((executor as any).decoderAlignSession).toBeUndefined();
    expect((executor as any).alignmentReferenceEncoderSession).toBeUndefined();
    expect((executor as any).alignmentReferenceDecoderAlignSession).toBeUndefined();
    expect((executor as any).decoderAlignLoadPromise).toBeUndefined();
    expect((executor as any).alignmentReferenceEncoderLoadPromise).toBeUndefined();
    expect((executor as any).alignmentReferenceDecoderAlignLoadPromise).toBeUndefined();
  });
});

describe('Challenger Stress: Chunking Audio Boundaries', () => {
  it('Boundary: 0 samples audio duration', () => {
    const audio: AudioBufferLike = {
      sampleRate: 16000,
      durationSeconds: 0,
      channels: [new Float32Array(0)],
      numberOfChannels: 1,
      numberOfFrames: 0,
    };

    expect(shouldChunkWhisperAudio(audio, {})).toBe(false);
  });

  it('Boundary: 1 sample audio duration', () => {
    const audio: AudioBufferLike = {
      sampleRate: 16000,
      durationSeconds: 1 / 16000,
      channels: [new Float32Array(1)],
      numberOfChannels: 1,
      numberOfFrames: 1,
    };

    expect(shouldChunkWhisperAudio(audio, {})).toBe(false);
  });

  it('Boundary: Exact 30s window boundary (480,000 samples at 16kHz)', () => {
    const exact30s: AudioBufferLike = {
      sampleRate: 16000,
      durationSeconds: 30.0,
      channels: [new Float32Array(480000)],
      numberOfChannels: 1,
      numberOfFrames: 480000,
    };

    // Exactly 30.0s is NOT > 30s, so chunking is not triggered
    expect(shouldChunkWhisperAudio(exact30s, {})).toBe(false);
  });

  it('Boundary: Just below 30s boundary (479,999 samples)', () => {
    const justUnder: AudioBufferLike = {
      sampleRate: 16000,
      durationSeconds: 479999 / 16000,
      channels: [new Float32Array(479999)],
      numberOfChannels: 1,
      numberOfFrames: 479999,
    };

    expect(shouldChunkWhisperAudio(justUnder, {})).toBe(false);
  });

  it('Boundary: Just above 30s boundary (480,001 samples)', () => {
    const justOver: AudioBufferLike = {
      sampleRate: 16000,
      durationSeconds: 480001 / 16000,
      channels: [new Float32Array(480001)],
      numberOfChannels: 1,
      numberOfFrames: 480001,
    };

    expect(shouldChunkWhisperAudio(justOver, {})).toBe(true);
  });

  it('Boundary: Custom chunkLengthSeconds & maxInputDurationSeconds', () => {
    const audio10s: AudioBufferLike = {
      sampleRate: 16000,
      durationSeconds: 10.0,
      channels: [new Float32Array(160000)],
      numberOfChannels: 1,
      numberOfFrames: 160000,
    };
    const audio10_1s: AudioBufferLike = {
      sampleRate: 16000,
      durationSeconds: 10.001,
      channels: [new Float32Array(160016)],
      numberOfChannels: 1,
      numberOfFrames: 160016,
    };

    expect(shouldChunkWhisperAudio(audio10s, { chunkLengthSeconds: 10 })).toBe(false);
    expect(shouldChunkWhisperAudio(audio10_1s, { chunkLengthSeconds: 10 })).toBe(true);

    expect(shouldChunkWhisperAudio(audio10s, { maxInputDurationSeconds: 10 })).toBe(false);
    expect(shouldChunkWhisperAudio(audio10_1s, { maxInputDurationSeconds: 10 })).toBe(true);
  });

  it('Options: windowing disabled or unsafeAllowOverMaxWindow overrides chunking', () => {
    const longAudio: AudioBufferLike = {
      sampleRate: 16000,
      durationSeconds: 120.0,
      channels: [new Float32Array(16000 * 120)],
      numberOfChannels: 1,
      numberOfFrames: 16000 * 120,
    };

    expect(shouldChunkWhisperAudio(longAudio, { windowing: 'disabled' })).toBe(false);
    expect(shouldChunkWhisperAudio(longAudio, { unsafeAllowOverMaxWindow: true })).toBe(false);
  });

  it('transcribeWhisperLongAudio executes planned chunks and sets unsafeAllowOverMaxWindow', async () => {
    const sampleRate = 16000;
    const totalSamples = 16000 * 65; // 65 seconds
    const audio: AudioBufferLike = {
      sampleRate,
      durationSeconds: 65,
      channels: [new Float32Array(totalSamples)],
      numberOfChannels: 1,
      numberOfFrames: totalSamples,
    };

    const chunkCalls: { start: number; frames: number; unsafe: boolean }[] = [];
    const transcribeChunk = async (
      chunkAudio: AudioBufferLike,
      opts: WhisperSeq2SeqTranscriptionOptions,
    ): Promise<WhisperNativeTranscript> => {
      chunkCalls.push({
        start: 0,
        frames: chunkAudio.numberOfFrames,
        unsafe: !!opts.unsafeAllowOverMaxWindow,
      });
      return {
        utteranceText: `chunk-${chunkCalls.length}`,
        isFinal: true,
        language: 'en',
        segments: [{
          index: 0,
          text: `chunk-${chunkCalls.length}`,
          startTime: 0,
          endTime: chunkAudio.durationSeconds,
          confidence: 0.9,
        }],
      };
    };

    const transcript = await transcribeWhisperLongAudio(
      audio,
      { chunkLengthSeconds: 30 },
      transcribeChunk,
    );

    expect(chunkCalls.length).toBeGreaterThan(1);
    expect(chunkCalls.every((c) => c.unsafe === true)).toBe(true);
    expect(transcript.utteranceText).toContain('chunk-1');
    expect(transcript.segments.length).toBe(chunkCalls.length);
  });

  it('transcribeWhisperLongAudio handles 1 sample gracefully', async () => {
    const audio: AudioBufferLike = {
      sampleRate: 16000,
      durationSeconds: 1 / 16000,
      channels: [new Float32Array([0.5])],
      numberOfChannels: 1,
      numberOfFrames: 1,
    };

    let called = false;
    const result = await transcribeWhisperLongAudio(audio, {}, async (chunkAudio) => {
      called = true;
      expect(chunkAudio.numberOfFrames).toBe(1);
      return {
        utteranceText: 'one-sample',
        isFinal: true,
        language: 'en',
        segments: [{ index: 0, text: 'one-sample', startTime: 0, endTime: 1 / 16000, confidence: 1.0 }],
      };
    });

    expect(called).toBe(true);
    expect(result.utteranceText).toBe('one-sample');
  });

  it('transcribeWhisperLongAudio with 0 samples triggers validation error in planWhisperChunks', async () => {
    const audio: AudioBufferLike = {
      sampleRate: 16000,
      durationSeconds: 0,
      channels: [new Float32Array(0)],
      numberOfChannels: 1,
      numberOfFrames: 0,
    };

    await expect(
      transcribeWhisperLongAudio(audio, {}, async () => {
        throw new Error('Should not reach transcribeChunk');
      }),
    ).rejects.toThrow('audioLengthSamples must be a positive integer.');
  });

  it('mergeWhisperChunkTranscripts combines chunks correctly', () => {
    const t1: WhisperNativeTranscript = {
      utteranceText: 'hello',
      isFinal: true,
      language: 'en',
      segments: [{ index: 0, text: 'hello', startTime: 0, endTime: 2, confidence: 0.9 }],
    };
    const t2: WhisperNativeTranscript = {
      utteranceText: 'world',
      isFinal: true,
      language: 'en',
      segments: [{ index: 0, text: 'world', startTime: 0, endTime: 2, confidence: 0.9 }],
    };
    const merged = mergeWhisperChunkTranscripts([
      { chunkStartTime: 0, transcript: t1 },
      { chunkStartTime: 10, transcript: t2 },
    ]);
    expect(merged.utteranceText).toBe('hello world');
    expect(merged.segments[1]?.startTime).toBe(10);
  });
});

describe('Challenger Stress: Alignment & finalizeMergedWhisperTranscript', () => {
  it('hasVerifiedWhisperDecoderAlignment requires boolean true', () => {
    expect(hasVerifiedWhisperDecoderAlignment(true)).toBe(true);
    expect(hasVerifiedWhisperDecoderAlignment(false)).toBe(false);
    expect(hasVerifiedWhisperDecoderAlignment(undefined)).toBe(false);
  });

  it('executeAlignmentReferenceEncoder throws when encoder produces 0 output tensors', async () => {
    const mockOrtSession: OrtSessionLike = {
      inputNames: ['input'],
      outputNames: ['output'],
      run: vi.fn().mockResolvedValue({}),
      release: vi.fn(),
    };
    const loadedState: any = {
      ort: {
        Tensor: class {
          constructor(public type: any, public data: any, public dims: any) {}
        },
      },
    };
    const alignmentRuntime: any = { decoderAlignSession: mockOrtSession };
    const ownedTensors = new Set<OrtTensorLike>();

    await expect(
      executeAlignmentReferenceEncoder(
        loadedState,
        mockOrtSession,
        alignmentRuntime,
        new Float32Array(80 * 3000),
        80,
        3000,
        ownedTensors,
      ),
    ).rejects.toThrow('Whisper alignment-reference encoder returned no output tensor.');
  });

  it('finalizeMergedWhisperTranscript filters special tokens when returnSpecialTokens is false', async () => {
    const tokenizer: any = {
      decode: vi.fn((ids: number[]) => (ids[0] === 50257 ? '<|endoftext|>' : 'word')),
      isTimestampTokenId: vi.fn(() => false),
      isSpecialTokenId: vi.fn((id: number) => id === 50257),
      getTokenId: vi.fn((token: string) => {
        if (token === '<|0.00|>') return 50364;
        if (token === '<|30.00|>') return 51864;
        return undefined;
      }),
    };

    const tokenDetails: WhisperNativeToken[] = [
      { id: 100, text: 'hello', special: false, index: 0, startTime: 0, endTime: 0.5, confidence: 0.9 },
      { id: 50257, text: '<|endoftext|>', special: true, index: 1, startTime: 0.5, endTime: 0.6, confidence: 1.0 },
    ];

    const audio: AudioBufferLike = {
      sampleRate: 16000,
      durationSeconds: 1.0,
      channels: [new Float32Array(16000)],
      numberOfChannels: 1,
      numberOfFrames: 16000,
    };

    const resFiltered = await finalizeMergedWhisperTranscript({
      loaded: {} as any,
      configLanguages: ['en'],
      encoderHiddenStates: {} as any,
      tokenizer,
      tokenDetails,
      options: { returnSpecialTokens: false },
      audio,
      language: 'en',
      warnings: [],
    });

    expect(resFiltered.tokens?.length).toBe(1);
    expect(resFiltered.tokens?.[0]?.text).toBe('hello');

    const resSpecial = await finalizeMergedWhisperTranscript({
      loaded: {} as any,
      configLanguages: ['en'],
      encoderHiddenStates: {} as any,
      tokenizer,
      tokenDetails,
      options: { returnSpecialTokens: true },
      audio,
      language: 'en',
      warnings: [],
    });

    expect(resSpecial.tokens?.length).toBe(2);
    expect(resSpecial.tokens?.[1]?.text).toBe('<|endoftext|>');
  });
});

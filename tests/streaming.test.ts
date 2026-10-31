import { createSpeechRuntime } from '@asrjs/speech-recognition';
import { createWasmBackend } from '@asrjs/speech-recognition';
import { DefaultStreamingTranscriber, RollingAudioWindow } from '@asrjs/speech-recognition/inference';
import { createNemoTdtModelFamily } from '@asrjs/speech-recognition/models/nemo-tdt';
import { createParakeetPresetFactory } from '@asrjs/speech-recognition/presets/parakeet';
import { describe, expect, it } from 'vitest';

describe('streaming orchestration', () => {
  it('resets state without disposing the caller-owned session', async () => {
    let transcribeCalls = 0;
    let disposed = false;
    const session = {
      async transcribe() {
        transcribeCalls += 1;
        return {
          text: 'hello',
          warnings: [],
          meta: { detailLevel: 'text' as const, isFinal: true },
        };
      },
      dispose() {
        disposed = true;
      },
    };
    const transcriber = new DefaultStreamingTranscriber(session, { maxWindowMs: 1000 });

    await transcriber.pushAudio(new Float32Array(8000));
    await transcriber.reset();
    expect(disposed).toBe(false);
    await transcriber.pushAudio(new Float32Array(8000));

    expect(transcribeCalls).toBe(2);
    await transcriber.dispose();
    await transcriber.dispose();
    expect(disposed).toBe(true);
    await expect(transcriber.pushAudio(new Float32Array(8000))).rejects.toThrow(/disposed/);
  });

  it('does not commit an in-flight result after reset', async () => {
    let resolveFirst!: (result: { text: string; warnings: []; meta: { detailLevel: 'text'; isFinal: true } }) => void;
    let transcribeCalls = 0;
    const firstResult = new Promise<{ text: string; warnings: []; meta: { detailLevel: 'text'; isFinal: true } }>((resolve) => {
      resolveFirst = resolve;
    });
    const session = {
      async transcribe() {
        transcribeCalls += 1;
        if (transcribeCalls === 1) return firstResult;
        return { text: 'new', warnings: [], meta: { detailLevel: 'text' as const, isFinal: true } };
      },
      dispose() {},
    };
    const transcriber = new DefaultStreamingTranscriber(session, { maxWindowMs: 1000 });

    const staleCall = transcriber.pushAudio(new Float32Array(8000));
    await Promise.resolve();
    await transcriber.reset();
    resolveFirst({ text: 'stale', warnings: [], meta: { detailLevel: 'text', isFinal: true } });

    const stale = await staleCall;
    expect(stale.text).toBe('');
    expect(transcriber.getState().committedText).toBe('');
    expect((await transcriber.pushAudio(new Float32Array(8000))).text).toBe('new');
  });

  it('serializes concurrent audio pushes through one rolling window', async () => {
    let resolveFirst!: (result: { text: string; warnings: []; meta: { detailLevel: 'text'; isFinal: true } }) => void;
    let transcribeCalls = 0;
    const firstResult = new Promise<{ text: string; warnings: []; meta: { detailLevel: 'text'; isFinal: true } }>((resolve) => {
      resolveFirst = resolve;
    });
    const session = {
      async transcribe() {
        transcribeCalls += 1;
        if (transcribeCalls === 1) return firstResult;
        return { text: 'second', warnings: [], meta: { detailLevel: 'text' as const, isFinal: true } };
      },
      dispose() {},
    };
    const transcriber = new DefaultStreamingTranscriber(session, { maxWindowMs: 1000 });

    const first = transcriber.pushAudio(new Float32Array(8000));
    const second = transcriber.pushAudio(new Float32Array(8000));
    await Promise.resolve();
    expect(transcribeCalls).toBe(1);

    resolveFirst({ text: 'first', warnings: [], meta: { detailLevel: 'text', isFinal: true } });
    await first;
    await second;
    expect(transcribeCalls).toBe(2);
    await transcriber.dispose();
  });

  it('maintains partial and final transcript state', async () => {
    const runtime = createSpeechRuntime({
      backends: [createWasmBackend()],
      modelFamilies: [createNemoTdtModelFamily()],
      presets: [createParakeetPresetFactory()],
    });
    const model = await runtime.loadModel({
      preset: 'parakeet',
      modelId: 'parakeet-tdt-0.6b-v3',
    });
    const session = await model.createSession();
    const transcriber = new DefaultStreamingTranscriber(session, {
      detail: 'words',
      overlapMs: 250,
      maxWindowMs: 1000,
    });

    const partial = await transcriber.pushAudio(new Float32Array(8000));
    expect(partial.kind).toBe('partial');
    expect(partial.previewText.length).toBeGreaterThan(0);
    expect(transcriber.getState().isFinalized).toBe(false);

    const final = await transcriber.finalize();
    expect(final.kind).toBe('final');
    expect(final.committedText.length).toBeGreaterThan(0);
    expect(transcriber.getState().isFinalized).toBe(true);
  });

  it('trims the rolling audio window while preserving overlap', () => {
    const window = new RollingAudioWindow({
      maxWindowMs: 1000,
      overlapMs: 250,
    });

    window.push(new Float32Array(8000));
    window.push(new Float32Array(8000), 0.5);
    window.push(new Float32Array(8000), 1.0);

    expect(window.getBufferedDurationSeconds()).toBeLessThanOrEqual(1.1);
    expect(window.getBufferedDurationSeconds()).toBeGreaterThan(0.25);
  });

  describe('Issue #223: Multi-window streaming and Eviction-Before-Drop', () => {
    it('preserves all words and monotonically offsets timestamps across window evictions', async () => {
      // Stream 4 chunks of 0.5s (8000 samples @ 16 kHz), total 2.0s audio.
      // Window is max 1.0s with 0.25s overlap.
      const session = {
        async transcribe(input: any) {
          const numFrames = input.numberOfFrames ?? 0;
          const duration = numFrames / 16000;
          // Generate words corresponding to the audio in the window
          const words: any[] = [];
          if (duration >= 0.4) {
            words.push({ text: 'wordA', startTime: 0.1, endTime: 0.4 });
          }
          if (duration >= 0.9) {
            words.push({ text: 'wordB', startTime: 0.6, endTime: 0.9 });
          }
          return {
            text: words.map((w) => w.text).join(' '),
            words,
            warnings: [],
            meta: { detailLevel: 'words' as const, isFinal: true },
          };
        },
        dispose() {},
      };

      const transcriber = new DefaultStreamingTranscriber(session as never, {
        maxWindowMs: 1000,
        overlapMs: 250,
        detail: 'words',
      });

      // Push 4 chunks of 8000 frames (0.5s each)
      await transcriber.pushAudio(new Float32Array(8000));
      await transcriber.pushAudio(new Float32Array(8000));
      // Window now has 1.0s audio. Next push will trigger eviction-before-drop!
      await transcriber.pushAudio(new Float32Array(8000));
      await transcriber.pushAudio(new Float32Array(8000));

      const final = await transcriber.finalize();
      expect(final.kind).toBe('final');
      expect(final.text.length).toBeGreaterThan(0);
      expect(final.words).toBeDefined();
      expect(final.words!.length).toBeGreaterThanOrEqual(2);

      // Verify word timestamps are monotonically advancing and properly offset
      let lastEnd = 0;
      for (const word of final.words!) {
        expect(word.startTime).toBeGreaterThanOrEqual(lastEnd - 0.15);
        expect(word.endTime).toBeGreaterThan(word.startTime);
        lastEnd = word.endTime;
      }

      await transcriber.dispose();
    });

    it('ensures emitPartials=false does not drop untranscribed audio before committing', async () => {
      let transcribeCount = 0;
      const session = {
        async transcribe(input: any) {
          transcribeCount++;
          const duration = input.numberOfFrames / 16000;
          return {
            text: `spoken_${duration.toFixed(1)}`,
            words: [{ text: `spoken_${duration.toFixed(1)}`, startTime: 0.1, endTime: Math.max(0.2, duration - 0.1) }],
            warnings: [],
            meta: { detailLevel: 'words' as const, isFinal: true },
          };
        },
        dispose() {},
      };

      const transcriber = new DefaultStreamingTranscriber(session as never, {
        maxWindowMs: 1000,
        overlapMs: 250,
        emitPartials: false,
        detail: 'words',
      });

      // Push 3 chunks of 0.5s. With emitPartials: false, partials are blank,
      // but eviction must still transcribe and commit before dropping audio.
      const p1 = await transcriber.pushAudio(new Float32Array(8000));
      expect(p1.text).toBe('');

      const p2 = await transcriber.pushAudio(new Float32Array(8000));
      expect(p2.text).toBe('');

      // Exceeds 1000ms window -> eviction before drop must transcribe chunk before eviction
      const p3 = await transcriber.pushAudio(new Float32Array(8000));
      expect(p3.text).toBe('');

      const final = await transcriber.finalize();
      expect(final.kind).toBe('final');
      expect(final.text).toContain('spoken');
      expect(transcribeCount).toBeGreaterThanOrEqual(2);

      await transcriber.dispose();
    });
  });

  describe('Issue #224: Streaming VAD silence accumulation', () => {
    it('auto-finalizes when cumulative silence across 100x256-sample chunks reaches minFinalSilenceMs (350 ms)', async () => {
      let speechState = true;
      const vad = {
        analyze: async () => ({
          isSpeech: speechState,
          speechProbability: speechState ? 0.95 : 0.05,
        }),
      };

      const session = {
        async transcribe() {
          return {
            text: 'speech segment',
            warnings: [],
            meta: { detailLevel: 'text' as const, isFinal: true },
          };
        },
        dispose() {},
      };

      const transcriber = new DefaultStreamingTranscriber(session as never, {
        vad: vad as never,
        minFinalSilenceMs: 350,
      });

      // 1. Initial speech chunk (256 frames @ 16 kHz = 16 ms)
      speechState = true;
      await transcriber.pushAudio(new Float32Array(256));
      expect(transcriber.getState().isFinalized).toBe(false);

      // 2. Feed silence chunks of 256 samples (16 ms each).
      // 21 chunks * 16 ms = 336 ms (< 350 ms) -> not finalized
      speechState = false;
      for (let i = 0; i < 21; i++) {
        await transcriber.pushAudio(new Float32Array(256));
        expect(transcriber.getState().isFinalized).toBe(false);
      }

      // 3. 22nd chunk: 22 * 16 ms = 352 ms (>= 350 ms) -> must auto-finalize!
      const pFinal = await transcriber.pushAudio(new Float32Array(256));
      expect(pFinal.kind).toBe('final');
      expect(transcriber.getState().isFinalized).toBe(true);

      await transcriber.dispose();
    });

    it('resets the silence accumulator when speech resumes mid-silence', async () => {
      let speechState = true;
      const vad = {
        analyze: async () => ({
          isSpeech: speechState,
          speechProbability: speechState ? 0.95 : 0.05,
        }),
      };

      const session = {
        async transcribe() {
          return {
            text: 'speech segment',
            warnings: [],
            meta: { detailLevel: 'text' as const, isFinal: true },
          };
        },
        dispose() {},
      };

      const transcriber = new DefaultStreamingTranscriber(session as never, {
        vad: vad as never,
        minFinalSilenceMs: 350,
      });

      // Initial speech
      speechState = true;
      await transcriber.pushAudio(new Float32Array(256));

      // 15 silent chunks = 240 ms silence
      speechState = false;
      for (let i = 0; i < 15; i++) {
        await transcriber.pushAudio(new Float32Array(256));
        expect(transcriber.getState().isFinalized).toBe(false);
      }

      // Speech resumes -> resets silence accumulator
      speechState = true;
      await transcriber.pushAudio(new Float32Array(256));
      expect(transcriber.getState().isFinalized).toBe(false);

      // 15 more silent chunks = 240 ms silence -> must NOT finalize because accumulator was reset
      speechState = false;
      for (let i = 0; i < 15; i++) {
        await transcriber.pushAudio(new Float32Array(256));
        expect(transcriber.getState().isFinalized).toBe(false);
      }

      // 6 more silent chunks -> 15 + 6 = 21 chunks * 16 ms = 336 ms < 350 ms
      for (let i = 0; i < 6; i++) {
        await transcriber.pushAudio(new Float32Array(256));
        expect(transcriber.getState().isFinalized).toBe(false);
      }
      // 22nd silent chunk: 22 * 16 ms = 352 ms >= 350 ms -> auto-finalizes!
      const final = await transcriber.pushAudio(new Float32Array(256));
      expect(final.kind).toBe('final');
      expect(transcriber.getState().isFinalized).toBe(true);

      await transcriber.dispose();
    });
  });
});


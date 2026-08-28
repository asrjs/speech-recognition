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
});

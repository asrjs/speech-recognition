import { createSpeechRuntime } from '@asrjs/speech-recognition';
import { createWasmBackend } from '@asrjs/speech-recognition';
import { DefaultStreamingTranscriber, RollingAudioWindow } from '@asrjs/speech-recognition/inference';
import { createNemoTdtModelFamily } from '@asrjs/speech-recognition/models/nemo-tdt';
import { createParakeetPresetFactory } from '@asrjs/speech-recognition/presets/parakeet';
import { describe, expect, it } from 'vitest';

describe('streaming orchestration', () => {
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

import { createSpeechRuntime } from '@asrjs/speech-recognition';
import { createWasmBackend } from '@asrjs/speech-recognition';
import { DefaultStreamingTranscriber } from '@asrjs/speech-recognition/inference';
import { createNemoTdtModelFamily } from '@asrjs/speech-recognition/models/nemo-tdt';
import { createParakeetPresetFactory } from '@asrjs/speech-recognition/presets/parakeet';

export async function streamingExample() {
  const runtime = createSpeechRuntime({
    backends: [createWasmBackend()],
    modelFamilies: [createNemoTdtModelFamily()],
    presets: [createParakeetPresetFactory({ useManifestSource: true })],
  });

  let streaming: DefaultStreamingTranscriber | undefined;
  let failure: { error: unknown } | undefined;
  try {
    const model = await runtime.loadModel({
      preset: 'parakeet',
      modelId: 'parakeet-tdt-0.6b-v3',
    });
    const session = await model.createSession();
    // Windowed offline inference, not a native streaming model.
    streaming = new DefaultStreamingTranscriber(session, {
      detail: 'words',
      overlapMs: 500,
      maxWindowMs: 5000,
    });
    const partial = await streaming.pushAudio(new Float32Array(8000));
    const final = await streaming.finalize();

    return { partial, final };
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    const cleanupErrors: unknown[] = [];
    try {
      await streaming?.dispose();
    } catch (error) {
      cleanupErrors.push(error);
    }
    try {
      // Runtime owns loaded models and their remaining sessions, even if
      // acquiring the session/wrapper or releasing the wrapper failed.
      await runtime.dispose();
    } catch (error) {
      cleanupErrors.push(error);
    }
    if (cleanupErrors.length > 0) {
      throw new AggregateError(
        [...(failure ? [failure.error] : []), ...cleanupErrors],
        'Streaming example resource cleanup failed',
        { cause: failure ? failure.error : cleanupErrors[0] },
      );
    }
  }
}

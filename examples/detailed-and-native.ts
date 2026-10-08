import { createSpeechRuntime } from '@asrjs/speech-recognition';
import { createWasmBackend } from '@asrjs/speech-recognition';
import { createNemoTdtModelFamily } from '@asrjs/speech-recognition/models/nemo-tdt';
import { createParakeetPresetFactory } from '@asrjs/speech-recognition/presets/parakeet';

export async function detailedAndNativeExample() {
  const runtime = createSpeechRuntime({
    backends: [createWasmBackend()],
    modelFamilies: [createNemoTdtModelFamily()],
    presets: [createParakeetPresetFactory({ useManifestSource: true })],
  });

  let failure: { error: unknown } | undefined;
  try {
    const model = await runtime.loadModel({
      preset: 'parakeet',
      modelId: 'parakeet-tdt-0.6b-v3',
    });
    const session = await model.createSession();
    // Await before cleanup: returning the pending promise would release live resources.
    return await session.transcribe(new Float32Array(32000), {
      detail: 'detailed',
      responseFlavor: 'canonical+native',
      returnFrameIndices: true,
      returnLogProbs: true,
      returnTdtSteps: true,
    });
  } catch (error) {
    failure = { error };
    throw error;
  } finally {
    try {
      // The runtime owns loaded models; the TDT model owns its sessions.
      await runtime.dispose();
    } catch (cleanupError) {
      throw new AggregateError(
        [...(failure ? [failure.error] : []), cleanupError],
        'Detailed/native example resource cleanup failed',
        { cause: failure ? failure.error : cleanupError },
      );
    }
  }
}

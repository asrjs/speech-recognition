import { createSpeechRuntime } from '@asrjs/speech-recognition';
import { createWasmBackend } from '@asrjs/speech-recognition';
import { createWebGpuBackend } from '@asrjs/speech-recognition';
import { createNemoTdtModelFamily } from '@asrjs/speech-recognition/models/nemo-tdt';
import { createParakeetPresetFactory } from '@asrjs/speech-recognition/presets/parakeet';

export async function chooseBackendExample(): Promise<string> {
  const runtime = createSpeechRuntime();
  runtime
    .registerBackend(createWebGpuBackend())
    .registerBackend(createWasmBackend())
    .registerModelFamily(createNemoTdtModelFamily())
    .registerPreset(createParakeetPresetFactory());

  try {
    const backend = await runtime.selectBackend({
      preferredBackendIds: ['webgpu', 'wasm'],
      requiredPrecision: 'fp32',
      preferAcceleration: true,
    });

    // Availability is a device probe, not model/graph compatibility evidence.
    return backend.id;
  } finally {
    await runtime.dispose();
  }
}

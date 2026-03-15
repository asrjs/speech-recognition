import { createSpeechRuntime } from '@asrjs/speech-recognition';
import { createWasmBackend } from '@asrjs/speech-recognition';
import { createWebGpuBackend } from '@asrjs/speech-recognition';
import { createWebGlBackend } from '@asrjs/speech-recognition';
import { createNemoTdtModelFamily } from '@asrjs/speech-recognition/models/nemo-tdt';
import { createParakeetPresetFactory } from '@asrjs/speech-recognition/presets/parakeet';

export async function chooseBackendExample(): Promise<string> {
  const runtime = createSpeechRuntime();
  runtime
    .registerBackend(createWebGpuBackend())
    .registerBackend(createWasmBackend())
    .registerBackend(createWebGlBackend())
    .registerModelFamily(createNemoTdtModelFamily())
    .registerPreset(createParakeetPresetFactory());

  const backend = await runtime.selectBackend({
    preferredBackendIds: ['webgpu', 'wasm'],
    requiredPrecision: 'fp32',
    preferAcceleration: true
  });

  return backend.id;
}

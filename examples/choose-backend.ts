import { createSpeechRuntime } from 'asr.js';
import { createWasmBackend } from 'asr.js';
import { createWebGpuBackend } from 'asr.js';
import { createWebGlBackend } from 'asr.js';
import { createParakeetPresetFactory } from 'asr.js';

export async function chooseBackendExample(): Promise<string> {
  const runtime = createSpeechRuntime();
  runtime
    .registerBackend(createWebGpuBackend())
    .registerBackend(createWasmBackend())
    .registerBackend(createWebGlBackend())
    .registerModelFamily(createParakeetPresetFactory());

  const backend = await runtime.selectBackend({
    preferredBackendIds: ['webgpu', 'wasm'],
    requiredPrecision: 'fp32',
    preferAcceleration: true
  });

  return backend.id;
}

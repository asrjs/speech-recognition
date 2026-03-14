import { createWebGlBackend, createWebGpuBackend, createWebNnBackend, createWasmBackend } from '../inference/index.js';
import { createMedAsrPresetFactory, createParakeetPresetFactory, createWhisperPresetFactory } from '../presets/index.js';
import type { SpeechRuntimeHooks } from '../types/index.js';
import { createSpeechRuntime, type DefaultSpeechRuntime } from './session.js';

export interface CreateBuiltInSpeechRuntimeOptions {
  readonly hooks?: SpeechRuntimeHooks;
  readonly useManifestSources?: boolean;
}

export function registerBuiltInBackends(runtime: DefaultSpeechRuntime): DefaultSpeechRuntime {
  runtime.registerBackend(createWebGpuBackend());
  runtime.registerBackend(createWasmBackend());
  runtime.registerBackend(createWebNnBackend());
  runtime.registerBackend(createWebGlBackend());
  return runtime;
}

export function registerBuiltInModelFamilies(
  runtime: DefaultSpeechRuntime,
  options: CreateBuiltInSpeechRuntimeOptions = {}
): DefaultSpeechRuntime {
  runtime.registerModelFamily(createParakeetPresetFactory({
    useManifestSource: options.useManifestSources ?? true
  }));
  runtime.registerModelFamily(createMedAsrPresetFactory());
  runtime.registerModelFamily(createWhisperPresetFactory());
  return runtime;
}

export function createBuiltInSpeechRuntime(
  options: CreateBuiltInSpeechRuntimeOptions = {}
): DefaultSpeechRuntime {
  const runtime = createSpeechRuntime({
    hooks: options.hooks
  });

  registerBuiltInBackends(runtime);
  registerBuiltInModelFamilies(runtime, options);

  return runtime;
}

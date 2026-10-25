import { describe, expect, it } from 'vitest';
import * as root from '../src/index.js';
import * as audio from '../src/audio/index.js';
import * as pipeline from '../src/pipeline/index.js';
import * as wasm from '../src/inference/backends/wasm/index.js';
import * as native from '../src/inference/backends/native-cpu/index.js';
import exportsBeforeRefactor from './fixtures/root-runtime-exports.json';

describe('public root API compatibility', () => {
  it('retains the complete runtime export surface from before the import refactor', () => {
    // Keep the historical baseline intact; enumerate intentional API additions.
    expect(Object.keys(root).sort()).toEqual(
      [...exportsBeforeRefactor, 'transcribeSpeechFromAudioSource'].sort(),
    );
    expect(root.transcribeSpeechFromAudioSource).toBeTypeOf('function');
  });
  it('shares constructors and functions across root and public subpaths', () => {
    expect(root.PcmAudioBuffer).toBe(audio.PcmAudioBuffer);
    expect(root.WhisperMelProcessor).toBe(audio.WhisperMelProcessor);
    expect(root.PipelineAbortedError).toBe(pipeline.PipelineAbortedError);
    expect(root.ProductionWhisperPipeline).toBe(pipeline.ProductionWhisperPipeline);
    expect(root.createWasmBackend).toBe(wasm.createWasmBackend);
    expect(root.probeWasmCapabilities).toBe(wasm.probeWasmCapabilities);
    expect(root.createNativeCpuBackend).toBe(native.createNativeCpuBackend);
    expect(root.createWasmBackend()).not.toBeInstanceOf(Promise);
    expect(root.createSpeechRuntime()).not.toBeInstanceOf(Promise);
    expect(root.listSpeechModels()).not.toBeInstanceOf(Promise);
  });
});

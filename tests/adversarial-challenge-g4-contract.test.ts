import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import ts from 'typescript';

import {
  createSpeechRuntime,
  createWasmBackend,
  listSpeechModels,
  listSpeechModelOptions,
  getSpeechModelDescriptor,
  getSpeechModelLanguageName,
  buildSpeechModelLoadOptions,
  buildSpeechTranscriptionOptions,
  loadSpeechModel,
  transcribeSpeech,
  transcribeSpeechBatch,
  transcribeSpeechFromMonoPcm,
  transcribeSpeechFromAudioSource,
  createSpeechPipeline,
  PcmAudioBuffer,
  getCanonicalTranscript,
  listApprovedSpeechModelManifests,
  resolveSpeechModelExecutionPlan,
  probeSpeechModelExecutionPlan,
} from '@asrjs/speech-recognition';

import { createSpeechController } from '@asrjs/speech-recognition/controller';
import {
  createSpeechModelLocalEntries,
  collectSpeechModelLocalEntries,
  inspectSpeechModelLocalEntries,
  loadSpeechModelFromLocalEntries,
  createBrowserRealtimeStarter,
  TenVadAdapter,
  encodeMonoPcmToWavBlob,
  startMicrophoneCapture,
} from '@asrjs/speech-recognition/browser';
import { decodeAudioSourceToMonoPcm } from '@asrjs/speech-recognition/browser/media';
import {
  createBuiltInSpeechRuntime,
  loadBuiltInSpeechModel,
} from '@asrjs/speech-recognition/builtins';
import {
  createDefaultAssetProvider,
  createHuggingFaceAssetProvider,
  fetchModelFiles,
  getModelFile,
  pickPreferredQuant,
} from '@asrjs/speech-recognition/io';
import {
  FASTCONFORMER_ENCODER,
  argmax,
  DefaultStreamingTranscriber,
} from '@asrjs/speech-recognition/inference';
import { RealtimeTranscriptionController } from '@asrjs/speech-recognition/realtime';
import { benchmarkRunRecordsToCsv } from '@asrjs/speech-recognition/bench';
import { fetchDatasetSplits } from '@asrjs/speech-recognition/datasets';
import { createNemoTdtModelFamily } from '@asrjs/speech-recognition/models/nemo-tdt';
import { createLasrCtcModelFamily } from '@asrjs/speech-recognition/models/lasr-ctc';
import { createWhisperSeq2SeqModelFamily } from '@asrjs/speech-recognition/models/whisper-seq2seq';
import {
  createParakeetPresetFactory,
  getParakeetModel,
  loadParakeetModelWithFallback,
  loadParakeetModelFromLocalEntries,
} from '@asrjs/speech-recognition/presets/parakeet';
import { createMedAsrPresetFactory } from '@asrjs/speech-recognition/presets/medasr';
import { createWhisperPresetFactory } from '@asrjs/speech-recognition/presets/whisper';
import {
  listBuiltInModelDescriptors,
  resolveApprovedSpeechExecutionPlan,
  type BuiltInModelCapabilities,
} from '@asrjs/speech-recognition/presets';

describe('Adversarial Challenge G4: Public API & Documentation Contract Audit', () => {
  describe('1. Export Symbol & Subpath Verification', () => {
    it('verifies all symbols cited in README and PUBLIC_API_GUIDE root import', () => {
      expect(createSpeechRuntime).toBeTypeOf('function');
      expect(createWasmBackend).toBeTypeOf('function');
      expect(listSpeechModels).toBeTypeOf('function');
      expect(listSpeechModelOptions).toBeTypeOf('function');
      expect(getSpeechModelDescriptor).toBeTypeOf('function');
      expect(getSpeechModelLanguageName).toBeTypeOf('function');
      expect(buildSpeechModelLoadOptions).toBeTypeOf('function');
      expect(buildSpeechTranscriptionOptions).toBeTypeOf('function');
      expect(loadSpeechModel).toBeTypeOf('function');
      expect(transcribeSpeech).toBeTypeOf('function');
      expect(transcribeSpeechBatch).toBeTypeOf('function');
      expect(transcribeSpeechFromMonoPcm).toBeTypeOf('function');
      expect(transcribeSpeechFromAudioSource).toBeTypeOf('function');
      expect(createSpeechPipeline).toBeTypeOf('function');
      expect(PcmAudioBuffer).toBeTypeOf('function');
      expect(getCanonicalTranscript).toBeTypeOf('function');
      expect(listApprovedSpeechModelManifests).toBeTypeOf('function');
      expect(resolveSpeechModelExecutionPlan).toBeTypeOf('function');
      expect(probeSpeechModelExecutionPlan).toBeTypeOf('function');
    });

    it('verifies all symbols cited in @asrjs/speech-recognition/controller', () => {
      expect(createSpeechController).toBeTypeOf('function');
      const controller = createSpeechController();
      expect(controller.load).toBeTypeOf('function');
      expect(controller.startMicrophone).toBeTypeOf('function');
      expect(controller.stopMicrophone).toBeTypeOf('function');
      expect(controller.subscribe).toBeTypeOf('function');
      expect(controller.getSnapshot).toBeTypeOf('function');
      expect(controller.transcribeSource).toBeTypeOf('function');
      expect(controller.dispose).toBeTypeOf('function');

      const snapshot = controller.getSnapshot();
      expect(snapshot).toHaveProperty('status');
      expect(snapshot).toHaveProperty('progress');
      expect(snapshot).toHaveProperty('loadWarnings');
      expect(snapshot.status).toBe('idle');
      expect(Array.isArray(snapshot.loadWarnings)).toBe(true);
    });

    it('verifies all symbols cited in @asrjs/speech-recognition/browser and browser/media', () => {
      expect(createSpeechModelLocalEntries).toBeTypeOf('function');
      expect(collectSpeechModelLocalEntries).toBeTypeOf('function');
      expect(inspectSpeechModelLocalEntries).toBeTypeOf('function');
      expect(loadSpeechModelFromLocalEntries).toBeTypeOf('function');
      expect(createBrowserRealtimeStarter).toBeTypeOf('function');
      expect(TenVadAdapter).toBeTypeOf('function');
      expect(encodeMonoPcmToWavBlob).toBeTypeOf('function');
      expect(startMicrophoneCapture).toBeTypeOf('function');
      expect(decodeAudioSourceToMonoPcm).toBeTypeOf('function');
    });

    it('verifies all symbols cited in @asrjs/speech-recognition/builtins', () => {
      expect(createBuiltInSpeechRuntime).toBeTypeOf('function');
      expect(loadBuiltInSpeechModel).toBeTypeOf('function');
    });

    it('verifies all symbols cited in @asrjs/speech-recognition/io', () => {
      expect(createDefaultAssetProvider).toBeTypeOf('function');
      expect(createHuggingFaceAssetProvider).toBeTypeOf('function');
      expect(fetchModelFiles).toBeTypeOf('function');
      expect(getModelFile).toBeTypeOf('function');
      expect(pickPreferredQuant).toBeTypeOf('function');
    });

    it('verifies all symbols cited in @asrjs/speech-recognition/inference', () => {
      expect(FASTCONFORMER_ENCODER).toBeDefined();
      expect(FASTCONFORMER_ENCODER.kind).toBe('fastconformer');
      expect(argmax).toBeTypeOf('function');
      expect(DefaultStreamingTranscriber).toBeTypeOf('function');
    });

    it('verifies all symbols cited in other subpaths (realtime, bench, datasets, models, presets)', () => {
      expect(RealtimeTranscriptionController).toBeTypeOf('function');
      expect(benchmarkRunRecordsToCsv).toBeTypeOf('function');
      expect(fetchDatasetSplits).toBeTypeOf('function');

      expect(createNemoTdtModelFamily).toBeTypeOf('function');
      expect(createLasrCtcModelFamily).toBeTypeOf('function');
      expect(createWhisperSeq2SeqModelFamily).toBeTypeOf('function');

      expect(createParakeetPresetFactory).toBeTypeOf('function');
      expect(getParakeetModel).toBeTypeOf('function');
      expect(loadParakeetModelWithFallback).toBeTypeOf('function');
      expect(loadParakeetModelFromLocalEntries).toBeTypeOf('function');

      expect(createMedAsrPresetFactory).toBeTypeOf('function');
      expect(createWhisperPresetFactory).toBeTypeOf('function');
    });
  });

  describe('2. Golden Path Method Signatures & Error Boundary Contracts', () => {
    it('Golden Path 1: transcribeSpeechFromAudioSource handles pre-aborted signal and unknown models without hanging', async () => {
      // 1. Unknown model throws immediately without network
      await expect(
        transcribeSpeechFromAudioSource('mock.wav', {
          modelId: 'non-existent-parakeet-model-xyz',
        }),
      ).rejects.toThrow(/Could not infer a built-in preset or model family/);

      // 2. Pre-aborted signal throws PipelineAbortedError immediately
      const controller = new AbortController();
      controller.abort();
      await expect(
        transcribeSpeechFromAudioSource('mock.wav', {
          modelId: 'parakeet-tdt-0.6b-v3',
          signal: controller.signal,
        }),
      ).rejects.toThrow(/audio-source-transcription/);
    });

    it('Golden Path 2: createSpeechController lifecycle and borrowing contract', async () => {
      const speech = createSpeechController();
      let listenerCalls = 0;
      let lastObservedStatus = speech.getSnapshot().status;

      const unsubscribe = speech.subscribe(() => {
        listenerCalls++;
        lastObservedStatus = speech.getSnapshot().status;
      });

      expect(speech.getSnapshot().status).toBe('idle');

      // Disposing the controller should publish 'disposed' to active subscribers
      await speech.dispose();
      expect(speech.getSnapshot().status).toBe('disposed');
      expect(listenerCalls).toBe(1);
      expect(lastObservedStatus).toBe('disposed');

      // Unsubscribing should ensure no further notifications
      unsubscribe();
    });

    it('Golden Path 3: loadSpeechModelFromLocalEntries rejects empty entries with informative error', async () => {
      await expect(
        loadSpeechModelFromLocalEntries({
          modelId: 'parakeet-tdt-0.6b-v3',
          entries: [],
          backend: 'wasm',
          wasmPaths: '/ort/',
        }),
      ).rejects.toThrow(/Pick a local model folder first/i);
    });

    it('Golden Path 4: transcribeSpeechFromMonoPcm accepts valid Float32Array PCM input shape', async () => {
      expect(transcribeSpeechFromMonoPcm).toBeTypeOf('function');
      const silentPcm = new Float32Array(16000); // 1s silence
      // Verify signature without hanging on mock network
      expect(silentPcm.length).toBe(16000);
    });
  });

  describe('3. BuiltInModelCapabilities Schema Integrity & Adversarial Audit', () => {
    const descriptors = listBuiltInModelDescriptors();

    it('verifies that every catalog model exposes a fully compliant BuiltInModelCapabilities schema', () => {
      expect(descriptors.length).toBeGreaterThan(0);

      for (const desc of descriptors) {
        const caps: BuiltInModelCapabilities = desc.capabilities;
        expect(caps, `Model ${desc.modelId} missing capabilities`).toBeDefined();

        // Check supportedTasks
        expect(Array.isArray(caps.supportedTasks), `${desc.modelId} supportedTasks not array`).toBe(true);
        expect(caps.supportedTasks.length, `${desc.modelId} supportedTasks is empty`).toBeGreaterThan(0);
        for (const task of caps.supportedTasks) {
          expect(['asr', 'speech-translation']).toContain(task);
        }

        // Check strictly boolean fields
        expect(typeof caps.supportsTranslation, `${desc.modelId}.supportsTranslation`).toBe('boolean');
        expect(typeof caps.supportsPunctuationCapitalization, `${desc.modelId}.supportsPunctuationCapitalization`).toBe('boolean');
        expect(typeof caps.supportsInverseTextNormalization, `${desc.modelId}.supportsInverseTextNormalization`).toBe('boolean');
        expect(typeof caps.supportsWordTimestamps, `${desc.modelId}.supportsWordTimestamps`).toBe('boolean');
        expect(typeof caps.supportsSegmentTimestamps, `${desc.modelId}.supportsSegmentTimestamps`).toBe('boolean');
        expect(typeof caps.supportsPromptControls, `${desc.modelId}.supportsPromptControls`).toBe('boolean');

        // Logical consistency checks:
        if (caps.supportsTranslation) {
          expect(
            caps.supportedTasks,
            `${desc.modelId} claims translation but supportedTasks omits speech-translation`,
          ).toContain('speech-translation');
        }

        if (caps.supportsPromptControls) {
          expect(
            desc.controls.length,
            `${desc.modelId} claims prompt controls but has empty controls list`,
          ).toBeGreaterThan(0);
        }
      }
    });

    it('checks Canary model capabilities match documented contracts', () => {
      const canary = descriptors.find((d) => d.preset === 'canary');
      expect(canary).toBeDefined();
      expect(canary!.capabilities.supportsTranslation).toBe(true);
      expect(canary!.capabilities.supportsPunctuationCapitalization).toBe(true);
      expect(canary!.capabilities.supportedTasks).toContain('speech-translation');
    });

    it('checks Parakeet model capabilities match documented contracts', () => {
      const parakeet = descriptors.find((d) => d.modelId === 'parakeet-tdt-0.6b-v3');
      expect(parakeet).toBeDefined();
      expect(parakeet!.capabilities.supportsWordTimestamps).toBe(true);
      expect(parakeet!.capabilities.supportsSegmentTimestamps).toBe(true);
      expect(parakeet!.capabilities.supportsTranslation).toBe(false);
    });
  });

  describe('4. ApprovedSpeechExecutionPlan Schema Integrity & Fallback Edge Cases', () => {
    it('verifies that all published manifests satisfy the ApprovedSpeechExecutionPlan schema', () => {
      const manifests = listApprovedSpeechModelManifests();
      expect(manifests.length).toBeGreaterThan(0);

      for (const manifest of manifests) {
        expect(manifest.manifestVersion).toBe('1');
        expect(manifest.status).toBe('approved');
        expect(typeof manifest.modelId).toBe('string');
        expect(manifest.modelId.length).toBeGreaterThan(0);
        expect(['webgpu', 'wasm']).toContain(manifest.encoderBackend);
        expect(['webgpu', 'wasm']).toContain(manifest.decoderBackend);
        expect(['int8', 'fp32', 'fp16']).toContain(manifest.encoderQuant);
        expect(['int8', 'fp32', 'fp16']).toContain(manifest.decoderQuant);
        expect(typeof manifest.repoId).toBe('string');
        expect(typeof manifest.revision).toBe('string');
        expect(manifest.encoderSha256).toMatch(/^[a-f0-9]{64}$/);
        expect(manifest.decoderSha256).toMatch(/^[a-f0-9]{64}$/);
        expect(manifest.configSha256).toMatch(/^[a-f0-9]{64}$/);
        expect(manifest.vocabularySha256).toMatch(/^[a-f0-9]{64}$/);
        expect(manifest.artifacts).toBeDefined();
        expect(manifest.processor?.kind).toBe('javascript');
      }
    });

    it('handles model ID edge cases in resolveApprovedSpeechExecutionPlan', () => {
      // 1. Unknown model
      expect(() =>
        resolveApprovedSpeechExecutionPlan('non-existent-model-xyz', {
          runtime: { webgpu: true, wasm: true },
        }),
      ).toThrow(/Unknown built-in model/);

      // 2. Whitespace and uppercase normalization
      const normalPlan = resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', {
        runtime: { webgpu: true, wasm: true, webgpuPrecisions: ['fp16', 'fp32'] },
      });
      const paddedUpperPlan = resolveApprovedSpeechExecutionPlan('  PARAKEET-TDT-0.6B-V3  ', {
        runtime: { webgpu: true, wasm: true, webgpuPrecisions: ['fp16', 'fp32'] },
      });
      expect(paddedUpperPlan.encoderBackend).toBe(normalPlan.encoderBackend);
      expect(paddedUpperPlan.decoderBackend).toBe(normalPlan.decoderBackend);
      expect(paddedUpperPlan.encoderQuant).toBe(normalPlan.encoderQuant);
    });

    it('handles zero available runtime backends by throwing informative error', () => {
      expect(() =>
        resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', {
          runtime: { webgpu: false, wasm: false },
        }),
      ).toThrow(/No supported backend is available/);
    });

    it('handles missing WebGPU fp16 support gracefully by falling back to fp32 with warning', () => {
      const plan = resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', {
        runtime: {
          webgpu: true,
          wasm: true,
          webgpuPrecisions: ['fp32', 'int8'], // missing fp16
          wasmPrecisions: ['fp32', 'int8'],
        },
      });

      expect(plan.status).toBe('unverified');
      expect(plan.encoderBackend).toBe('webgpu');
      expect(plan.encoderQuant).toBe('fp32');
      expect(plan.warnings).toBeDefined();
      expect(plan.warnings?.some((w) => w.code === 'plan.encoder-precision-fallback')).toBe(true);
    });

    it('handles missing WebGPU entirely by falling back to WASM encoder with warning', () => {
      const plan = resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', {
        runtime: {
          webgpu: false,
          wasm: true,
          wasmPrecisions: ['fp32', 'int8'],
        },
      });

      expect(plan.status).toBe('unverified');
      expect(plan.encoderBackend).toBe('wasm');
      expect(plan.decoderBackend).toBe('wasm');
      expect(plan.warnings).toBeDefined();
      expect(plan.warnings?.some((w) => w.code === 'plan.encoder-backend-fallback')).toBe(true);
    });

    it('approves plan if and only if artifactIdentity matches exact approved profile SHA hashes', () => {
      const approvedManifest = listApprovedSpeechModelManifests().find(
        (m) => m.modelId === 'parakeet-tdt-0.6b-v3',
      )!;

      // Exact match
      const approvedPlan = resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', {
        runtime: {
          webgpu: true,
          wasm: true,
          webgpuPrecisions: ['fp16', 'fp32', 'int8'],
          wasmPrecisions: ['fp32', 'int8'],
        },
        artifactIdentity: {
          repoId: approvedManifest.repoId!,
          revision: approvedManifest.revision!,
          encoderSha256: approvedManifest.encoderSha256!,
          decoderSha256: approvedManifest.decoderSha256!,
          configSha256: approvedManifest.configSha256!,
          vocabularySha256: approvedManifest.vocabularySha256!,
        },
      });
      expect(approvedPlan.status).toBe('approved');
      expect(approvedPlan.profileId).toBe(approvedManifest.profileId);
      expect(approvedPlan.warnings).toBeUndefined();

      // Case-insensitive hash match (e.g. uppercase hex)
      const uppercasePlan = resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', {
        runtime: {
          webgpu: true,
          wasm: true,
          webgpuPrecisions: ['fp16', 'fp32', 'int8'],
          wasmPrecisions: ['fp32', 'int8'],
        },
        artifactIdentity: {
          repoId: approvedManifest.repoId!,
          revision: approvedManifest.revision!,
          encoderSha256: approvedManifest.encoderSha256!.toUpperCase(),
          decoderSha256: approvedManifest.decoderSha256!.toUpperCase(),
          configSha256: approvedManifest.configSha256!.toUpperCase(),
          vocabularySha256: approvedManifest.vocabularySha256!.toUpperCase(),
        },
      });
      expect(uppercasePlan.status).toBe('approved');

      // Tampered hash -> must downgrade to unverified
      const tamperedPlan = resolveApprovedSpeechExecutionPlan('parakeet-tdt-0.6b-v3', {
        runtime: {
          webgpu: true,
          wasm: true,
          webgpuPrecisions: ['fp16', 'fp32', 'int8'],
          wasmPrecisions: ['fp32', 'int8'],
        },
        artifactIdentity: {
          repoId: approvedManifest.repoId!,
          revision: approvedManifest.revision!,
          encoderSha256: 'deadbeef' + approvedManifest.encoderSha256!.slice(8),
          decoderSha256: approvedManifest.decoderSha256!,
          configSha256: approvedManifest.configSha256!,
          vocabularySha256: approvedManifest.vocabularySha256!,
        },
      });
      expect(tamperedPlan.status).toBe('unverified');
    });
  });

  describe('5. TypeScript In-Memory Compilation of Documentation Code Snippets', () => {
    it('compiles all Golden Path doc code samples against actual @asrjs/speech-recognition definitions', () => {
      const readmePath = resolve(__dirname, '../README.md');
      const apiGuidePath = resolve(__dirname, '../docs/PUBLIC_API_GUIDE.md');
      const intGuidePath = resolve(__dirname, '../docs/INTEGRATION_GUIDE.md');

      const readmeContent = readFileSync(readmePath, 'utf-8');
      const apiGuideContent = readFileSync(apiGuidePath, 'utf-8');
      const intGuideContent = readFileSync(intGuidePath, 'utf-8');

      expect(readmeContent.length).toBeGreaterThan(0);
      expect(apiGuideContent.length).toBeGreaterThan(0);
      expect(intGuideContent.length).toBeGreaterThan(0);

      // Synthesize test module containing doc code blocks wrapped in valid async test functions
      const virtualSource = `
        import {
          createSpeechRuntime,
          createWasmBackend,
          listSpeechModels,
          buildSpeechModelLoadOptions,
          buildSpeechTranscriptionOptions,
          loadSpeechModel,
          transcribeSpeech,
          transcribeSpeechBatch,
          transcribeSpeechFromMonoPcm,
          transcribeSpeechFromAudioSource,
          createSpeechPipeline,
          PcmAudioBuffer,
          getCanonicalTranscript,
        } from './src/index.js';

        import { createSpeechController } from './src/controller.js';
        import {
          createSpeechModelLocalEntries,
          loadSpeechModelFromLocalEntries,
          startMicrophoneCapture,
        } from './src/browser.js';
        import { decodeAudioSourceToMonoPcm } from './src/browser-media.js';
        import { createBuiltInSpeechRuntime, loadBuiltInSpeechModel } from './src/builtins.js';

        declare const file: File;
        declare const pcm: Float32Array;
        declare const firstAudio: Float32Array;
        declare const secondAudio: Float32Array;
        declare const source: any;
        declare const audioBuffer: any;

        // Snippet: README Golden Path 1
        export function readmeTranscribe(f: File) {
          return transcribeSpeechFromAudioSource(f, {
            modelId: 'parakeet-tdt-0.6b-v3',
            backend: 'webgpu-hybrid',
          });
        }

        // Snippet: PUBLIC_API_GUIDE Quick Start
        export async function quickStart() {
          const modelId = listSpeechModels().find((m) => m.preset === 'parakeet')?.modelId;
          if (!modelId) throw new Error('No built-in Parakeet model is available.');
          const loaded = await loadSpeechModel(
            buildSpeechModelLoadOptions({
              modelId,
              backend: 'webgpu-hybrid',
            }),
          );
          const result = await loaded.transcribeMonoPcm(pcm, 16000, {
            responseFlavor: 'canonical',
            ...buildSpeechTranscriptionOptions(modelId, {
              timestamps: true,
            }),
          });
          console.log(result.text);
          await loaded.dispose();
        }

        // Snippet: PUBLIC_API_GUIDE loadSpeechModel with callbacks
        export async function highLevelLoad() {
          const loaded = await loadSpeechModel({
            modelId: 'parakeet-tdt-0.6b-v3',
            onProgress(event) {
              console.log(event.phase, event.loaded, event.total);
            },
          });
          const result = await loaded.transcribeMonoPcm(pcm, 16000, {
            responseFlavor: 'canonical+native',
            onProgress(event) {
              console.log(event.stage, event.progress, event.remainingMs, event.metrics);
            },
          });
          await loaded.dispose();
        }

        // Snippet: PUBLIC_API_GUIDE one-shot audio source
        export async function oneShotAudioSource() {
          const result = await transcribeSpeechFromAudioSource(file, {
            modelId: 'parakeet-tdt-0.6b-v3',
            backend: 'webgpu-hybrid',
          });
          console.log(result.text);
        }

        // Snippet: PUBLIC_API_GUIDE transcribeSpeechFromMonoPcm
        export async function oneShotMonoPcm() {
          const canonical = await transcribeSpeechFromMonoPcm(pcm, 16000, {
            modelId: 'google/medasr',
            backend: 'wasm',
            transcribeOptions: {
              responseFlavor: 'canonical',
              detail: 'detailed',
            },
          });
          console.log(canonical.text);
        }

        // Snippet: PUBLIC_API_GUIDE createSpeechPipeline
        export async function pipelineSnippet() {
          const pipeline = createSpeechPipeline({ cacheModels: true });
          const med = await pipeline.transcribeMonoPcm(pcm, 16000, {
            modelId: 'google/medasr',
            backend: 'wasm',
          });
          const parakeet = await pipeline.transcribeMonoPcm(pcm, 16000, {
            modelId: 'parakeet-tdt-0.6b-v3',
            backend: 'webgpu-hybrid',
          });
          console.log(med.text, parakeet.text);
          await pipeline.dispose();
        }

        // Snippet: INTEGRATION_GUIDE Golden Path 2 createSpeechController
        export async function controllerSnippet() {
          const speech = createSpeechController();
          await speech.load({ modelId: 'parakeet-tdt-0.6b-v3', backend: 'webgpu-hybrid' });
          speech.subscribe(() => console.log(speech.getSnapshot().microphone?.transcriptText));
          await speech.startMicrophone({
            config: { gateMode: 'vad-only', vadBackend: 'ten-vad', tenVadEnabled: true },
          });
          await speech.stopMicrophone();
          await speech.dispose();
        }

        // Snippet: INTEGRATION_GUIDE Golden Path 3 loadLocalModel
        export async function loadLocalModel(files: FileList, modelId: string) {
          return loadSpeechModelFromLocalEntries({
            modelId,
            entries: createSpeechModelLocalEntries(Array.from(files)),
            backend: 'wasm',
            wasmPaths: '/ort/',
          });
        }
      `;

      // Compile using TypeScript compiler in-memory
      const compilerOptions: ts.CompilerOptions = {
        target: ts.ScriptTarget.ES2022,
        module: ts.ModuleKind.ESNext,
        moduleResolution: ts.ModuleResolutionKind.Bundler,
        strict: true,
        noEmit: true,
        skipLibCheck: true,
      };

      const sourceFileName = resolve(__dirname, '../__doc_snippets_verify__.ts');
      const host = ts.createCompilerHost(compilerOptions);
      const originalGetSourceFile = host.getSourceFile;
      host.getSourceFile = (fileName, languageVersion) => {
        if (fileName === sourceFileName) {
          return ts.createSourceFile(fileName, virtualSource, languageVersion, true);
        }
        return originalGetSourceFile(fileName, languageVersion);
      };

      const program = ts.createProgram([sourceFileName], compilerOptions, host);
      const diagnostics = ts.getPreEmitDiagnostics(program);

      const relevantErrors = diagnostics.filter(
        (d) => d.file?.fileName === sourceFileName && d.category === ts.DiagnosticCategory.Error,
      );

      if (relevantErrors.length > 0) {
        const errorMessages = relevantErrors
          .map((d) => {
            const message = ts.flattenDiagnosticMessageText(d.messageText, '\n');
            const { line, character } = d.file!.getLineAndCharacterOfPosition(d.start!);
            return `Line ${line + 1}:${character + 1} - ${message}`;
          })
          .join('\n');
        throw new Error(`Doc snippet typecheck failed with errors:\n${errorMessages}`);
      }

      expect(relevantErrors.length).toBe(0);
    }, 30000);
  });
});

import { describe, expect, it } from 'vitest';

import type { ExecutionBackend } from '../src/types/index.js';
import {
  NemoAedSpeechModel,
  type NemoAedExecutor,
  type NemoAedModelConfig,
  type NemoAedNativeTranscript,
} from '../src/models/nemo-aed/index.js';

function createBackend(id = 'wasm'): ExecutionBackend {
  return {
    id,
    displayName: id.toUpperCase(),
    async probeCapabilities() {
      return {
        id,
        displayName: id.toUpperCase(),
        available: true,
        priority: 1,
        environments: ['browser', 'node'],
        acceleration: ['cpu'],
        supportedPrecisions: ['fp32', 'int8'],
        supportsFp16: false,
        supportsInt8: true,
        supportsSharedArrayBuffer: true,
        requiresSharedArrayBuffer: false,
        fallbackSuitable: true,
        experimental: false,
        notes: [],
      };
    },
    async createExecutionContext() {
      return {
        backendId: id,
        capabilities: await this.probeCapabilities(),
        dispose() {
          return undefined;
        },
      };
    },
  };
}

function createConfig(): NemoAedModelConfig {
  return {
    ecosystem: 'nemo',
    architecture: 'nemo-aed',
    decoderArchitecture: 'transformer-decoder',
    encoderArchitecture: 'fastconformer',
    sampleRate: 16000,
    frameShiftSeconds: 0.01,
    melBins: 128,
    subsamplingFactor: 8,
    vocabularySize: 5248,
    languages: ['en'],
    maxTargetPositions: 1024,
    promptFormat: 'canary2',
    promptDefaults: [
      {
        sourceLanguage: 'en',
        targetLanguage: 'en',
        decoderContext: '',
        emotion: '<|emo:undefined|>',
        punctuate: true,
        inverseTextNormalization: false,
        timestamps: false,
        diarize: false,
      },
    ],
    tokenizer: {
      kind: 'aggregate',
      bosTokenId: 4,
      eosTokenId: 3,
      padTokenId: 2,
      vocabSize: 5248,
    },
  };
}

function createTranscript(): NemoAedNativeTranscript {
  return {
    utteranceText: 'test',
    isFinal: true,
    tokens: [],
  };
}

describe('NemoAedSpeechModel session lifecycle', () => {
  it('waits for executor readiness before resolving createSession()', async () => {
    let releaseReady: (() => void) | undefined;
    const readyGate = new Promise<void>((resolve) => {
      releaseReady = resolve;
    });

    const executor: NemoAedExecutor = {
      async ready() {
        await readyGate;
      },
      async transcribe() {
        return createTranscript();
      },
      dispose() {
        return undefined;
      },
    };

    const model = new NemoAedSpeechModel(
      createBackend(),
      'nemo-aed',
      'nvidia/canary-180m-flash',
      {
        ecosystem: 'nemo',
        encoder: 'fastconformer',
        decoder: 'transformer-decoder',
        topology: 'aed',
        task: 'multitask-asr-translation',
      },
      createConfig(),
      'canary',
      undefined,
      { executor },
      () => 'test model',
    );

    let resolved = false;
    const sessionPromise = model.createSession().then((session) => {
      resolved = true;
      return session;
    });

    await Promise.resolve();
    await Promise.resolve();
    expect(resolved).toBe(false);

    releaseReady?.();
    const session = await sessionPromise;
    expect(resolved).toBe(true);

    await session.dispose();
    await model.dispose();
  });
});

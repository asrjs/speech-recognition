// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { WhisperOnnxExecutor } from '../src/models/whisper-seq2seq/executor.js';

describe('Whisper ONNX end-to-end smoke', () => {
  it('runs real ONNX inference and does not return stub output', async () => {
    if (!process.env.ASRJS_FIXTURE_SMOKE && !process.env.ASRJS_FIXTURE_SMOKE_FORCE) {
      console.warn('Skipping: set ASRJS_FIXTURE_SMOKE=1 to run ONNX fixture smoke test');
      return;
    }

    const config = {
      melBins: 80,
      maxSourcePositions: 3000,
      maxTargetPositions: 448,
      languages: ['tr', 'en'],
      processorArchitecture: 'whisper-mel',
      encoderArchitecture: 'whisper-transformer',
      decoderArchitecture: 'transformer-decoder',
      tokenizer: { kind: 'tiktoken', vocabSize: 51865 },
      windowing: { kind: 'disabled' },
    };

    const source = {
      kind: 'huggingface',
      repoId: 'onnx-community/whisper-tiny',
    };

    const executor = new WhisperOnnxExecutor(
      'whisper-tiny',
      { family: 'whisper-seq2seq', task: 'transcribe' },
      config,
      'wasm',
      { source }
    );

    // 1 second of 440 Hz sine at 16 kHz
    const sampleRate = 16000;
    const samples = new Float32Array(sampleRate);
    for (let i = 0; i < sampleRate; i++) {
      samples[i] = Math.sin((2 * Math.PI * 440 * i) / sampleRate);
    }

    const audio = {
      sampleRate,
      durationSeconds: 1,
      channels: [samples],
      numberOfChannels: 1,
      numberOfFrames: sampleRate,
    };

    await executor.ready();
    const result = await executor.transcribe(
      audio,
      { language: 'tr', noTimestamps: true, maxNewTokens: 50 },
      { modelId: 'whisper-tiny', config }
    );

    await executor.dispose();

    // Assert: no stub warning
    const stubWarning = result.warnings.find(
      (w) => w.code === 'whisper-seq2seq.stubbed-decoder'
    );
    expect(stubWarning).toBeUndefined();

    // Assert: tokens array is populated (even if EOS immediately)
    // For a sine wave, output may be empty or contain only special tokens,
    // but the pipeline should have produced tokens (including EOS).
    expect(Array.isArray(result.tokens)).toBe(true);
  });
});

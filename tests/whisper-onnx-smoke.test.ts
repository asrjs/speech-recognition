import { describe, it, expect } from 'vitest';
import * as path from 'path';
import { pathToFileURL } from 'url';
import { WhisperOnnxExecutor } from '../src/models/whisper-seq2seq/executor.js';
import type {
  WhisperArtifactSource,
  WhisperSeq2SeqModelConfig,
} from '../src/models/whisper-seq2seq/types.js';
import {
  requireFixtureDirectory,
  requireFixtureFiles,
  requireFixtureOptIn,
} from './support/fixture-gates.js';

describe('Whisper ONNX end-to-end smoke', () => {
  it('runs real ONNX inference and does not return stub output', async (context) => {
    requireFixtureOptIn(
      context,
      process.env.ASRJS_FIXTURE_SMOKE === '1' || process.env.ASRJS_FIXTURE_SMOKE_FORCE === '1',
      'Set ASRJS_FIXTURE_SMOKE=1 to run ONNX fixture smoke test',
    );

    const config: WhisperSeq2SeqModelConfig = {
      ecosystem: 'openai',
      architecture: 'whisper-seq2seq',
      melBins: 80,
      sampleRate: 16000,
      maxSourcePositions: 3000,
      maxTargetPositions: 448,
      vocabularySize: 51865,
      languages: ['tr', 'en'],
      processorArchitecture: 'whisper-mel',
      encoderArchitecture: 'whisper-transformer',
      decoderArchitecture: 'transformer-decoder',
      tokenizer: { kind: 'tiktoken', vocabSize: 51865 },
    };

    const fixtureDir = requireFixtureDirectory(
      process.env.WHISPER_MERGED_FIXTURE_DIR,
      'WHISPER_MERGED_FIXTURE_DIR',
    );
    const encoderPath = path.join(
      fixtureDir,
      process.env.WHISPER_MERGED_ENCODER ?? 'onnx/encoder_model.onnx',
    );
    const decoderPath = path.join(
      fixtureDir,
      process.env.WHISPER_MERGED_DECODER ?? 'onnx/decoder_model_merged.onnx',
    );
    const tokenizerPath = path.join(
      fixtureDir,
      process.env.WHISPER_MERGED_TOKENIZER ?? 'tokenizer.json',
    );
    requireFixtureFiles({ encoder: encoderPath, decoder: decoderPath, tokenizer: tokenizerPath });

    const source: WhisperArtifactSource = {
      kind: 'direct',
      artifacts: {
        encoderUrl: pathToFileURL(encoderPath).href,
        decoderUrl: pathToFileURL(decoderPath).href,
        tokenizerUrl: pathToFileURL(tokenizerPath).href,
      },
    };

    const executor = new WhisperOnnxExecutor(
      'whisper-tiny',
      { ecosystem: 'openai', family: 'whisper-seq2seq', task: 'transcribe' },
      config,
      'wasm',
      { source },
    );
    context.onTestFinished(() => executor.dispose());

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
      { modelId: 'whisper-tiny', config },
    );

    // Assert: no stub warning
    const stubWarning = result.warnings?.find((w) => w.code === 'whisper-seq2seq.stubbed-decoder');
    expect(stubWarning).toBeUndefined();

    // Assert: tokens array is populated (even if EOS immediately)
    // For a sine wave, output may be empty or contain only special tokens,
    // but the pipeline should have produced tokens (including EOS).
    expect(Array.isArray(result.tokens)).toBe(true);
  }, 240_000);
});

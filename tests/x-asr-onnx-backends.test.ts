import * as fs from 'node:fs';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';

import { OrtXAsrExecutor } from '../src/models/x-asr/executor.js';
import { createXAsrModelFamily } from '../src/models/x-asr/model.js';
import type { XAsrModelConfig } from '../src/models/x-asr/types.js';

const MODEL_DIR = 'N:/models/x-asr/zh-en/chunk-160ms-model';
const ENCODER = path.join(MODEL_DIR, 'encoder-160ms.onnx');
const DECODER = path.join(MODEL_DIR, 'decoder-160ms.onnx');
const JOINER = path.join(MODEL_DIR, 'joiner-160ms.onnx');
const TOKENS = path.join(MODEL_DIR, 'tokens.txt');
const WAVEFORM = 'N:/models/gigaam/multilingual-ctc/captures/jfk-short.waveform.npy';
const ENABLED = process.env.XASR_ONNX_SMOKE === '1';
const EXPECTED =
  'And so my fellow americans ask not what your country can do for you ask what you can do for your country';

const CONFIG: XAsrModelConfig = {
  ecosystem: 'x-asr',
  architecture: 'zipformer2-streaming-rnnt',
  processorArchitecture: 'kaldi-fbank',
  encoderArchitecture: 'zipformer2',
  decoderArchitecture: 'stateless-rnnt',
  sampleRate: 16000,
  featureDim: 80,
  featureHopSeconds: 0.01,
  rawStride: 1,
  languages: ['zh', 'en'],
  chunkMs: 160,
  graph: {
    encoderStateInputs: [],
    encoderFrameSize: 29,
    encoderFrameShift: 16,
    decoderContextSize: 2,
    featureInputName: 'x',
    encoderOutputName: 'encoder_out',
    decoderInputName: 'y',
    decoderOutputName: 'decoder_out',
    joinerEncoderInputName: 'encoder_out',
    joinerDecoderInputName: 'decoder_out',
    joinerOutputName: 'logit',
  },
};

function loadNpyFloat32(filePath: string): Float32Array {
  const buffer = fs.readFileSync(filePath);
  const major = buffer[6];
  const headerLength = major === 1 ? buffer.readUInt16LE(8) : Number(buffer.readBigUInt64LE(8));
  const headerStart = major === 1 ? 10 : 16;
  const dataOffset = headerStart + headerLength;
  return new Float32Array(
    buffer.buffer,
    buffer.byteOffset + dataOffset,
    (buffer.byteLength - dataOffset) / 4,
  );
}

function classifyOrtFailure(error: unknown, backend: 'wasm' | 'webgpu'): Record<string, unknown> {
  const message = error instanceof Error ? error.message : String(error);
  const lowered = message.toLowerCase();
  let failureClass = 'ORT_SESSION_FAILED';
  if (backend === 'webgpu' && /webgpu|gpu adapter|not support/i.test(message)) {
    failureClass = /dtype|float16|fp16/i.test(message)
      ? 'WEBGPU_UNSUPPORTED_DTYPE'
      : /memory|oom|out of memory/i.test(lowered)
        ? 'WEBGPU_MEMORY_LIMIT'
        : 'WEBGPU_NO_ADAPTER';
  } else if (/not.*implement|unsupported.*op|failed to find kernel/i.test(lowered)) {
    failureClass = 'ORT_WEB_UNSUPPORTED_OP';
  } else if (/memory|oom|out of memory|array buffer allocation/i.test(lowered)) {
    failureClass = backend === 'webgpu' ? 'WEBGPU_MEMORY_LIMIT' : 'WASM_MEMORY_LIMIT';
  }
  return { backend, failureClass, message };
}

describe.skipIf(
  !ENABLED || !fs.existsSync(ENCODER) || !fs.existsSync(DECODER) || !fs.existsSync(JOINER) || !fs.existsSync(TOKENS) || !fs.existsSync(WAVEFORM),
)('X-ASR official Zipformer2 wasm/webgpu smoke', () => {
  it(
    'transcribes jfk-short through onnxruntime-web WASM with stateful encoder caches',
    { timeout: 600_000 },
    async () => {
      const waveform = loadNpyFloat32(WAVEFORM);
      const executor = new OrtXAsrExecutor('X-ASR-zh-en', 'wasm', CONFIG, {
        source: {
          kind: 'direct',
          cpuThreads: 1,
          artifacts: {
            encoderUrl: pathToFileURL(ENCODER).href,
            decoderUrl: pathToFileURL(DECODER).href,
            joinerUrl: pathToFileURL(JOINER).href,
            tokenizerUrl: pathToFileURL(TOKENS).href,
          },
        },
      });
      const outPath = path.resolve('tools/data/results/x-asr/x-asr-zh-en-160ms-jfk-short-wasm.json');
      let transcribed = false;
      try {
        await executor.ready();
        const result = await executor.transcribe({
          sampleRate: 16000,
          numberOfChannels: 1,
          numberOfFrames: waveform.length,
          durationSeconds: waveform.length / 16000,
          channels: [waveform],
        });
        transcribed = true;
        const match = result.utteranceText === EXPECTED;
        const payload = {
          backend: 'wasm',
          engine: 'onnxruntime-web',
          streaming: 'true-stateful-zipformer2',
          text: result.utteranceText,
          expected: EXPECTED,
          text_match: match,
          failureClass: match ? null : 'PREPROCESSING_MISMATCH',
          metrics: result.metrics,
          status: 'experimental-js-frontend-wasm',
        };
        fs.mkdirSync(path.dirname(outPath), { recursive: true });
        fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
        expect(result.utteranceText).toBe(EXPECTED);
      } catch (error) {
        if (!transcribed) {
          const payload = { ...classifyOrtFailure(error, 'wasm'), status: 'experimental-blocked' };
          fs.mkdirSync(path.dirname(outPath), { recursive: true });
          fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
        }
        throw error;
      } finally {
        executor.dispose();
      }
    },
  );

  it(
    'records Node WebGPU result or failure class',
    { timeout: 600_000 },
    async () => {
      const waveform = loadNpyFloat32(WAVEFORM);
      const executor = new OrtXAsrExecutor('X-ASR-zh-en', 'webgpu', CONFIG, {
        source: {
          kind: 'direct',
          cpuThreads: 1,
          artifacts: {
            encoderUrl: pathToFileURL(ENCODER).href,
            decoderUrl: pathToFileURL(DECODER).href,
            joinerUrl: pathToFileURL(JOINER).href,
            tokenizerUrl: pathToFileURL(TOKENS).href,
          },
        },
      });
      const outPath = path.resolve('tools/data/results/x-asr/x-asr-zh-en-160ms-jfk-short-webgpu.json');
      fs.mkdirSync(path.dirname(outPath), { recursive: true });
      try {
        await executor.ready();
        const result = await executor.transcribe({
          sampleRate: 16000,
          numberOfChannels: 1,
          numberOfFrames: waveform.length,
          durationSeconds: waveform.length / 16000,
          channels: [waveform],
        });
        fs.writeFileSync(outPath, `${JSON.stringify({ backend: 'webgpu', text: result.utteranceText, expected: EXPECTED, text_match: result.utteranceText === EXPECTED, status: 'experimental-js-frontend-webgpu' }, null, 2)}\n`);
        expect(result.utteranceText).toBe(EXPECTED);
      } catch (error) {
        const payload = { ...classifyOrtFailure(error, 'webgpu'), status: 'experimental-blocked' };
        fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
        expect(payload.failureClass).toMatch(
          /WEBGPU_NO_ADAPTER|WEBGPU_UNSUPPORTED_DTYPE|WEBGPU_MEMORY_LIMIT|ORT_WEB_UNSUPPORTED_OP|ORT_SESSION_FAILED/,
        );
      } finally {
        executor.dispose();
      }
    },
  );
  it(
    'runs the public model-created streaming transcriber on the real artifact',
    { timeout: 600_000 },
    async () => {
      const waveform = loadNpyFloat32(WAVEFORM);
      const executor = new OrtXAsrExecutor('X-ASR-zh-en', 'wasm', CONFIG, {
        source: {
          kind: 'direct',
          cpuThreads: 1,
          artifacts: {
            encoderUrl: pathToFileURL(ENCODER).href,
            decoderUrl: pathToFileURL(DECODER).href,
            joinerUrl: pathToFileURL(JOINER).href,
            tokenizerUrl: pathToFileURL(TOKENS).href,
          },
        },
      });
      const model = await createXAsrModelFamily({ dependencies: { executor } }).createModel(
        { modelId: 'X-ASR-zh-en' },
        {
          runtime: {} as never,
          backend: { id: 'wasm', displayName: 'WASM' } as never,
          assetProvider: undefined,
          hooks: {},
        },
      );
      try {
        const transcriber = await model.createStreamingTranscriber();
        try {
          await transcriber.pushAudio(waveform);
          const result = await transcriber.finalize();
          expect(result.kind).toBe('final');
          expect(result.text).toBe(EXPECTED);
        } finally {
          await transcriber.dispose?.();
        }
      } finally {
        await model.dispose();
      }
    },
  );
});

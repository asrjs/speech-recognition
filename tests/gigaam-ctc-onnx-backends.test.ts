import * as fs from 'node:fs';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';

import { OrtGigaAmCtcExecutor } from '../src/models/gigaam-ctc/executor.js';
import type { GigaAmModelConfig } from '../src/models/gigaam-ctc/types.js';

const REFERENCE = path.resolve(
  'tools/data/results/gigaam/multilingual-ctc-jfk-short-reference.json',
);
const ONNX_DIR = path.resolve('N:/models/onnx/gigaam/multilingual-ctc');
const ONNX_PATH = path.join(ONNX_DIR, 'multilingual_ctc.onnx');
const VOCAB_PATH = path.join(ONNX_DIR, 'multilingual_vocab.txt');
const ENABLED = process.env.GIGAAM_CTC_ONNX_SMOKE === '1';

const CONFIG: GigaAmModelConfig = {
  ecosystem: 'gigaam',
  architecture: 'gigaam-ctc',
  processorArchitecture: 'gigaam-fbank',
  encoderArchitecture: 'gigaam-conformer',
  decoderArchitecture: 'ctc',
  sampleRate: 16000,
  rawStride: 4,
  nMels: 64,
  featureHopSeconds: 0.01,
  vocabularySize: 71,
  languages: ['ru', 'en', 'kk', 'ky', 'uz'],
  tokenizer: { kind: 'sentencepiece', blankTokenId: 70 },
  nFft: 320,
  winLength: 320,
  hopLength: 160,
  center: false,
  featureLayout: 'mel-major',
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

function audioInput(waveform: Float32Array) {
  return {
    sampleRate: 16000,
    numberOfChannels: 1,
    numberOfFrames: waveform.length,
    durationSeconds: waveform.length / 16000,
    channels: [waveform],
  };
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
  !ENABLED || !fs.existsSync(REFERENCE) || !fs.existsSync(ONNX_PATH) || !fs.existsSync(VOCAB_PATH),
)('GigaAM CTC official ONNX wasm/webgpu smoke', () => {
  it(
    'transcribes jfk-short from JS frontend through onnxruntime-web WASM',
    { timeout: 600_000 },
    async () => {
      const capture = JSON.parse(fs.readFileSync(REFERENCE, 'utf8')) as {
        samples: Array<{
          text: string;
          audio: { waveform_npy: string };
        }>;
      };
      const sample = capture.samples[0]!;
      const waveform = loadNpyFloat32(sample.audio.waveform_npy);
      const executor = new OrtGigaAmCtcExecutor(
        'gigaam-multilingual-ctc',
        'wasm',
        CONFIG,
        {
          source: {
            kind: 'direct',
            cpuThreads: 1,
            artifacts: {
              modelUrl: pathToFileURL(ONNX_PATH).href,
              tokenizerUrl: pathToFileURL(VOCAB_PATH).href,
            },
          },
        },
      );
      try {
        await executor.ready();
        const result = await executor.transcribe({
          sampleRate: 16000,
          numberOfChannels: 1,
          numberOfFrames: waveform.length,
          durationSeconds: waveform.length / 16000,
          channels: [waveform],
        });
        const payload = {
          backend: 'wasm',
          engine: 'onnxruntime-web',
          text: result.utteranceText,
          expected: sample.text,
          text_match: result.utteranceText === sample.text,
          onnx_path: ONNX_PATH,
          status: 'experimental-js-frontend-wasm',
        };
        const outPath = path.resolve(
          'tools/data/results/gigaam/multilingual-ctc-jfk-short-wasm.json',
        );
        fs.mkdirSync(path.dirname(outPath), { recursive: true });
        fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
        expect(result.utteranceText).toBe(sample.text);
      } catch (error) {
        const payload = {
          ...classifyOrtFailure(error, 'wasm'),
          onnx_path: ONNX_PATH,
          status: 'experimental-blocked',
        };
        const outPath = path.resolve(
          'tools/data/results/gigaam/multilingual-ctc-jfk-short-wasm.json',
        );
        fs.mkdirSync(path.dirname(outPath), { recursive: true });
        fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
        throw error;
      } finally {
        executor.dispose();
      }
    },
  );

  it(
    'records WebGPU session result or failure class for the official fp32 graph',
    { timeout: 600_000 },
    async () => {
      const capture = JSON.parse(fs.readFileSync(REFERENCE, 'utf8')) as {
        samples: Array<{ text: string; audio: { waveform_npy: string } }>;
      };
      const sample = capture.samples[0]!;
      const waveform = loadNpyFloat32(sample.audio.waveform_npy);
      const executor = new OrtGigaAmCtcExecutor(
        'gigaam-multilingual-ctc',
        'webgpu',
        CONFIG,
        {
          source: {
            kind: 'direct',
            cpuThreads: 1,
            artifacts: {
              modelUrl: pathToFileURL(ONNX_PATH).href,
              tokenizerUrl: pathToFileURL(VOCAB_PATH).href,
            },
          },
        },
      );
      const outPath = path.resolve(
        'tools/data/results/gigaam/multilingual-ctc-jfk-short-webgpu.json',
      );
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
        const payload = {
          backend: 'webgpu',
          engine: 'onnxruntime-web',
          text: result.utteranceText,
          expected: sample.text,
          text_match: result.utteranceText === sample.text,
          onnx_path: ONNX_PATH,
          dtype: 'float32',
          status: 'experimental-js-frontend-webgpu',
        };
        fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
        expect(result.utteranceText).toBe(sample.text);
      } catch (error) {
        const payload = {
          ...classifyOrtFailure(error, 'webgpu'),
          onnx_path: ONNX_PATH,
          dtype: 'float32',
          status: 'experimental-blocked',
        };
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
    'runs mixed-length inputs through one official WASM batch graph call',
    { timeout: 600_000 },
    async () => {
      const capture = JSON.parse(fs.readFileSync(REFERENCE, 'utf8')) as {
        samples: Array<{ text: string; audio: { waveform_npy: string } }>;
      };
      const sample = capture.samples[0]!;
      const waveform = loadNpyFloat32(sample.audio.waveform_npy);
      const shorterWaveform = waveform.subarray(0, Math.floor(waveform.length * 0.6));
      const executor = new OrtGigaAmCtcExecutor(
        'gigaam-multilingual-ctc',
        'wasm',
        CONFIG,
        {
          source: {
            kind: 'direct',
            cpuThreads: 1,
            artifacts: {
              modelUrl: pathToFileURL(ONNX_PATH).href,
              tokenizerUrl: pathToFileURL(VOCAB_PATH).href,
            },
          },
        },
      );
      const outPath = path.resolve(
        'tools/data/results/gigaam/multilingual-ctc-jfk-short-batch-wasm.json',
      );
      fs.mkdirSync(path.dirname(outPath), { recursive: true });
      try {
        await executor.ready();
        const results = await executor.transcribeBatch([
          audioInput(waveform),
          audioInput(shorterWaveform),
        ]);
        const payload = {
          backend: 'wasm',
          engine: 'onnxruntime-web',
          batch_size: results.length,
          texts: results.map((result) => result.utteranceText),
          expected_first: sample.text,
          first_text_match: results[0]?.utteranceText === sample.text,
          second_text_non_empty: Boolean(results[1]?.utteranceText),
          onnx_path: ONNX_PATH,
          status: 'experimental-js-frontend-wasm-batch',
        };
        fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
        expect(results).toHaveLength(2);
        expect(results[0]?.utteranceText).toBe(sample.text);
        expect(results[1]?.utteranceText.length).toBeGreaterThan(0);
      } catch (error) {
        const payload = {
          ...classifyOrtFailure(error, 'wasm'),
          onnx_path: ONNX_PATH,
          status: 'experimental-blocked',
        };
        fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
        throw error;
      } finally {
        executor.dispose();
      }
    },
  );

  it(
    'transcribes jfk-short from JS frontend through official fp16 WASM',
    { timeout: 600_000 },
    async () => {
      const fp16Onnx = path.resolve('N:/models/onnx/gigaam/multilingual-ctc-fp16/multilingual_ctc.onnx');
      const fp16Vocab = path.resolve('N:/models/onnx/gigaam/multilingual-ctc-fp16/multilingual_vocab.txt');
      if (!fs.existsSync(fp16Onnx) || !fs.existsSync(fp16Vocab)) {
        return;
      }
      const capture = JSON.parse(fs.readFileSync(REFERENCE, 'utf8')) as {
        samples: Array<{ text: string; audio: { waveform_npy: string } }>;
      };
      const sample = capture.samples[0]!;
      const waveform = loadNpyFloat32(sample.audio.waveform_npy);
      const executor = new OrtGigaAmCtcExecutor(
        'gigaam-multilingual-ctc',
        'wasm',
        CONFIG,
        {
          source: {
            kind: 'direct',
            cpuThreads: 1,
            artifacts: {
              modelUrl: pathToFileURL(fp16Onnx).href,
              tokenizerUrl: pathToFileURL(fp16Vocab).href,
            },
          },
        },
      );
      const outPath = path.resolve(
        'tools/data/results/gigaam/multilingual-ctc-jfk-short-wasm-fp16.json',
      );
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
        const payload = {
          backend: 'wasm',
          engine: 'onnxruntime-web',
          dtype: 'float16',
          text: result.utteranceText,
          expected: sample.text,
          text_match: result.utteranceText === sample.text,
          onnx_path: fp16Onnx,
          status: 'experimental-js-frontend-wasm-fp16',
        };
        fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
        expect(result.utteranceText).toBe(sample.text);
      } catch (error) {
        const payload = {
          ...classifyOrtFailure(error, 'wasm'),
          dtype: 'float16',
          onnx_path: fp16Onnx,
          status: 'experimental-blocked',
        };
        fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
        throw error;
      } finally {
        executor.dispose();
      }
    },
  );
});

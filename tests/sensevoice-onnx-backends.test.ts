import * as fs from 'node:fs';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';

import { OrtSenseVoiceExecutor } from '../src/models/sensevoice/executor.js';

const ONNX_DIR = 'N:/models/onnx/sensevoice/small';
const ONNX_PATH = path.join(ONNX_DIR, 'model.onnx');
const VOCAB_PATH = path.join(ONNX_DIR, 'vocab.txt');
const CMVN_PATH = path.join(ONNX_DIR, 'am.mvn');
const WAVEFORM_PATH = 'N:/models/sensevoice/SenseVoiceSmall/captures/jfk-short.waveform.npy';
const ENABLED = process.env.SENSEVOICE_ONNX_SMOKE === '1';
const EXPECTED =
  'and so my fellow americans ask not what your country can do for you ask what you can do for your country';

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
  !ENABLED || !fs.existsSync(ONNX_PATH) || !fs.existsSync(VOCAB_PATH) || !fs.existsSync(CMVN_PATH) || !fs.existsSync(WAVEFORM_PATH),
)('SenseVoice official ONNX wasm/webgpu smoke', () => {
  it(
    'transcribes jfk-short from JS frontend through onnxruntime-web WASM',
    { timeout: 600_000 },
    async () => {
      const waveform = loadNpyFloat32(WAVEFORM_PATH);
      const executor = new OrtSenseVoiceExecutor('sensevoice-small', 'wasm', {
        source: {
          kind: 'direct',
          cpuThreads: 1,
          artifacts: {
            modelUrl: pathToFileURL(ONNX_PATH).href,
            tokenizerUrl: pathToFileURL(VOCAB_PATH).href,
            cmvnUrl: pathToFileURL(CMVN_PATH).href,
          },
        },
      });
      const outPath = path.resolve('tools/data/results/sensevoice/sensevoice-small-jfk-short-wasm.json');
      try {
        await executor.ready();
        const result = await executor.transcribe(
          {
            sampleRate: 16000,
            numberOfChannels: 1,
            numberOfFrames: waveform.length,
            durationSeconds: waveform.length / 16000,
            channels: [waveform],
          },
          { language: 'en', useItn: false },
        );
        const payload = {
          backend: 'wasm',
          engine: 'onnxruntime-web',
          text: result.utteranceText,
          expected: EXPECTED,
          text_match: result.utteranceText === EXPECTED,
          language: result.language,
          metadata: result.metadata,
          metrics: result.metrics,
          onnx_path: ONNX_PATH,
          status: 'experimental-js-frontend-wasm',
        };
        fs.mkdirSync(path.dirname(outPath), { recursive: true });
        fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
        expect(result.utteranceText).toBe(EXPECTED);
      } catch (error) {
        const payload = {
          ...classifyOrtFailure(error, 'wasm'),
          onnx_path: ONNX_PATH,
          status: 'experimental-blocked',
        };
        fs.mkdirSync(path.dirname(outPath), { recursive: true });
        fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
        throw error;
      } finally {
        executor.dispose();
      }
    },
  );

  it(
    'records Node WebGPU session result or failure class for the official graph',
    { timeout: 600_000 },
    async () => {
      const waveform = loadNpyFloat32(WAVEFORM_PATH);
      const executor = new OrtSenseVoiceExecutor('sensevoice-small', 'webgpu', {
        source: {
          kind: 'direct',
          cpuThreads: 1,
          artifacts: {
            modelUrl: pathToFileURL(ONNX_PATH).href,
            tokenizerUrl: pathToFileURL(VOCAB_PATH).href,
            cmvnUrl: pathToFileURL(CMVN_PATH).href,
          },
        },
      });
      const outPath = path.resolve('tools/data/results/sensevoice/sensevoice-small-jfk-short-webgpu.json');
      fs.mkdirSync(path.dirname(outPath), { recursive: true });
      try {
        await executor.ready();
        const result = await executor.transcribe(
          {
            sampleRate: 16000,
            numberOfChannels: 1,
            numberOfFrames: waveform.length,
            durationSeconds: waveform.length / 16000,
            channels: [waveform],
          },
          { language: 'en', useItn: false },
        );
        const payload = {
          backend: 'webgpu',
          engine: 'onnxruntime-web',
          text: result.utteranceText,
          expected: EXPECTED,
          text_match: result.utteranceText === EXPECTED,
          onnx_path: ONNX_PATH,
          status: 'experimental-js-frontend-webgpu',
        };
        fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
        expect(result.utteranceText).toBe(EXPECTED);
      } catch (error) {
        const payload = {
          ...classifyOrtFailure(error, 'webgpu'),
          onnx_path: ONNX_PATH,
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
});

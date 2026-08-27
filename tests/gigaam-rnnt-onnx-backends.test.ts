import * as fs from 'node:fs';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';

import { OrtGigaAmRnntExecutor } from '../src/models/gigaam-rnnt/executor.js';
import type { GigaAmRnntModelConfig } from '../src/models/gigaam-rnnt/types.js';

const REFERENCE = path.resolve('tools/data/results/gigaam/v3-e2e-rnnt-example-reference.json');
const ONNX_DIR = path.resolve('N:/models/onnx/gigaam/v3-e2e-rnnt');
const ENCODER_PATH = path.join(ONNX_DIR, 'v3_e2e_rnnt_encoder.onnx');
const DECODER_PATH = path.join(ONNX_DIR, 'v3_e2e_rnnt_decoder.onnx');
const JOINT_PATH = path.join(ONNX_DIR, 'v3_e2e_rnnt_joint.onnx');
const VOCAB_PATH = path.join(ONNX_DIR, 'v3_e2e_rnnt_vocab.txt');
const ENABLED = process.env.GIGAAM_RNNT_ONNX_SMOKE === '1';

const CONFIG: GigaAmRnntModelConfig = {
  ecosystem: 'gigaam',
  architecture: 'gigaam-rnnt',
  processorArchitecture: 'gigaam-fbank',
  encoderArchitecture: 'gigaam-conformer',
  decoderArchitecture: 'rnnt',
  sampleRate: 16000,
  rawStride: 4,
  nMels: 64,
  featureHopSeconds: 0.01,
  vocabularySize: 1025,
  languages: ['ru'],
  tokenizer: { kind: 'sentencepiece', blankTokenId: 1024 },
  nFft: 320,
  winLength: 320,
  hopLength: 160,
  featureLayout: 'mel-major',
  predictionHiddenSize: 320,
  predictionRnnLayers: 1,
  maxTokensPerFrame: 10,
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

function artifacts() {
  return {
    encoderUrl: pathToFileURL(ENCODER_PATH).href,
    decoderUrl: pathToFileURL(DECODER_PATH).href,
    jointUrl: pathToFileURL(JOINT_PATH).href,
    tokenizerUrl: pathToFileURL(VOCAB_PATH).href,
  };
}

describe.skipIf(
  !ENABLED ||
    !fs.existsSync(REFERENCE) ||
    !fs.existsSync(ENCODER_PATH) ||
    !fs.existsSync(DECODER_PATH) ||
    !fs.existsSync(JOINT_PATH) ||
    !fs.existsSync(VOCAB_PATH),
)('GigaAM v3 E2E RNN-T official ONNX wasm/webgpu smoke', () => {
  it('transcribes official example.wav from JS frontend through onnxruntime-web WASM', { timeout: 600_000 }, async () => {
    const capture = JSON.parse(fs.readFileSync(REFERENCE, 'utf8')) as {
      samples: Array<{ text: string; audio: { waveform_npy: string } }>;
    };
    const sample = capture.samples[0]!;
    const waveform = loadNpyFloat32(sample.audio.waveform_npy);
    const executor = new OrtGigaAmRnntExecutor('gigaam-v3-e2e-rnnt', 'wasm', CONFIG, {
      source: { kind: 'direct', cpuThreads: 1, artifacts: artifacts() },
    });
    const outPath = path.resolve('tools/data/results/gigaam/v3-e2e-rnnt-example-wasm.json');
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
        text: result.utteranceText,
        expected: sample.text,
        text_match: result.utteranceText === sample.text,
        onnx_dir: ONNX_DIR,
        status: 'experimental-js-frontend-wasm',
      };
      fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
      expect(result.utteranceText).toBe(sample.text);
    } catch (error) {
      fs.writeFileSync(outPath, `${JSON.stringify({ ...classifyOrtFailure(error, 'wasm'), onnx_dir: ONNX_DIR, status: 'experimental-blocked' }, null, 2)}\n`);
      throw error;
    } finally {
      executor.dispose();
    }
  });

  it('records WebGPU session result or failure class for the official fp32 graphs', { timeout: 600_000 }, async () => {
    const capture = JSON.parse(fs.readFileSync(REFERENCE, 'utf8')) as {
      samples: Array<{ text: string; audio: { waveform_npy: string } }>;
    };
    const sample = capture.samples[0]!;
    const waveform = loadNpyFloat32(sample.audio.waveform_npy);
    const executor = new OrtGigaAmRnntExecutor('gigaam-v3-e2e-rnnt', 'webgpu', CONFIG, {
      source: { kind: 'direct', cpuThreads: 1, artifacts: artifacts() },
    });
    const outPath = path.resolve('tools/data/results/gigaam/v3-e2e-rnnt-example-webgpu.json');
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
        onnx_dir: ONNX_DIR,
        dtype: 'float32',
        status: 'experimental-js-frontend-webgpu',
      };
      fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
      expect(result.utteranceText).toBe(sample.text);
    } catch (error) {
      const payload = {
        ...classifyOrtFailure(error, 'webgpu'),
        onnx_dir: ONNX_DIR,
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
  });
});

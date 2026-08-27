import * as fs from 'node:fs';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';
import { describe, expect, it } from 'vitest';

import { OrtQwen3AsrExecutor, parseOfficialQwen3AsrConfig } from '../src/models/qwen-asr/index.js';
import {
  officialQwen3AsrEncoderFilename,
  parseOfficialQwen3AsrEncoderVariant,
} from '../src/models/qwen-asr/official.js';

const ONNX_DIR = 'N:/models/onnx/qwen3-asr-0.6b-official';
const TOKENIZER = path.join(ONNX_DIR, 'tokenizer/tokenizer.json');
const WAVEFORM = 'N:/models/gigaam/multilingual-ctc/captures/jfk-short.waveform.npy';
const ENABLED = process.env.QWEN_OFFICIAL_ONNX_SMOKE === '1';
const EXPECTED =
  'And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.';

function firstExisting(...files: string[]): string | undefined {
  return files.find((file) => fs.existsSync(file));
}

function fileMeta(filePath: string | undefined): { path?: string; size_bytes?: number } {
  if (!filePath || !fs.existsSync(filePath)) return {};
  return { path: filePath, size_bytes: fs.statSync(filePath).size };
}

const FP32_PREFILL = path.join(ONNX_DIR, 'decoder-prefill.onnx');
const FP32_STEP = path.join(ONNX_DIR, 'decoder-step.onnx');
const FP32_PREFILL_DATA = path.join(ONNX_DIR, 'decoder-prefill.onnx.data');
const FP32_STEP_DATA = path.join(ONNX_DIR, 'decoder-step.onnx.data');

type SmokeArtifacts = {
  readonly dtype: 'float16' | 'float32';
  readonly encoder: string;
  readonly prefill: string;
  readonly step: string;
  readonly prefillData: string;
  readonly stepData: string;
};

function encoderCandidates(): string[] {
  const requested = parseOfficialQwen3AsrEncoderVariant(process.env.QWEN_OFFICIAL_ENCODER);
  if (requested === 'static-t1100') {
    return [
      path.join(ONNX_DIR, 'audio-encoder-static-t1100-fp16.onnx'),
      path.join(ONNX_DIR, officialQwen3AsrEncoderFilename('static-t1100')),
    ];
  }
  return [path.join(ONNX_DIR, officialQwen3AsrEncoderFilename('dynamic'))];
}

function fp16Artifacts(): SmokeArtifacts | undefined {
  const prefill = path.join(ONNX_DIR, 'decoder-prefill-fp16.onnx');
  const step = path.join(ONNX_DIR, 'decoder-step-fp16.onnx');
  const data = firstExisting(
    path.join(ONNX_DIR, 'decoder-fp16.onnx.data'),
    path.join(ONNX_DIR, 'decoder-prefill-fp16.onnx.data'),
  );
  const encoder = firstExisting(...encoderCandidates());
  if (!encoder || !fs.existsSync(prefill) || !fs.existsSync(step) || !data) return undefined;
  return {
    dtype: 'float16',
    encoder,
    prefill,
    step,
    prefillData: data,
    stepData: firstExisting(path.join(ONNX_DIR, 'decoder-fp16.onnx.data'), path.join(ONNX_DIR, 'decoder-step-fp16.onnx.data')) ?? data,
  };
}

function fp32Artifacts(): SmokeArtifacts | undefined {
  const encoder = firstExisting(...encoderCandidates());
  if (!encoder || ![FP32_PREFILL, FP32_STEP, FP32_PREFILL_DATA, FP32_STEP_DATA].every((file) => fs.existsSync(file))) {
    return undefined;
  }
  return {
    dtype: 'float32',
    encoder,
    prefill: FP32_PREFILL,
    step: FP32_STEP,
    prefillData: FP32_PREFILL_DATA,
    stepData: FP32_STEP_DATA,
  };
}

const PRIMARY = fp16Artifacts() ?? fp32Artifacts();
const T1050_SAMPLES = 1050 * 160;

const CONFIG = parseOfficialQwen3AsrConfig();

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
  } else if (
    /not.*implement|unsupported.*op|failed to find kernel|type error|does not match expected type|does not exist/i.test(
      lowered,
    )
  ) {
    failureClass = 'ORT_WEB_UNSUPPORTED_OP';
  } else if (/bad_alloc|memory|oom|out of memory|array buffer allocation|too big|cannot allocate/i.test(lowered)) {
    failureClass = backend === 'webgpu' ? 'WEBGPU_MEMORY_LIMIT' : 'WASM_MEMORY_LIMIT';
  } else if (/too large|model_too_large|file size/i.test(lowered)) {
    failureClass = 'MODEL_TOO_LARGE';
  }
  return { backend, failureClass, message };
}

function rssMb(): number | undefined {
  const rss = process.memoryUsage().rss;
  return Math.round(rss / (1024 * 1024));
}

function createExecutor(backend: 'wasm' | 'webgpu', artifacts: SmokeArtifacts): OrtQwen3AsrExecutor {
  const encoderData = firstExisting(`${artifacts.encoder}.data`);
  return new OrtQwen3AsrExecutor('Qwen/Qwen3-ASR-0.6B', CONFIG, backend, {
    source: {
      kind: 'direct',
      cpuThreads: 1,
      cacheOutputLocation: 'cpu',
      artifacts: {
        encoderUrl: pathToFileURL(artifacts.encoder).href,
        decoderUrl: pathToFileURL(artifacts.prefill).href,
        decoderStepUrl: pathToFileURL(artifacts.step).href,
        tokenizerUrl: pathToFileURL(TOKENIZER).href,
        encoderDataUrl: encoderData ? pathToFileURL(encoderData).href : undefined,
        encoderDataPath: encoderData ? path.basename(encoderData) : undefined,
        decoderPrefillDataUrl: pathToFileURL(artifacts.prefillData).href,
        decoderPrefillDataPath: path.basename(artifacts.prefillData),
        decoderStepDataUrl: pathToFileURL(artifacts.stepData).href,
        decoderStepDataPath: path.basename(artifacts.stepData),
      },
    },
  });
}

function artifactPayload(artifacts: SmokeArtifacts) {
  return {
    dtype: artifacts.dtype,
    encoder: fileMeta(artifacts.encoder),
    prefill: fileMeta(artifacts.prefill),
    step: fileMeta(artifacts.step),
    prefill_data: fileMeta(artifacts.prefillData),
    step_data: fileMeta(artifacts.stepData),
  };
}

async function transcribeWasm(artifacts: SmokeArtifacts, waveform: Float32Array) {
  const executor = createExecutor('wasm', artifacts);
  let stage = 'init';
  try {
    await executor.ready();
    stage = 'ready';
    const result = await executor.transcribe(
      {
        sampleRate: 16000,
        numberOfChannels: 1,
        numberOfFrames: waveform.length,
        durationSeconds: waveform.length / 16000,
        channels: [waveform],
      },
      { maxNewTokens: 64 },
      {
        modelId: 'Qwen/Qwen3-ASR-0.6B',
        classification: { ecosystem: 'qwen', task: 'multilingual-asr' },
        config: CONFIG,
      },
    );
    return { result, stage, error: undefined as unknown };
  } catch (error) {
    return { result: undefined, stage, error };
  } finally {
    await executor.dispose();
  }
}

describe.skipIf(
  !ENABLED ||
    !PRIMARY ||
    !fs.existsSync(TOKENIZER) ||
    !fs.existsSync(WAVEFORM),
)('Qwen3-ASR official explicit-KV wasm/webgpu smoke', () => {
  it(
    'transcribes jfk-short T=1050 through the default dynamic encoder on WASM',
    { timeout: 900_000 },
    async () => {
      const full = loadNpyFloat32(WAVEFORM);
      const waveform = full.subarray(0, Math.min(full.length, T1050_SAMPLES));
      const outPath = path.resolve('tools/data/results/qwen/qwen3-asr-0.6b-jfk-short-wasm.json');
      const attempts: SmokeArtifacts[] = [];
      const fp16 = fp16Artifacts();
      const fp32 = fp32Artifacts();
      if (fp16) attempts.push(fp16);
      if (fp32 && fp32.prefill !== fp16?.prefill) attempts.push(fp32);
      let lastFailure: Record<string, unknown> | undefined;
      fs.mkdirSync(path.dirname(outPath), { recursive: true });
      for (const artifacts of attempts) {
        const { result, stage, error } = await transcribeWasm(artifacts, waveform);
        if (result) {
          const match = result.utteranceText === EXPECTED;
          fs.writeFileSync(outPath, `${JSON.stringify({
            backend: 'wasm',
            engine: 'onnxruntime-web',
            sequential_sessions: true,
            encoder_variant: path.basename(artifacts.encoder),
            input_samples: waveform.length,
            input_frames: Math.floor(waveform.length / 160),
            padded_frames: result.metrics?.encoderFrameCount,
            artifacts: artifactPayload(artifacts),
            fp16_wasm_fallback: artifacts.dtype === 'float32' && Boolean(fp16),
            rss_mb: rssMb(),
            text: result.utteranceText,
            language: result.language,
            expected: EXPECTED,
            text_match: match,
            failureClass: match ? null : 'PREPROCESSING_MISMATCH',
            metrics: result.metrics,
            prior_failure: lastFailure,
            status: 'experimental',
          }, null, 2)}\n`);
          expect(path.basename(artifacts.encoder)).toContain(
            process.env.QWEN_OFFICIAL_ENCODER === 'static-t1100' ? 'static-t1100' : 'dynamic',
          );
          expect(waveform.length).toBe(T1050_SAMPLES);
          expect(result.utteranceText).toBe(EXPECTED);
          return;
        }
        lastFailure = {
          ...classifyOrtFailure(error, 'wasm'),
          stage,
          encoder_only_wasm: stage === 'ready',
          artifacts: artifactPayload(artifacts),
          rss_mb: rssMb(),
        };
      }
      fs.writeFileSync(outPath, `${JSON.stringify({
        ...lastFailure,
        sequential_sessions: true,
        status: 'experimental-blocked',
      }, null, 2)}\n`);
      if (
        lastFailure?.failureClass !== 'WASM_MEMORY_LIMIT'
        && lastFailure?.failureClass !== 'MODEL_TOO_LARGE'
        && lastFailure?.failureClass !== 'ORT_WEB_UNSUPPORTED_OP'
      ) {
        throw new Error(String(lastFailure?.message ?? 'Qwen WASM smoke failed'));
      }
    },
  );

  it(
    'records Node WebGPU result or failure class',
    { timeout: 900_000 },
    async () => {
      const waveform = loadNpyFloat32(WAVEFORM);
      const artifacts = PRIMARY;
      if (!artifacts) return;
      const executor = createExecutor('webgpu', artifacts);
      const outPath = path.resolve('tools/data/results/qwen/qwen3-asr-0.6b-jfk-short-webgpu.json');
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
          { maxNewTokens: 64 },
          {
            modelId: 'Qwen/Qwen3-ASR-0.6B',
            classification: { ecosystem: 'qwen', task: 'multilingual-asr' },
            config: CONFIG,
          },
        );
        const match = result.utteranceText === EXPECTED;
        fs.writeFileSync(outPath, `${JSON.stringify({
          backend: 'webgpu',
          engine: 'onnxruntime-web',
          artifacts: artifactPayload(artifacts),
          text: result.utteranceText,
          language: result.language,
          expected: EXPECTED,
          text_match: match,
          failureClass: match ? null : 'PREPROCESSING_MISMATCH',
          metrics: result.metrics,
          status: 'experimental',
        }, null, 2)}\n`);
        expect(result.utteranceText).toBe(EXPECTED);
      } catch (error) {
        const payload = {
          ...classifyOrtFailure(error, 'webgpu'),
          status: 'experimental-blocked',
        };
        fs.writeFileSync(outPath, `${JSON.stringify(payload, null, 2)}\n`);
        if (payload.failureClass !== 'WEBGPU_NO_ADAPTER' && payload.failureClass !== 'WEBGPU_MEMORY_LIMIT') {
          throw error;
        }
      } finally {
        await executor.dispose();
      }
    },
  );
});

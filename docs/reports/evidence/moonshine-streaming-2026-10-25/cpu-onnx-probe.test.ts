import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import * as ort from 'onnxruntime-node';
import { describe, it } from 'vitest';
import { MoonshineStreamingSession } from '../../../../src/models/moonshine/streaming-session.js';
import { MoonshineTokenizer } from '../../../../src/models/moonshine/session.js';

const project = resolve('speech-recognition');
const graphDir = resolve('demos/tmp/real-e2e/remaining-models-20261024/native-moonshine-streaming-tiny');
const snapshotDir = graphDir;
const captureDir = resolve('demos/tmp/real-e2e/remaining-models-20261025/moonshine-streaming-v17-mic-retest/webgpu/realtime/mic-pcm-capture');
const officialPath = resolve(project, 'tmp/moonshine-vad-native-cpu-2026-10-25.json');
const outputPath = process.env.MOONSHINE_PROBE_OUTPUT;
if (!outputPath) throw new Error('Set MOONSHINE_PROBE_OUTPUT to a fresh receipt path.');
const cache = new Map();
async function graph(name) {
  let promise = cache.get(name);
  if (!promise) {
    promise = ort.InferenceSession.create(resolve(graphDir, name), { executionProviders: ['cpu'] });
    cache.set(name, promise);
  }
  return promise;
}
function floatSamples(path) {
  const bytes = readFileSync(path);
  const owned = Uint8Array.from(bytes).buffer;
  return new Float32Array(owned);
}
function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}
function makeSession(tokenizer, trace) {
  return new MoonshineStreamingSession({
    ort,
    chunkedEncoder: { run: async (feeds) => (await graph('encoder_chunked_model.onnx')).run(feeds) },
    wholeEncoder: { run: async (feeds) => {
      const input = feeds.input_values;
      const mask = feeds.attention_mask;
      trace.inputDims = input.dims;
      trace.maskDims = mask.dims;
      trace.maskCount = Number(mask.data.reduce((sum, value) => sum + value, 0));
      trace.actualTail = Array.from(input.data).slice(0, trace.maskCount);
      const out = await (await graph('encoder_model.onnx')).run(feeds);
      trace.wholeOutputDims = out.last_hidden_state.dims;
      return out;
    } },
    prefill: { run: async (feeds) => {
      const out = await (await graph('decoder_prefill_model.onnx')).run(feeds);
      trace.decoder.push({
        stage: 'prefill',
        argmax: argmax(out.logits.data),
        kv: tensorSummary(out.present_k_0),
      });
      return out;
    } },
    step: { run: async (feeds) => {
      const out = await (await graph('decoder_step_model.onnx')).run(feeds);
      trace.decoder.push({
        stage: 'step',
        inputToken: Number(feeds.input_ids.data[0]),
        argmax: argmax(out.logits.data),
        kv: tensorSummary(out.present_k_0),
      });
      return out;
    } },
    tokenizer,
    bosTokenId: 1,
    eosTokenId: 2,
    emitFrames: 256,
  });
}
function argmax(data) {
  let best = -Infinity;
  let index = -1;
  for (let i = 0; i < data.length; i += 1) if (data[i] > best) { best = data[i]; index = i; }
  return index;
}
function tensorSummary(tensor) {
  return { type: tensor.type, dataType: tensor.data.constructor.name, dims: tensor.dims };
}

describe('temporary Moonshine VAD CPU ORT probe', () => {
  it('compares exact captured VAD segments with the official whole-utterance receipt', async () => {
    const official = JSON.parse(readFileSync(officialPath, 'utf8'));
    const metas = readdirSync(captureDir).filter((name) => name.endsWith('.json')).map((name) => {
      const path = resolve(captureDir, name);
      return { ...JSON.parse(readFileSync(path, 'utf8')), metadataPath: path };
    });
    const tokenizerJson = JSON.parse(readFileSync(resolve(snapshotDir, 'tokenizer.json'), 'utf8'));
    const tokenizer = MoonshineTokenizer.fromTokenizerJson(tokenizerJson, 0);
    const outputFlows = [];
    for (const flow of official.flows) {
      const segments = [];
      for (const expected of flow.segments) {
        const meta = metas.find((candidate) => candidate.pcmSha256 === expected.pcmSha256);
        if (!meta) throw new Error(`No captured PCM metadata for ${flow.flow} request ${expected.requestId} sha ${expected.pcmSha256}`);
        if (meta.frames !== expected.frames) throw new Error(`Frame count mismatch for ${flow.flow} request ${expected.requestId}`);
        const pcmBytes = readFileSync(meta.pcmPath);
        const actualPcmSha256 = sha256(pcmBytes);
        if (actualPcmSha256 !== expected.pcmSha256 || actualPcmSha256 !== meta.pcmSha256) {
          throw new Error(`Captured PCM SHA-256 mismatch for ${flow.flow} request ${expected.requestId}: ${actualPcmSha256}`);
        }
        const pcm = floatSamples(meta.pcmPath);
        const trace = { decoder: [] };
        const session = makeSession(tokenizer, trace);
        const start = performance.now();
        await session.pushAudio(pcm);
        const committedSamples = session.committedEncoderFrames * 4 * 80;
        const result = await session.finalize();
        const expectedTail = Array.from(pcm.subarray(Math.max(0, committedSamples - 480 * 80)));
        const tailMatchesOriginalAudio = trace.actualTail?.length === expectedTail.length
          && trace.actualTail.every((sample, index) => sample === expectedTail[index]);
        segments.push({
          requestId: expected.requestId,
          pcmSha256: actualPcmSha256,
          frames: pcm.length,
          officialText: expected.native.text,
          streamingOnnxText: result.text,
          officialTokenIds: expected.native.tokens,
          streamingOnnxTokenIds: result.tokenIds,
          streamingOnnxTokenIdsWithSpecialTokens: [1, ...result.tokenIds, 2],
          textEqual: result.text === expected.native.text,
          tokenIdsEqualAfterExplicitBosEosNormalization: JSON.stringify([1, ...result.tokenIds, 2]) === JSON.stringify(expected.native.tokens),
          tailSamples: trace.actualTail?.length,
          expectedTailSamples: expectedTail.length,
          tailMatchesOriginalAudio,
          decoderTrace: expected.requestId === 9 ? trace.decoder : undefined,
          wholeOutputDims: trace.wholeOutputDims,
          inputDims: trace.inputDims,
          maskDims: trace.maskDims,
          maskCount: trace.maskCount,
          encoderFrames: result.encoderFrames,
          seconds: Number(((performance.now() - start) / 1000).toFixed(3)),
        });
      }
      const joined = segments.map((segment) => segment.streamingOnnxText.trim()).filter(Boolean).join(' ');
      outputFlows.push({
        flow: flow.flow,
        browserText: flow.browserTranscript,
        officialJoinedText: flow.nativeJoinedInRequestOrder,
        streamingOnnxJoinedText: joined,
        officialJoinedEqualsStreamingOnnx: joined === flow.nativeJoinedInRequestOrder,
        segments,
      });
    }
    const output = {
      status: 'CPU_ONNX_STREAMING_EXECUTION_COMPLETED_NOT_PARITY_PASS',
      scope: 'Exact graph folder staged by the browser v17 run; fp32 four-graph ONNX through MoonshineStreamingSession on byte-hash-verified exact captured PCM; CPU ORT only. No browser/WebGPU claim.',
      graphDir,
      graphSha256: Object.fromEntries(['encoder_model.onnx','encoder_chunked_model.onnx','decoder_prefill_model.onnx','decoder_step_model.onnx'].map((name) => [name, sha256(readFileSync(resolve(graphDir, name)))])),
      captureDir,
      officialReceipt: officialPath,
      tokenizer: resolve(snapshotDir, 'tokenizer.json'),
      flows: outputFlows,
    };
    mkdirSync(resolve(project, 'tmp'), { recursive: true });
    writeFileSync(outputPath, JSON.stringify(output, null, 2) + '\n');
    console.log(JSON.stringify({ outputPath, flows: outputFlows.map((flow) => ({ flow: flow.flow, officialJoinedEqualsStreamingOnnx: flow.officialJoinedEqualsStreamingOnnx, segments: flow.segments.map(({ requestId, frames, officialText, streamingOnnxText, textEqual, tokenIdsEqualAfterExplicitBosEosNormalization, tailSamples, expectedTailSamples, tailMatchesOriginalAudio }) => ({ requestId, frames, officialText, streamingOnnxText, textEqual, tokenIdsEqualAfterExplicitBosEosNormalization, tailSamples, expectedTailSamples, tailMatchesOriginalAudio })) })) }, null, 2));
  }, 900000);
});

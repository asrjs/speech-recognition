import * as ort from 'onnxruntime-web';
import type { FireRedRuntimeOptions } from '../types.js';
import {
  DEFAULT_MODEL_URLS,
  STREAM_CACHE_BATCH,
  STREAM_CACHE_LAYERS,
  STREAM_CACHE_LEN,
  STREAM_CACHE_PROJ,
} from './constants.js';
import { FireRedRuntimeError } from './errors.js';
import { loadBinaryResource } from './loader.js';
import { getDefaultAssetCache } from './asset-cache.js';

export interface BackendRunResult {
  readonly probs: Float32Array;
}

export interface StreamBackendRunResult extends BackendRunResult {
  readonly caches: Float32Array;
}

export interface FireRedBackend {
  runStream(feat: Float32Array, caches: Float32Array): Promise<StreamBackendRunResult>;
  runVad(feat: Float32Array): Promise<BackendRunResult>;
  runAed(feat: Float32Array): Promise<BackendRunResult>;
  dispose(): Promise<void>;
}

function resolveViteDevOrtWasmPath(): string | null {
  if (typeof window === 'undefined') {
    return null;
  }
  const marker = '/@fs/';
  const metaUrl = import.meta.url;
  const markerIndex = metaUrl.indexOf(marker);
  if (markerIndex < 0) {
    return null;
  }
  const afterMarker = metaUrl.slice(markerIndex + marker.length).split('?')[0]?.split('#')[0] ?? '';
  const decodedPath = decodeURIComponent(afterMarker);
  const sourceSuffix = '/src/runtime/firered-vad/core/backend.ts';
  if (!decodedPath.endsWith(sourceSuffix)) {
    return null;
  }
  const repoRoot = decodedPath.slice(0, -sourceSuffix.length);
  const ortDist = `${repoRoot}/node_modules/onnxruntime-web/dist/`;
  return `/@fs/${encodeURI(ortDist)}`;
}

function configureOrtEnvironment(options: FireRedRuntimeOptions): void {
  ort.env.wasm.numThreads = options.wasmNumThreads ?? 1;
  if (options.wasmPaths) {
    ort.env.wasm.wasmPaths = options.wasmPaths;
  } else if (!ort.env.wasm.wasmPaths && typeof window !== 'undefined') {
    const localVitePath = resolveViteDevOrtWasmPath();
    ort.env.wasm.wasmPaths =
      localVitePath ??
      `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions?.common ?? '1.24.1'}/dist/`;
  }
}

function getModelUrls(options: FireRedRuntimeOptions) {
  return {
    vadUrl: options.modelUrls?.vadUrl ?? DEFAULT_MODEL_URLS.vadUrl,
    streamVadWithCacheUrl:
      options.modelUrls?.streamVadWithCacheUrl ?? DEFAULT_MODEL_URLS.streamVadWithCacheUrl,
    aedUrl: options.modelUrls?.aedUrl ?? DEFAULT_MODEL_URLS.aedUrl,
  };
}

function flattenFeatures(frames: Float32Array[]): Float32Array {
  const merged = new Float32Array(frames.length * (frames[0]?.length ?? 0));
  let offset = 0;
  for (const frame of frames) {
    merged.set(frame, offset);
    offset += frame.length;
  }
  return merged;
}

function ensureFloat32Array(value: unknown, outputName: string): Float32Array {
  if (!(value instanceof ort.Tensor)) {
    throw new FireRedRuntimeError(`ONNX output "${outputName}" is missing.`);
  }
  if (!(value.data instanceof Float32Array)) {
    throw new FireRedRuntimeError(`ONNX output "${outputName}" is not Float32Array.`);
  }
  return value.data;
}

class OrtFireRedBackend implements FireRedBackend {
  private readonly streamSession: ort.InferenceSession;
  private readonly vadSession: ort.InferenceSession;
  private readonly aedSession: ort.InferenceSession;

  constructor(
    streamSession: ort.InferenceSession,
    vadSession: ort.InferenceSession,
    aedSession: ort.InferenceSession,
  ) {
    this.streamSession = streamSession;
    this.vadSession = vadSession;
    this.aedSession = aedSession;
  }

  async runStream(feat: Float32Array, caches: Float32Array): Promise<StreamBackendRunResult> {
    const time = Math.floor(feat.length / 80);
    const feed: Record<string, ort.Tensor> = {
      feat: new ort.Tensor('float32', feat, [1, time, 80]),
      caches_in: new ort.Tensor('float32', caches, [
        STREAM_CACHE_LAYERS,
        STREAM_CACHE_BATCH,
        STREAM_CACHE_PROJ,
        STREAM_CACHE_LEN,
      ]),
    };
    const outputs = await this.streamSession.run(feed);
    const probs = ensureFloat32Array(outputs.probs, 'probs');
    const nextCaches = ensureFloat32Array(outputs.caches_out, 'caches_out');
    return {
      probs,
      caches: nextCaches,
    };
  }

  async runVad(feat: Float32Array): Promise<BackendRunResult> {
    const time = Math.floor(feat.length / 80);
    const feed = { feat: new ort.Tensor('float32', feat, [1, time, 80]) };
    const outputs = await this.vadSession.run(feed);
    return { probs: ensureFloat32Array(outputs.probs, 'probs') };
  }

  async runAed(feat: Float32Array): Promise<BackendRunResult> {
    const time = Math.floor(feat.length / 80);
    const feed = { feat: new ort.Tensor('float32', feat, [1, time, 80]) };
    const outputs = await this.aedSession.run(feed);
    return { probs: ensureFloat32Array(outputs.probs, 'probs') };
  }

  async dispose(): Promise<void> {
    await Promise.all([this.streamSession.release(), this.vadSession.release(), this.aedSession.release()]);
  }
}

export async function createOrtFireRedBackend(options: FireRedRuntimeOptions = {}): Promise<FireRedBackend> {
  configureOrtEnvironment(options);
  const modelUrls = getModelUrls(options);
  const cache = options.cacheAssets === false ? undefined : getDefaultAssetCache(true);

  const [streamModel, vadModel, aedModel] = await Promise.all([
    loadBinaryResource(modelUrls.streamVadWithCacheUrl, cache),
    loadBinaryResource(modelUrls.vadUrl, cache),
    loadBinaryResource(modelUrls.aedUrl, cache),
  ]);

  const sessionOptions: ort.InferenceSession.SessionOptions = {
    executionProviders: ['wasm'],
  };
  const [streamSession, vadSession, aedSession] = await Promise.all([
    ort.InferenceSession.create(streamModel, sessionOptions),
    ort.InferenceSession.create(vadModel, sessionOptions),
    ort.InferenceSession.create(aedModel, sessionOptions),
  ]);

  return new OrtFireRedBackend(streamSession, vadSession, aedSession);
}

export function flattenFeatFrames(frames: Float32Array[]): Float32Array {
  return flattenFeatures(frames);
}

export function createZeroStreamCache(): Float32Array {
  return new Float32Array(STREAM_CACHE_LAYERS * STREAM_CACHE_BATCH * STREAM_CACHE_PROJ * STREAM_CACHE_LEN);
}

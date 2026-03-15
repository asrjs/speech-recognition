import { normalizePcmInput, type PcmAudioBuffer } from '../audio/index.js';
import type { AudioInputLike } from '../types/index.js';

export interface AudioCacheEntry<TValue> {
  readonly key: string;
  readonly value: TValue;
  readonly sizeBytes: number;
  readonly sampleRate: number;
  readonly durationSeconds: number;
  readonly createdAt: number;
  readonly lastAccessedAt: number;
}

export interface AudioCacheStats {
  readonly entries: number;
  readonly sizeBytes: number;
  readonly maxSizeBytes: number;
  readonly hitCount: number;
  readonly missCount: number;
}

export interface AudioFeatureCacheOptions<TValue> {
  readonly maxSizeMB?: number;
  readonly estimateSizeBytes?: (value: TValue) => number;
  readonly keyResolver?: (audio: PcmAudioBuffer) => string;
}

function defaultEstimateSizeBytes(value: unknown): number {
  if (ArrayBuffer.isView(value)) {
    return value.byteLength;
  }

  if (typeof value === 'string') {
    return value.length * 2;
  }

  return 1024;
}

export function createAudioCacheKey(input: AudioInputLike): string {
  const audio = normalizePcmInput(input).toMono();
  const samples = audio.channels[0] ?? new Float32Array(0);
  const count = samples.length;
  if (count === 0) {
    return '0_0';
  }

  const fullHashThreshold = 1_000_000;
  const edgeSampleCount = 1024;
  const targetSampledPoints = 131072;

  const mix32 = (state: number, sample: number): number => {
    const quantized = Math.max(
      -32768,
      Math.min(32767, Math.round((Number.isFinite(sample) ? sample : 0) * 32768)),
    );
    let key = quantized & 0xffff;
    key = Math.imul(key, 0xcc9e2d51);
    key = (key << 15) | (key >>> 17);
    key = Math.imul(key, 0x1b873593);
    state ^= key;
    state = (state << 13) | (state >>> 19);
    return (Math.imul(state, 5) + 0xe6546b64) | 0;
  };

  const fmix32 = (state: number): number => {
    state ^= state >>> 16;
    state = Math.imul(state, 0x85ebca6b);
    state ^= state >>> 13;
    state = Math.imul(state, 0xc2b2ae35);
    state ^= state >>> 16;
    return state >>> 0;
  };

  let state = count | 0;
  if (count <= fullHashThreshold) {
    for (let index = 0; index < count; index += 1) {
      state = mix32(state, samples[index] ?? 0);
    }
  } else {
    const edgeCount = Math.min(edgeSampleCount, Math.floor(count / 2));
    for (let index = 0; index < edgeCount; index += 1) {
      state = mix32(state, samples[index] ?? 0);
    }

    const middleStart = edgeCount;
    const middleEnd = count - edgeCount;
    const middleLength = Math.max(0, middleEnd - middleStart);
    const remainingBudget = Math.max(1, targetSampledPoints - edgeCount * 2);
    const stride = Math.max(1, Math.floor(middleLength / remainingBudget));

    for (let index = middleStart; index < middleEnd; index += stride) {
      state = mix32(state, samples[index] ?? 0);
    }

    for (let index = Math.max(edgeCount, count - edgeCount); index < count; index += 1) {
      state = mix32(state, samples[index] ?? 0);
    }
  }

  return `${count}_${fmix32(state)}`;
}

export class AudioFeatureCache<TValue> {
  private readonly entries = new Map<string, AudioCacheEntry<TValue>>();
  private readonly maxSizeBytes: number;
  private readonly estimateSizeBytes: (value: TValue) => number;
  private readonly keyResolver: (audio: PcmAudioBuffer) => string;
  private totalSizeBytes = 0;
  private hitCount = 0;
  private missCount = 0;

  constructor(options: AudioFeatureCacheOptions<TValue> = {}) {
    const maxSizeMB = options.maxSizeMB ?? 50;
    if (!Number.isFinite(maxSizeMB) || maxSizeMB <= 0) {
      throw new Error('AudioFeatureCache expected `maxSizeMB` to be a positive number.');
    }

    this.maxSizeBytes = Math.floor(maxSizeMB * 1024 * 1024);
    this.estimateSizeBytes =
      options.estimateSizeBytes ?? ((value) => defaultEstimateSizeBytes(value));
    this.keyResolver = options.keyResolver ?? ((audio) => createAudioCacheKey(audio));
  }

  has(input: AudioInputLike | string): boolean {
    const key =
      typeof input === 'string' ? input : this.keyResolver(normalizePcmInput(input).toMono());
    return this.entries.has(key);
  }

  get(input: AudioInputLike | string): TValue | undefined {
    const key =
      typeof input === 'string' ? input : this.keyResolver(normalizePcmInput(input).toMono());
    const entry = this.entries.get(key);
    if (!entry) {
      this.missCount += 1;
      return undefined;
    }

    this.hitCount += 1;
    this.entries.delete(key);
    this.entries.set(key, {
      ...entry,
      lastAccessedAt: Date.now(),
    });
    return entry.value;
  }

  set(input: AudioInputLike | string, value: TValue, audioMeta?: AudioInputLike): string {
    const audio =
      typeof input === 'string'
        ? normalizePcmInput(audioMeta ?? new Float32Array(0)).toMono()
        : normalizePcmInput(input).toMono();
    const key = typeof input === 'string' ? input : this.keyResolver(audio);
    const sizeBytes = this.estimateSizeBytes(value);
    if (!Number.isFinite(sizeBytes) || sizeBytes <= 0) {
      throw new Error('AudioFeatureCache size estimator must return a positive finite byte size.');
    }

    if (sizeBytes > this.maxSizeBytes) {
      return key;
    }

    const existing = this.entries.get(key);
    if (existing) {
      this.totalSizeBytes -= existing.sizeBytes;
      this.entries.delete(key);
    }

    while (this.totalSizeBytes + sizeBytes > this.maxSizeBytes && this.entries.size > 0) {
      this.evictOldest();
    }

    const timestamp = Date.now();
    this.entries.set(key, {
      key,
      value,
      sizeBytes,
      sampleRate: audio.sampleRate,
      durationSeconds: audio.durationSeconds,
      createdAt: timestamp,
      lastAccessedAt: timestamp,
    });
    this.totalSizeBytes += sizeBytes;
    return key;
  }

  async getOrCreate(
    input: AudioInputLike,
    createValue: (audio: PcmAudioBuffer, key: string) => Promise<TValue> | TValue,
  ): Promise<{ value: TValue; key: string; cached: boolean }> {
    const audio = normalizePcmInput(input).toMono();
    const key = this.keyResolver(audio);
    const cached = this.get(key);
    if (cached !== undefined) {
      return { value: cached, key, cached: true };
    }

    const value = await createValue(audio, key);
    this.set(key, value, audio);
    return { value, key, cached: false };
  }

  delete(input: AudioInputLike | string): boolean {
    const key =
      typeof input === 'string' ? input : this.keyResolver(normalizePcmInput(input).toMono());
    const entry = this.entries.get(key);
    if (!entry) {
      return false;
    }

    this.entries.delete(key);
    this.totalSizeBytes -= entry.sizeBytes;
    return true;
  }

  clear(): void {
    this.entries.clear();
    this.totalSizeBytes = 0;
    this.hitCount = 0;
    this.missCount = 0;
  }

  getStats(): AudioCacheStats {
    return {
      entries: this.entries.size,
      sizeBytes: this.totalSizeBytes,
      maxSizeBytes: this.maxSizeBytes,
      hitCount: this.hitCount,
      missCount: this.missCount,
    };
  }

  private evictOldest(): void {
    let oldestKey: string | undefined;
    let oldestTimestamp = Number.POSITIVE_INFINITY;

    for (const [key, entry] of this.entries) {
      if (entry.lastAccessedAt < oldestTimestamp) {
        oldestTimestamp = entry.lastAccessedAt;
        oldestKey = key;
      }
    }

    if (!oldestKey) {
      return;
    }

    const entry = this.entries.get(oldestKey);
    if (!entry) {
      return;
    }

    this.entries.delete(oldestKey);
    this.totalSizeBytes -= entry.sizeBytes;
  }
}

export class IncrementalCache<TValue> extends AudioFeatureCache<TValue> {}

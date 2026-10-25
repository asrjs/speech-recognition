import { decodeWavToMonoPcm } from '../audio/wav.js';
import {
  rethrowIfAssetAborted,
  subscribeToAbortSignal,
  throwIfAssetAborted,
  toFetchAbortSignal,
} from '../io/abort.js';
import type { AbortSignalLike } from '../types/index.js';

export interface AudioBufferLikeForDecode {
  readonly numberOfChannels: number;
  readonly length: number;
  readonly sampleRate: number;
  getChannelData(channel: number): Float32Array;
}

export interface AudioContextLikeForDecode {
  decodeAudioData(buffer: ArrayBuffer): Promise<AudioBufferLikeForDecode>;
  close?: () => Promise<void> | void;
}

export type BrowserAudioDecodeStrategy = 'audiocontext-target-rate' | 'audiocontext-native-rate';

export type BrowserAudioResampler = 'browser-audiocontext' | 'linear-parity' | 'none';

export type BrowserAudioResamplerQuality = 'linear' | null;

export interface DecodedMonoAudio {
  readonly pcm: Float32Array;
  readonly sampleRate: number;
  readonly durationSec: number;
  readonly numberOfChannels: number;
  readonly metrics: AudioPreparationMetrics;
}

export interface AudioPreparationMetrics {
  readonly backend: 'browser';
  readonly strategy: BrowserAudioDecodeStrategy;
  readonly inputSampleRate?: number;
  readonly outputSampleRate: number;
  readonly decodeMs: number;
  readonly downmixMs: number;
  readonly resampleMs: number;
  readonly totalMs: number;
  readonly wallMs: number;
  readonly resampler: BrowserAudioResampler;
  readonly resamplerQuality: BrowserAudioResamplerQuality;
  readonly audioDurationSec: number;
}

export interface DecodeAudioSourceOptions {
  readonly targetSampleRate?: number;
  readonly signal?: AbortSignalLike | null;
  readonly createAudioContext?: (sampleRate?: number) => AudioContextLikeForDecode;
  readonly fetchImpl?: typeof fetch;
  readonly strategy?: 'browser-target-rate' | 'native-rate';
  readonly resamplerQuality?: 'linear';
}

export interface EncodeMonoPcmToWavBlobOptions {
  readonly sampleRate?: number;
}

const WAV_HEADER_SIZE = 12;

function writeAscii(view: DataView, offset: number, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}

export function mixAudioBufferChannelsToMono(audioBuffer: AudioBufferLikeForDecode): Float32Array {
  const channelCount = Math.max(1, audioBuffer.numberOfChannels);
  if (channelCount === 1) {
    return new Float32Array(audioBuffer.getChannelData(0));
  }

  const mono = new Float32Array(audioBuffer.length);
  for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
    const channel = audioBuffer.getChannelData(channelIndex);
    for (let frameIndex = 0; frameIndex < audioBuffer.length; frameIndex += 1) {
      mono[frameIndex] = (mono[frameIndex] ?? 0) + (channel[frameIndex] ?? 0) / channelCount;
    }
  }
  return mono;
}

export async function resolveAudioSourceToBlob(
  source: string | Blob,
  options: Pick<DecodeAudioSourceOptions, 'fetchImpl' | 'signal'> = {},
): Promise<Blob> {
  if (typeof source !== 'string') {
    throwIfAssetAborted(options.signal, 'audio-source');
    return source;
  }

  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  if (typeof fetchImpl !== 'function') {
    throw new Error('resolveAudioSourceToBlob requires fetch when source is a URL.');
  }

  throwIfAssetAborted(options.signal, 'audio-fetch');
  const nativeSignal = toFetchAbortSignal(options.signal);
  const abortController =
    options.signal && !nativeSignal && typeof AbortController !== 'undefined'
      ? new AbortController()
      : undefined;
  const unsubscribeAbort = abortController
    ? subscribeToAbortSignal(options.signal, () => abortController.abort())
    : undefined;
  try {
    const response = await fetchImpl(source, {
      ...((nativeSignal ?? abortController?.signal)
        ? { signal: nativeSignal ?? abortController?.signal }
        : {}),
    });
    throwIfAssetAborted(options.signal, 'audio-fetch');
    if (!response.ok) {
      throw new Error(`Audio fetch failed: ${response.status}`);
    }
    const blob = await response.blob();
    throwIfAssetAborted(options.signal, 'audio-fetch');
    return blob;
  } catch (error) {
    rethrowIfAssetAborted(error, 'audio-fetch');
    throw error;
  } finally {
    unsubscribeAbort?.();
  }
}

export function encodeMonoPcmToWavBlob(
  pcm: Float32Array,
  options: EncodeMonoPcmToWavBlobOptions = {},
): Blob {
  const sampleRate = options.sampleRate ?? 16000;
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const dataSize = pcm.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeAscii(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, 'WAVE');
  writeAscii(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeAscii(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let index = 0; index < pcm.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, pcm[index] ?? 0));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function nowMs(): number {
  return globalThis?.performance?.now?.() ?? Date.now();
}

function readAscii(view: DataView, offset: number, length: number): string {
  let output = '';
  for (let index = 0; index < length; index += 1) {
    output += String.fromCharCode(view.getUint8(offset + index));
  }
  return output;
}

function isWavArrayBuffer(arrayBuffer: ArrayBuffer): boolean {
  if (arrayBuffer.byteLength < WAV_HEADER_SIZE) {
    return false;
  }
  const view = new DataView(arrayBuffer);
  return readAscii(view, 0, 4) === 'RIFF' && readAscii(view, 8, 4) === 'WAVE';
}

export function resampleLinear(audio: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) {
    return audio;
  }
  if (!Number.isFinite(fromRate) || !Number.isFinite(toRate) || fromRate <= 0 || toRate <= 0) {
    throw new Error(`Invalid resample rates: from=${fromRate}, to=${toRate}`);
  }

  const ratio = toRate / fromRate;
  const outputLength = Math.max(1, Math.round(audio.length * ratio));
  const output = new Float32Array(outputLength);
  const sourceScale = fromRate / toRate;

  for (let index = 0; index < outputLength; index += 1) {
    const sourcePosition = index * sourceScale;
    const leftIndex = Math.floor(sourcePosition);
    const rightIndex = Math.min(leftIndex + 1, audio.length - 1);
    const fraction = sourcePosition - leftIndex;
    output[index] = (audio[leftIndex] ?? 0) * (1 - fraction) + (audio[rightIndex] ?? 0) * fraction;
  }

  return output;
}

export async function decodeAudioSourceToMonoPcm(
  source: string | Blob,
  options: DecodeAudioSourceOptions = {},
): Promise<DecodedMonoAudio> {
  const startMs = nowMs();
  const targetSampleRate = options.targetSampleRate ?? 16000;
  throwIfAssetAborted(options.signal, 'audio-source');
  const blob = await resolveAudioSourceToBlob(source, options);
  const arrayBuffer = await blob.arrayBuffer();
  throwIfAssetAborted(options.signal, 'audio-read');
  const strategy = options.strategy ?? 'browser-target-rate';
  const resamplerQuality = options.resamplerQuality ?? 'linear';

  const createAudioContext =
    options.createAudioContext ??
    ((sampleRate?: number) => {
      if (typeof globalThis.AudioContext !== 'function') {
        throw new Error('decodeAudioSourceToMonoPcm requires AudioContext in this environment.');
      }
      return sampleRate
        ? new globalThis.AudioContext({ sampleRate })
        : new globalThis.AudioContext();
    });

  if (strategy === 'native-rate' && isWavArrayBuffer(arrayBuffer)) {
    const decodeStartMs = nowMs();
    const decoded = decodeWavToMonoPcm(arrayBuffer);
    throwIfAssetAborted(options.signal, 'audio-decode');
    const decodeMs = nowMs() - decodeStartMs;
    const resampleStartMs = nowMs();
    const pcm = resampleLinear(decoded.pcm, decoded.sampleRate, targetSampleRate);
    const resampleMs = nowMs() - resampleStartMs;
    const wallMs = nowMs() - startMs;
    const durationSec = decoded.pcm.length / decoded.sampleRate;

    return {
      pcm,
      sampleRate: targetSampleRate,
      durationSec,
      numberOfChannels: decoded.numberOfChannels,
      metrics: {
        backend: 'browser',
        strategy: 'audiocontext-native-rate',
        inputSampleRate: decoded.sampleRate,
        outputSampleRate: targetSampleRate,
        decodeMs,
        downmixMs: 0,
        resampleMs,
        totalMs: decodeMs + resampleMs,
        wallMs,
        resampler: decoded.sampleRate === targetSampleRate ? 'none' : 'linear-parity',
        resamplerQuality: decoded.sampleRate === targetSampleRate ? null : resamplerQuality,
        audioDurationSec: durationSec,
      },
    };
  }

  const audioContext =
    strategy === 'native-rate' ? createAudioContext() : createAudioContext(targetSampleRate);
  let primaryError: unknown;
  let failed = false;
  try {
    const decodeStartMs = nowMs();
    const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
    throwIfAssetAborted(options.signal, 'audio-decode');
    const decodeMs = nowMs() - decodeStartMs;
    const downmixStartMs = nowMs();
    const mixedMono = mixAudioBufferChannelsToMono(decoded);
    const downmixMs = nowMs() - downmixStartMs;
    const resampleStartMs = nowMs();
    const pcm =
      strategy === 'native-rate'
        ? resampleLinear(mixedMono, decoded.sampleRate, targetSampleRate)
        : mixedMono;
    const resampleMs = strategy === 'native-rate' ? nowMs() - resampleStartMs : 0;
    const wallMs = nowMs() - startMs;
    const durationSec = decoded.length / decoded.sampleRate;
    return {
      pcm,
      sampleRate: targetSampleRate,
      durationSec,
      numberOfChannels: decoded.numberOfChannels,
      metrics: {
        backend: 'browser',
        strategy:
          strategy === 'native-rate' ? 'audiocontext-native-rate' : 'audiocontext-target-rate',
        inputSampleRate: strategy === 'native-rate' ? decoded.sampleRate : undefined,
        outputSampleRate: targetSampleRate,
        decodeMs,
        downmixMs,
        resampleMs,
        totalMs: decodeMs + downmixMs + resampleMs,
        wallMs,
        resampler:
          strategy === 'native-rate'
            ? decoded.sampleRate === targetSampleRate
              ? 'none'
              : 'linear-parity'
            : 'browser-audiocontext',
        resamplerQuality:
          strategy === 'native-rate' && decoded.sampleRate !== targetSampleRate
            ? resamplerQuality
            : null,
        audioDurationSec: durationSec,
      },
    };
  } catch (error) {
    primaryError = error;
    failed = true;
    throw error;
  } finally {
    try {
      await audioContext.close?.();
    } catch (cleanupError) {
      if (failed) {
        throw new AggregateError([primaryError, cleanupError], 'Audio decode and cleanup failed.', {
          cause: primaryError,
        });
      }
      throw cleanupError;
    }
  }
}

export async function decodeAudioBufferToMonoPcm(
  sourceBuffer: ArrayBuffer,
  options: Omit<DecodeAudioSourceOptions, 'fetchImpl'> = {},
): Promise<DecodedMonoAudio> {
  return await decodeAudioSourceToMonoPcm(new Blob([sourceBuffer]), options);
}

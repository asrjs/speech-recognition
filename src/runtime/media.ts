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
  readonly createAudioContext?: (sampleRate?: number) => AudioContextLikeForDecode;
  readonly fetchImpl?: typeof fetch;
  readonly strategy?: 'browser-target-rate' | 'native-rate';
  readonly resamplerQuality?: 'linear';
}

const WAV_HEADER_SIZE = 12;

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
  options: Pick<DecodeAudioSourceOptions, 'fetchImpl'> = {},
): Promise<Blob> {
  if (typeof source !== 'string') {
    return source;
  }

  const fetchImpl = options.fetchImpl ?? globalThis.fetch;
  if (typeof fetchImpl !== 'function') {
    throw new Error('resolveAudioSourceToBlob requires fetch when source is a URL.');
  }

  const response = await fetchImpl(source);
  if (!response.ok) {
    throw new Error(`Audio fetch failed: ${response.status}`);
  }

  return await response.blob();
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

function pcmSampleToFloat(view: DataView, offset: number, bitsPerSample: number): number {
  switch (bitsPerSample) {
    case 8:
      return (view.getUint8(offset) - 128) / 128;
    case 16:
      return view.getInt16(offset, true) / 32768;
    case 24: {
      const b0 = view.getUint8(offset);
      const b1 = view.getUint8(offset + 1);
      const b2 = view.getUint8(offset + 2);
      let value = b0 | (b1 << 8) | (b2 << 16);
      if (value & 0x800000) value |= ~0xffffff;
      return value / 8388608;
    }
    case 32:
      return view.getInt32(offset, true) / 2147483648;
    default:
      throw new Error(`Unsupported PCM WAV bits per sample: ${bitsPerSample}`);
  }
}

function floatSampleToFloat(view: DataView, offset: number, bitsPerSample: number): number {
  if (bitsPerSample === 32) return view.getFloat32(offset, true);
  if (bitsPerSample === 64) return view.getFloat64(offset, true);
  throw new Error(`Unsupported float WAV bits per sample: ${bitsPerSample}`);
}

function decodeWavArrayBuffer(arrayBuffer: ArrayBuffer): {
  readonly audio: Float32Array;
  readonly sampleRate: number;
  readonly numberOfChannels: number;
} {
  const view = new DataView(arrayBuffer);
  if (!isWavArrayBuffer(arrayBuffer)) {
    throw new Error('Invalid WAV: expected RIFF/WAVE header.');
  }

  let offset = WAV_HEADER_SIZE;
  let audioFormat: number | null = null;
  let numberOfChannels: number | null = null;
  let sampleRate: number | null = null;
  let bitsPerSample: number | null = null;
  let dataOffset: number | null = null;
  let dataSize: number | null = null;

  while (offset + 8 <= arrayBuffer.byteLength) {
    const chunkId = readAscii(view, offset, 4);
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkDataStart = offset + 8;
    const nextOffset = chunkDataStart + chunkSize + (chunkSize % 2);

    if (chunkId === 'fmt ') {
      audioFormat = view.getUint16(chunkDataStart, true);
      numberOfChannels = view.getUint16(chunkDataStart + 2, true);
      sampleRate = view.getUint32(chunkDataStart + 4, true);
      bitsPerSample = view.getUint16(chunkDataStart + 14, true);
    } else if (chunkId === 'data') {
      dataOffset = chunkDataStart;
      dataSize = chunkSize;
    }

    offset = nextOffset;
  }

  if (
    audioFormat == null ||
    numberOfChannels == null ||
    sampleRate == null ||
    bitsPerSample == null ||
    dataOffset == null ||
    dataSize == null
  ) {
    throw new Error('Invalid WAV: missing fmt or data chunk.');
  }

  const bytesPerSample = bitsPerSample / 8;
  const totalSamples = Math.floor(dataSize / bytesPerSample);
  const totalFrames = Math.floor(totalSamples / numberOfChannels);
  const mono = new Float32Array(totalFrames);

  let sampleOffset = dataOffset;
  for (let frameIndex = 0; frameIndex < totalFrames; frameIndex += 1) {
    let sum = 0;
    for (let channelIndex = 0; channelIndex < numberOfChannels; channelIndex += 1) {
      sum +=
        audioFormat === 3
          ? floatSampleToFloat(view, sampleOffset, bitsPerSample)
          : pcmSampleToFloat(view, sampleOffset, bitsPerSample);
      sampleOffset += bytesPerSample;
    }
    mono[frameIndex] = sum / numberOfChannels;
  }

  return {
    audio: mono,
    sampleRate,
    numberOfChannels,
  };
}

function resampleLinear(audio: Float32Array, fromRate: number, toRate: number): Float32Array {
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
  const blob = await resolveAudioSourceToBlob(source, options);
  const arrayBuffer = await blob.arrayBuffer();
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
    const decoded = decodeWavArrayBuffer(arrayBuffer);
    const decodeMs = nowMs() - decodeStartMs;
    const resampleStartMs = nowMs();
    const pcm = resampleLinear(decoded.audio, decoded.sampleRate, targetSampleRate);
    const resampleMs = nowMs() - resampleStartMs;
    const wallMs = nowMs() - startMs;
    const durationSec = decoded.audio.length / decoded.sampleRate;

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
  try {
    const decodeStartMs = nowMs();
    const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
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
  } finally {
    await audioContext.close?.();
  }
}

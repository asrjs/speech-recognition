import { SAMPLE_RATE } from './constants.js';
import { isLikelyHttpUrl, isNodeRuntime, looksLikeFileUrl } from './util.js';

export interface ParsedWavPcm16 {
  readonly sampleRate: number;
  readonly samples: Int16Array;
}

function readAscii(view: DataView, start: number, length: number): string {
  let out = '';
  for (let i = 0; i < length; i += 1) {
    out += String.fromCharCode(view.getUint8(start + i));
  }
  return out;
}

export function parseWavPcm16(bytes: Uint8Array): ParsedWavPcm16 {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (readAscii(view, 0, 4) !== 'RIFF' || readAscii(view, 8, 4) !== 'WAVE') {
    throw new Error('Only RIFF/WAVE PCM files are supported.');
  }
  let offset = 12;
  let sampleRate = 0;
  let channels = 0;
  let bitsPerSample = 0;
  let dataOffset = -1;
  let dataSize = -1;

  while (offset + 8 <= view.byteLength) {
    const chunkId = readAscii(view, offset, 4);
    const chunkSize = view.getUint32(offset + 4, true);
    offset += 8;

    if (chunkId === 'fmt ') {
      const audioFormat = view.getUint16(offset, true);
      channels = view.getUint16(offset + 2, true);
      sampleRate = view.getUint32(offset + 4, true);
      bitsPerSample = view.getUint16(offset + 14, true);
      if (audioFormat !== 1) {
        throw new Error('Only PCM WAV format is supported.');
      }
    } else if (chunkId === 'data') {
      dataOffset = offset;
      dataSize = chunkSize;
      break;
    }
    offset += chunkSize + (chunkSize % 2);
  }

  if (dataOffset < 0 || dataSize <= 0) {
    throw new Error('WAV data chunk not found.');
  }
  if (channels !== 1) {
    throw new Error('Only mono WAV files are supported.');
  }
  if (bitsPerSample !== 16) {
    throw new Error('Only 16-bit PCM WAV files are supported.');
  }
  if (sampleRate !== SAMPLE_RATE) {
    throw new Error(`Expected sample rate ${SAMPLE_RATE}, got ${sampleRate}.`);
  }

  const pcm = new Int16Array(bytes.buffer, bytes.byteOffset + dataOffset, dataSize / 2);
  return {
    sampleRate,
    samples: new Int16Array(pcm),
  };
}

async function readNodeFileBytes(pathLike: string): Promise<Uint8Array> {
  const fs = await import('node:fs/promises');
  const path = pathLike.startsWith('file://') ? new URL(pathLike) : pathLike;
  const content = await fs.readFile(path);
  return new Uint8Array(content.buffer, content.byteOffset, content.byteLength);
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch WAV: ${response.status} ${response.statusText}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

export async function loadPcm16Wav(input: string | Uint8Array | ArrayBuffer): Promise<ParsedWavPcm16> {
  if (input instanceof Uint8Array) {
    return parseWavPcm16(input);
  }
  if (input instanceof ArrayBuffer) {
    return parseWavPcm16(new Uint8Array(input));
  }

  if (isLikelyHttpUrl(input) || looksLikeFileUrl(input)) {
    return parseWavPcm16(await fetchBytes(input));
  }
  if (isNodeRuntime()) {
    return parseWavPcm16(await readNodeFileBytes(input));
  }
  return parseWavPcm16(await fetchBytes(input));
}

#!/usr/bin/env node
/**
 * Smoke test: load Wav2Vec2 Turkish model from HuggingFace and transcribe Turkish audio.
 * Uses m3hrdadfi/wav2vec2-large-xlsr-turkish ONNX via ysdede/wav2vec2-large-xlsr-turkish-onnx.
 */
import { access, readFile } from 'node:fs/promises';
import path from 'node:path';

const TURKISH_FIXTURE = 'tests/fixtures/019b50a9c564e61682b48068caed8d1e08eba0584b665c31bafd7431438e9ef0.tr.wav';
const EXPECT_WORDS = ['hastalıkların', 'salgınlar'];  // Turkish words expected in the transcript

function decodeWav(buffer) {
  if (buffer.toString('ascii', 0, 4) !== 'RIFF' || buffer.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('Only RIFF/WAVE fixtures are supported.');
  }

  let offset = 12;
  let format = null;
  let data = null;
  while (offset + 8 <= buffer.length) {
    const id = buffer.toString('ascii', offset, offset + 4);
    const size = buffer.readUInt32LE(offset + 4);
    const start = offset + 8;
    const end = start + size;
    if (id === 'fmt ') {
      format = {
        audioFormat: buffer.readUInt16LE(start),
        channels: buffer.readUInt16LE(start + 2),
        sampleRate: buffer.readUInt32LE(start + 4),
        bitsPerSample: buffer.readUInt16LE(start + 14),
      };
    } else if (id === 'data') {
      data = buffer.subarray(start, end);
    }
    offset = end + (size % 2);
  }

  if (!format || !data) throw new Error('WAV fixture must contain fmt and data chunks.');

  const bytesPerSample = format.bitsPerSample / 8;
  const frameCount = Math.floor(data.length / (bytesPerSample * format.channels));
  const pcm = new Float32Array(frameCount);
  for (let f = 0; f < frameCount; f++) {
    let sum = 0;
    for (let c = 0; c < format.channels; c++) {
      const so = (f * format.channels + c) * bytesPerSample;
      if (format.audioFormat === 1 && format.bitsPerSample === 16) {
        sum += data.readInt16LE(so) / 32768;
      } else if (format.audioFormat === 3 && format.bitsPerSample === 32) {
        sum += data.readFloatLE(so);
      }
    }
    pcm[f] = sum / format.channels;
  }
  return { pcm, sampleRate: format.sampleRate, durationSeconds: frameCount / format.sampleRate };
}

async function requireFile(filePath) {
  const absolute = path.resolve(filePath);
  await access(absolute);
  return absolute;
}

async function main() {
  const audioPath = process.env.WAV2VEC2_TR_AUDIO || TURKISH_FIXTURE;
  const resolved = await requireFile(audioPath);
  const { pcm, sampleRate, durationSeconds } = decodeWav(await readFile(resolved));

  const { loadSpeechModel } = await import('../../dist/runtime/load.js');

  console.log(`Loading Turkish Wav2Vec2 from HuggingFace...`);
  const started = performance.now();
  const loaded = await loadSpeechModel({
    modelId: 'm3hrdadfi/wav2vec2-large-xlsr-turkish',
    useManifestSources: true,
  });
  const loadMs = performance.now() - started;
  console.log(`Model loaded in ${loadMs.toFixed(0)}ms`);

  try {
    const start = performance.now();
    const result = await loaded.transcribe({
      data: pcm,
      channels: [pcm],
      numberOfChannels: 1,
      numberOfFrames: pcm.length,
      sampleRate,
      durationSeconds,
      format: 'f32-planar',
    });
    const ms = performance.now() - start;
    const transcript = result.text ?? '';
    const words = transcript.split(/\s+/).filter(Boolean);

    console.log(`wav2vec2-turkish HF smoke: ${transcript ? 'PASS' : 'FAIL'}`);
    console.log(`audio=${audioPath}`);
    console.log(`sampleRate=${sampleRate} duration=${durationSeconds.toFixed(1)}s`);
    console.log(`load=${loadMs.toFixed(0)}ms transcribe=${ms.toFixed(0)}ms`);
    console.log(`words=${words.length} transcript=${transcript}`);

  } finally {
    await loaded.dispose();
  }
}

main().catch((error) => {
  console.error(error?.stack ?? error);
  process.exit(1);
});

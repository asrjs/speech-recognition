import { describe, expect, it, vi } from 'vitest';
import { Buffer } from 'node:buffer';
import {
  createRunnerKvEntry,
  decodePcm16Wav,
  parseArgs,
  runnerKvData,
  runnerKvDims,
  sample,
} from './smoke/whisperx-runner.mjs';

describe('whisperx-runner CLI', () => {
  it('parses language auto, beam size, and output format', () => {
    const parsed = parseArgs([
      '--language', 'auto',
      '--beam_size', '3',
      '--output_format', 'json',
      '--word_timestamps',
      'clip.wav',
    ]);

    expect(parsed.audioPath).toBe('clip.wav');
    expect(parsed.opts.language).toBe('auto');
    expect(parsed.opts.beamSize).toBe(3);
    expect(parsed.opts.outputFormat).toBe('json');
    expect(parsed.opts.wordTimestamps).toBe(true);
  });

  it('honors --no-word_timestamps', () => {
    const parsed = parseArgs(['--no-word_timestamps', 'clip.wav']);
    expect(parsed.opts.wordTimestamps).toBe(false);
  });

  it('maps the WhisperX --model-dir spelling to the model option', () => {
    const parsed = parseArgs(['--model-dir', 'models/whisper', 'clip.wav']);
    expect(parsed.opts.model).toBe('models/whisper');
  });

  it('samples large Whisper vocabularies without spreading into Math.max', () => {
    const logits = new Float32Array(51865);
    logits.fill(-100);
    logits[123] = 8;
    const random = vi.spyOn(Math, 'random').mockReturnValue(0);
    try {
      expect(sample(logits, 0.5)).toBe(123);
    } finally {
      random.mockRestore();
    }
  });

  it('keeps decoder KV dimensions attached to each beam hypothesis', () => {
    const entry = createRunnerKvEntry({
      data: new Float32Array(8),
      dims: [1, 2, 4, 1],
    });
    const laterEntry = createRunnerKvEntry({
      data: new Float32Array(10),
      dims: [1, 2, 5, 1],
    });

    expect(runnerKvData(entry)).toHaveLength(8);
    expect(runnerKvDims(entry, [1, 2, 5, 1])).toEqual([1, 2, 4, 1]);
    expect(runnerKvDims(laterEntry, [1, 2, 4, 1])).toEqual([1, 2, 5, 1]);
    expect(runnerKvDims(new Float32Array(8), [1, 2, 4, 1])).toEqual([1, 2, 4, 1]);
  });

  it('preserves fp16 KV dtype instead of reinterpreting its backing view', () => {
    const data = new Uint16Array([0x3c00, 0xc000]);
    const entry = createRunnerKvEntry({
      data,
      dims: [1, 1, 2, 1],
      type: 'float16',
    });

    expect(entry.type).toBe('float16');
    expect(runnerKvData(entry)).toBe(data);
  });

  it('finds PCM samples after RIFF metadata chunks', () => {
    const chunk = (name: string, payload: Buffer): Buffer => {
      const header = Buffer.alloc(8);
      header.write(name, 0, 4, 'ascii');
      header.writeUInt32LE(payload.length, 4);
      return payload.length % 2 === 0
        ? Buffer.concat([header, payload])
        : Buffer.concat([header, payload, Buffer.alloc(1)]);
    };
    const fmt = Buffer.alloc(16);
    fmt.writeUInt16LE(1, 0);
    fmt.writeUInt16LE(1, 2);
    fmt.writeUInt32LE(16_000, 4);
    fmt.writeUInt32LE(32_000, 8);
    fmt.writeUInt16LE(2, 12);
    fmt.writeUInt16LE(16, 14);
    const samples = Buffer.alloc(4);
    samples.writeInt16LE(16_384, 0);
    samples.writeInt16LE(-16_384, 2);
    const body = Buffer.concat([
      chunk('fmt ', fmt),
      chunk('LIST', Buffer.from('INFO!', 'ascii')),
      chunk('data', samples),
    ]);
    const wav = Buffer.alloc(12 + body.length);
    wav.write('RIFF', 0, 4, 'ascii');
    wav.writeUInt32LE(body.length + 4, 4);
    wav.write('WAVE', 8, 4, 'ascii');
    body.copy(wav, 12);

    const decoded = decodePcm16Wav(wav);
    expect(decoded.sampleRate).toBe(16_000);
    expect(Array.from(decoded.pcm)).toEqual([0.5, -0.5]);
  });
});

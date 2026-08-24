import { describe, expect, it, vi } from 'vitest';
import {
  createRunnerKvEntry,
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
});

import {
  extractAudioUrl,
  getConfigsAndSplits,
  normalizeDatasetRow,
  normalizeReferenceText,
} from '@asrjs/speech-recognition/datasets';
import { describe, expect, it } from 'vitest';

describe('dataset utility edge cases', () => {
  it('extracts the first usable audio URL from nested values', () => {
    expect(extractAudioUrl(null)).toBeNull();
    expect(extractAudioUrl(undefined)).toBeNull();
    expect(extractAudioUrl('')).toBeNull();
    expect(extractAudioUrl('https://example/audio.wav')).toBe('https://example/audio.wav');
    expect(extractAudioUrl([null, undefined, { url: 'https://example/array.wav' }])).toBe(
      'https://example/array.wav',
    );
    expect(extractAudioUrl({ src: 'src.wav', url: 'url.wav', path: 'path.wav' })).toBe('src.wav');
    expect(extractAudioUrl({ name: 'audio.wav' })).toBeNull();
  });

  it('normalizes reference text markers and surrounding whitespace', () => {
    expect(normalizeReferenceText(null)).toBe('');
    expect(normalizeReferenceText('  Hello   NEWLINE   World  ')).toBe('Hello\nWorld');
    expect(normalizeReferenceText('A PARAGRAPH B')).toBe('A\nB');
  });

  it('normalizes plain and wrapped dataset rows', () => {
    const plain = {
      audio: 'audio.wav',
      text: 'hello world',
      speaker: 'Alice',
      gender: 'female',
      speed: 1.2,
      volume: 0.8,
      sample_rate: 44100,
    };

    expect(normalizeDatasetRow(plain, 5)).toEqual({
      rowIndex: 5,
      audioUrl: 'audio.wav',
      referenceText: 'hello world',
      speaker: 'Alice',
      gender: 'female',
      speed: 1.2,
      volume: 0.8,
      sampleRate: 44100,
      raw: plain,
    });

    const wrappedRow = { transcript: 'wrapped text' };
    expect(normalizeDatasetRow({ row: wrappedRow, row_idx: 10 }, 0)).toMatchObject({
      rowIndex: 10,
      audioUrl: null,
      referenceText: 'wrapped text',
      speaker: '',
      gender: '',
      sampleRate: 16000,
      raw: wrappedRow,
    });
  });

  it('deduplicates splits by config while ignoring incomplete entries', () => {
    const result = getConfigsAndSplits([
      { config: 'en', split: 'train' },
      { config: 'en', split: 'test' },
      { config: 'en', split: 'train' },
      { config: 'fr', split: 'train' },
      { config: 'es' },
      { split: 'train' },
    ]);

    expect(result.get('en')).toEqual(['train', 'test']);
    expect(result.get('fr')).toEqual(['train']);
    expect(result.has('es')).toBe(false);
  });
});

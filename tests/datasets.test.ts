import { describe, expect, it } from 'vitest';
import {
  extractAudioUrl,
  normalizeReferenceText,
  normalizeDatasetRow,
  getConfigsAndSplits,
} from '../src/datasets.js';

describe('datasets utility functions', () => {
  describe('extractAudioUrl', () => {
    it('returns null for falsy values', () => {
      expect(extractAudioUrl(null)).toBeNull();
      expect(extractAudioUrl(undefined)).toBeNull();
      expect(extractAudioUrl('')).toBeNull();
    });

    it('returns the string itself if passed a string', () => {
      expect(extractAudioUrl('http://example.com/audio.wav')).toBe('http://example.com/audio.wav');
    });

    it('returns the first extracted url from an array', () => {
      expect(extractAudioUrl(['http://example.com/audio1.wav', 'http://example.com/audio2.wav'])).toBe(
        'http://example.com/audio1.wav',
      );
      expect(extractAudioUrl([null, undefined, 'http://example.com/audio.wav'])).toBe(
        'http://example.com/audio.wav',
      );
    });

    it('returns null if array contains no valid urls', () => {
      expect(extractAudioUrl([null, ''])).toBeNull();
      expect(extractAudioUrl([])).toBeNull();
    });

    it('extracts url from an object with src, url, or path', () => {
      expect(extractAudioUrl({ src: 'http://example.com/audio-src.wav' })).toBe(
        'http://example.com/audio-src.wav',
      );
      expect(extractAudioUrl({ url: 'http://example.com/audio-url.wav' })).toBe(
        'http://example.com/audio-url.wav',
      );
      expect(extractAudioUrl({ path: 'http://example.com/audio-path.wav' })).toBe(
        'http://example.com/audio-path.wav',
      );
      // Tests fallback order (src ?? url ?? path ?? null)
      expect(extractAudioUrl({ url: 'url.wav', path: 'path.wav' })).toBe('url.wav');
    });

    it('returns null for objects without valid keys', () => {
      expect(extractAudioUrl({ name: 'audio.wav' })).toBeNull();
      expect(extractAudioUrl({})).toBeNull();
    });
  });

  describe('normalizeReferenceText', () => {
    it('handles falsy values', () => {
      expect(normalizeReferenceText(null)).toBe('');
      expect(normalizeReferenceText(undefined)).toBe('');
    });

    it('replaces PARAGRAPH with double newline', () => {
      expect(normalizeReferenceText('Hello PARAGRAPH World')).toBe('Hello\nWorld');
    });

    it('replaces NEWLINE with single newline', () => {
      expect(normalizeReferenceText('Hello NEWLINE World')).toBe('Hello\nWorld');
    });

    it('normalizes whitespace around newlines', () => {
      expect(normalizeReferenceText('Hello   \nWorld')).toBe('Hello\nWorld');
      expect(normalizeReferenceText('Hello\n   World')).toBe('Hello\nWorld');
      expect(normalizeReferenceText('Hello   \n   World')).toBe('Hello\nWorld');
    });

    it('trims the output', () => {
      expect(normalizeReferenceText('  Hello World  ')).toBe('Hello World');
    });
  });

  describe('normalizeDatasetRow', () => {
    it('normalizes a plain row object', () => {
      const row = {
        audio: 'audio.wav',
        text: 'hello world',
        speaker: 'Alice',
        gender: 'female',
        speed: 1.2,
        volume: 0.8,
        sample_rate: 44100,
      };

      const result = normalizeDatasetRow(row, 5);

      expect(result).toEqual({
        rowIndex: 5,
        audioUrl: 'audio.wav',
        referenceText: 'hello world',
        speaker: 'Alice',
        gender: 'female',
        speed: 1.2,
        volume: 0.8,
        sampleRate: 44100,
        raw: row,
      });
    });

    it('normalizes a wrapped row object', () => {
      const row = { transcript: 'hello' };
      const wrapper = {
        row,
        row_idx: 10,
      };

      const result = normalizeDatasetRow(wrapper, 0);

      expect(result).toEqual({
        rowIndex: 10,
        audioUrl: null,
        referenceText: 'hello',
        speaker: '',
        gender: '',
        speed: NaN,
        volume: NaN,
        sampleRate: 16000,
        raw: row,
      });
    });

    it('extracts reference text with fallback priority', () => {
      const row1 = { transcription: 'one', text: 'two', transcript: 'three' };
      expect(normalizeDatasetRow(row1).referenceText).toBe('one');

      const row2 = { text: 'two', transcript: 'three' };
      expect(normalizeDatasetRow(row2).referenceText).toBe('two');

      const row3 = { transcript: 'three' };
      expect(normalizeDatasetRow(row3).referenceText).toBe('three');
    });

    it('defaults sample_rate to 16000', () => {
      const row = { text: 'hi' };
      expect(normalizeDatasetRow(row).sampleRate).toBe(16000);
    });
  });

  describe('getConfigsAndSplits', () => {
    it('groups splits by config and deduplicates', () => {
      const splits = [
        { config: 'en', split: 'train' },
        { config: 'en', split: 'test' },
        { config: 'en', split: 'train' }, // duplicate split
        { config: 'fr', split: 'train' },
        { config: 'fr', split: 'validation' },
        { dataset: 'ignored' }, // missing config and split
        { config: 'es' }, // missing split
        { split: 'train' }, // missing config
      ];

      const result = getConfigsAndSplits(splits);

      expect(result.size).toBe(2);
      expect(result.get('en')).toEqual(['train', 'test']);
      expect(result.get('fr')).toEqual(['train', 'validation']);
      expect(result.get('es')).toBeUndefined();
    });

    it('returns an empty map for empty array', () => {
      expect(getConfigsAndSplits([])).toEqual(new Map());
    });
  });
});
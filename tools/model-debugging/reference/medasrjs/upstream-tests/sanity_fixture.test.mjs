import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, test, expect } from 'vitest';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const WAV_PATH = path.join(__dirname, 'fixtures', 'sanity_sample.wav');
const LABEL_PATH = path.join(__dirname, 'fixtures', 'sanity_sample.label.json');

describe('sanity fixture assets', () => {
  test('wav and label are present and usable', () => {
    expect(fs.existsSync(WAV_PATH)).toBe(true);
    expect(fs.statSync(WAV_PATH).size).toBeGreaterThan(0);

    expect(fs.existsSync(LABEL_PATH)).toBe(true);
    const label = JSON.parse(fs.readFileSync(LABEL_PATH, 'utf-8'));
    expect(typeof label.transcription).toBe('string');
    expect(label.transcription.trim().length).toBeGreaterThan(0);
  });
});

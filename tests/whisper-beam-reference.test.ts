import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

interface BeamReference {
  readonly tokens: readonly number[];
  readonly text: string;
}

interface WhisperBeamReferenceFixture {
  readonly audio: {
    readonly fixture: string;
    readonly sha256: string;
    readonly sample_rate: number;
    readonly duration_seconds: number;
    readonly num_samples: number;
  };
  readonly prompt_ids: readonly number[];
  readonly decode: {
    readonly language: string;
    readonly task: string;
    readonly no_timestamps: boolean;
    readonly max_new_tokens: number;
    readonly do_sample: boolean;
  };
  readonly beams: Readonly<Record<string, BeamReference>>;
}

const fixturePath = fileURLToPath(
  new URL('../tools/data/results/whisper/whisper-large-v3-turbo-jfk2-beams.json', import.meta.url),
);
const fixture = JSON.parse(readFileSync(fixturePath, 'utf8')) as WhisperBeamReferenceFixture;
const audioPath = fileURLToPath(new URL('./fixtures/jfk2.en.wav', import.meta.url));

function sha256(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex').toUpperCase();
}

describe('Whisper HF beam reference fixture', () => {
  it('keeps audio provenance and deterministic decode policy explicit', () => {
    expect(fixture.audio.fixture).toBe('tests/fixtures/jfk2.en.wav');
    expect(sha256(audioPath)).toBe(fixture.audio.sha256);
    expect(fixture.audio.sample_rate).toBe(16_000);
    expect(fixture.audio.num_samples).toBe(176_000);
    expect(fixture.decode).toEqual({
      language: 'en',
      task: 'transcribe',
      no_timestamps: true,
      max_new_tokens: 128,
      do_sample: false,
      length_penalty: 1,
      early_stopping: false,
    });
  });

  it('records the complete prompt, EOS, and stable text for beams 1, 2, and 5', () => {
    expect(fixture.prompt_ids).toEqual([50258, 50259, 50360, 50364]);
    expect(Object.keys(fixture.beams).sort()).toEqual(['1', '2', '5']);

    const referenceText = fixture.beams['1']!.text;
    for (const beamSize of ['1', '2', '5']) {
      const beam = fixture.beams[beamSize]!;
      expect(beam.tokens.slice(0, fixture.prompt_ids.length)).toEqual(fixture.prompt_ids);
      expect(beam.tokens.at(-1)).toBe(50257);
      expect(beam.tokens.slice(fixture.prompt_ids.length, -1).every((id) => id < 50364)).toBe(true);
      expect(beam.text).toBe(referenceText);
    }

    expect(fixture.beams['2']!.tokens).toEqual(fixture.beams['1']!.tokens);
    expect(fixture.beams['5']!.tokens).toEqual(fixture.beams['1']!.tokens);
  });
});

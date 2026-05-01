import { describe, expect, it } from 'vitest';
import * as whisperExports from '../../src/presets/whisper.js';

describe('whisper index exports', () => {
  it('exports expected items from factory', () => {
    expect(typeof whisperExports.createWhisperPresetFactory).toBe('function');
  });

  it('exports expected items from manifest', () => {
    expect(whisperExports.WHISPER_PRESET_MANIFESTS).toBeDefined();
    expect(typeof whisperExports.resolveWhisperPresetManifest).toBe('function');
  });
});

import { describe, expect, it } from 'vitest';
import {
  parseArgs,
  runOrThrow,
  validateCommandArg,
  validatePath,
  validateSampleRate,
} from '../tools/model-debugging/reference/hf-parakeet-port/node-nemo-shared-audio-parity.mjs';

describe('node-nemo-shared-audio-parity security validation', () => {
  describe('validatePath', () => {
    it('accepts valid paths', () => {
      expect(validatePath('tools/data/fixtures/audio/librivox.org.wav', 'audio')).toContain('librivox.org.wav');
      expect(validatePath('/models/onnx/nemo/model', 'nodeModel')).toBe('/models/onnx/nemo/model');
    });

    it('rejects paths containing null bytes or invalid types', () => {
      expect(() => validatePath('audio\0file.wav', 'audio')).toThrow(/Null bytes/);
      expect(() => validatePath(123 as any, 'audio')).toThrow(/Invalid audio/);
      expect(() => validatePath('', 'audio')).toThrow(/Empty audio/);
    });
  });

  describe('validateSampleRate', () => {
    it('accepts positive integer sample rates', () => {
      expect(validateSampleRate(16000)).toBe(16000);
      expect(validateSampleRate('44100')).toBe(44100);
    });

    it('rejects invalid, negative, non-integer, or non-numeric sample rates', () => {
      expect(() => validateSampleRate(0)).toThrow(/Invalid sample rate/);
      expect(() => validateSampleRate(-16000)).toThrow(/Invalid sample rate/);
      expect(() => validateSampleRate(16000.5)).toThrow(/Invalid sample rate/);
      expect(() => validateSampleRate('abc')).toThrow(/Invalid sample rate/);
      expect(() => validateSampleRate(NaN)).toThrow(/Invalid sample rate/);
    });
  });

  describe('validateCommandArg', () => {
    it('accepts safe strings', () => {
      expect(validateCommandArg('python', 'command')).toBe('python');
      expect(validateCommandArg('--input', 'arg')).toBe('--input');
    });

    it('rejects non-strings or strings with null bytes', () => {
      expect(() => validateCommandArg('python\0.exe', 'command')).toThrow(/Null bytes/);
      expect(() => validateCommandArg(null as any, 'arg')).toThrow(/Invalid arg/);
    });
  });

  describe('parseArgs', () => {
    it('returns default options when no arguments provided', () => {
      const opts = parseArgs([]);
      expect(opts.sampleRate).toBe(16000);
      expect(opts.keepArtifacts).toBe(false);
      expect(opts.audio).toContain('librivox.org.wav');
    });

    it('parses valid custom arguments', () => {
      const opts = parseArgs([
        '--audio', 'custom.wav',
        '--sample-rate', '22050',
        '--node-model', '/node-path',
        '--onnx-asr-model-path', '/onnx-path',
        '--nemo-model', 'nvidia/nemo-model',
        '--keep-artifacts',
      ]);
      expect(opts.sampleRate).toBe(22050);
      expect(opts.keepArtifacts).toBe(true);
      expect(opts.audio).toContain('custom.wav');
      expect(opts.nodeModel).toBe('/node-path');
      expect(opts.onnxAsrModelPath).toBe('/onnx-path');
      expect(opts.nemoModel).toBe('nvidia/nemo-model');
    });

    it('throws error when flag value contains null bytes', () => {
      expect(() => parseArgs(['--audio', 'malicious\0.wav'])).toThrow(/Null bytes/);
    });
  });

  describe('runOrThrow', () => {
    it('validates command and arguments before execution', () => {
      expect(() => runOrThrow('node\0', ['-v'], 'test')).toThrow(/Null bytes/);
      expect(() => runOrThrow('node', ['-v\0'], 'test')).toThrow(/Null bytes/);
    });
  });
});

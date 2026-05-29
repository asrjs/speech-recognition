/**
 * Tests for VAD backend adapters.
 * T1: FireRed backend (full-file) + TenVAD backend (energy-fallback).
 */

import { describe, it, expect } from 'vitest';
import { TenVadBackend } from '../src/chunking/backends/ten-vad.js';
import { FireRedVadBackend } from '../src/chunking/backends/firered-vad.js';
import type { WhisperVadBackend } from '../src/chunking/types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Generate synthetic audio: 2s silence + 3s speech (440Hz sine) + 2s silence. */
function makeSpeechAudio(sampleRate: number): Float32Array {
  const totalSamples = sampleRate * 7; // 7 seconds
  const audio = new Float32Array(totalSamples);

  // Silence: 0-2s
  // Speech: 2-5s (440 Hz sine)
  // Silence: 5-7s
  for (let i = 0; i < totalSamples; i++) {
    const t = i / sampleRate;
    if (t >= 2.0 && t < 5.0) {
      audio[i] = Math.sin(2 * Math.PI * 440 * t) * 0.3;
    }
  }
  return audio;
}

// ---------------------------------------------------------------------------
// TenVAD Backend
// ---------------------------------------------------------------------------

describe('TenVadBackend', () => {
  it('can be created with defaults', async () => {
    const backend = await TenVadBackend.create();
    expect(backend).toBeDefined();
  });

  it('detects speech in synthetic audio', async () => {
    const backend = await TenVadBackend.create({ threshold: 0.5 });
    const audio = makeSpeechAudio(16000);
    const segments = await backend.segment(audio, 16000, 0.5);
    expect(segments.length).toBeGreaterThan(0);
    // The speech segment should be roughly 2-5s
    if (segments.length > 0) {
      expect(segments[0]!.startSeconds).toBeGreaterThan(1.0);
      expect(segments[0]!.endSeconds).toBeLessThan(6.0);
    }
  });

  it('returns empty for all-silence audio', async () => {
    const backend = await TenVadBackend.create({ threshold: 0.5 });
    const silence = new Float32Array(16000 * 3); // 3s silence
    const segments = await backend.segment(silence, 16000, 0.5);
    expect(segments.length).toBe(0);
  });

  it('respects min speech duration', async () => {
    const backend = await TenVadBackend.create({
      threshold: 0.3,
      minSpeechDurationMs: 2000, // require 2s minimum
    });
    const audio = makeSpeechAudio(16000);
    const segments = await backend.segment(audio, 16000, 0.3);
    // Speech is ~3s > 2s minimum → should be detected
    expect(segments.length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// FireRed Backend (interface contract only — model not loaded)
// ---------------------------------------------------------------------------

describe('FireRedVadBackend', () => {
  it('has static create method matching interface', () => {
    expect(FireRedVadBackend.create).toBeInstanceOf(Function);
  });

  it('implements WhisperVadBackend interface contract', () => {
    // Type check: the class implements the interface
    const _backend: WhisperVadBackend = {} as FireRedVadBackend;
    expect(_backend).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// WhisperVadBackend interface via mock
// ---------------------------------------------------------------------------

describe('WhisperVadBackend mock', () => {
  it('accepts mock implementation', async () => {
    const mock: WhisperVadBackend = {
      async segment(audio, sampleRate, _threshold) {
        const duration = audio.length / sampleRate;
        if (duration < 0.25) return [];
        return [{ startSeconds: 0, endSeconds: duration, durationSeconds: duration }];
      },
    };
    const result = await mock.segment(new Float32Array(16000 * 5), 16000, 0.5);
    expect(result).toHaveLength(1);
    expect(result[0]!.durationSeconds).toBeCloseTo(5.0, 0);
  });
});

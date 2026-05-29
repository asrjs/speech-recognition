/**
 * Tests for enhanced Whisper types.
 * Phase 1: type compilation + default value factories.
 */

import { describe, it, expect } from 'vitest';

// Import the types (even just importing verifies they compile)
import {
  type QualityVerdict,
  type QualityGateResult,
  type SegmentQualityMetrics,
  type EnhancedDecodeResult,
  type EnhancedDecodeOptions,
  type VadSegmenterConfig,
  makeDefaultEnhancedDecodeOptions,
  makeDefaultVadSegmenterConfig,
} from '../src/models/whisper-seq2seq/enhanced-types.js';

describe('enhanced-types compilation', () => {
  it('exports QualityVerdict type', () => {
    const v: QualityVerdict = 'accept';
    expect(v).toBe('accept');
  });

  it('exports QualityGateResult interface', () => {
    const r: QualityGateResult = {
      verdict: 'reject',
      compressionRatio: 3.2,
      avgLogProb: -2.1,
      noSpeechProb: 0.1,
      entropy: 3.0,
      reason: 'compression_ratio_too_high',
    };
    expect(r.verdict).toBe('reject');
    expect(r.compressionRatio).toBe(3.2);
  });

  it('exports SegmentQualityMetrics interface', () => {
    const m: SegmentQualityMetrics = {
      compressionRatio: 1.5,
      avgLogProb: -0.8,
      noSpeechProb: 0.05,
      entropy: 1.2,
      temperature: 0.4,
    };
    expect(m.temperature).toBe(0.4);
  });

  it('exports EnhancedDecodeResult interface', () => {
    const metric: SegmentQualityMetrics = {
      compressionRatio: 2.0,
      avgLogProb: -0.5,
      noSpeechProb: 0.02,
      entropy: 1.1,
      temperature: 0.0,
    };
    const r: EnhancedDecodeResult = {
      tokens: [50364, 400, 370, 50257],
      text: 'hello',
      metrics: metric,
    };
    expect(r.text).toBe('hello');
    expect(r.tokens).toHaveLength(4);
  });
});

describe('EnhancedDecodeOptions defaults', () => {
  it('provides all defaults when called with no args', () => {
    const opts = makeDefaultEnhancedDecodeOptions();
    expect(opts.compressionRatioThreshold).toBe(2.4);
    expect(opts.logProbThreshold).toBe(-1.0);
    expect(opts.noSpeechThreshold).toBe(0.6);
    expect(opts.entropyThreshold).toBe(2.4);
    expect(opts.temperatureFallback).toBe(true);
    expect(opts.temperatures).toEqual([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
    expect(opts.conditionOnPreviousText).toBe(true);
    expect(opts.maxContextTokens).toBeUndefined();
  });

  it('merges user overrides with defaults', () => {
    const opts = makeDefaultEnhancedDecodeOptions({
      compressionRatioThreshold: 3.0,
      temperatureFallback: false,
      temperatures: [0.0, 0.5],
    });
    expect(opts.compressionRatioThreshold).toBe(3.0);
    expect(opts.logProbThreshold).toBe(-1.0);
    expect(opts.temperatureFallback).toBe(false);
    expect(opts.temperatures).toEqual([0.0, 0.5]);
  });
});

describe('VadSegmenterConfig defaults', () => {
  it('provides defaults for ten-vad backend', () => {
    const cfg = makeDefaultVadSegmenterConfig({ backend: 'ten-vad' });
    expect(cfg.backend).toBe('ten-vad');
    expect(cfg.speechThreshold).toBe(0.5);
    expect(cfg.minSpeechDurationMs).toBe(250);
    expect(cfg.minSilenceDurationMs).toBe(100);
    expect(cfg.speechPadMs).toBe(400);
    expect(cfg.maxSegmentDurationMs).toBe(29000);
  });

  it('provides defaults for firered-vad backend', () => {
    const cfg = makeDefaultVadSegmenterConfig({ backend: 'firered-vad' });
    expect(cfg.backend).toBe('firered-vad');
  });

  it('merges user overrides', () => {
    const cfg = makeDefaultVadSegmenterConfig({
      backend: 'ten-vad',
      speechThreshold: 0.7,
      minSpeechDurationMs: 500,
    });
    expect(cfg.speechThreshold).toBe(0.7);
    expect(cfg.minSpeechDurationMs).toBe(500);
    expect(cfg.minSilenceDurationMs).toBe(100); // unchanged default
  });
});

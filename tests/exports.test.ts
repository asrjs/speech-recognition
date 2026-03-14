import { describe, expect, it } from 'vitest';

describe('public exports', () => {
  it('exposes the single-package API surface', async () => {
    const module = await import('asr.js');

    expect(module.TranscriptDetailLevel).toBeUndefined();
    expect(module.DefaultSpeechRuntime).toBeTypeOf('function');
    expect(module.PcmAudioBuffer).toBeTypeOf('function');
    expect(module.FASTCONFORMER_ENCODER.kind).toBe('fastconformer');
    expect(module.CTC_HEAD_DECODER.kind).toBe('ctc-head');
    expect(module.estimateFrameBasedProcessorDescriptor).toBeTypeOf('function');
    expect(module.StubSentencePieceTokenizer).toBeTypeOf('function');
    expect(module.createDecodingDescriptor).toBeTypeOf('function');
    expect(module.DefaultStreamingTranscriber).toBeTypeOf('function');
    expect(module.createWasmBackend).toBeTypeOf('function');
    expect(module.createWebGpuBackend).toBeTypeOf('function');
    expect(module.createWebNnBackend).toBeTypeOf('function');
    expect(module.createWebGlBackend).toBeTypeOf('function');
    expect(module.createModelClassification).toBeTypeOf('function');
    expect(module.AudioFeatureCache).toBeTypeOf('function');
    expect(module.AudioChunker).toBeTypeOf('function');
    expect(module.LayeredAudioBuffer).toBeTypeOf('function');
    expect(module.FrameAlignedTokenMerger).toBeTypeOf('function');
    expect(module.LcsPtfaTokenMerger).toBeTypeOf('function');
    expect(module.createNemoTdtModelFamily).toBeTypeOf('function');
    expect(module.createHfCtcModelFamily).toBeTypeOf('function');
    expect(module.createWhisperSeq2SeqModelFamily).toBeTypeOf('function');
    expect(module.createParakeetPresetFactory).toBeTypeOf('function');
    expect(module.createMedAsrPresetFactory).toBeTypeOf('function');
    expect(module.createWhisperPresetFactory).toBeTypeOf('function');
  });
});

import { describe, expect, it } from 'vitest';

describe('public exports', () => {
  it('keeps the root API focused on runtime-critical surfaces', async () => {
    const module = await import('@asrjs/speech-recognition');

    expect(module.TranscriptDetailLevel).toBeUndefined();
    expect(module.DefaultSpeechRuntime).toBeTypeOf('function');
    expect(module.createSpeechRuntime).toBeTypeOf('function');
    expect(module.listSpeechModels).toBeTypeOf('function');
    expect(module.listSpeechModelOptions).toBeTypeOf('function');
    expect(module.getSpeechModelDescriptor).toBeTypeOf('function');
    expect(module.getSpeechModelLanguageName).toBeTypeOf('function');
    expect(module.resolveSpeechModelComponentBackends).toBeTypeOf('function');
    expect(module.detectSpeechModelRepoQuantizations).toBeTypeOf('function');
    expect(module.buildSpeechModelLoadOptions).toBeTypeOf('function');
    expect(module.buildSpeechTranscriptionOptions).toBeTypeOf('function');
    expect(module.loadSpeechModel).toBeTypeOf('function');
    expect(module.transcribeSpeech).toBeTypeOf('function');
    expect(module.transcribeSpeechFromMonoPcm).toBeTypeOf('function');
    expect(module.createSpeechPipeline).toBeTypeOf('function');
    expect(module.PcmAudioBuffer).toBeTypeOf('function');
    expect(module.createWasmBackend).toBeTypeOf('function');
    expect(module.createWebGpuBackend).toBeTypeOf('function');
    expect(module.createWebNnBackend).toBeTypeOf('function');
    expect(module.createWebGlBackend).toBeTypeOf('function');
    expect(module.createNemoTdtTranscriptNormalizer).toBeTypeOf('function');
    expect(module.getCanonicalTranscript).toBeTypeOf('function');

    expect(module.createBuiltInSpeechRuntime).toBeUndefined();
    expect(module.fetchModelFiles).toBeUndefined();
    expect(module.argmax).toBeUndefined();
    expect(module.FASTCONFORMER_ENCODER).toBeUndefined();
    expect(module.DefaultStreamingTranscriber).toBeUndefined();
    expect(module.createNemoTdtModelFamily).toBeUndefined();
    expect(module.createParakeetPresetFactory).toBeUndefined();
    expect(module.ParakeetModel).toBeUndefined();
    expect(module.decodeAudioSourceToMonoPcm).toBeUndefined();
    expect(module.AudioRingBuffer).toBeUndefined();
    expect(module.fetchDatasetSplits).toBeUndefined();
    expect(module.benchmarkRunRecordsToCsv).toBeUndefined();
  });

  it('exposes layered helper, model, and preset subpaths', async () => {
    const builtins = await import('@asrjs/speech-recognition/builtins');
    const io = await import('@asrjs/speech-recognition/io');
    const ioNode = await import('@asrjs/speech-recognition/io/node');
    const inference = await import('@asrjs/speech-recognition/inference');
    const browser = await import('@asrjs/speech-recognition/browser');
    const realtime = await import('@asrjs/speech-recognition/realtime');
    const bench = await import('@asrjs/speech-recognition/bench');
    const datasets = await import('@asrjs/speech-recognition/datasets');
    const nemoTdt = await import('@asrjs/speech-recognition/models/nemo-tdt');
    const nemoAed = await import('@asrjs/speech-recognition/models/nemo-aed');
    const lasrCtc = await import('@asrjs/speech-recognition/models/lasr-ctc');
    const whisperModel = await import('@asrjs/speech-recognition/models/whisper-seq2seq');
    const presets = await import('@asrjs/speech-recognition/presets');
    const canaryPreset = await import('@asrjs/speech-recognition/presets/canary');
    const parakeetPreset = await import('@asrjs/speech-recognition/presets/parakeet');
    const medasrPreset = await import('@asrjs/speech-recognition/presets/medasr');
    const whisperPreset = await import('@asrjs/speech-recognition/presets/whisper');

    expect(builtins.createBuiltInSpeechRuntime).toBeTypeOf('function');
    expect(builtins.loadBuiltInSpeechModel).toBeTypeOf('function');
    expect(builtins.registerBuiltInModelFamilies).toBeTypeOf('function');
    expect(builtins.registerBuiltInPresets).toBeTypeOf('function');

    expect(io.fetchModelFiles).toBeTypeOf('function');
    expect(io.getModelFile).toBeTypeOf('function');
    expect(io.pickPreferredQuant).toBeTypeOf('function');
    expect(io.createDefaultAssetProvider).toBeTypeOf('function');
    expect(io.createHuggingFaceAssetProvider).toBeTypeOf('function');
    expect(io.createNodeFileSystemAssetProvider).toBeUndefined();

    expect(ioNode.createNodeFileSystemAssetProvider).toBeTypeOf('function');
    expect(ioNode.createDefaultNodeAssetProvider).toBeTypeOf('function');

    expect(inference.FASTCONFORMER_ENCODER.kind).toBe('fastconformer');
    expect(inference.CTC_HEAD_DECODER.kind).toBe('ctc-head');
    expect(inference.argmax).toBeTypeOf('function');
    expect(inference.confidenceFromLogits).toBeTypeOf('function');
    expect(inference.DefaultStreamingTranscriber).toBeTypeOf('function');
    expect(inference.FrameAlignedTokenMerger).toBeTypeOf('function');
    expect(inference.LcsPtfaTokenMerger).toBeTypeOf('function');

    expect(browser.decodeAudioSourceToMonoPcm).toBeTypeOf('function');
    expect(browser.createSpeechModelLocalEntries).toBeTypeOf('function');
    expect(browser.collectSpeechModelLocalEntries).toBeTypeOf('function');
    expect(browser.inspectSpeechModelLocalEntries).toBeTypeOf('function');
    expect(browser.loadSpeechModelFromLocalEntries).toBeTypeOf('function');
    expect(browser.createBrowserRealtimeStarter).toBeTypeOf('function');
    expect(browser.createBrowserRealtimeMicrophoneController).toBeTypeOf('function');
    expect(browser.createBrowserRealtimeMonitor).toBeTypeOf('function');
    expect(browser.renderBrowserRealtimeWaveformFrame).toBeTypeOf('function');
    expect(browser.TenVadAdapter).toBeTypeOf('function');
    expect(browser.resolveSupportedTenVadHopSize).toBeTypeOf('function');
    expect(browser.startMicrophoneCapture).toBeTypeOf('function');
    expect(browser.startMicrophoneRingCapture).toBeTypeOf('function');
    expect(browser.encodeMonoPcmToWavBlob).toBeTypeOf('function');
    expect(realtime.AudioFeatureCache).toBeTypeOf('function');
    expect(realtime.AudioRingBuffer).toBeTypeOf('function');
    expect(realtime.NoiseFloorTracker).toBeTypeOf('function');
    expect(realtime.RoughSpeechGate).toBeTypeOf('function');
    expect(realtime.RealtimeTranscriptionController).toBeTypeOf('function');
    expect(realtime.StreamingSpeechDetector).toBeTypeOf('function');
    expect(realtime.VoiceActivityProbabilityBuffer).toBeTypeOf('function');
    expect(realtime.listStreamingPresets).toBeTypeOf('function');
    expect(bench.summarizeNumericSeries).toBeTypeOf('function');
    expect(bench.benchmarkRunRecordsToCsv).toBeTypeOf('function');
    expect(datasets.fetchDatasetSplits).toBeTypeOf('function');
    expect(datasets.fetchRandomRows).toBeTypeOf('function');

    expect(nemoTdt.createNemoTdtModelFamily).toBeTypeOf('function');
    expect(nemoAed.createNemoAedModelFamily).toBeTypeOf('function');
    expect(lasrCtc.createLasrCtcModelFamily).toBeTypeOf('function');
    expect(whisperModel.createWhisperSeq2SeqModelFamily).toBeTypeOf('function');
    expect(presets.listBuiltInModelDescriptors).toBeTypeOf('function');
    expect(presets.getBuiltInModelDescriptor).toBeTypeOf('function');
    expect(presets.buildBuiltInHubLoadOptions).toBeTypeOf('function');
    expect(presets.buildBuiltInTranscriptionOptions).toBeTypeOf('function');
    expect(presets.resolveBuiltInModelComponentBackends).toBeTypeOf('function');

    expect(canaryPreset.createCanaryPresetFactory).toBeTypeOf('function');
    expect(canaryPreset.CanaryModel).toBeTypeOf('function');
    expect(canaryPreset.getCanaryModel).toBeTypeOf('function');
    expect(canaryPreset.transcribeCanary).toBeTypeOf('function');
    expect(parakeetPreset.createParakeetPresetFactory).toBeTypeOf('function');
    expect(parakeetPreset.ParakeetModel).toBeTypeOf('function');
    expect(parakeetPreset.getParakeetModel).toBeTypeOf('function');
    expect(parakeetPreset.loadParakeetModelWithFallback).toBeTypeOf('function');
    expect(medasrPreset.createMedAsrPresetFactory).toBeTypeOf('function');
    expect(whisperPreset.createWhisperPresetFactory).toBeTypeOf('function');
  });
});

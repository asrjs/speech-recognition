import { createLocalArtifactAdapter } from './local-artifact-adapter.js';

export const senseVoiceBuiltInLocalModelAdapter = createLocalArtifactAdapter({
  preset: 'sensevoice',
  required: ['model.onnx', 'vocab.txt', 'am.mvn'],
  encoderArtifact: 'model.onnx',
  decoderArtifact: 'model.onnx',
  tokenizerArtifact: 'vocab.txt',
  resolveLoadOptions(options, urls) {
    return {
      source: {
        kind: 'direct',
        artifacts: {
          modelUrl: urls['model.onnx']!,
          tokenizerUrl: urls['vocab.txt']!,
          cmvnUrl: urls['am.mvn']!,
        },
        cpuThreads: options.cpuThreads,
        wasmPaths: options.wasmPaths,
        enableProfiling: options.enableProfiling,
      },
    };
  },
});

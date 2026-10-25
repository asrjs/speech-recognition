import { createLocalArtifactAdapter } from './local-artifact-adapter.js';

export const gigaAmCtcBuiltInLocalModelAdapter = createLocalArtifactAdapter({
  preset: 'gigaam-ctc',
  required: ['multilingual_ctc.onnx', 'multilingual_vocab.txt'],
  encoderArtifact: 'multilingual_ctc.onnx',
  decoderArtifact: 'multilingual_ctc.onnx',
  tokenizerArtifact: 'multilingual_vocab.txt',
  resolveLoadOptions(options, urls) {
    return {
      source: { kind: 'direct', artifacts: {
        modelUrl: urls['multilingual_ctc.onnx']!,
        tokenizerUrl: urls['multilingual_vocab.txt']!,
      }, cpuThreads: options.cpuThreads, wasmPaths: options.wasmPaths, enableProfiling: options.enableProfiling },
    };
  },
});

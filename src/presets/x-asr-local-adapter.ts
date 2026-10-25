import { createLocalArtifactAdapter } from './local-artifact-adapter.js';

export const xAsrBuiltInLocalModelAdapter = createLocalArtifactAdapter({
  preset: 'x-asr',
  required: ['encoder-160ms.onnx', 'decoder-160ms.onnx', 'joiner-160ms.onnx', 'tokens.txt'],
  encoderArtifact: 'encoder-160ms.onnx',
  decoderArtifact: 'decoder-160ms.onnx',
  tokenizerArtifact: 'tokens.txt',
  resolveLoadOptions(options, urls) {
    return {
      source: {
        kind: 'direct',
        artifacts: {
          encoderUrl: urls['encoder-160ms.onnx']!,
          decoderUrl: urls['decoder-160ms.onnx']!,
          joinerUrl: urls['joiner-160ms.onnx']!,
          tokenizerUrl: urls['tokens.txt']!,
        },
        cpuThreads: options.cpuThreads,
        wasmPaths: options.wasmPaths,
        enableProfiling: options.enableProfiling,
      },
      config: { chunkMs: 160 },
    };
  },
});

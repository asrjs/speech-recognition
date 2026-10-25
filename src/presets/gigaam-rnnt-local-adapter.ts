import { createLocalArtifactAdapter } from './local-artifact-adapter.js';

export const gigaAmRnntBuiltInLocalModelAdapter = createLocalArtifactAdapter({
  preset: 'gigaam-rnnt',
  required: ['v3_e2e_rnnt_encoder.onnx', 'v3_e2e_rnnt_decoder.onnx', 'v3_e2e_rnnt_joint.onnx', 'v3_e2e_rnnt_vocab.txt'],
  encoderArtifact: 'v3_e2e_rnnt_encoder.onnx',
  decoderArtifact: 'v3_e2e_rnnt_decoder.onnx',
  tokenizerArtifact: 'v3_e2e_rnnt_vocab.txt',
  resolveLoadOptions(options, urls) {
    return {
      source: {
        kind: 'direct',
        encoderBackend: options.encoderBackend,
        decoderBackend: options.decoderBackend,
        jointBackend: options.decoderBackend,
        cpuThreads: options.cpuThreads,
        wasmPaths: options.wasmPaths,
        enableProfiling: options.enableProfiling,
        artifacts: {
          encoderUrl: urls['v3_e2e_rnnt_encoder.onnx']!,
          decoderUrl: urls['v3_e2e_rnnt_decoder.onnx']!,
          jointUrl: urls['v3_e2e_rnnt_joint.onnx']!,
          tokenizerUrl: urls['v3_e2e_rnnt_vocab.txt']!,
        },
      },
    };
  },
});

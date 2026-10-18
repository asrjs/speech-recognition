import {
  STREAMING_GATE_MODES,
  mergeStreamingConfig,
  type StreamingDetectorConfigOverrides,
} from '@asrjs/speech-recognition/realtime';

// Compiled by the public-consumer typecheck, including saved legacy settings.
export const vadConfigurations: StreamingDetectorConfigOverrides[] = Object.values(
  STREAMING_GATE_MODES,
).map((gateMode) => ({ gateMode, tenVadEnabled: true }));

export const neuralOnly = mergeStreamingConfig('generic-streaming', {
  gateMode: 'vad-only', tenVadEnabled: true,
});
export const hybrid = mergeStreamingConfig('generic-streaming', {
  gateMode: 'hybrid', tenVadEnabled: true,
});

import {
  createSpeechRuntime,
  createNativeCpuBackend,
  PcmAudioBuffer,
} from '@asrjs/speech-recognition';
import * as browser from '@asrjs/speech-recognition/browser';
import { createNemoTdtModelFamily } from '@asrjs/speech-recognition/models/nemo-tdt';

const runtime = createSpeechRuntime();
runtime.registerBackend(createNativeCpuBackend());
runtime.registerModelFamily(createNemoTdtModelFamily());
const audio = PcmAudioBuffer.fromMono(new Float32Array(160), 16000);
void audio;
void browser;
void runtime.dispose();

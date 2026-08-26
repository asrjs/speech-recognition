import { FireRedFbank } from '../../runtime/firered-vad/core/fbank.js';

/**
 * X-ASR uses the same 16 kHz/80-bin Kaldi fbank family as sherpa-onnx.
 * The stream wrapper owns the unconsumed sample tail; this class only turns
 * complete frames into frame-major [frame, feature] data.
 */
export class XAsrJsFrontend {
  private readonly fbank = new FireRedFbank({
    num_bins: 80,
    sample_rate: 16000,
    frame_length: 400,
    frame_shift: 160,
    remove_dc_offset: true,
    pre_emphasis: true,
    use_log: true,
    stateful_pre_emphasis: false,
  });

  process(audio: Float32Array): Float32Array {
    const frames = this.fbank.compute(audio);
    const output = new Float32Array(frames.length * 80);
    frames.forEach((frame, index) => output.set(frame, index * 80));
    return output;
  }
}

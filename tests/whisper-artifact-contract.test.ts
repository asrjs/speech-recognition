import { describe, expect, it } from 'vitest';
import { inspectWhisperArtifactContract } from '../tools/model-debugging/scripts/whisper-artifact-contract.mjs';

const whisperConfig = {
  model_type: 'whisper',
  architectures: ['WhisperForConditionalGeneration'],
};

describe('Whisper artifact capability contract', () => {
  it('requires the explicit causal marker for splitgraph attention DTW', () => {
    const contract = inspectWhisperArtifactContract({
      config: whisperConfig,
      manifest: {
        format: 'whisper-browser-self-export-v1',
        alignment_export: {
          causal_self_attention: true,
          attention_values: 'logits',
          attention_layout: 'selected_heads',
        },
      },
      graphs: [
        {
          path: 'encoder_model.onnx',
          loaded: true,
          input_names: ['input_features'],
          output_names: ['last_hidden_state'],
        },
        {
          path: 'decoder_init.onnx',
          loaded: true,
          input_names: ['input_ids'],
          output_names: ['logits'],
        },
        {
          path: 'decoder_step.onnx',
          loaded: true,
          input_names: ['input_ids'],
          output_names: ['logits'],
        },
        {
          path: 'decoder_align.onnx',
          loaded: true,
          input_names: ['input_ids'],
          output_names: ['alignment'],
        },
      ],
    });

    expect(contract.layout).toBe('splitgraph');
    expect(contract.alignment.claim).toBe('splitgraph-causal-attention-dtw');
    expect(contract.alignment.causal_self_attention_verified).toBe(true);
  });

  it('downgrades legacy splitgraph alignment without a causal marker', () => {
    const contract = inspectWhisperArtifactContract({
      config: whisperConfig,
      manifest: { format: 'whisper-browser-self-export-v1' },
      graphs: [
        { path: 'decoder_init.onnx', loaded: true, input_names: [], output_names: ['logits'] },
        { path: 'decoder_step.onnx', loaded: true, input_names: [], output_names: ['logits'] },
        { path: 'decoder_align.onnx', loaded: true, input_names: [], output_names: ['alignment'] },
      ],
    });

    expect(contract.alignment.claim).toBe('splitgraph-generated-timestamp-fallback');
    expect(contract.alignment.causal_self_attention_marker_present).toBe(false);
    expect(
      contract.checks.find((check) => check.id === 'splitgraph-causal-alignment-marker')?.status,
    ).toBe('warn');
  });

  it('requires cross-attention outputs from every merged decoder variant', () => {
    const contract = inspectWhisperArtifactContract({
      config: whisperConfig,
      graphs: [
        {
          path: 'onnx/decoder_model_merged.onnx',
          loaded: true,
          input_names: ['input_ids'],
          output_names: ['logits', 'cross_attentions.0', 'cross_attentions.1'],
        },
        {
          path: 'onnx/decoder_model_merged_quantized.onnx',
          loaded: true,
          input_names: ['input_ids'],
          output_names: ['logits'],
        },
      ],
    });

    expect(contract.layout).toBe('merged');
    expect(contract.merged_cross_attention_available).toBe(true);
    expect(contract.merged_cross_attention_verified).toBe(false);
    expect(contract.alignment.claim).toBe('merged-generated-timestamp-fallback');
  });
});

/**
 * Derive the timestamp/alignment capability of a local Whisper artifact.
 *
 * This is deliberately metadata-only. It does not run inference and it never
 * treats a decoder_align filename or a generic merged decoder as proof of
 * causal attention alignment.
 */

function basename(value) {
  return (
    String(value ?? '')
      .replaceAll('\\', '/')
      .split('/')
      .at(-1) ?? ''
  );
}

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function graphName(graph) {
  return basename(graph?.path);
}

function isWhisperConfig(config) {
  if (!isRecord(config)) return false;
  if (config.model_type === 'whisper') return true;
  return (
    Array.isArray(config.architectures) &&
    config.architectures.some(
      (value) => typeof value === 'string' && value.toLowerCase().includes('whisper'),
    )
  );
}

function graphSummary(graph) {
  const outputNames = Array.isArray(graph?.output_names) ? graph.output_names : [];
  const crossAttentionOutputs = outputNames
    .filter((name) => typeof name === 'string' && /^cross_attentions\.\d+$/.test(name))
    .sort((left, right) => Number(left.split('.').at(-1)) - Number(right.split('.').at(-1)));
  return {
    path: graph?.path,
    loaded: graph?.loaded === true,
    input_count: Array.isArray(graph?.input_names) ? graph.input_names.length : 0,
    output_count: outputNames.length,
    cross_attention_outputs: crossAttentionOutputs,
    has_cross_attention_outputs: crossAttentionOutputs.length > 0,
  };
}

/**
 * @param {{ config?: object, manifest?: object, graphs?: object[] }} input
 */
export function inspectWhisperArtifactContract(input = {}) {
  const config = isRecord(input.config) ? input.config : undefined;
  const manifest = isRecord(input.manifest) ? input.manifest : undefined;
  const graphs = Array.isArray(input.graphs) ? input.graphs : [];
  const whisperDetected =
    isWhisperConfig(config) || manifest?.format === 'whisper-browser-self-export-v1';

  const decoderInit = graphs.find((graph) => graphName(graph) === 'decoder_init.onnx');
  const decoderStep = graphs.find((graph) => graphName(graph) === 'decoder_step.onnx');
  const decoderAlign = graphs.find((graph) => graphName(graph) === 'decoder_align.onnx');
  const splitGraph = Boolean(decoderInit || decoderStep || decoderAlign);
  const mergedDecoders = graphs
    .filter((graph) => /^decoder_model_merged(?:_.*)?\.onnx$/i.test(graphName(graph)))
    .map(graphSummary);

  const rawAlignmentExport = manifest?.alignment_export;
  const alignmentExport = isRecord(rawAlignmentExport) ? rawAlignmentExport : undefined;
  const markerPresent = typeof alignmentExport?.causal_self_attention === 'boolean';
  const causalSelfAttentionVerified = alignmentExport?.causal_self_attention === true;
  const hasMergedCrossAttention =
    mergedDecoders.length > 0 && mergedDecoders.every((graph) => graph.has_cross_attention_outputs);
  const hasAnyMergedCrossAttention = mergedDecoders.some(
    (graph) => graph.has_cross_attention_outputs,
  );

  let alignmentClaim = 'no-whisper-alignment-artifact';
  if (splitGraph) {
    alignmentClaim =
      causalSelfAttentionVerified && decoderAlign
        ? 'splitgraph-causal-attention-dtw'
        : 'splitgraph-generated-timestamp-fallback';
  } else if (mergedDecoders.length > 0) {
    alignmentClaim = hasMergedCrossAttention
      ? 'merged-cross-attention-dtw-eligible'
      : 'merged-generated-timestamp-fallback';
  }

  const checks = [
    {
      id: 'whisper-model-config',
      status: whisperDetected ? 'pass' : 'warn',
      message: whisperDetected
        ? 'Whisper model metadata detected.'
        : 'Whisper model metadata was not detected; contract claims are informational.',
    },
    {
      id: 'splitgraph-causal-alignment-marker',
      status:
        splitGraph && decoderAlign
          ? causalSelfAttentionVerified
            ? 'pass'
            : 'warn'
          : 'not_applicable',
      message:
        splitGraph && decoderAlign
          ? causalSelfAttentionVerified
            ? 'decoder_align declares causal_self_attention=true.'
            : 'decoder_align is present without an explicit causal_self_attention=true marker.'
          : 'No splitgraph decoder_align artifact is present.',
    },
    {
      id: 'merged-cross-attention-outputs',
      status:
        mergedDecoders.length > 0 ? (hasMergedCrossAttention ? 'pass' : 'warn') : 'not_applicable',
      message:
        mergedDecoders.length > 0
          ? hasMergedCrossAttention
            ? 'Every merged decoder exports cross_attentions.* outputs.'
            : 'At least one merged decoder does not export cross_attentions.*; word timestamps use interpolation.'
          : 'No merged decoder artifact is present.',
    },
  ];

  return {
    schema_version: 1,
    detected: whisperDetected,
    layout:
      splitGraph && mergedDecoders.length > 0
        ? 'mixed'
        : splitGraph
          ? 'splitgraph'
          : mergedDecoders.length > 0
            ? 'merged'
            : 'unknown',
    alignment: {
      claim: alignmentClaim,
      decoder_align_graph: decoderAlign?.path,
      causal_self_attention_marker_present: markerPresent,
      causal_self_attention_verified: Boolean(decoderAlign && causalSelfAttentionVerified),
      attention_values: alignmentExport?.attention_values,
      attention_layout: alignmentExport?.attention_layout,
    },
    merged_decoders: mergedDecoders,
    merged_cross_attention_verified: hasMergedCrossAttention,
    merged_cross_attention_available: hasAnyMergedCrossAttention,
    checks,
  };
}

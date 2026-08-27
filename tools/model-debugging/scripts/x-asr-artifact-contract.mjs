/**
 * Validate the local file contract of the X-ASR streaming deployment.
 *
 * This intentionally checks packaging and graph discovery only. Numerical
 * parity and streaming state semantics require an approved local artifact.
 */
import path from 'node:path';

const VARIANTS = [160, 480, 960, 1920];

function normalize(value) {
  return String(value ?? '').replace(/\\/g, '/');
}

function relative(modelDir, filePath) {
  // Artifact reports may contain Windows paths even when the audit runs in a
  // Linux CI job (or vice versa). Normalize before computing the relative
  // path so contract matching is independent of the host path flavor.
  return normalize(path.posix.relative(normalize(modelDir), normalize(filePath)));
}

function findExact(modelDir, files, expected) {
  return files.find((filePath) => relative(modelDir, filePath) === expected);
}

function findVariantFile(modelDir, files, variantDir, prefix, suffix) {
  const root = `${variantDir}/`;
  return files.find((filePath) => {
    const normalized = relative(modelDir, filePath);
    return (
      normalized.startsWith(root) &&
      normalized.startsWith(`${root}${prefix}`) &&
      normalized.endsWith(suffix)
    );
  });
}

function graphBoundaryCheck(name, graph) {
  if (!graph)
    return {
      status: 'missing',
      code: 'ONNX_GRAPH_INVALID',
      message: `${name} graph was not discovered.`,
    };
  if (!graph.loaded)
    return { status: 'failed', code: 'ONNX_GRAPH_INVALID', message: `${name} graph did not load.` };
  const inputs = graph.input_names ?? [];
  const outputs = graph.output_names ?? [];
  const minimumInputs = name === 'encoder' ? 2 : name === 'joiner' ? 2 : 1;
  const minimumOutputs = name === 'encoder' ? 2 : 1;
  if (inputs.length < minimumInputs || outputs.length < minimumOutputs) {
    return {
      status: 'failed',
      code: 'ONNX_GRAPH_INVALID',
      message: `${name} graph boundary is incomplete (inputs=${inputs.length}, outputs=${outputs.length}).`,
    };
  }
  return {
    status: 'pass',
    code: null,
    message: `${name} graph exposes the expected input/output boundary.`,
  };
}

export function inspectXAsrArtifactContract({ modelDir, files, graphs }) {
  const graphByPath = new Map(graphs.map((graph) => [normalize(graph.path), graph]));
  const variants = VARIANTS.map((chunkMs) => {
    const variantDir = `chunk-${chunkMs}ms-model`;
    const encoder = findVariantFile(modelDir, files, variantDir, 'encoder-', '.onnx');
    const decoder = findVariantFile(modelDir, files, variantDir, 'decoder-', '.onnx');
    const joiner = findVariantFile(modelDir, files, variantDir, 'joiner-', '.onnx');
    const tokens = findExact(modelDir, files, `${variantDir}/tokens.txt`);
    const required = { encoder, decoder, joiner, tokens };
    const missing = Object.entries(required)
      .filter(([, filePath]) => !filePath)
      .map(([name]) => name);
    const graphReports = Object.fromEntries(
      Object.entries({ encoder, decoder, joiner })
        .filter(([, filePath]) => filePath)
        .map(([name, filePath]) => {
          const graph = graphByPath.get(relative(modelDir, filePath));
          return [
            name,
            graph
              ? {
                  path: graph.path,
                  loaded: graph.loaded,
                  input_names: graph.input_names ?? [],
                  output_names: graph.output_names ?? [],
                  input_metadata: graph.input_metadata ?? [],
                  output_metadata: graph.output_metadata ?? [],
                }
              : {
                  path: relative(modelDir, filePath),
                  loaded: false,
                  error: 'Graph was not discovered by the ONNX audit.',
                },
          ];
        }),
    );
    const boundaryChecks = Object.entries(graphReports).map(([name, graph]) => ({
      name,
      ...graphBoundaryCheck(name, graph),
    }));
    return {
      chunk_ms: chunkMs,
      directory: variantDir,
      files: Object.fromEntries(
        Object.entries(required).map(([name, filePath]) => [
          name,
          filePath ? relative(modelDir, filePath) : null,
        ]),
      ),
      missing,
      graphs: graphReports,
      boundary_checks: boundaryChecks,
      tokens_present: Boolean(tokens),
      ok: missing.length === 0 && boundaryChecks.every((check) => check.status === 'pass'),
    };
  });
  const failures = variants.flatMap((variant) => [
    ...variant.missing.map((name) => `${variant.directory}: missing ${name}`),
    ...Object.values(variant.graphs)
      .filter((graph) => !graph.loaded)
      .map((graph) => `${variant.directory}: graph failed to load (${graph.path})`),
    ...variant.boundary_checks
      .filter((check) => check.status !== 'pass')
      .map((check) => `${variant.directory}: ${check.code} ${check.message}`),
  ]);
  return {
    schema_version: 1,
    model: 'X-ASR-zh-en',
    local_only: true,
    expected_chunk_ms: VARIANTS,
    variants,
    failures,
    ok: failures.length === 0,
  };
}

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
  return normalize(path.relative(modelDir, filePath));
}

function findExact(modelDir, files, expected) {
  return files.find((filePath) => relative(modelDir, filePath) === expected);
}

function findVariantFile(modelDir, files, variantDir, prefix, suffix) {
  const root = `${variantDir}/`;
  return files.find((filePath) => {
    const normalized = relative(modelDir, filePath);
    return normalized.startsWith(root) && normalized.startsWith(`${root}${prefix}`) && normalized.endsWith(suffix);
  });
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
    const missing = Object.entries(required).filter(([, filePath]) => !filePath).map(([name]) => name);
    const graphReports = Object.fromEntries(
      Object.entries({ encoder, decoder, joiner })
        .filter(([, filePath]) => filePath)
        .map(([name, filePath]) => {
          const graph = graphByPath.get(relative(modelDir, filePath));
          return [name, graph ? {
            path: graph.path,
            loaded: graph.loaded,
            input_names: graph.input_names ?? [],
            output_names: graph.output_names ?? [],
          } : { path: relative(modelDir, filePath), loaded: false, error: 'Graph was not discovered by the ONNX audit.' }];
        }),
    );
    return {
      chunk_ms: chunkMs,
      directory: variantDir,
      files: Object.fromEntries(Object.entries(required).map(([name, filePath]) => [name, filePath ? relative(modelDir, filePath) : null])),
      missing,
      graphs: graphReports,
      tokens_present: Boolean(tokens),
      ok: missing.length === 0 && Object.values(graphReports).every((graph) => graph.loaded),
    };
  });
  const failures = variants.flatMap((variant) => [
    ...variant.missing.map((name) => `${variant.directory}: missing ${name}`),
    ...Object.values(variant.graphs).filter((graph) => !graph.loaded).map((graph) => `${variant.directory}: graph failed to load (${graph.path})`),
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

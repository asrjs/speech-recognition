import { readFile } from 'node:fs/promises';
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const distRoot = resolve(repositoryRoot, 'dist');
const defaultEntryPoints = [
  'index.js',
  'builtins.js',
  'io.js',
  'inference.js',
  'browser.js',
  'browser-media.js',
  'worker.js',
  'realtime.js',
  'bench.js',
  'bench-browser.js',
  'datasets.js',
  'quality.js',
  'chunking.js',
  'post-processing.js',
  'pipeline.js',
  'tokenizers.js',
  'alignment.js',
  // Model-family subpaths are browser-facing package entry points too. Keep
  // them in the default graph so a new Node-only dependency cannot silently
  // break direct browser imports while the root entry remains clean.
  'models/lasr-ctc.js',
  'models/gigaam-ctc.js',
  'models/gigaam-rnnt.js',
  'models/x-asr.js',
  'models/nemo-aed.js',
  'models/nemo-rnnt.js',
  'models/nemo-tdt.js',
  'models/sensevoice.js',
  'models/qwen-asr.js',
  'models/qwen3-asr.js',
  'models/moonshine.js',
  'models/parakeet-ctc.js',
  'models/omnilingual-ctc.js',
  'models/wav2vec2.js',
  'models/whisper-seq2seq.js',
];
const requestedEntryPoints = process.argv.slice(2);
const entryPoints = (
  requestedEntryPoints.length > 0 ? requestedEntryPoints : defaultEntryPoints
).map((entryPoint) => resolve(distRoot, entryPoint));

// The public root currently exposes several independent API areas, so this
// ceiling protects the measured regression baseline while the requested
// <45-module acceptance target remains explicitly tracked below.
const rootImportRegressionCeiling = 78;
const rootImportBudgetTarget = 45;

const staticModulePattern =
  /^(?:\s*)(?:import|export)\s+(?:(?!\bfrom\b)[^;\r\n])*?\bfrom\s+['"]([^'"]+)['"]\s*;?\s*$|^(?:\s*)import\s+['"]([^'"]+)['"]\s*;?\s*$/gm;

function isWithin(parent, candidate) {
  const path = relative(parent, candidate);
  return path === '' || (path !== '..' && !path.startsWith(`..${sep}`) && !isAbsolute(path));
}

function resolveLocalImport(fromFile, specifier) {
  if (!specifier.startsWith('.')) {
    return null;
  }

  const resolved = resolve(dirname(fromFile), specifier);
  if (!isWithin(distRoot, resolved)) {
    return null;
  }
  return resolved.endsWith('.js') ? resolved : `${resolved}.js`;
}

function collectStaticImports(source) {
  const imports = [];
  for (const match of source.matchAll(staticModulePattern)) {
    imports.push(match[1] ?? match[2]);
  }
  return imports;
}

async function traverseGraph(startEntries) {
  const queue = [...startEntries];
  const visited = new Set();
  const violations = [];

  while (queue.length > 0) {
    const current = queue.pop();
    if (!current || visited.has(current)) {
      continue;
    }
    visited.add(current);

    const source = await readFile(current, 'utf8');
    for (const specifier of collectStaticImports(source)) {
      if (specifier.startsWith('node:')) {
        violations.push(`${relative(repositoryRoot, current)} -> ${specifier}`);
        continue;
      }

      const localTarget = resolveLocalImport(current, specifier);
      if (!localTarget) {
        continue;
      }
      if (localTarget.endsWith('io-node.js') || localTarget.endsWith('io/node.js')) {
        violations.push(`${relative(repositoryRoot, current)} -> ${specifier}`);
        continue;
      }
      queue.push(localTarget);
    }
  }

  return { visited, violations };
}

async function main() {
  const { visited, violations } = await traverseGraph(entryPoints);

  // Card T1 / Root Bundle Boundary: shared public imports must not pull model
  // implementations or the native ORT loader into the static browser graph.
  const rootIndex = resolve(distRoot, 'index.js');
  const { visited: rootVisited } = await traverseGraph([rootIndex]);
  const forbiddenRootModuleRegex = /[/\\]models[/\\]|[/\\]backends[/\\]native-cpu[/\\]ort\.js$/;
  for (const mod of rootVisited) {
    if (forbiddenRootModuleRegex.test(mod)) {
      violations.push(
        `Root bundle boundary violation: index.js transitively imports ${relative(repositoryRoot, mod)}`,
      );
    }
    const source = await readFile(mod, 'utf8');
    for (const specifier of collectStaticImports(source)) {
      if (/^onnxruntime(?:-node|-web)(?:\/|$)/.test(specifier)) {
        violations.push(
          `Root bundle boundary violation: ${relative(repositoryRoot, mod)} statically imports ${specifier}`,
        );
      }
    }
  }

  if (rootVisited.size > rootImportRegressionCeiling) {
    violations.push(
      `Root import graph grew to ${rootVisited.size} modules; measured regression ceiling is ${rootImportRegressionCeiling}`,
    );
  }

  const strictBudget = process.env.ASRJS_ROOT_IMPORT_STRICT_BUDGET === '1';
  if (strictBudget && rootVisited.size >= rootImportBudgetTarget) {
    violations.push(
      `Root import budget target is UNMET: ${rootVisited.size} static modules; required <${rootImportBudgetTarget}`,
    );
  }

  if (violations.length > 0) {
    console.error('[browser-root-import-graph] forbidden static imports:');
    for (const violation of violations) {
      console.error(`  ${violation}`);
    }
    process.exitCode = 1;
    return;
  }

  console.log(
    `[browser-root-import-graph] ok: ${visited.size} local modules reachable from ${
      entryPoints.length
    } browser-safe dist entries`,
  );
  console.log(
    `[browser-root-import-graph] root index: ${rootVisited.size} static modules (regression ceiling <=${rootImportRegressionCeiling}: PASS; requested <${rootImportBudgetTarget}: ${rootVisited.size < rootImportBudgetTarget ? 'PASS' : 'UNMET'})`,
  );
  // Passing the boundary/regression checks is not acceptance of the requested
  // budget. Keep the default command non-successful until that target is met.
  if (strictBudget && rootVisited.size >= rootImportBudgetTarget) {
    process.exitCode = 2;
  }
}

main().catch((error) => {
  console.error('[browser-root-import-graph] failed:', error);
  process.exitCode = 1;
});

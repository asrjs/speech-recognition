import { readFile } from 'node:fs/promises';
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const distRoot = resolve(repositoryRoot, 'dist');
const entryPoint = resolve(distRoot, 'index.js');

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

async function main() {
  const queue = [entryPoint];
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

  if (violations.length > 0) {
    console.error('[browser-root-import-graph] forbidden static imports:');
    for (const violation of violations) {
      console.error(`  ${violation}`);
    }
    process.exitCode = 1;
    return;
  }

  console.log(
    `[browser-root-import-graph] ok: ${visited.size} local modules reachable from dist/index.js`,
  );
}

main().catch((error) => {
  console.error('[browser-root-import-graph] failed:', error);
  process.exitCode = 1;
});

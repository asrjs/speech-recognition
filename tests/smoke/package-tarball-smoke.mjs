import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repositoryRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const packageJsonPath = resolve(repositoryRoot, 'package.json');
const packageJson = JSON.parse(readFileSync(packageJsonPath, 'utf8'));
const npmCommand = process.platform === 'win32' ? (process.env.ComSpec ?? 'cmd.exe') : 'npm';
const npmArgs =
  process.platform === 'win32'
    ? ['/d', '/s', '/c', 'npm', 'pack', '--json', '--dry-run']
    : ['pack', '--json', '--dry-run'];

function normalizePackagePath(path) {
  return path.replaceAll('\\', '/').replace(/^\.\//, '');
}

function collectExportTargets(value, targets = []) {
  if (typeof value === 'string') {
    targets.push(value);
    return targets;
  }

  if (value && typeof value === 'object') {
    for (const nestedValue of Object.values(value)) {
      collectExportTargets(nestedValue, targets);
    }
  }

  return targets;
}

function runPackDryRun() {
  const output = execFileSync(npmCommand, npmArgs, {
    cwd: repositoryRoot,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'inherit'],
  });
  const result = JSON.parse(output);
  assert.equal(result.length, 1, 'npm pack should describe exactly one package');
  return result[0];
}

function assertPublishedFile(fileMap, target, reason) {
  const normalizedTarget = normalizePackagePath(target);
  assert.ok(!normalizedTarget.includes('*'), `wildcard target must be expanded: ${target}`);
  const entry = fileMap.get(normalizedTarget);
  assert.ok(entry, `${reason} is missing from npm pack: ${normalizedTarget}`);
  assert.ok(entry.size > 0, `${reason} is empty in npm pack: ${normalizedTarget}`);
}

function assertDeclarationPair(fileMap, runtimePath, reason) {
  const declarationPath = runtimePath.replace(/\.js$/, '.d.ts');
  assertPublishedFile(fileMap, runtimePath, reason);
  assertPublishedFile(fileMap, declarationPath, `${reason} declaration`);
}

function assertWildcardCoverage(fileMap, prefix, extension, reason) {
  const runtimeFiles = [...fileMap.keys()].filter(
    (path) =>
      path.startsWith(prefix) &&
      path.endsWith(extension) &&
      !path.slice(prefix.length).includes('/'),
  );
  assert.ok(runtimeFiles.length > 0, `${reason} has no published runtime files`);
  for (const runtimePath of runtimeFiles) {
    assertDeclarationPair(fileMap, runtimePath, `${reason} ${runtimePath}`);
  }
}

function main() {
  const packed = runPackDryRun();
  assert.equal(packed.name, packageJson.name);
  assert.equal(packed.version, packageJson.version);

  const fileMap = new Map(packed.files.map((entry) => [normalizePackagePath(entry.path), entry]));
  assertPublishedFile(fileMap, 'README.md', 'README');
  assertPublishedFile(fileMap, 'package.json', 'package metadata');

  for (const path of fileMap.keys()) {
    const isTypeScriptSource = /\.tsx?(?:\.map)?$/.test(path) && !/\.d\.ts(?:\.map)?$/.test(path);
    assert.ok(!isTypeScriptSource, `source leaked into npm pack: ${path}`);
    assert.ok(!path.startsWith('tests/'), `tests leaked into npm pack: ${path}`);
    assert.ok(!path.startsWith('docs/'), `docs leaked into npm pack: ${path}`);
  }

  const exportTargets = collectExportTargets(packageJson.exports);
  for (const target of exportTargets) {
    if (target.includes('*')) {
      continue;
    }
    assertPublishedFile(fileMap, target, `export target ${target}`);
  }
  assertPublishedFile(fileMap, packageJson.main, 'main entry');
  assertPublishedFile(fileMap, packageJson.types, 'types entry');

  assertWildcardCoverage(fileMap, 'dist/models/', '.js', 'models wildcard');
  assertWildcardCoverage(fileMap, 'dist/presets/', '.js', 'presets wildcard');

  for (const modelRuntimePath of [
    'dist/models/whisper-seq2seq/index.js',
    'dist/models/gigaam-rnnt/index.js',
  ]) {
    assertDeclarationPair(fileMap, modelRuntimePath, 'explicit model subpath');
  }

  console.log(
    `[package-tarball-smoke] ok: ${packed.entryCount} published files; all export targets and declarations are present`,
  );
}

try {
  main();
} catch (error) {
  console.error('[package-tarball-smoke] failed:', error);
  process.exitCode = 1;
}

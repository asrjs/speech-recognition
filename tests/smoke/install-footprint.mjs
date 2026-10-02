// Offline package-manager measurement only; never loads native bindings or ASR models.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import {
  closeSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  readFileSync,
  readdirSync,
  writeFileSync,
  writeSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parsePackDryRun } from './package-tarball-smoke.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');

export function consumerManifest(metadata, archive, native) {
  const version = metadata.peerDependencies?.['onnxruntime-node'];
  assert.equal(typeof version, 'string', 'Missing pinned native peer');
  assert.match(
    version,
    /^\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?$/,
    'Native peer must be an exact version',
  );
  assert.equal(metadata.peerDependenciesMeta?.['onnxruntime-node']?.optional, true);
  return {
    name: native ? 'asrjs-footprint-native' : 'asrjs-footprint-browser',
    private: true,
    type: 'module',
    dependencies: {
      [metadata.name]: `file:${archive}`,
      ...(native ? { 'onnxruntime-node': version } : {}),
    },
  };
}

export function measureTree(directory) {
  const total = { files: 0, bytes: 0 };
  function visit(path) {
    const stat = lstatSync(path);
    assert.ok(!stat.isSymbolicLink(), 'Installed footprint must not follow a link');
    if (stat.isDirectory()) for (const name of readdirSync(path).sort()) visit(join(path, name));
    else {
      assert.ok(stat.isFile(), 'Installed footprint contains a non-file');
      total.files++;
      total.bytes += stat.size;
      assert.ok(Number.isSafeInteger(total.bytes));
    }
  }
  visit(directory);
  return total;
}

export function compareInstalls(browser, native) {
  if (browser.status !== 'PASS_INSTALL_ONLY' || native.status !== 'PASS_INSTALL_ONLY') return null;
  for (const run of [browser, native])
    for (const key of ['files', 'bytes'])
      assert.ok(Number.isSafeInteger(run.footprint?.[key]) && run.footprint[key] >= 0);
  return {
    files: native.footprint.files - browser.footprint.files,
    bytes: native.footprint.bytes - browser.footprint.bytes,
  };
}

export function reserveReceipt(output) {
  return openSync(output, 'wx');
}

function npm(args, cwd, env) {
  const cli =
    env.ASRJS_NPM_CLI ?? join(dirname(process.execPath), 'node_modules/npm/bin/npm-cli.js');
  assert.ok(lstatSync(cli).isFile(), 'Set ASRJS_NPM_CLI to the installed npm-cli.js');
  const result = spawnSync(process.execPath, [cli, ...args], {
    cwd,
    env,
    encoding: 'utf8',
    timeout: 60000,
    maxBuffer: 4 * 1024 * 1024,
  });
  if (result.error) throw result.error;
  assert.equal(result.signal, null, 'npm terminated by signal');
  if (result.status !== 0)
    throw new Error(`npm ${args[0]} exited ${result.status}: ${result.stderr}`);
  return result.stdout;
}

function installedConsumer(directory, archive, metadata, native, env) {
  const manifest = consumerManifest(metadata, `../${basename(archive)}`, native);
  mkdirSync(directory);
  writeFileSync(join(directory, 'package.json'), JSON.stringify(manifest), { flag: 'wx' });
  const run = { status: 'FAIL_INSTALL', directory, nativeRequested: native, manifest };
  try {
    npm(
      ['install', '--offline', '--ignore-scripts', '--omit=dev', '--no-audit', '--no-fund'],
      directory,
      env,
    );
    const installed = JSON.parse(
      readFileSync(join(directory, 'node_modules', metadata.name, 'package.json')),
    );
    assert.equal(installed.name, metadata.name);
    assert.equal(installed.version, metadata.version);
    const require = createRequire(join(directory, 'package.json'));
    let nativePath = null;
    try {
      nativePath = require.resolve('onnxruntime-node');
    } catch (error) {
      if (error.code !== 'MODULE_NOT_FOUND') throw error;
    }
    assert.equal(
      nativePath !== null,
      native,
      'Actual native resolution differs from requested install',
    );
    if (nativePath) {
      const local = relative(join(directory, 'node_modules'), nativePath);
      assert.ok(
        local && local !== '..' && !local.startsWith(`..${sep}`),
        'Native resolution escaped consumer',
      );
      const peer = JSON.parse(
        readFileSync(join(directory, 'node_modules/onnxruntime-node/package.json')),
      );
      assert.equal(peer.version, metadata.peerDependencies['onnxruntime-node']);
    }
    const lockBytes = readFileSync(join(directory, 'package-lock.json'));
    run.lockSha256 = sha(lockBytes);
    run.packages = Object.fromEntries(
      Object.entries(JSON.parse(lockBytes).packages)
        .filter(([path]) => path)
        .map(([path, value]) => [path, value.version]),
    );
    run.nativeResolution = nativePath;
    run.footprint = measureTree(join(directory, 'node_modules'));
    run.status = 'PASS_INSTALL_ONLY';
  } catch (error) {
    run.error = String(error);
  }
  return run;
}

export function main(args = process.argv.slice(2)) {
  assert.equal(args.length, 2, 'Usage: node tests/smoke/install-footprint.mjs --output NEW.json');
  assert.equal(args[0], '--output');
  const output = resolve(args[1]);
  const fd = reserveReceipt(output); // Refuse overwrite before any packing or installation.
  const report = {
    schemaVersion: 1,
    status: 'FAIL_SETUP',
    scope: 'OFFLINE_INSTALL_FOOTPRINT_ONLY',
    scriptsExecuted: false,
    networkAllowed: false,
    node: process.version,
    platform: process.platform,
    toolSha256: sha(readFileSync(fileURLToPath(import.meta.url))),
    limitations: [
      'Not inference, runtime/provider acceptance, or a published registry installation',
      'Logical file bytes, not allocated disk blocks; install scripts are disabled',
    ],
  };
  try {
    const packageBytes = readFileSync(join(root, 'package.json'));
    const metadata = JSON.parse(packageBytes);
    consumerManifest(metadata, 'unused.tgz', false);
    assert.ok(lstatSync(join(root, 'dist/index.js')).isFile(), 'Build first');
    const fixture = mkdtempSync(join(tmpdir(), 'asrjs-offline-install-'));
    report.fixture = fixture; // Retain both consumers for independent inspection.
    const env = {
      ...process.env,
      NODE_TLS_REJECT_UNAUTHORIZED: '1',
      npm_config_offline: 'true',
      npm_config_ignore_scripts: 'true',
      npm_config_audit: 'false',
      npm_config_fund: 'false',
      npm_config_update_notifier: 'false',
    };
    delete env.NODE_PATH;
    report.npm = npm(['--version'], root, env).trim();
    report.cache = npm(['config', 'get', 'cache'], root, env).trim();
    report.packageJsonSha256 = sha(packageBytes);
    // Launch npm's JS CLI without a shell, including on Windows/paths with spaces.
    const packed = parsePackDryRun(
      npm(['pack', root, '--json', '--offline', '--ignore-scripts'], fixture, env),
    );
    assert.equal(basename(packed.filename), packed.filename);
    const archive = join(fixture, packed.filename);
    const archiveBytes = readFileSync(archive);
    assert.equal(archiveBytes.length, packed.size);
    report.archive = {
      sha256: sha(archiveBytes),
      bytes: packed.size,
      unpackedBytes: packed.unpackedSize,
      files: packed.entryCount,
    };
    report.browser = installedConsumer(join(fixture, 'browser'), archive, metadata, false, env);
    report.native = installedConsumer(join(fixture, 'native'), archive, metadata, true, env);
    report.delta = compareInstalls(report.browser, report.native);
    report.status = report.delta
      ? 'PASS_OFFLINE_INSTALL_COMPARISON_ONLY'
      : 'FAIL_INSTALL_COMPARISON';
    if (report.delta)
      for (const [path, version] of Object.entries(report.browser.packages))
        assert.equal(report.native.packages[path], version, `Baseline dependency changed: ${path}`);
    assert.equal(
      sha(readFileSync(join(root, 'package.json'))),
      report.packageJsonSha256,
      'Package metadata changed during measurement',
    );
    assert.equal(
      sha(readFileSync(archive)),
      report.archive.sha256,
      'Archive changed during installation',
    );
    assert.equal(
      sha(readFileSync(fileURLToPath(import.meta.url))),
      report.toolSha256,
      'Measurement tool changed during execution',
    );
  } catch (error) {
    report.status = 'FAIL_SETUP';
    report.delta = null;
    report.error = String(error);
  } finally {
    try {
      writeSync(fd, JSON.stringify(report, null, 2));
    } finally {
      closeSync(fd);
    }
  }
  console.log(
    JSON.stringify({ status: report.status, output, fixture: report.fixture, delta: report.delta }),
  );
  return report.status === 'PASS_OFFLINE_INSTALL_COMPARISON_ONLY' ? 0 : 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    process.exitCode = main();
  } catch (error) {
    console.error(String(error));
    process.exitCode = 1;
  }
}

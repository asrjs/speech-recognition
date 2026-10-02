import { mkdtempSync, mkdirSync, writeFileSync, symlinkSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { describe, expect, it } from 'vitest';
import {
  consumerManifest,
  measureTree,
  compareInstalls,
  reserveReceipt,
} from './smoke/install-footprint.mjs';

const metadata = {
  name: '@asrjs/speech-recognition',
  version: '0.2.0',
  peerDependencies: { 'onnxruntime-node': '1.25.0-dev.20260228-6e72d31970' },
  peerDependenciesMeta: { 'onnxruntime-node': { optional: true } },
};

describe('offline installed footprint controls (no installation or inference)', () => {
  it('uses the same archive and only explicitly adds the pinned native peer', () => {
    const browser = consumerManifest(metadata, '../speech.tgz', false);
    const native = consumerManifest(metadata, '../speech.tgz', true);
    expect(browser.dependencies).toEqual({ [metadata.name]: 'file:../speech.tgz' });
    expect(native.dependencies).toEqual({ ...browser.dependencies, ...metadata.peerDependencies });
    expect(browser.private).toBe(true);
    expect(Object.hasOwn(browser, 'scripts')).toBe(false);
  });

  it('fails closed on missing or nonoptional native peer contracts', () => {
    expect(() => consumerManifest({ ...metadata, peerDependencies: {} }, 'a.tgz', true)).toThrow();
    expect(() =>
      consumerManifest({ ...metadata, peerDependenciesMeta: {} }, 'a.tgz', false),
    ).toThrow();
  });

  it('counts file bytes, including hidden metadata, and rejects external links', () => {
    const root = mkdtempSync(join(tmpdir(), 'asrjs-footprint-unit-'));
    mkdirSync(join(root, 'pkg'));
    writeFileSync(join(root, 'pkg/a'), 'abc');
    writeFileSync(join(root, '.package-lock.json'), '12345');
    expect(measureTree(root)).toEqual({ files: 2, bytes: 8 });
    symlinkSync(join(root, 'pkg'), join(root, 'linked'), 'junction');
    expect(() => measureTree(root)).toThrow('link');
  });

  it('does not turn partial/missing/failed installs into footprint success', () => {
    const pass = { status: 'PASS_INSTALL_ONLY', footprint: { files: 10, bytes: 100 } };
    expect(compareInstalls(pass, { ...pass, footprint: { files: 12, bytes: 145 } })).toEqual({
      files: 2,
      bytes: 45,
    });
    expect(
      compareInstalls(pass, { status: 'FAIL_INSTALL', footprint: { files: 12, bytes: 145 } }),
    ).toBeNull();
    expect(compareInstalls({ status: 'FAIL_INSTALL' }, pass)).toBeNull();
    expect(() => compareInstalls(pass, { ...pass, footprint: { files: -1, bytes: 0 } })).toThrow();
  });

  it('refuses an existing receipt without modifying its evidence', () => {
    const root = mkdtempSync(join(tmpdir(), 'asrjs-footprint-receipt-'));
    const output = join(root, 'receipt.json');
    writeFileSync(output, 'preserve');
    expect(() => reserveReceipt(output)).toThrow();
    expect(readFileSync(output, 'utf8')).toBe('preserve');
  });

  it('actual CLI rejects existing evidence before package-manager work', () => {
    const root = mkdtempSync(join(tmpdir(), 'asrjs-footprint-cli-'));
    const output = join(root, 'receipt.json');
    writeFileSync(output, 'preserve CLI evidence');
    const result = spawnSync(
      process.execPath,
      ['tests/smoke/install-footprint.mjs', '--output', output],
      {
        encoding: 'utf8',
        timeout: 10000,
        env: { ...process.env, ASRJS_NPM_CLI: 'intentionally missing' },
      },
    );
    expect(result.error).toBeUndefined();
    expect(result.signal).toBeNull();
    expect(result.status).toBe(1);
    expect(result.stderr).toContain('EEXIST');
    expect(result.stdout).not.toContain('PASS');
    expect(readFileSync(output, 'utf8')).toBe('preserve CLI evidence');
  });

  it('actual CLI records requested missing npm as FAIL, never SKIP or a zero footprint', () => {
    const root = mkdtempSync(join(tmpdir(), 'asrjs-footprint-cli-'));
    const output = join(root, 'receipt.json');
    const result = spawnSync(
      process.execPath,
      ['tests/smoke/install-footprint.mjs', '--output', output],
      {
        encoding: 'utf8',
        timeout: 10000,
        env: { ...process.env, ASRJS_NPM_CLI: join(root, 'absent-npm.js') },
      },
    );
    expect(result.error).toBeUndefined();
    expect(result.signal).toBeNull();
    expect(result.status).toBe(1);
    const report = JSON.parse(readFileSync(output, 'utf8'));
    expect(report.status).toBe('FAIL_SETUP');
    expect(report.delta).toBeNull();
    expect(report.error).toContain('ENOENT');
    expect(report.browser).toBeUndefined();
    expect(report.toolSha256).toMatch(/^[0-9a-f]{64}$/);
  });
});

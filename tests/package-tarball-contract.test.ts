import { describe, expect, it } from 'vitest';
import * as smoke from './smoke/package-tarball-smoke.mjs';

const metadata = { name: '@asrjs/speech-recognition', version: '0.2.0' };
const packed = () => ({
  ...metadata,
  files: [{ path: 'package.json', size: 10 }],
  entryCount: 1,
});

describe('npm pack dry-run contract (not publish acceptance)', () => {
  it('accepts the legacy one-package array', () => {
    expect(smoke.parsePackDryRun(JSON.stringify([packed()]), metadata)).toEqual(packed());
  });

  it('accepts npm 12 output keyed by the exact package name', () => {
    expect(smoke.parsePackDryRun(JSON.stringify({ [metadata.name]: packed() }), metadata)).toEqual(
      packed(),
    );
  });

  it.each(
    [
      null,
      [],
      [packed(), packed()],
      {},
      { wrong: packed() },
      { [metadata.name]: packed(), another: packed() },
      [{ ...packed(), name: 'another-package' }],
      [{ ...packed(), version: 'wrong-version' }],
      [{ ...packed(), files: null }],
      [{ ...packed(), files: [] }],
      [{ ...packed(), files: [{ path: '', size: 10 }] }],
      [{ ...packed(), files: [{ path: 'package.json', size: null }] }],
      [{ ...packed(), entryCount: 2 }],
    ].map((value) => ({ value })),
  )('rejects malformed, multiple or mismatched package reports: $value', ({ value }) => {
    expect(() => smoke.parsePackDryRun(JSON.stringify(value), metadata)).toThrow('npm pack');
  });
});

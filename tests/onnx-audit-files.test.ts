import {
  shouldExcludeAuditFile,
  shouldSkipAuditDirectory,
} from '../tools/model-debugging/scripts/node-audit-onnx-artifact.mjs';
import { describe, expect, it } from 'vitest';

describe('ONNX artifact audit file selection', () => {
  it('skips repository and dependency directories case-insensitively', () => {
    expect(shouldSkipAuditDirectory('.git')).toBe(true);
    expect(shouldSkipAuditDirectory('NODE_MODULES')).toBe(true);
    expect(shouldSkipAuditDirectory('chunk-160ms-model')).toBe(false);
  });

  it('excludes only the requested output path', () => {
    expect(shouldExcludeAuditFile('C:\\models\\report.json', 'C:\\models\\report.json')).toBe(true);
    expect(shouldExcludeAuditFile('C:\\models\\other.json', 'C:\\models\\report.json')).toBe(false);
    expect(shouldExcludeAuditFile('C:\\models\\report.json', undefined)).toBe(false);
  });
});

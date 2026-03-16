import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

export function resolveArtifactPath(refDir, fileName) {
  const rawPath = path.join(refDir, fileName);
  if (fs.existsSync(rawPath)) {
    return rawPath;
  }

  const gzipPath = `${rawPath}.gz`;
  if (fs.existsSync(gzipPath)) {
    return gzipPath;
  }

  return null;
}

export function hasArtifact(refDir, fileName) {
  return resolveArtifactPath(refDir, fileName) !== null;
}

export function readJsonArtifact(refDir, fileName) {
  const resolvedPath = resolveArtifactPath(refDir, fileName);
  if (!resolvedPath) {
    throw new Error(`Reference artifact not found: ${path.join(refDir, fileName)}`);
  }

  const raw = fs.readFileSync(resolvedPath);
  const decoded = resolvedPath.endsWith('.gz') ? zlib.gunzipSync(raw).toString('utf8') : raw.toString('utf8');
  return JSON.parse(decoded);
}

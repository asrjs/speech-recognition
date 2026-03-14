import { defineConfig, searchForWorkspaceRoot } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import fs from 'fs';
import { execSync } from 'child_process';

function readJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
}

function getShortCommitHash(repoRoot) {
  try {
    return execSync('git rev-parse --short HEAD', {
      encoding: 'utf8',
      cwd: repoRoot,
    }).trim();
  } catch {
    return null;
  }
}

const repoRoot = path.resolve(__dirname, '../..');
const localPkg = readJson(path.resolve(repoRoot, 'package.json'));
const localVersion = localPkg?.version;

const shortHash = getShortCommitHash(repoRoot);
const asrVersion = localVersion || 'unknown';
const asrSource = shortHash ? `dev-${shortHash}` : 'dev';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    fs: {
      // Allow serving files from the repo root (needed for PARAKEET_LOCAL mode).
      allow: [
        searchForWorkspaceRoot(process.cwd()),
      ],
    },
  },
  resolve: {
    alias: {
      // This demo always exercises the local asr.js source tree.
      'asr.js': path.resolve(__dirname, '../../src/index.ts'),
    },
  },
  define: {
    global: 'globalThis',
    __PARAKEET_VERSION__: JSON.stringify(asrVersion),
    __PARAKEET_SOURCE__: JSON.stringify(asrSource),
  },
});

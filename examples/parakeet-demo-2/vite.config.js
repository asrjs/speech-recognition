import { defineConfig, searchForWorkspaceRoot } from 'vite';
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
const pkg = readJson(path.resolve(repoRoot, 'package.json'));
const shortHash = getShortCommitHash(repoRoot);

export default defineConfig({
  server: {
    port: 5174,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    fs: {
      allow: [searchForWorkspaceRoot(process.cwd())],
    },
  },
  resolve: {
    alias: {
      'asr.js': path.resolve(__dirname, '../../src/index.ts'),
    },
  },
  define: {
    global: 'globalThis',
    __ASRJS_VERSION__: JSON.stringify(pkg?.version || 'unknown'),
    __ASRJS_SOURCE__: JSON.stringify(shortHash ? `dev-${shortHash}` : 'dev'),
  },
});

import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';

const certDir = fileURLToPath(new URL('./certs', import.meta.url));
const nodeRuntimeStub = fileURLToPath(new URL('./src/node-runtime-stub.js', import.meta.url));

// Source-checkout dev paths follow this config, never the shell's cwd or a drive.
export function resolveDemoPaths(configUrl) {
  const repoRoot = fileURLToPath(new URL('../../', configUrl));
  const ortDist = fileURLToPath(new URL('../../node_modules/onnxruntime-web/dist/', configUrl));
  const encodedDist = ortDist
    .replace(/\\/g, '/')
    .split('/')
    .map((part) => encodeURIComponent(part).replace(/%3A/gi, ':'))
    .join('/');
  return { repoRoot, wasmBase: `/@fs/${encodedDist}` };
}

const demoPaths = resolveDemoPaths(import.meta.url);

export default defineConfig({
  define: {
    'import.meta.env.ASRJS_ORT_WASM_BASE': JSON.stringify(demoPaths.wasmBase),
  },
  resolve: {
    alias: [
      {
        find: '../../io/node.js',
        replacement: nodeRuntimeStub,
      },
    ],
  },
  server: {
    fs: {
      allow: [demoPaths.repoRoot, 'N:/models'],
    },
    https: fs.existsSync(`${certDir}/key.pem`)
      ? { key: fs.readFileSync(`${certDir}/key.pem`), cert: fs.readFileSync(`${certDir}/cert.pem`) }
      : false,
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
});

import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';

const certDir = fileURLToPath(new URL('./certs', import.meta.url));
const nodeRuntimeStub = fileURLToPath(new URL('./src/node-runtime-stub.js', import.meta.url));

export default defineConfig({
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
      allow: ['N:/github/asrjs/speech-recognition', 'N:/models'],
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
